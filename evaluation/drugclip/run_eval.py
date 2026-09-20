"""DrugCLIP zero-shot scoring and CV-selected projection-head fine-tuning."""
import argparse, json, os, sys
import numpy as np, pandas as pd, torch, torch.nn as nn
from sklearn.model_selection import StratifiedKFold

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics import all_metrics
import dc_model as D

here = os.environ["COMPARATOR_WORKDIR"]
import verify_tables
TARGETS = ["588689", "540297-493091", "434954-2097", "463203-2650",
           "493248-485317", "504329", "624273-588549", "1053173-743445"]
BUDGETS = [40, 100, 300]


class FocalLoss(nn.Module):
    """Binary focal loss."""
    def __init__(self, alpha=0.8, gamma=2.0):
        super().__init__()
        self.alpha, self.gamma = alpha, gamma

    def forward(self, logit, y):
        p = torch.sigmoid(logit)
        ce = nn.functional.binary_cross_entropy_with_logits(logit, y, reduction="none")
        pt = p * y + (1 - p) * (1 - y)
        at = self.alpha * y + (1 - self.alpha) * (1 - y)
        return (at * (1 - pt) ** self.gamma * ce).mean()


def head_ft(model, rep_tr, y_tr, rep_ev, poc_rep, seed, epochs=200, lr=1e-3,
            l2sp=1e-3, device="cuda"):
    """Fine-tune mol_project / pocket_project / logit_scale on N labelled compounds."""
    import copy
    torch.manual_seed(seed)
    mp = copy.deepcopy(model.mol_project).to(device).float()
    pp = copy.deepcopy(model.pocket_project).to(device).float()
    scale = nn.Parameter(model.logit_scale.detach().clone().to(device).float())
    # Penalize trainable projection parameters against their pretrained values.
    ref = {("M." + k): v.detach().clone() for k, v in mp.named_parameters()}
    ref.update({("P." + k): v.detach().clone() for k, v in pp.named_parameters()})
    params = list(mp.parameters()) + list(pp.parameters()) + [scale]
    opt = torch.optim.Adam(params, lr=lr)
    loss_fn = FocalLoss()
    xt = torch.tensor(rep_tr, dtype=torch.float32, device=device)
    yt = torch.tensor(y_tr, dtype=torch.float32, device=device)
    pr = torch.tensor(poc_rep, dtype=torch.float32, device=device)[None]
    mp.train(); pp.train()
    for _ in range(epochs):
        opt.zero_grad()
        m = mp(xt); m = m / m.norm(dim=1, keepdim=True)
        q = pp(pr); q = q / q.norm(dim=1, keepdim=True)
        logit = scale.exp() * (m @ q.T).squeeze(1)
        loss = loss_fn(logit, yt)
        if l2sp:
            pen = sum(((q - ref["M." + k]) ** 2).sum() for k, q in mp.named_parameters())
            pen = pen + sum(((q - ref["P." + k]) ** 2).sum() for k, q in pp.named_parameters())
            loss = loss + l2sp * pen
        loss.backward()
        opt.step()
    mp.eval(); pp.eval()
    with torch.no_grad():
        q = pp(pr); q = q / q.norm(dim=1, keepdim=True)
        out = []
        for i in range(0, len(rep_ev), 20000):
            x = torch.tensor(rep_ev[i:i + 20000], dtype=torch.float32, device=device)
            m = mp(x); m = m / m.norm(dim=1, keepdim=True)
            out.append((scale.exp() * (m @ q.T).squeeze(1)).cpu().numpy())
    return np.concatenate(out)


HEAD_FT_GRID = [
    {"lr": lr, "epochs": ep, "l2sp": l2}
    for lr in (1e-4, 1e-3, 1e-2)
    for ep in (50, 200, 800)
    for l2 in (0.0, 1e-2)
]


NFOLD = 3


def can_cv(y):
    """Can a stratified NFOLD split be formed from these labels?"""
    return int(np.bincount(y.astype(int), minlength=2).min()) >= NFOLD


def select_head_ft_cfg(model, rep_tr, y_tr, poc_rep, grid=HEAD_FT_GRID, seed=0):
    """Select hyperparameters by stratified CV within the training labels."""
    from sklearn.metrics import average_precision_score
    if not can_cv(y_tr):
        return None, float("nan"), "undefined_too_few_actives"
    skf = StratifiedKFold(NFOLD, shuffle=True, random_state=seed)
    folds = [(a, b) for a, b in skf.split(rep_tr, y_tr)
             if len(set(y_tr[a])) > 1 and len(set(y_tr[b])) > 1]
    if not folds:
        return None, float("nan"), "undefined_no_usable_fold"
    best, best_cfg = -np.inf, None
    for cfg in grid:
        sc = [average_precision_score(
                  y_tr[vb],
                  head_ft(model, rep_tr[va], y_tr[va], rep_tr[vb], poc_rep, seed, **cfg))
              for va, vb in folds]
        m = float(np.mean(sc))
        if m > best:
            best, best_cfg = m, cfg
    return best_cfg, best, "cv"




def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--arms", default="zeroshot,headft")
    ap.add_argument("--out", default=f"{here}/results")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    arms = args.arms.split(",")
    if not set(arms) <= {"zeroshot", "headft"}:
        raise ValueError("Supported arms: zeroshot,headft")

    with open(f"{here}/tables/reference_digest.json") as fh:
        global REF
        REF = json.load(fh)
    z = np.load(f"{here}/embs/all.npz")
    rep_all, emb_all, have = z["rep"], z["emb"], z["have"]
    smi_idx = pd.read_csv(f"{here}/smiles_index.csv")
    smi2id = dict(zip(smi_idx.smiles, smi_idx.smi_id))
    pk = np.load(f"{here}/embs/pockets.npz")
    poc = {t: (r, e) for t, r, e in zip(pk["target"], pk["rep"], pk["emb"])}

    # Use official inference embeddings for zero-shot scores.
    off = np.load(f"{here}/results/official_zeroshot.npz", allow_pickle=True)
    off_emb = np.zeros_like(emb_all, dtype=np.float32)
    off_ids = np.array([smi2id[x] for x in off["smiles"]])
    off_emb[off_ids] = off["emb"].astype(np.float32)
    off_have = np.zeros(len(emb_all), dtype=bool)
    off_have[off_ids] = True
    off_poc = {t: e for t, e in zip(off["pocket"], off["pocket_emb"])}
    print(f"official zero-shot embeddings: {off_have.sum()}/{len(off_have)}", flush=True)

    model = None
    if "headft" in arms:
        model, _, _ = D.build_model()

    rows, scores_out = [], {}
    for t in TARGETS:
        df = pd.read_csv(f"{here}/tables/{t}.csv", dtype={"complex_id": str})
        df["smi_id"] = df.smiles.map(smi2id)
        ref = REF["targets"][t]
        got = verify_tables.digest(df)
        assert got == ref["sha256"], f"{t}: compound IDs, labels or splits do not match"
        assert len(df) == ref["n_rows"], f"{t}: {len(df)} rows vs {ref['n_rows']}"
        assert int(df.is_eval.sum()) == ref["n_eval"], f"{t}: eval size mismatch"
        for nb in BUDGETS:
            n_got = int(df[f"train_{nb}"].sum())
            assert n_got == ref[f"n_train_{nb}"] == nb, \
                f"{t}: train_{nb} has {n_got} compounds, expected {nb}"
        ok = have[df.smi_id.values] & off_have[df.smi_id.values]
        drop = int((~ok).sum())
        if drop:
            raise SystemExit(
                f"{t}: {drop} missing embeddings; complete molecule encoding before evaluation."
            )
        df = df[ok].reset_index(drop=True)
        ev = df[df.is_eval]
        y = ev.is_binder.values
        emb_ev = off_emb[ev.smi_id.values]          # official inference path
        rep_ev = rep_all[ev.smi_id.values].astype(np.float32)
        poc_rep, _ = poc[t]
        poc_emb = off_poc[t]                        # official inference path
        poc_emb = poc_emb / np.linalg.norm(poc_emb)

        if "zeroshot" in arms:
            s = emb_ev @ poc_emb
            m = all_metrics(y, s)
            m.update(target=t, arm="drugclip_zeroshot", train_size=0, seed=-1, n_dropped=drop,
                     path="official")
            rows.append(m)
            scores_out[f"{t}|drugclip_zeroshot|0|-1"] = s
        for col, arm, n in [("base", "boltz_base", 0)] + [
                (f"headft_{n}", "boltz_headft", n) for n in BUDGETS]:
            if col in ev and ev[col].notna().all():
                m = all_metrics(y, ev[col].values)
                m.update(target=t, arm=arm, train_size=n, seed=-1, n_dropped=drop)
                rows.append(m)

        for n in BUDGETS:
            tr = df[df[f"train_{n}"]]
            y_tr = tr.is_binder.values
            n_train_used = len(tr)
            rep_tr = rep_all[tr.smi_id.values].astype(np.float32)
            cfg, cv_ap, cv_note = (None, float("nan"), "skipped")
            if "headft" in arms:
                cfg, cv_ap, cv_note = select_head_ft_cfg(model, rep_tr, y_tr, poc_rep)
                print(f"  {t} N={n} selected {cfg} (cv AP {cv_ap:.4f}, {cv_note})", flush=True)
            for seed in range(args.seeds):
                if "headft" in arms and cfg is not None:
                    s = head_ft(model, rep_tr, y_tr, rep_ev, poc_rep, seed, **cfg)
                    m = all_metrics(y, s)
                    m.update(target=t, arm="drugclip_headft", train_size=n, seed=seed,
                             n_train_active=int(y_tr.sum()), n_train_used=n_train_used,
                             n_dropped=drop, hp_lr=cfg["lr"], hp_epochs=cfg["epochs"],
                             hp_l2sp=cfg["l2sp"], hp_cv_ap=cv_ap, hp_source=cv_note)
                    rows.append(m)
                    scores_out[f"{t}|drugclip_headft|{n}|{seed}"] = s
        print("done", t, flush=True)
        pd.DataFrame(rows).to_csv(f"{args.out}/metrics.csv", index=False)
    np.savez_compressed(f"{args.out}/scores.npz", **scores_out)
    print("wrote", f"{args.out}/metrics.csv")


if __name__ == "__main__":
    main()
