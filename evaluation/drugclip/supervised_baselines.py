"""Three-fold CV LightGBM on ECFP, CheMeleon and pooled Boltz features."""
import argparse, json, os, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import average_precision_score
import lightgbm as lgb

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from metrics import all_metrics
import verify_tables

here = os.environ["COMPARATOR_WORKDIR"]
ANALYSIS = os.environ["FEATURE_CACHE_DIR"]
TARGETS = ["588689", "540297-493091", "434954-2097", "463203-2650",
           "493248-485317", "504329", "624273-588549", "1053173-743445"]
BUDGETS = [40, 100, 300]
NFOLD = 3

GRID = [dict(n_estimators=ne, num_leaves=nl, learning_rate=lr,
             min_child_samples=mcs, subsample=0.8, colsample_bytree=0.8,
             class_weight="balanced", verbosity=-1, force_col_wise=True)
        for ne in (100, 200, 600)
        for nl in (7, 15, 31)
        for lr in (0.02, 0.05, 0.15)
        for mcs in (5,)]


def load_features(target, which, ids_wanted):
    if "_" in which:  # concatenated feature sets, e.g. ecfp_graw
        parts = [load_features(target, w, ids_wanted) for w in which.split("_")]
        return np.hstack(parts)
    if which == "ecfp":
        from rdkit import Chem, RDLogger
        from rdkit.Chem import rdFingerprintGenerator
        RDLogger.DisableLog("rdApp.*")
        gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)
        tab = pd.read_csv(f"{here}/tables/{target}.csv",
                          usecols=["complex_id", "smiles"], dtype={"complex_id": str})
        smi = dict(zip(tab.complex_id, tab.smiles))
        X = np.zeros((len(ids_wanted), 2048), dtype=np.float32)
        for i, cid in enumerate(ids_wanted):
            m = Chem.MolFromSmiles(smi[cid])
            if m is not None:
                X[i] = np.asarray(gen.GetFingerprintAsNumPy(m), dtype=np.float32)
        return X
    if which == "graw":
        p = f"{ANALYSIS}/graw/{target}.npz"
        key = "graw"
    elif which == "chemeleon":
        p = f"{ANALYSIS}/chemeleon/{target}.npz"
        key = "emb"
    else:
        raise ValueError(which)
    z = np.load(p, allow_pickle=True)
    pos = {str(v): i for i, v in enumerate(z["ids"])}
    idx = np.array([pos[c] for c in ids_wanted])
    return z[key][idx].astype(np.float32)


def can_cv(y):
    c = np.bincount(y.astype(int), minlength=2)
    return c.min() >= NFOLD


def fit_predict(params, Xtr, ytr, Xev, seeds):
    preds = []
    for s in seeds:
        clf = lgb.LGBMClassifier(random_state=s, n_jobs=int(os.environ.get("LGB_JOBS", "4")),
                                 **params)
        clf.fit(Xtr, ytr)
        preds.append(clf.predict_proba(Xev)[:, 1])
    return np.mean(preds, axis=0)


def select(Xtr, ytr, seed=0):
    skf = StratifiedKFold(NFOLD, shuffle=True, random_state=seed)
    folds = [(a, b) for a, b in skf.split(Xtr, ytr)
             if len(set(ytr[a])) > 1 and len(set(ytr[b])) > 1]
    if not folds:
        return None, float("nan")
    best, best_p = -np.inf, None
    for p in GRID:
        sc = []
        for a, b in folds:
            clf = lgb.LGBMClassifier(random_state=seed,
                                     n_jobs=int(os.environ.get("LGB_JOBS", "4")), **p)
            clf.fit(Xtr[a], ytr[a])
            sc.append(average_precision_score(ytr[b], clf.predict_proba(Xtr[b])[:, 1]))
        m = float(np.mean(sc))
        if m > best:
            best, best_p = m, p
    return best_p, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default=",".join(TARGETS))
    ap.add_argument("--features", default="ecfp,graw,chemeleon,ecfp_graw,chemeleon_graw")
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--out", default=f"{here}/results/supervised_baselines.csv")
    args = ap.parse_args()
    seeds = list(range(args.seeds))
    with open(f"{here}/tables/reference_digest.json") as fh:
        REF = json.load(fh)["targets"]

    rows = []
    for t in args.targets.split(","):
        df = pd.read_csv(f"{here}/tables/{t}.csv", dtype={"complex_id": str})
        assert verify_tables.digest(df) == REF[t]["sha256"], f"{t}: compound set drift"
        ids = df.complex_id.tolist()
        ev = df.is_eval.values
        y_ev = df.is_binder.values[ev]
        for feat in args.features.split(","):
            t0 = time.time()
            X = load_features(t, feat, ids)
            Xev = X[ev]
            for n in BUDGETS:
                tr = df[f"train_{n}"].values
                Xtr, ytr = X[tr], df.is_binder.values[tr]
                if not can_cv(ytr):
                    m = {k: float("nan") for k in
                         ("auroc", "auprc", "ef_1pct", "ef_5pct", "bedroc",
                          "recall_at_50", "recall_at_100", "hits_at_20", "hits_at_100")}
                    m.update(n_eval=int(ev.sum()), n_active=int(y_ev.sum()))
                    note, cvap = "undefined_too_few_actives", float("nan")
                else:
                    params, cvap = select(Xtr, ytr)
                    note = "cv"
                    s = fit_predict(params, Xtr, ytr, Xev, seeds)
                    m = all_metrics(y_ev, s)
                m.update(target=t, arm=f"{feat}_lgbm_cv", feature=feat, tuning="cv",
                         train_size=n, n_train_active=int(ytr.sum()),
                         cv_ap=cvap, note=note)
                rows.append(m)
                print(f"  {t} {feat} N={n} cv: AUPRC "
                      f"{m['auprc'] if isinstance(m['auprc'], float) else float('nan'):.4f} "
                      f"({note})", flush=True)
            print(f"[{t}/{feat}] {time.time()-t0:.0f}s", flush=True)
            os.makedirs(os.path.dirname(args.out), exist_ok=True)
            pd.DataFrame(rows).to_csv(args.out, index=False)
    print("wrote", args.out)


if __name__ == "__main__":
    main()
