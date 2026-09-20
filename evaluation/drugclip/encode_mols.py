"""Encode every conformer shard with the DrugCLIP molecule encoder."""
import argparse, glob, os, sys, time
import numpy as np, torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dc_model as D

here = os.environ["COMPARATOR_WORKDIR"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--confs", default=f"{here}/confs")
    ap.add_argument("--out", default=f"{here}/embs")
    ap.add_argument("--batch-size", type=int, default=64)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    model, mol_dict, pkt_dict = D.build_model()

    pout = f"{args.out}/pockets.npz"
    if not os.path.exists(pout):
        names, reps, embs = [], [], []
        for p in sorted(glob.glob(f"{here}/pockets/*.npz")):
            t = os.path.basename(p)[:-4]
            z = np.load(p)
            f = D.featurize_pocket(z["pocket_atoms"], z["pocket_coordinates"], pkt_dict)
            tok, dist = D.collate([f], pkt_dict.pad())
            r, e = D.encode(model, tok, dist, len(pkt_dict), "pocket")
            names.append(t); reps.append(r.numpy()[0]); embs.append(e.numpy()[0])
        np.savez(pout, target=np.array(names), rep=np.stack(reps).astype(np.float32),
                 emb=np.stack(embs).astype(np.float32))
        print("pockets ->", pout, flush=True)

    for sp in sorted(glob.glob(f"{args.confs}/shard_[0-9][0-9][0-9][0-9].npz")):
        sh = os.path.basename(sp)[:-4]
        out = f"{args.out}/{sh}.npz"
        if os.path.exists(out):
            continue
        t0 = time.time()
        z = np.load(sp, allow_pickle=True)
        ids, atoms, coords = z["smi_id"], z["atoms"], z["coords"]
        feats = [D.featurize_mol(a, c[0], mol_dict) for a, c in zip(atoms, coords)]
        order = np.argsort([len(f[0]) for f in feats])  # length-sorted batching
        reps = np.zeros((len(feats), 512), dtype=np.float16)
        embs = np.zeros((len(feats), 128), dtype=np.float16)
        B = args.batch_size
        for i in range(0, len(order), B):
            sel = order[i:i + B]
            tok, dist = D.collate([feats[j] for j in sel], mol_dict.pad())
            r, e = D.encode(model, tok, dist, len(mol_dict), "mol")
            reps[sel] = r.numpy().astype(np.float16)
            embs[sel] = e.numpy().astype(np.float16)
        np.savez(out + ".tmp.npz", smi_id=ids, rep=reps, emb=embs)
        os.rename(out + ".tmp.npz", out)
        print(f"{sh} n={len(ids)} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
