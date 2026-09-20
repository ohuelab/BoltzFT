"""Merge the per-shard embeddings into one lookup keyed by smi_id."""
import glob, os, sys
import numpy as np, pandas as pd

here = os.environ["COMPARATOR_WORKDIR"]


def load():
    idx = pd.read_csv(f"{here}/smiles_index.csv")
    n = len(idx)
    rep = np.zeros((n, 512), dtype=np.float16)
    emb = np.zeros((n, 128), dtype=np.float16)
    have = np.zeros(n, dtype=bool)
    for p in sorted(glob.glob(f"{here}/embs/shard_[0-9][0-9][0-9][0-9].npz")):
        z = np.load(p)
        ids = z["smi_id"]
        rep[ids] = z["rep"]
        emb[ids] = z["emb"]
        have[ids] = True
    return idx, rep, emb, have


if __name__ == "__main__":
    idx, rep, emb, have = load()
    out = f"{here}/embs/all.npz"
    np.savez(out, rep=rep, emb=emb, have=have)
    print("covered", int(have.sum()), "/", len(have), "->", out)
