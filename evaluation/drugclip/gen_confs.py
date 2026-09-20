"""Generate one deterministic ETKDGv3+MMFF conformer per unique SMILES (parallel)."""
import argparse, os, sys, time
import numpy as np, pandas as pd
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from conformer import smiles_to_conformers

here = os.environ["COMPARATOR_WORKDIR"]


def work(job):
    smi_id, smi, n_conf = job
    r = smiles_to_conformers(smi, n_conf=n_conf)
    if r is None:
        return smi_id, None, None
    atoms, coords = r
    return smi_id, atoms, coords


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--shard-size", type=int, default=20000)
    ap.add_argument("--out", default=f"{here}/confs")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    idx = pd.read_csv(f"{here}/smiles_index.csv")
    n = len(idx)
    nsh = (n + args.shard_size - 1) // args.shard_size
    print(f"{n} smiles -> {nsh} shards", flush=True)
    for sh in range(nsh):
        out = f"{args.out}/shard_{sh:04d}.npz"
        if os.path.exists(out):
            print("skip", sh, flush=True)
            continue
        part = idx.iloc[sh * args.shard_size:(sh + 1) * args.shard_size]
        jobs = [(int(r.smi_id), r.smiles, 1) for r in part.itertuples()]
        t0 = time.time()
        with Pool(args.workers) as pool:
            res = pool.map(work, jobs, chunksize=64)
        ids, atoms, coords, fails = [], [], [], []
        for smi_id, a, c in res:
            if a is None:
                fails.append(smi_id)
                continue
            ids.append(smi_id)
            atoms.append(a)
            coords.append(np.stack(c))
        aobj = np.empty(len(atoms), dtype=object)
        aobj[:] = atoms
        cobj = np.empty(len(coords), dtype=object)
        cobj[:] = coords
        np.savez_compressed(
            out + ".tmp.npz",
            smi_id=np.array(ids, dtype=np.int64),
            atoms=aobj,
            coords=cobj,
            failed=np.array(fails, dtype=np.int64),
        )
        os.rename(out + ".tmp.npz", out)
        print(f"shard {sh}/{nsh} n={len(ids)} fail={len(fails)} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
