#!/usr/bin/env python3
"""Build ranked compound tables, disjoint training/evaluation IDs, and scoring chunks."""
from __future__ import annotations
import argparse
import gzip
from pathlib import Path

import pandas as pd

ID_COL, SMILES_COL, LABEL_COL, SCORE_COL = "CID", "neut-smiles", "Active_v2", "affinity_probability_binary"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results-csv", type=Path, required=True)
    p.add_argument("--target", required=True)
    p.add_argument("--consolidated-dir", type=Path, required=True)
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--train-sizes", type=int, nargs="+", default=[40, 100, 300])
    p.add_argument("--val-tail", type=int, default=20)
    p.add_argument("--chunk-size", type=int, default=1000)
    args = p.parse_args()
    args.exclude_top = max(args.train_sizes)

    emb = args.consolidated_dir / "embeddings"
    pref = f"embeddings_{args.target}_"
    folded = {int(f.stem[len(pref):]) for f in emb.glob(f"{pref}*.npz")}
    if not folded:
        raise SystemExit(f"no embeddings under {emb}")

    res = pd.read_csv(args.results_csv)
    res[ID_COL] = res[ID_COL].astype(int)
    res = res[res[ID_COL].isin(folded)].copy()
    res = res.dropna(subset=[SCORE_COL]).drop_duplicates(ID_COL, keep="first")
    ids_path = Path(__file__).resolve().parents[1] / 'data/splits' / f'{args.target}.txt.gz'
    with gzip.open(ids_path, 'rt') as f:
        ordered = [int(cid.strip().removeprefix(args.target + '_')) for cid in f]
    res = res.set_index(ID_COL).loc[ordered].reset_index()
    res["boltz_rank"] = res.index + 1

    ml = pd.DataFrame({
        "complex_id": [f"{args.target}_{c}" for c in res[ID_COL]],
        "ligand_id": res[ID_COL].to_numpy(),
        "smiles": res[SMILES_COL].to_numpy(),
        "is_binder": res[LABEL_COL].astype(int).to_numpy(),
        "boltz_rank": res["boltz_rank"].to_numpy(),
        "affinity_probability_binary": res[SCORE_COL].to_numpy(),
        "dataset": "MF-PCBA",
        "target_id": args.target,
    })
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ml.to_csv(args.out_dir / "ml_table.csv", index=False)

    if len(ml) <= args.exclude_top + args.val_tail:
        raise ValueError("Insufficient compounds for disjoint training and validation")
    score_ids_all = ml["complex_id"].tolist()  # rank order, whole library
    (args.out_dir / "score_ids_all.txt").write_text("\n".join(score_ids_all) + "\n")

    eval_ids = score_ids_all[args.exclude_top:]
    (args.out_dir / "eval_ids.txt").write_text("\n".join(eval_ids) + "\n")

    counts = {}
    for n in sorted(args.train_sizes):
        ids = score_ids_all[:n]
        (args.out_dir / f"train_ids_top{n}.txt").write_text("\n".join(ids) + "\n")
        counts[n] = len(ids)

    val_ids = score_ids_all[-args.val_tail:]
    keep = set(score_ids_all[:max(args.train_sizes)]) | set(val_ids)
    ml[ml["complex_id"].isin(keep)].to_csv(args.out_dir / "ml_table_train.csv", index=False)

    # Score all compounds; evaluators select the IDs in eval_ids.txt.
    cdir = args.out_dir / "eval_chunks"
    cdir.mkdir(exist_ok=True)
    for i in range(0, len(score_ids_all), args.chunk_size):
        idx = i // args.chunk_size
        (cdir / f"chunk_{idx:03d}.txt").write_text("\n".join(score_ids_all[i:i + args.chunk_size]) + "\n")
    n_chunks = (len(score_ids_all) + args.chunk_size - 1) // args.chunk_size

    print(f"target={args.target}")
    print(f"folded={len(folded)}  ml_table rows={len(ml)}  n_score_ids={len(score_ids_all)}")
    print(f"n_excluded={args.exclude_top}  n_eval_ids={len(eval_ids)}")
    print(f"train_sizes={counts}  ml_table_train={int(ml['complex_id'].isin(keep).sum())}  eval_chunks={n_chunks}")
    print(f"is_binder balance: {ml['is_binder'].value_counts().to_dict()}")


if __name__ == "__main__":
    main()
