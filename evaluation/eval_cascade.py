#!/usr/bin/env python3
"""Evaluate No-FT candidate selection followed by head-FT reranking."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

EXP = Path(os.environ["WORKDIR"])
TARGETS = [
    "588689", "540297-493091", "434954-2097", "463203-2650",
    "493248-485317", "504329", "624273-588549", "1053173-743445",
]
SCORE_COLUMN = "affinity_probability_binary"
CUTS = [500, 1000, 2500, 5000, 10000, 25000]
BUDGETS = [100, 500]
SECONDS_PER_COMPOUND = 0.2


def eval_ids(target: str) -> list[str]:
    path = EXP / "runs" / target / "ft_inputs_full" / "eval_ids.txt"
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def labels(target: str) -> pd.Series:
    table = pd.read_csv(
        EXP / "runs" / target / "ft_inputs_full" / "ml_table.csv",
        usecols=["complex_id", "is_binder"], dtype={"complex_id": str},
    )
    return table.set_index("complex_id")["is_binder"].astype(int)


def chunked_arm(scores_dir: Path) -> pd.Series | None:
    chunks = sorted(scores_dir.glob("chunk_*.csv"))
    if not chunks:
        raise FileNotFoundError(f"No score chunks: {scores_dir}")
    frame = pd.concat(
        [pd.read_csv(c, dtype={"sample_id": str}) for c in chunks], ignore_index=True
    )
    if frame.sample_id.duplicated().any():
        raise ValueError(f"Duplicate score IDs: {scores_dir}")
    return frame.set_index("sample_id")[SCORE_COLUMN]


def hits_at(order: np.ndarray, y: np.ndarray, k: int) -> int:
    """Actives among the first k entries of a ranking."""
    return int(y[order[:k]].sum())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", default=TARGETS)
    parser.add_argument("--sizes", type=int, nargs="+", default=[40, 100, 300])
    parser.add_argument("--out-csv", type=Path,
                        default=EXP / "analysis/out/headft_cached/cascade.csv")
    args = parser.parse_args()

    rows: list[dict] = []
    for target in args.targets:
        ids = eval_ids(target)
        y_series = labels(target).reindex(ids)
        y = y_series.to_numpy(dtype=int)
        run = EXP / "runs" / target / "headft_affcache"

        s1 = chunked_arm(run / "base" / "scores")
        s1 = s1.reindex(ids).to_numpy(dtype=float)
        if not np.isfinite(s1).all():
            raise ValueError(f"{target}: missing No-FT scores")
        # Descending score; ties broken by index so the ordering is deterministic.
        order1 = np.lexsort((np.arange(len(s1)), -s1))

        for n in args.sizes:
            s2_series = chunked_arm(run / f"lightning_top{n}" / "scores")
            if s2_series is None:
                print(f"  {target}/N={n}: head-FT arm incomplete"); continue
            s2 = s2_series.reindex(ids).to_numpy(dtype=float)
            if not np.isfinite(s2).all():
                raise ValueError(f"{target}: missing head-FT scores")
            order_full = np.lexsort((np.arange(len(s2)), -s2))

            for k in BUDGETS:
                full_hits = hits_at(order_full, y, k)
                base_hits = hits_at(order1, y, k)
                # None selects the full library for this target.
                for m in CUTS + [None]:
                    survivors = order1[: (len(ids) if m is None else m)]
                    # re-rank survivors by the head-FT score
                    resort = survivors[np.lexsort((np.arange(len(survivors)),
                                                   -s2[survivors]))]
                    rows.append({
                        "target": target, "n_train": n, "budget_k": k,
                        "cut_M": len(survivors), "is_full": m is None,
                        "hits": hits_at(resort, y, k),
                        "ceiling": int(y[survivors].sum()),
                        "hits_full_headft": full_hits,
                        "hits_stage1_only": base_hits,
                        "n_active_total": int(y.sum()),
                        "gpu_hours": len(survivors) * SECONDS_PER_COMPOUND / 3600,
                    })

    if not rows:
        print("Nothing to evaluate."); return
    df = pd.DataFrame(rows)
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)

    for n in sorted(df.n_train.unique()):
        for k in BUDGETS:
            sub = df[(df.n_train == n) & (df.budget_k == k)]
            print(f"\n=== N={n}, final budget k={k} "
                  f"(stage 1 = No-FT, {sub.target.nunique()} targets) ===")
            print(f"{'stage-1 cut M':>14} {'head-FT GPU-h':>14} {'hits@k':>9} "
                  f"{'vs full':>9} {'targets at full':>16}")
            full = sub.groupby("target")["hits_full_headft"].first()
            s1only = sub.groupby("target")["hits_stage1_only"].first()
            for m in CUTS + [None]:
                g = (sub[sub.is_full] if m is None
                     else sub[(~sub.is_full) & (sub.cut_M == m)]).set_index("target")
                if g.empty:
                    continue
                tot, ftot = g["hits"].sum(), full.loc[g.index].sum()
                n_at_full = int((g["hits"] >= full.loc[g.index]).sum())
                label = "all (~50k)" if m is None else str(m)
                print(f"{label:>14} {g['gpu_hours'].sum():>14.2f} {tot:>9d} "
                      f"{tot / ftot:>8.3f} {n_at_full:>10d}/{len(g)}")
            print(f"{'stage 1 only':>14} {0.0:>14.2f} {s1only.sum():>9d} "
                  f"{s1only.sum() / full.sum():>8.3f}")
    print(f"\nWrote {len(df)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
