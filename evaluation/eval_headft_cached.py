#!/usr/bin/env python3
"""Evaluate No-FT and head-FT scores on the same held-out compound IDs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import all_metrics

EXP = Path(os.environ["WORKDIR"])
SCORE_COLUMN = "affinity_probability_binary"


def load_arm(scores_dir: Path, expected_chunks: int | None) -> pd.DataFrame | None:
    """Load complete, non-overlapping score chunks."""
    chunks = sorted(scores_dir.glob("chunk_*.csv"))
    if not chunks:
        return None
    if expected_chunks is not None and len(chunks) != expected_chunks:
        print(
            f"    SKIP {scores_dir.parent.name}: {len(chunks)}/{expected_chunks} "
            "chunks scored (partial arms are not evaluated)"
        )
        return None
    frame = pd.concat([pd.read_csv(path) for path in chunks], ignore_index=True)
    frame["sample_id"] = frame["sample_id"].astype(str)
    if frame["sample_id"].duplicated().any():
        raise ValueError(f"Duplicate sample_id in {scores_dir}")
    return frame[["sample_id", SCORE_COLUMN]]


def labels_for(target: str) -> pd.Series:
    ml_table = EXP / "runs" / target / "ft_inputs_full" / "ml_table.csv"
    # Preserve identifiers when joining scores and labels.
    table = pd.read_csv(ml_table, usecols=["complex_id", "is_binder"],
                        dtype={"complex_id": str})
    return table.set_index("complex_id")["is_binder"].astype(int)


def eval_ids_for(target: str) -> list[str]:
    """The fixed common eval set: all compounds minus the top-300 training pool."""
    path = EXP / "runs" / target / "ft_inputs_full" / "eval_ids.txt"
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def evaluate(targets: list[str], sizes: list[int]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for target in targets:
        print(f"[{target}]")
        run_dir = EXP / "runs" / target / "headft_affcache"
        labels = labels_for(target)
        n_chunks = len(
            list((EXP / "runs" / target / "ft_inputs_full" / "eval_chunks").glob("chunk_*.txt"))
        )

        arms: dict[str, tuple[pd.DataFrame, int]] = {}
        base = load_arm(run_dir / "base" / "scores", n_chunks)
        if base is not None:
            arms["base"] = (base, 0)
        for n_train in sizes:
            frame = load_arm(run_dir / f"lightning_top{n_train}" / "scores", n_chunks)
            if frame is not None:
                arms[f"headft_{n_train}"] = (frame, n_train)
        if len(arms) != len(sizes) + 1:
            raise ValueError(f"{target}: missing or incomplete score arms")
        if not arms:
            print("    no complete arms yet")
            continue

        eval_ids = eval_ids_for(target)
        for arm, (frame, n_train) in arms.items():
            merged = frame.set_index("sample_id").reindex(eval_ids)
            if merged[SCORE_COLUMN].isna().any():
                missing = int(merged[SCORE_COLUMN].isna().sum())
                raise ValueError(f"{target}/{arm}: {missing} eval ids were not scored")
            merged = merged.reset_index().rename(columns={"index": "sample_id"})
            merged["y"] = merged["sample_id"].map(labels)
            if merged["y"].isna().any():
                missing = int(merged["y"].isna().sum())
                raise ValueError(f"{target}/{arm}: {missing} scored ids have no label")
            metrics = all_metrics(
                merged["y"].to_numpy(dtype=int),
                merged[SCORE_COLUMN].to_numpy(dtype=float),
            )
            rows.append({"target": target, "arm": "base" if n_train == 0 else "lightning", "n_train": n_train, **metrics})
            print(
                f"    {arm:<24s} AP={metrics['auprc']:.4f} EF1%={metrics['ef_1pct']:.2f} "
                f"BEDROC={metrics['bedroc']:.3f} n={metrics['n_eval']}"
            )
    return pd.DataFrame(rows)


def summarize(frame: pd.DataFrame) -> None:
    """Per-N AP ratio of head-FT over the no-FT control, and the sign test."""
    if frame.empty:
        return
    base = frame[frame.arm == "base"].set_index("target")["auprc"]
    ft = frame[frame.arm != "base"]
    print("\n=== head-FT / no-FT AP ratio ===")
    for n_train, group in ft.groupby("n_train"):
        shared = group[group.target.isin(base.index)]
        if shared.empty:
            continue
        ratios = shared["auprc"].to_numpy() / base.loc[shared["target"]].to_numpy()
        wins = int((ratios > 1).sum())
        print(
            f"  N={n_train:<4d} geometric mean ratio {np.exp(np.log(ratios).mean()):.3f}  "
            f"wins {wins}/{len(ratios)}  "
            f"per-target {np.round(ratios, 3).tolist()}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--targets", nargs="+", required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[40, 100, 300])
    parser.add_argument(
        "--out-csv", type=Path, default=EXP / "analysis" / "out" / "headft_cached" / "metrics.csv"
    )
    args = parser.parse_args()

    frame = evaluate(args.targets, args.sizes)
    if frame.empty:
        print("Nothing to evaluate.")
        return
    args.out_csv.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(args.out_csv, index=False)
    summarize(frame)
    print(f"\nWrote {len(frame)} rows -> {args.out_csv}")


if __name__ == "__main__":
    main()
