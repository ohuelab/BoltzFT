#!/usr/bin/env python3
"""Validate released compound scores and reproduce the archived metrics."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path

import numpy as np
import pandas as pd

from metrics import all_metrics

ROOT = Path(__file__).resolve().parents[1]
RTOL, ATOL = 1e-10, 1e-12


def read(path):
    return pd.read_csv(path, dtype={"target": str, "complex_id": str, "ligand_id": str},
                       float_precision="round_trip")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def reference_rows():
    root = ROOT / "figures/data/baseline_cv"
    dc = read(root / "metrics.csv")
    dc = dc[dc.arm.isin(["boltz_base", "boltz_headft", "drugclip_zeroshot", "drugclip_headft"])]
    lgb = pd.concat([read(root / "supervised_baselines.csv"),
                     read(root / "supervised_baselines_concat.csv")], ignore_index=True)
    lgb = lgb[lgb.tuning.eq("cv")].copy()
    lgb["seed"] = -1
    rows = pd.concat([dc, lgb], ignore_index=True)
    require(not rows.duplicated(["target", "arm", "train_size", "seed"]).any(),
            "Duplicate reference conditions")
    return rows.set_index(["target", "arm", "train_size", "seed"])


def compare(observed, reference, label, checks):
    for metric, value in observed.items():
        if metric not in reference:
            continue
        expected = reference[metric]
        require(np.isfinite(value) and np.isfinite(expected), f"{label}: undefined {metric}")
        matches = bool(np.isclose(value, expected, rtol=RTOL, atol=ATOL))
        checks.append(dict(condition=label, metric=metric, observed=value,
                           reference=expected, abs_difference=abs(float(value) - float(expected)), matches=matches))
        require(matches, f"{label}: {metric} {value} != {expected}")


def cascade_rows(target, ev):
    y = ev.is_binder.to_numpy(int)
    s1 = ev.no_ft.to_numpy(float)
    order1 = np.lexsort((np.arange(len(y)), -s1))
    rows = []
    for n in (40, 100, 300):
        s2 = ev[f"headft_N{n}"].to_numpy(float)
        full = np.lexsort((np.arange(len(y)), -s2))
        for k in (100, 500):
            for m in (500, 1000, 2500, 5000, 10000, 25000, len(y)):
                survivors = order1[:m]
                reranked = survivors[np.lexsort((np.arange(len(survivors)), -s2[survivors]))]
                rows.append(dict(target=target, n_train=n, budget_k=k, cut_M=m,
                                 is_full=m == len(y), hits=int(y[reranked[:k]].sum()),
                                 ceiling=int(y[survivors].sum()),
                                 hits_full_headft=int(y[full[:k]].sum()),
                                 hits_stage1_only=int(y[order1[:k]].sum()),
                                 n_active_total=int(y.sum()), gpu_hours=m * 0.2 / 3600))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data-dir", type=Path, default=ROOT / "data/predictions")
    ap.add_argument("--out", type=Path, default=ROOT / "runs/prediction_verification")
    args = ap.parse_args()
    manifest = json.loads((args.data_dir / "manifest.json").read_text())
    # NumPy's default argsort determines tied-score metrics in the archived protocol.
    require(np.__version__ == manifest["metric_numpy_version"],
            f"Use NumPy {manifest['metric_numpy_version']} to preserve archived tie ordering")
    refs = reference_rows()
    controls = read(ROOT / "figures/data/trainer_control.csv").set_index(["target", "arm", "n_train"])
    require(not controls.index.duplicated().any(), "Duplicate trainer-control conditions")
    populations = read(ROOT / "figures/data/target_populations.csv").set_index("target")
    metrics, checks, cascades, seen = [], [], [], set()
    total_rows = total_eval = 0
    for target, info in manifest["targets"].items():
        path = args.data_dir / info["file"]
        df = read(path)
        require(df.columns.tolist() == manifest["columns"], f"{target}: column mismatch")
        require(df.complex_id.notna().all() and not df.complex_id.duplicated().any(),
                f"{target}: missing/duplicate compound IDs")
        require(df.target.eq(target).all(), f"{target}: target mismatch")
        require(df.smiles.notna().all() and df.ligand_id.notna().all(), f"{target}: missing molecule")
        require(df.is_binder.isin([0, 1]).all(), f"{target}: invalid labels")
        for col in ("train_40", "train_100", "train_300", "is_eval"):
            require(df[col].dtype == bool, f"{target}: invalid split flag {col}")
        require(len(df) == info["n_rows"] == populations.loc[target, "n_available"],
                f"{target}: population mismatch")
        require(np.array_equal(~df.train_300, df.is_eval), f"{target}: train/eval overlap")
        require((~df.train_40 | df.train_100).all() and (~df.train_100 | df.train_300).all(),
                f"{target}: nonnested training sets")
        for n in (40, 100, 300):
            require(int(df[f"train_{n}"].sum()) == n, f"{target}: wrong training budget")
            require(np.array_equal(df[f"train_{n}"], np.arange(len(df)) < n),
                    f"{target}: training IDs differ from the released split order")
            require(int(df.loc[df[f"train_{n}"], "is_binder"].sum()) ==
                    populations.loc[target, f"train_active_{n}"], f"{target}: training labels differ")
        with gzip.open(ROOT / "data/splits" / f"{target}.txt.gz", "rt") as fh:
            split_ids = fh.read().splitlines()
        require(df.complex_id.tolist() == split_ids, f"{target}: released split IDs differ")
        ev = df[df.is_eval]
        require(np.array_equal(ev.eval_order.to_numpy(), np.arange(len(ev))) and
                df.loc[~df.is_eval, "eval_order"].eq(-1).all(), f"{target}: evaluation order differs")
        require(len(ev) == info["n_eval"] == populations.loc[target, "n_eval"], f"{target}: eval count")
        require(int(ev.is_binder.sum()) == info["n_active_eval"] == populations.loc[target, "n_active"],
                f"{target}: active count")
        y = ev.is_binder.to_numpy(int)
        for spec in manifest["scores"]:
            key = (target, spec["arm"], spec["train_size"], spec["seed"])
            eligible = key in refs.index and np.isfinite(refs.loc[key, "auprc"])
            col = spec["column"]
            if spec["arm"] == "boltz_base":
                require(np.isfinite(df[col].to_numpy(float)).all(), f"{target}/{col}: missing No-FT scores")
            else:
                require(df.loc[~df.is_eval, col].isna().all(), f"{target}/{col}: training scores present")
            if not eligible:
                require(df[col].isna().all(), f"{target}/{col}: undefined condition has scores")
                # DrugCLIP omits ineligible CV rows; LightGBM archives NaN rows.
                n = spec["train_size"]
                require(n > 0, f"{target}/{col}: missing unconditional score")
                tr = df[df[f"train_{n}"]]
                require(min(int(tr.is_binder.sum()), len(tr) - int(tr.is_binder.sum())) < 3,
                        f"{target}/{col}: unexpected missing CV condition")
                continue
            s = ev[col].to_numpy(dtype=spec["dtype"])
            require(np.isfinite(s).all(), f"{target}/{col}: incomplete scores")
            met = all_metrics(y, s)
            label = f"{target}/{col}"
            compare(met, refs.loc[key], label, checks)
            if spec["arm"] in ("boltz_base", "boltz_headft"):
                arm = "base" if spec["arm"] == "boltz_base" else "lightning"
                compare(met, controls.loc[(target, arm, spec["train_size"])], label + "/control", checks)
            seen.add(key)
            metrics.append(dict(target=target, arm=spec["arm"], train_size=spec["train_size"],
                                seed=spec["seed"], **met))
        cascades.extend(cascade_rows(target, ev))
        total_rows += len(df)
        total_eval += len(ev)
        print(f"PASS {target}: {len(ev):,} evaluation compounds", flush=True)
    expected = set(refs[np.isfinite(refs.auprc)].index)
    require(seen == expected, f"Reference coverage differs: {seen.symmetric_difference(expected)}")
    cascade = pd.DataFrame(cascades)
    keys = ["target", "n_train", "budget_k", "cut_M"]
    ref_cascade = read(ROOT / "figures/data/cascade.csv").set_index(keys)
    require(not ref_cascade.index.duplicated().any(), "Duplicate cascade reference")
    observed = cascade.set_index(keys)
    require(set(observed.index) == set(ref_cascade.index), "Cascade coverage differs")
    for key, row in observed.iterrows():
        compare(row.to_dict(), ref_cascade.loc[key], f"cascade/{key}", checks)
    args.out.mkdir(parents=True, exist_ok=True)
    result = pd.DataFrame(metrics)
    result.to_csv(args.out / "metrics.csv", index=False)
    # DrugCLIP averages metrics over runs; LightGBM scores already average five models.
    result.groupby(["target", "arm", "train_size"], as_index=False).mean(numeric_only=True).drop(
        columns="seed").to_csv(args.out / "metrics_per_target.csv", index=False)
    cascade.to_csv(args.out / "cascade.csv", index=False)
    pd.DataFrame(checks).to_csv(args.out / "metric_checks.csv", index=False)
    summary = dict(status="passed", n_targets=len(manifest["targets"]), n_rows=total_rows,
                   n_eval=total_eval, n_score_conditions=len(seen), n_cascade_conditions=len(cascade),
                   n_metric_checks=len(checks), max_absolute_difference=max(c["abs_difference"] for c in checks),
                   rtol=RTOL, atol=ATOL, versions={"numpy": np.__version__, "pandas": pd.__version__})
    (args.out / "verification.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
