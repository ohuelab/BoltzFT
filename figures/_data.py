"""Load result tables and calculate target-level summaries."""
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import t

DATA = Path(__file__).resolve().parent / "data"
OUT = DATA.parent / "out"
OUT.mkdir(exist_ok=True)
METRICS = [("auprc", "AP"), ("ef_1pct", "EF@1%"), ("bedroc", "BEDROC")]


def read(name):
    return pd.read_csv(DATA / name, dtype={"target": str})


def populations():
    return read("target_populations.csv")


def targets():
    return populations().target.tolist()


def budgets():
    df = read("trainer_control.csv")
    return sorted(df.loc[df.arm == "lightning", "n_train"].unique())


def paired(metric, n):
    df = read("trainer_control.csv")
    base = df[df.arm == "base"].set_index("target").loc[targets(), metric]
    ft = df[(df.arm == "lightning") & (df.n_train == n)].set_index("target").loc[targets(), metric]
    return base.to_numpy(), ft.to_numpy()


def mean_interval(values, geometric=False):
    x = np.asarray(values, dtype=float)
    if not np.isfinite(x).all() or len(x) < 2:
        raise ValueError("An interval requires at least two finite target values")
    if geometric:
        if (x <= 0).any():
            raise ValueError("Log ratios require strictly positive values")
        x = np.log(x)
    m = x.mean()
    half = t.ppf(0.975, len(x) - 1) * x.std(ddof=1) / np.sqrt(len(x))
    result = np.array([m, m - half, m + half])
    return np.exp(result) if geometric else result


def validate():
    """Check condition uniqueness and evaluation populations."""
    df = read("trainer_control.csv")
    if df.duplicated(["target", "arm", "n_train"]).any():
        raise ValueError("Duplicate target/condition rows")
    pop = populations().set_index("target")
    for target, g in df.groupby("target"):
        for col in ["n_eval", "n_active"]:
            if not (g[col] == pop.loc[target, col]).all():
                raise ValueError(f"Evaluation population mismatch: {target}, {col}")
    if pop.train_eval_overlap.any():
        raise ValueError("Training/evaluation overlap")
    cascade = read("cascade.csv")
    if cascade.duplicated(["target", "n_train", "budget_k", "cut_M"]).any():
        raise ValueError("Duplicate cascade rows")
    for (n, k), g in cascade.groupby(["n_train", "budget_k"]):
        if set(g.target) != set(targets()):
            raise ValueError("Incomplete cascade")
        full = g[g.is_full].set_index("target")
        if not (full.loc[targets(), "cut_M"] == pop.loc[targets(), "n_eval"]).all():
            raise ValueError("Full rescoring does not cover evaluation population")
        if not (full.hits == full.hits_full_headft).all():
            raise ValueError("Cascade full-library reference mismatch")


if __name__ == "__main__":
    validate()
