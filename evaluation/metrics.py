"""Screening metrics used for Boltz-2 and supervised comparisons."""
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score


def enrichment_factor(y: np.ndarray, score: np.ndarray, frac: float) -> float:
    n = len(y)
    k = max(1, int(round(n * frac)))
    order = np.argsort(-score)
    top = y[order][:k]
    hit_rate_top = top.mean()
    base = y.mean()
    return float(hit_rate_top / base) if base > 0 else float("nan")


def bedroc(y: np.ndarray, score: np.ndarray, alpha: float = 20.0) -> float:
    """Truchon & Bayly (2007) BEDROC."""
    order = np.argsort(-score)
    y = y[order]
    n = len(y)
    na = int(y.sum())
    if na == 0 or na == n:
        return float("nan")
    ranks = np.where(y == 1)[0] + 1  # 1-based ranks of actives
    ra = na / n
    rie_num = np.sum(np.exp(-alpha * ranks / n)) / na
    rie_den = (1.0 / n) * (1 - np.exp(-alpha)) / (np.exp(alpha / n) - 1)
    rie = rie_num / rie_den
    return float(
        rie * ra * np.sinh(alpha / 2) / (np.cosh(alpha / 2) - np.cosh(alpha / 2 - alpha * ra))
        + 1.0 / (1 - np.exp(alpha * (1 - ra)))
    )


def recall_at_k(y: np.ndarray, score: np.ndarray, k: int) -> float:
    order = np.argsort(-score)
    top = y[order][:k]
    na = int(y.sum())
    return float(top.sum() / na) if na > 0 else float("nan")


def all_metrics(y: np.ndarray, score: np.ndarray) -> dict:
    m = {
        "auroc": float(roc_auc_score(y, score)) if len(set(y)) > 1 else float("nan"),
        "auprc": float(average_precision_score(y, score)) if len(set(y)) > 1 else float("nan"),
        "ef_1pct": enrichment_factor(y, score, 0.01),
        "ef_5pct": enrichment_factor(y, score, 0.05),
        "bedroc": bedroc(y, score),
        "recall_at_50": recall_at_k(y, score, 50),
        "recall_at_100": recall_at_k(y, score, 100),
        "hits_at_20": int(y[np.argsort(-score)[:20]].sum()),
        "hits_at_100": int(y[np.argsort(-score)[:100]].sum()),
        "n_eval": int(len(y)),
        "n_active": int(y.sum()),
    }
    return m
