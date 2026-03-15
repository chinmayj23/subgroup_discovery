from typing import Dict

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def safe_hist_kl(
    p_samples: np.ndarray,
    q_samples: np.ndarray,
    eps: float = 1e-9,
) -> float:
    p = np.asarray(p_samples, dtype=float)
    q = np.asarray(q_samples, dtype=float)
    p = p[np.isfinite(p)]
    q = q[np.isfinite(q)]
    if p.size == 0 or q.size == 0:
        return 0.0

    # Freedman-Diaconis-like adaptive bins with fallback.
    q75, q25 = np.percentile(q, [75, 25])
    iqr = float(q75 - q25)
    if iqr <= 0:
        n_bins = max(5, int(np.sqrt(q.size)))
    else:
        h = 2.0 * iqr / (q.size ** (1.0 / 3.0))
        n_bins = int(np.ceil((np.nanmax(q) - np.nanmin(q)) / max(h, eps)))
        n_bins = max(5, min(100, n_bins))

    bins = np.histogram_bin_edges(q, bins=n_bins)
    p_hist, _ = np.histogram(p, bins=bins, density=True)
    q_hist, _ = np.histogram(q, bins=bins, density=True)

    p_prob = (p_hist + eps) / (p_hist.sum() + eps * len(p_hist))
    q_prob = (q_hist + eps) / (q_hist.sum() + eps * len(q_hist))
    return float(np.sum(p_prob * np.log(p_prob / q_prob)))


def interestingness_metrics(
    target_values: np.ndarray,
    interesting_mask: np.ndarray,
) -> Dict[str, float]:
    y = np.asarray(target_values, dtype=float)
    mask = np.asarray(interesting_mask, dtype=bool)
    valid = np.isfinite(y)
    y = y[valid]
    mask = mask[valid]
    if y.size == 0:
        return {
            "support": 0.0,
            "n_total": 0,
            "n_interesting": 0,
            "mean_all": 0.0,
            "mean_interesting": 0.0,
            "mean_shift": 0.0,
            "kl_divergence": 0.0,
        }

    pos = y[mask]
    if pos.size == 0:
        return {
            "support": 0.0,
            "n_total": int(y.size),
            "n_interesting": 0,
            "mean_all": float(np.mean(y)),
            "mean_interesting": 0.0,
            "mean_shift": float(-np.mean(y)),
            "kl_divergence": 0.0,
        }

    mean_all = float(np.mean(y))
    mean_pos = float(np.mean(pos))
    return {
        "support": float(pos.size / y.size),
        "n_total": int(y.size),
        "n_interesting": int(pos.size),
        "mean_all": mean_all,
        "mean_interesting": mean_pos,
        "mean_shift": float(mean_pos - mean_all),
        "kl_divergence": safe_hist_kl(pos, y),
    }


def classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_score: np.ndarray = None,
) -> Dict[str, float]:
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    out = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
    }
    if y_score is not None:
        y_score = np.asarray(y_score, dtype=float)
        if len(np.unique(y_true)) > 1 and np.isfinite(y_score).all():
            out["roc_auc"] = float(roc_auc_score(y_true, y_score))
    return out
