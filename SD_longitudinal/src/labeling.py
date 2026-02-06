from dataclasses import dataclass
from typing import List

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.stats import entropy

from .common import LABEL_COLUMN


def freedman_diaconis_bins(data: np.ndarray) -> np.ndarray:
    data = np.asarray(data).ravel()
    if data.size < 2:
        return np.array([data.min(), data.max()])
    q75, q25 = np.percentile(data, [75, 25])
    iqr = q75 - q25
    if iqr == 0:
        nbins = max(1, int(np.sqrt(data.size)))
    else:
        h = 2 * iqr / (data.size ** (1 / 3))
        nbins = int(np.ceil((data.max() - data.min()) / h)) if h > 0 else 10
    nbins = max(1, nbins)
    return np.histogram_bin_edges(data, bins=nbins)


def hist_kl(p_samples: np.ndarray, q_samples: np.ndarray, global_bins: np.ndarray, eps: float = 1e-9) -> float:
    p, q = np.asarray(p_samples).ravel(), np.asarray(q_samples).ravel()
    p_hist, _ = np.histogram(p, bins=global_bins, density=True)
    q_hist, _ = np.histogram(q, bins=global_bins, density=True)
    p_prob = (p_hist + eps) / (p_hist.sum() + eps * len(p_hist))
    q_prob = (q_hist + eps) / (q_hist.sum() + eps * len(q_hist))
    return float(entropy(p_prob, q_prob))


def jaccard_similarity(mask1: np.ndarray, mask2: np.ndarray) -> float:
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    return intersection / union if union > 0 else 0.0


def label_original_method(
    df: pd.DataFrame,
    target: str,
    high_value_quantile: float = 0.94,
    two_tailed: bool = False,
) -> pd.DataFrame:
    y_col = df[target]
    if isinstance(y_col, pd.DataFrame):
        y_col = y_col.iloc[:, 0]
    y_col = pd.to_numeric(y_col, errors="coerce")
    y = y_col.values.reshape(-1, 1)

    try:
        Z = linkage(y, method="single")
        heights = Z[:, 2]
        diffs = np.diff(heights)
        threshold = heights[np.argmax(diffs)] if len(diffs) > 0 else 0
        clusters = fcluster(Z, t=threshold, criterion="distance")
        unique_clusters, counts = np.unique(clusters, return_counts=True)
        largest_cluster_label = unique_clusters[np.argmax(counts)]
        interesting_mask = clusters != largest_cluster_label
    except Exception:
        interesting_mask = np.zeros(len(y), dtype=bool)

    upper = y_col.quantile(high_value_quantile)
    interesting_mask |= y_col >= upper

    if two_tailed:
        lower = y_col.quantile(1.0 - high_value_quantile)
        interesting_mask |= y_col <= lower

    labeled_df = df.copy()
    labeled_df[LABEL_COLUMN] = interesting_mask
    return labeled_df


def label_syflow_method(
    df: pd.DataFrame,
    target: str,
    n_seeds: int = 1000,
    beta: float = 0.5,
    lambd_div: float = 2.0,
    top_percentile: float = 0.01,
    max_overlap: float = 0.95,
    min_subgroup_size: int = 50,
    max_subgroup_frac: float = 0.90,
    progress_every: int = 0,
) -> pd.DataFrame:
    y_col = df[target]
    if isinstance(y_col, pd.DataFrame):
        y_col = y_col.iloc[:, 0]
    y_col = pd.to_numeric(y_col, errors="coerce")
    y_full = y_col.values
    n_total = len(y_full)

    global_bins = freedman_diaconis_bins(y_full)
    candidates: List[dict] = []

    for seed in range(n_seeds):
        if progress_every and (seed % progress_every == 0 or seed == n_seeds - 1):
            print(f"  [TimeTribes] seed {seed + 1}/{n_seeds}")
        rng = np.random.RandomState(seed)
        sample_idx = rng.choice(n_total, size=n_total, replace=True)
        y_boot = y_full[sample_idx].reshape(-1, 1)

        Z = linkage(y_boot, method="single")
        heights = Z[:, 2]
        diffs = np.diff(heights)
        threshold = heights[np.argmax(diffs)] if len(diffs) > 0 else 0.0
        if threshold == 0:
            continue

        boot_clusters = fcluster(Z, t=threshold, criterion="distance")
        unique_labels = np.unique(boot_clusters)

        for label in unique_labels:
            cluster_vals = y_boot[boot_clusters == label]
            if len(cluster_vals) == 0:
                continue
            c_min, c_max = cluster_vals.min(), cluster_vals.max()

            mask = (y_full >= c_min) & (y_full <= c_max)
            n_sub = int(mask.sum())

            if n_sub < min_subgroup_size or n_sub > (n_total * max_subgroup_frac):
                continue

            sub_y = y_full[mask]
            kl_exc = hist_kl(sub_y, y_full, global_bins)
            size_term = (n_sub / n_total) ** beta
            quality = size_term * kl_exc

            candidates.append(
                {
                    "seed": seed,
                    "mask": mask,
                    "y_values": sub_y,
                    "size_term": size_term,
                    "quality": quality,
                    "range": (float(c_min), float(c_max)),
                    "selected": False,
                }
            )

    if len(candidates) == 0:
        labeled_df = df.copy()
        labeled_df[LABEL_COLUMN] = False
        return labeled_df

    n_select = int(len(candidates) * top_percentile)
    n_select = max(1, n_select)

    selected_candidates: List[dict] = []

    for _ in range(n_select):
        best_candidate = None
        best_score = -np.inf

        for cand in candidates:
            if cand["selected"]:
                continue

            is_duplicate = False
            for sel in selected_candidates:
                if jaccard_similarity(cand["mask"], sel["mask"]) > max_overlap:
                    is_duplicate = True
                    break
            if is_duplicate:
                cand["selected"] = True
                continue

            diversity = 0.0
            if len(selected_candidates) > 0:
                for sel in selected_candidates:
                    diversity += hist_kl(cand["y_values"], sel["y_values"], global_bins)
                diversity /= len(selected_candidates)

            total_score = cand["quality"] + (lambd_div * cand["size_term"] * diversity)
            if total_score > best_score:
                best_score = total_score
                best_candidate = cand

        if best_candidate is None:
            break

        best_candidate["selected"] = True
        selected_candidates.append(best_candidate)

    final_mask = np.zeros(n_total, dtype=bool)
    for cand in selected_candidates:
        final_mask |= cand["mask"]

    labeled_df = df.copy()
    labeled_df[LABEL_COLUMN] = final_mask
    return labeled_df
