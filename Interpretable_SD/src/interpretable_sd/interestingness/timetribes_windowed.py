from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, fcluster

from SD_longitudinal.src.common import LABEL_COLUMN
from SD_longitudinal.src.labeling import (
    freedman_diaconis_bins,
    hist_kl,
    jaccard_similarity,
)
from interpretable_sd.utils.metrics import interestingness_metrics


def _append_candidate(
    candidates: List[dict],
    y_full: np.ndarray,
    global_bins: np.ndarray,
    mask: np.ndarray,
    beta: float,
    min_subgroup_size: Optional[int],
    max_subgroup_frac: Optional[float],
    seed: int,
    source: str,
    interval: tuple[float, float],
) -> bool:
    n_total = len(y_full)
    n_sub = int(mask.sum())
    if min_subgroup_size is not None and n_sub < min_subgroup_size:
        return False
    if max_subgroup_frac is not None and n_sub > (n_total * max_subgroup_frac):
        return False

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
            "range": (float(interval[0]), float(interval[1])),
            "source": source,
            "selected": False,
        }
    )
    return True


def _generate_quantile_window_candidates(
    candidates: List[dict],
    y_full: np.ndarray,
    y_boot: np.ndarray,
    global_bins: np.ndarray,
    beta: float,
    min_subgroup_size: Optional[int],
    max_subgroup_frac: Optional[float],
    seed: int,
) -> int:
    # Small quantile windows produce stable, non-degenerate target intervals
    # when 1D linkage collapses to singleton or near-global candidates.
    widths = (0.01, 0.02, 0.04)
    starts = (0.00, 0.10, 0.20, 0.35, 0.50, 0.65, 0.80, 0.90)
    added = 0

    for width in widths:
        for start in starts:
            end = start + width
            if end > 1.0:
                continue
            low, high = np.quantile(y_boot, [start, end])
            mask = (y_full >= low) & (y_full <= high)
            added += int(
                _append_candidate(
                    candidates=candidates,
                    y_full=y_full,
                    global_bins=global_bins,
                    mask=mask,
                    beta=beta,
                    min_subgroup_size=min_subgroup_size,
                    max_subgroup_frac=max_subgroup_frac,
                    seed=seed,
                    source="quantile_window",
                    interval=(low, high),
                )
            )
    return added


def _label_timetribes_with_fallback(
    df: pd.DataFrame,
    target: str,
    n_seeds: int,
    beta: float,
    lambd_div: float,
    top_percentile: float,
    max_overlap: float,
    min_subgroup_size: Optional[int],
    max_subgroup_frac: Optional[float],
    progress_every: int,
) -> pd.DataFrame:
    y_col = df[target]
    if isinstance(y_col, pd.DataFrame):
        y_col = y_col.iloc[:, 0]
    y_col = pd.to_numeric(y_col, errors="coerce")
    y_full = y_col.values
    n_total = len(y_full)
    global_bins = freedman_diaconis_bins(y_full)
    candidates: List[dict] = []
    cluster_candidate_count = 0
    window_candidate_count = 0

    for seed in range(n_seeds):
        if progress_every and (seed % progress_every == 0 or seed == n_seeds - 1):
            print(f"  [TimeTribes] seed {seed + 1}/{n_seeds}")

        rng = np.random.RandomState(seed)
        sample_idx = rng.choice(n_total, size=n_total, replace=True)
        y_boot = y_full[sample_idx].reshape(-1, 1)
        valid_from_clusters = 0

        try:
            Z = linkage(y_boot, method="single")
            heights = Z[:, 2]
            diffs = np.diff(heights)
            threshold = heights[np.argmax(diffs)] if len(diffs) > 0 else 0.0

            if threshold > 0:
                boot_clusters = fcluster(Z, t=threshold, criterion="distance")
                for label in np.unique(boot_clusters):
                    cluster_vals = y_boot[boot_clusters == label]
                    if len(cluster_vals) == 0:
                        continue
                    c_min, c_max = cluster_vals.min(), cluster_vals.max()
                    mask = (y_full >= c_min) & (y_full <= c_max)
                    added = _append_candidate(
                        candidates=candidates,
                        y_full=y_full,
                        global_bins=global_bins,
                        mask=mask,
                        beta=beta,
                        min_subgroup_size=min_subgroup_size,
                        max_subgroup_frac=max_subgroup_frac,
                        seed=seed,
                        source="cluster",
                        interval=(float(c_min), float(c_max)),
                    )
                    valid_from_clusters += int(added)
        except Exception:
            valid_from_clusters = 0

        cluster_candidate_count += valid_from_clusters
        window_candidate_count += _generate_quantile_window_candidates(
            candidates=candidates,
            y_full=y_full,
            y_boot=y_boot.ravel(),
            global_bins=global_bins,
            beta=beta,
            min_subgroup_size=min_subgroup_size,
            max_subgroup_frac=max_subgroup_frac,
            seed=seed,
        )

    print(
        "[Interestingness][TimeTribes] candidates | "
        f"cluster_valid={cluster_candidate_count} "
        f"window_valid={window_candidate_count}"
    )

    if len(candidates) == 0:
        labeled_df = df.copy()
        labeled_df[LABEL_COLUMN] = False
        return labeled_df

    n_select = max(1, int(len(candidates) * top_percentile))
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
            if selected_candidates:
                diversity = float(
                    np.mean(
                        [
                            hist_kl(cand["y_values"], sel["y_values"], global_bins)
                            for sel in selected_candidates
                        ]
                    )
                )

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


def run_timetribes_windowed(
    samples_df: pd.DataFrame,
    target_col: str,
    cfg: Dict,
) -> Dict:
    print(
        "[Interestingness][TimeTribes] "
        f"rows={len(samples_df)} target={target_col} "
        f"n_seeds={cfg.get('n_seeds', 500)} top_pct={cfg.get('top_percentile', 0.005)} "
        f"min_size={cfg.get('min_subgroup_size', None)} max_frac={cfg.get('max_subgroup_frac', None)}"
    )
    min_subgroup_size = cfg.get("min_subgroup_size")
    max_subgroup_frac = cfg.get("max_subgroup_frac")
    labeled_df = _label_timetribes_with_fallback(
        samples_df,
        target=target_col,
        n_seeds=int(cfg.get("n_seeds", 500)),
        beta=float(cfg.get("beta", 0.5)),
        lambd_div=float(cfg.get("lambd_div", 0.0)),
        top_percentile=float(cfg.get("top_percentile", 0.0005)),
        max_overlap=float(cfg.get("max_overlap", 0.95)),
        min_subgroup_size=None if min_subgroup_size is None else int(min_subgroup_size),
        max_subgroup_frac=None if max_subgroup_frac is None else float(max_subgroup_frac),
        progress_every=int(cfg.get("progress_every", 0)),
    )
    metrics = interestingness_metrics(
        labeled_df[target_col].values,
        labeled_df[LABEL_COLUMN].values,
    )
    print(
        "[Interestingness][TimeTribes] done | "
        f"interesting={metrics.get('n_interesting', 0)}/{metrics.get('n_total', 0)} "
        f"support={metrics.get('support', 0.0):.4f} "
        f"kl={metrics.get('kl_divergence', 0.0):.4f}"
    )
    return {
        "labeled_df": labeled_df,
        "metrics": metrics,
    }
