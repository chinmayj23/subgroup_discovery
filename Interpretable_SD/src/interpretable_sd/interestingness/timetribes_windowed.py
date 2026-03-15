from typing import Dict

import pandas as pd

from SD_longitudinal.src.common import LABEL_COLUMN
from SD_longitudinal.src.labeling import label_syflow_method
from interpretable_sd.utils.metrics import interestingness_metrics


def run_timetribes_windowed(
    samples_df: pd.DataFrame,
    target_col: str,
    cfg: Dict,
) -> Dict:
    print(
        "[Interestingness][TimeTribes] "
        f"rows={len(samples_df)} target={target_col} "
        f"n_seeds={cfg.get('n_seeds', 400)} top_pct={cfg.get('top_percentile', 0.01)}"
    )
    labeled_df = label_syflow_method(
        samples_df,
        target=target_col,
        n_seeds=int(cfg.get("n_seeds", 400)),
        beta=float(cfg.get("beta", 0.5)),
        lambd_div=float(cfg.get("lambd_div", 2.0)),
        top_percentile=float(cfg.get("top_percentile", 0.01)),
        max_overlap=float(cfg.get("max_overlap", 0.95)),
        min_subgroup_size=int(cfg.get("min_subgroup_size", 20)),
        max_subgroup_frac=float(cfg.get("max_subgroup_frac", 0.9)),
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
