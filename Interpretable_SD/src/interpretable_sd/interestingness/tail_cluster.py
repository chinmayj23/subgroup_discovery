from typing import Dict

import pandas as pd

from SD_longitudinal.src.common import LABEL_COLUMN
from SD_longitudinal.src.labeling import label_original_method
from interpretable_sd.utils.metrics import interestingness_metrics


def run_tail_cluster(
    samples_df: pd.DataFrame,
    target_col: str,
    cfg: Dict,
) -> Dict:
    print(
        "[Interestingness][TailCluster] "
        f"rows={len(samples_df)} target={target_col} "
        f"q={cfg.get('high_value_quantile', 0.95)} two_tailed={cfg.get('two_tailed', False)}"
    )
    labeled_df = label_original_method(
        samples_df,
        target=target_col,
        high_value_quantile=float(cfg.get("high_value_quantile", 0.95)),
        two_tailed=bool(cfg.get("two_tailed", False)),
    )
    metrics = interestingness_metrics(
        labeled_df[target_col].values,
        labeled_df[LABEL_COLUMN].values,
    )
    print(
        "[Interestingness][TailCluster] done | "
        f"interesting={metrics.get('n_interesting', 0)}/{metrics.get('n_total', 0)} "
        f"support={metrics.get('support', 0.0):.4f} "
        f"kl={metrics.get('kl_divergence', 0.0):.4f}"
    )
    return {
        "labeled_df": labeled_df,
        "metrics": metrics,
    }
