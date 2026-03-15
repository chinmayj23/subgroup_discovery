from pathlib import Path
from typing import Dict

import pandas as pd

from SD_longitudinal.src.common import LABEL_COLUMN
from interpretable_sd.utils.io import save_json


def save_syflow_like_report(
    out_path: Path,
    method_name: str,
    labeled_df: pd.DataFrame,
    target_col: str,
    interestingness_metrics_payload: Dict,
    rule_metrics_payload: Dict,
    baseline_payload: Dict,
) -> None:
    positives = (
        labeled_df[labeled_df[LABEL_COLUMN] == True]
        .sort_values(target_col, ascending=False)
        .head(25)
    )
    subgroup_preview = []
    cols = [
        c
        for c in [
            "segment_id",
            "segment_start_year",
            "segment_end_year",
            "segment_length",
            "sample_id",
            "window_start_year",
            "window_end_year",
            "window_length",
            target_col,
        ]
        if c in positives.columns
    ]
    for _, row in positives[cols].iterrows():
        subgroup_preview.append({k: row[k] for k in cols})

    payload = {
        "method": method_name,
        "n_samples": int(len(labeled_df)),
        "n_interesting": int(labeled_df[LABEL_COLUMN].sum()),
        "interestingness_metrics": interestingness_metrics_payload,
        "rule_module_metrics": rule_metrics_payload,
        "baselines": baseline_payload,
        "subgroup_preview": subgroup_preview,
    }
    save_json(out_path, payload)
