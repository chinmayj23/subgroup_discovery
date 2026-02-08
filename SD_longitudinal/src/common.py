import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

LABEL_COLUMN = "is_interesting_subgroup"


def load_json(path: str) -> dict:
    # Handle UTF-8 BOM safely (common on Windows-edited JSON files)
    raw = Path(path).read_text(encoding="utf-8-sig")
    return json.loads(raw)


def save_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def drop_by_prefix(columns: Iterable[str], prefixes: Iterable[str]) -> List[str]:
    prefixes = tuple(prefixes) if prefixes else tuple()
    return [c for c in columns if prefixes and c.startswith(prefixes)]


def prepare_xy(
    df: pd.DataFrame,
    target: str,
    label_column: str = LABEL_COLUMN,
    exclude_columns: Optional[Iterable[str]] = None,
    exclude_prefixes: Optional[Iterable[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found")

    y = df[label_column].astype(int)
    drop_cols = [target, label_column]
    if exclude_columns:
        drop_cols.extend([c for c in exclude_columns if c in df.columns])

    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    if exclude_prefixes:
        prefix_cols = drop_by_prefix(feature_df.columns, exclude_prefixes)
        feature_df = feature_df.drop(columns=prefix_cols)

    numeric_df = feature_df.select_dtypes(include=[np.number]).copy()
    if numeric_df.empty:
        return np.array([]), np.array([]), []

    valid_mask = ~(numeric_df.isna().any(axis=1) | y.isna())
    numeric_df = numeric_df[valid_mask]
    y = y[valid_mask]

    return numeric_df.values, y.values, list(numeric_df.columns)


def prepare_xy_with_target(
    df: pd.DataFrame,
    target: str,
    label_column: str = LABEL_COLUMN,
    exclude_columns: Optional[Iterable[str]] = None,
    exclude_prefixes: Optional[Iterable[str]] = None,
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found")

    y = df[label_column].astype(int)
    target_series = pd.to_numeric(df[target], errors="coerce")

    drop_cols = [target, label_column]
    if exclude_columns:
        drop_cols.extend([c for c in exclude_columns if c in df.columns])

    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    if exclude_prefixes:
        prefix_cols = drop_by_prefix(feature_df.columns, exclude_prefixes)
        feature_df = feature_df.drop(columns=prefix_cols)

    numeric_df = feature_df.select_dtypes(include=[np.number]).copy()
    if numeric_df.empty:
        return np.array([]), np.array([]), [], np.array([])

    valid_mask = ~(numeric_df.isna().any(axis=1) | y.isna() | target_series.isna())
    numeric_df = numeric_df[valid_mask]
    y = y[valid_mask]
    target_series = target_series[valid_mask]

    return numeric_df.values, y.values, list(numeric_df.columns), target_series.values
