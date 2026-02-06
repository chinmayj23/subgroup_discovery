import re
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
import polars as pl


def _ensure_date(pldf: pl.DataFrame, date_col: str) -> pl.DataFrame:
    if pldf[date_col].dtype in [pl.Date, pl.Datetime]:
        return pldf
    if pldf[date_col].dtype in [pl.Int32, pl.Int64, pl.Float64]:
        return pldf.with_columns(pl.col(date_col).cast(pl.Int64))
    return pldf.with_columns(
        pl.col(date_col).cast(pl.Utf8).str.to_date("%Y-%m-%d", strict=False).alias(date_col)
    )


def _humanize_indicator(name: str) -> str:
    return (
        name.replace("%", " percent")
        .replace("_", " ")
        .replace("  ", " ")
        .strip()
        .title()
    )


def build_question_map(feature_names: Iterable[str]) -> Tuple[Dict[str, str], Dict[str, str]]:
    question_map: Dict[str, str] = {}
    feature_base_map: Dict[str, str] = {}

    for feat in feature_names:
        if feat.startswith("label_target_"):
            continue

        m_latest = re.match(r"^latest_(.+)$", feat)
        if m_latest:
            base = m_latest.group(1)
            feature_base_map[feat] = base
            question_map[feat] = f"Current {_humanize_indicator(base)}"
            continue

        m = re.match(r"^(lag|delta|pct_change|trend|vol)(\d+)y_(.+)$", feat)
        if not m:
            continue
        kind, years, base = m.group(1), m.group(2), m.group(3)
        feature_base_map[feat] = base
        base_h = _humanize_indicator(base)

        if kind == "lag":
            question_map[feat] = f"{base_h} {years} years ago"
        elif kind == "delta":
            question_map[feat] = f"Change in {base_h} over last {years} years"
        elif kind == "pct_change":
            question_map[feat] = f"Percent change in {base_h} over last {years} years"
        elif kind == "trend":
            question_map[feat] = f"Average yearly change in {base_h} over last {years} years"
        elif kind == "vol":
            question_map[feat] = f"Volatility of {base_h} over last {years} years"

    return question_map, feature_base_map


def build_longitudinal_features(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    target_col: str,
    lags: List[int],
    windows: List[int],
    label_window: int,
    min_history: int,
    mode: str = "snapshot",
    panel_stride: int = 1,
    include_pct_change: bool = True,
    include_volatility: bool = False,
    indicator_include: Optional[List[str]] = None,
    indicator_exclude: Optional[List[str]] = None,
) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, str]]:
    if indicator_include:
        indicator_include = set(indicator_include)
    if indicator_exclude:
        indicator_exclude = set(indicator_exclude)

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c not in [id_col, date_col]]

    if indicator_include:
        numeric_cols = [c for c in numeric_cols if c in indicator_include]
    if indicator_exclude:
        numeric_cols = [c for c in numeric_cols if c not in indicator_exclude]

    # Always include target for labeling features
    if target_col not in numeric_cols and target_col in df.columns:
        numeric_cols.append(target_col)

    pldf = pl.from_pandas(df[[id_col, date_col] + numeric_cols])
    pldf = _ensure_date(pldf, date_col)
    pldf = pldf.filter(pl.col(date_col).is_not_null())

    pldf = pldf.with_columns(pl.col(target_col).cast(pl.Float64, strict=False))
    pldf = pldf.filter(pl.col(target_col).is_not_null())

    pldf = pldf.sort([id_col, date_col])

    feature_exprs: List[pl.Expr] = []
    for col in numeric_cols:
        feature_exprs.append(pl.col(col).alias(f"latest_{col}"))
        for lag in lags:
            shifted = pl.col(col).shift(lag).over(id_col)
            feature_exprs.append(shifted.alias(f"lag{lag}y_{col}"))
            feature_exprs.append((pl.col(col) - shifted).alias(f"delta{lag}y_{col}"))
            if include_pct_change:
                feature_exprs.append(
                    ((pl.col(col) - shifted) / (shifted + 1e-9)).alias(
                        f"pct_change{lag}y_{col}"
                    )
                )
        for window in windows:
            shifted = pl.col(col).shift(window).over(id_col)
            feature_exprs.append(((pl.col(col) - shifted) / window).alias(f"trend{window}y_{col}"))
            if include_volatility:
                feature_exprs.append(
                    pl.col(col).rolling_std(window).over(id_col).alias(f"vol{window}y_{col}")
                )

    label_exprs: List[pl.Expr] = [
        pl.col(target_col).alias("label_target_value"),
        (pl.col(target_col) - pl.col(target_col).shift(label_window).over(id_col)).alias(
            f"label_target_delta{label_window}"
        ),
        ((pl.col(target_col) - pl.col(target_col).shift(label_window).over(id_col)) / label_window).alias(
            f"label_target_trend{label_window}"
        ),
    ]

    pldf = pldf.with_columns(feature_exprs + label_exprs)
    # Drop raw numeric columns to avoid leakage and keep questions explicit.
    pldf = pldf.drop(numeric_cols)

    required_lag = max([min_history, label_window, max(lags) if lags else 1])
    required_col = f"lag{required_lag}y_{target_col}"
    if required_col in pldf.columns:
        pldf = pldf.filter(pl.col(required_col).is_not_null())

    if mode == "snapshot":
        pldf = pldf.group_by(id_col, maintain_order=True).last()
    else:
        # Polars requires a column for cum_count; use id_col as a stable proxy
        pldf = pldf.with_columns(pl.col(id_col).cum_count().over(id_col).alias("_t_idx"))
        if panel_stride > 1:
            pldf = pldf.filter((pl.col("_t_idx") % panel_stride) == 0)
        pldf = pldf.drop(["_t_idx"])

    out_df = pldf.to_pandas()
    question_map, feature_base_map = build_question_map(out_df.columns)
    return out_df, question_map, feature_base_map
