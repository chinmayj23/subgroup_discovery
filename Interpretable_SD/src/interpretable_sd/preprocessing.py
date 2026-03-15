from typing import Dict, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from SD_longitudinal.src.common import LABEL_COLUMN, drop_by_prefix
from SD_longitudinal.src.longitudinal_features import build_longitudinal_features


WindowSpec = Union[int, str]


def extract_year(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        years = pd.to_numeric(series, errors="coerce")
    else:
        parsed = pd.to_datetime(series, errors="coerce")
        years = parsed.dt.year
        if years.notna().mean() < 0.8:
            fallback = pd.to_numeric(series.astype(str).str.extract(r"(\d{4})")[0], errors="coerce")
            years = years.fillna(fallback)
    return years.astype("Int64")


def normalize_time_key(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        numeric = pd.to_numeric(series, errors="coerce")
        return numeric.round().astype("Int64").astype(str)

    parsed = pd.to_datetime(series, errors="coerce")
    out = pd.Series(index=series.index, dtype=object)
    parsed_mask = parsed.notna()
    out.loc[parsed_mask] = parsed.loc[parsed_mask].dt.strftime("%Y-%m-%d")
    out.loc[~parsed_mask] = series.loc[~parsed_mask].astype(str).str.strip()
    return out.fillna("")


def apply_target_transform(df: pd.DataFrame, target_cfg: Dict) -> Tuple[pd.DataFrame, str]:
    cfg = dict(target_cfg)
    mode = str(cfg.get("mode", "column")).lower()
    out = df.copy()

    if mode == "column":
        target_col = cfg["column"]
        return out, target_col

    if mode == "ratio":
        numerator = cfg["numerator"]
        denominator = cfg["denominator"]
        out_col = cfg.get("output_col", f"{numerator}_over_{denominator}")
        num = pd.to_numeric(out[numerator], errors="coerce")
        den = pd.to_numeric(out[denominator], errors="coerce")
        out[out_col] = num / (den + 1e-9)
        out.loc[~np.isfinite(out[out_col]), out_col] = np.nan
        return out, out_col

    raise ValueError(f"Unsupported target transform mode: {mode}")


def build_panel_feature_table(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    target_col: str,
    feature_cfg: Dict,
) -> Tuple[pd.DataFrame, Dict[str, str], Dict[str, str]]:
    print(
        "[Preprocess] build_panel_feature_table "
        f"target={target_col} lags={feature_cfg.get('lags')} windows={feature_cfg.get('windows')} "
        f"panel_stride={feature_cfg.get('panel_stride', 1)}"
    )
    lags = list(feature_cfg.get("lags", [1, 3, 5, 10]))
    windows = list(feature_cfg.get("windows", [3, 5, 10]))
    label_window = int(feature_cfg.get("label_window", max(lags) if lags else 1))
    features_df, question_map, feature_base_map = build_longitudinal_features(
        df,
        id_col=id_col,
        date_col=date_col,
        target_col=target_col,
        lags=lags,
        windows=windows,
        label_window=label_window,
        min_history=int(feature_cfg.get("min_history", 0)),
        mode="panel",
        panel_stride=int(feature_cfg.get("panel_stride", 1)),
        include_pct_change=bool(feature_cfg.get("include_pct_change", True)),
        include_volatility=bool(feature_cfg.get("include_volatility", True)),
        indicator_include=feature_cfg.get("indicator_include"),
        indicator_exclude=feature_cfg.get("indicator_exclude"),
        enforce_min_history=bool(feature_cfg.get("enforce_min_history", False)),
    )
    features_df["_year"] = extract_year(features_df[date_col])
    features_df["_time_key"] = normalize_time_key(features_df[date_col])
    print(
        "[Preprocess] panel features ready | "
        f"rows={len(features_df)} cols={len(features_df.columns)}"
    )
    return features_df, question_map, feature_base_map


def _window_agg(values: np.ndarray, mode: str) -> float:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return np.nan
    if mode == "last":
        return float(arr[-1])
    if mode == "delta":
        return float(arr[-1] - arr[0])
    if mode == "trend":
        denom = max(1, arr.size - 1)
        return float((arr[-1] - arr[0]) / denom)
    return float(np.mean(arr))


def build_window_samples(
    panel_df: pd.DataFrame,
    id_col: str,
    date_col: str,
    target_value_col: str = "label_target_value",
    window_specs: Iterable[WindowSpec] = (1, "all"),
    target_aggregation: str = "mean",
    require_full_window: bool = True,
) -> pd.DataFrame:
    print(
        "[Preprocess] build_window_samples "
        f"window_specs={list(window_specs)} agg={target_aggregation} full_only={require_full_window}"
    )
    records: List[Dict] = []

    for entity, grp in panel_df.groupby(id_col, sort=False):
        g = grp.sort_values(date_col).reset_index(drop=True)
        y = pd.to_numeric(g[target_value_col], errors="coerce").values
        years = extract_year(g[date_col])
        n = len(g)
        if n == 0:
            continue

        for end_idx in range(n):
            for spec in window_specs:
                if isinstance(spec, str) and spec.lower() == "all":
                    if end_idx != n - 1:
                        continue
                    start_idx = 0
                    window_tag = "all"
                else:
                    win = int(spec)
                    start_idx = end_idx - win + 1
                    if require_full_window and start_idx < 0:
                        continue
                    start_idx = max(0, start_idx)
                    window_tag = str(win)

                segment = y[start_idx : end_idx + 1]
                if np.isfinite(segment).sum() == 0:
                    continue

                base_row = g.iloc[end_idx].to_dict()
                w_start_year = years.iloc[start_idx]
                w_end_year = years.iloc[end_idx]
                window_len = int(end_idx - start_idx + 1)

                arr = np.asarray(segment, dtype=float)
                arr = arr[np.isfinite(arr)]
                if arr.size == 0:
                    continue
                delta = float(arr[-1] - arr[0])
                trend = float(delta / max(1, arr.size - 1))

                base_row.update(
                    {
                        "window_spec": window_tag,
                        "window_length": window_len,
                        "window_start_year": None if pd.isna(w_start_year) else int(w_start_year),
                        "window_end_year": None if pd.isna(w_end_year) else int(w_end_year),
                        "window_target_mean": float(np.mean(arr)),
                        "window_target_delta": delta,
                        "window_target_trend": trend,
                        "window_target_std": float(np.std(arr)),
                        "window_target_score": _window_agg(segment, target_aggregation),
                        "sample_id": (
                            f"{entity}__{window_tag}__"
                            f"{'' if pd.isna(w_start_year) else int(w_start_year)}_"
                            f"{'' if pd.isna(w_end_year) else int(w_end_year)}_"
                            f"{end_idx}"
                        ),
                    }
                )
                records.append(base_row)

    if not records:
        print("[Preprocess] window samples empty.")
        return pd.DataFrame()
    out = pd.DataFrame(records)
    out = out.dropna(subset=[id_col, date_col, "window_target_score"])
    print(
        "[Preprocess] window samples ready | "
        f"rows={len(out)} subjects={out[id_col].nunique() if id_col in out.columns else 0}"
    )
    return out.reset_index(drop=True)


def _smooth_positive_runs(labels: np.ndarray, min_interesting_run_length: int) -> np.ndarray:
    if min_interesting_run_length <= 1 or labels.size == 0:
        return labels.astype(bool)

    out = labels.astype(bool).copy()
    start = 0
    n = len(out)
    while start < n:
        end = start + 1
        while end < n and out[end] == out[start]:
            end += 1
        run_len = end - start
        if out[start] and run_len < min_interesting_run_length:
            out[start:end] = False
        start = end
    return out


def build_interest_segments(
    raw_labeled_df: pd.DataFrame,
    panel_df: pd.DataFrame,
    id_col: str,
    date_col: str,
    target_col: str,
    target_aggregation: str = "mean",
    min_interesting_run_length: int = 1,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    print(
        "[Preprocess] build_interest_segments "
        f"agg={target_aggregation} min_interesting_run_length={min_interesting_run_length}"
    )
    required = [id_col, date_col, target_col, LABEL_COLUMN]
    raw = raw_labeled_df.dropna(subset=[c for c in [id_col, date_col, target_col] if c in raw_labeled_df.columns]).copy()
    if raw.empty:
        print("[Preprocess] no labeled rows available for segmentation.")
        return pd.DataFrame(), pd.DataFrame()

    raw["_year"] = extract_year(raw[date_col])
    raw["_time_key"] = normalize_time_key(raw[date_col])
    raw = raw.sort_values([id_col, date_col]).reset_index(drop=True)

    run_rows: List[pd.DataFrame] = []
    for entity, grp in raw.groupby(id_col, sort=False):
        g = grp.sort_values(date_col).copy()
        labels = _smooth_positive_runs(
            g[LABEL_COLUMN].astype(bool).to_numpy(),
            min_interesting_run_length=int(min_interesting_run_length),
        )
        g[LABEL_COLUMN] = labels
        run_id = np.zeros(len(g), dtype=int)
        for i in range(1, len(g)):
            run_id[i] = run_id[i - 1] + int(labels[i] != labels[i - 1])
        g["_segment_idx"] = run_id
        run_rows.append(g)

    row_labeled_df = pd.concat(run_rows, ignore_index=True) if run_rows else pd.DataFrame(columns=required)
    if row_labeled_df.empty:
        print("[Preprocess] no rows after segmentation smoothing.")
        return row_labeled_df, pd.DataFrame()

    segment_records: List[Dict] = []
    for (entity, seg_idx), grp in row_labeled_df.groupby([id_col, "_segment_idx"], sort=False):
        g = grp.sort_values(date_col)
        arr = pd.to_numeric(g[target_col], errors="coerce").to_numpy(dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            continue
        start_year = g["_year"].iloc[0]
        end_year = g["_year"].iloc[-1]
        start_date = g[date_col].iloc[0]
        end_date = g[date_col].iloc[-1]
        delta = float(arr[-1] - arr[0])
        segment_records.append(
            {
                id_col: entity,
                "_segment_idx": int(seg_idx),
                LABEL_COLUMN: bool(g[LABEL_COLUMN].iloc[0]),
                "segment_id": (
                    f"{entity}__seg{int(seg_idx):02d}__"
                    f"{'' if pd.isna(start_year) else int(start_year)}_"
                    f"{'' if pd.isna(end_year) else int(end_year)}__"
                    f"{'I' if bool(g[LABEL_COLUMN].iloc[0]) else 'N'}"
                ),
                "segment_start_date": start_date,
                "segment_end_date": end_date,
                "segment_start_year": None if pd.isna(start_year) else int(start_year),
                "segment_end_year": None if pd.isna(end_year) else int(end_year),
                "segment_length": int(len(g)),
                "segment_target_mean": float(np.mean(arr)),
                "segment_target_delta": delta,
                "segment_target_trend": float(delta / max(1, arr.size - 1)),
                "segment_target_std": float(np.std(arr)),
                "segment_target_score": _window_agg(arr, target_aggregation),
            }
        )

    if not segment_records:
        print("[Preprocess] no segment records created.")
        return row_labeled_df, pd.DataFrame()

    segment_stats = pd.DataFrame(segment_records)

    panel_aligned = panel_df.copy()
    if "_time_key" not in panel_aligned.columns:
        panel_aligned["_time_key"] = normalize_time_key(panel_aligned[date_col])
    panel_aligned = panel_aligned.merge(
        row_labeled_df[[id_col, "_time_key", LABEL_COLUMN, "_segment_idx"]],
        on=[id_col, "_time_key"],
        how="inner",
        suffixes=("", "_label"),
    )
    if panel_aligned.empty:
        print("[Preprocess] panel/label alignment empty after merge.")
        return row_labeled_df, pd.DataFrame()

    panel_aligned = panel_aligned.sort_values([id_col, date_col])
    segment_last_rows = (
        panel_aligned.groupby([id_col, "_segment_idx"], sort=False, as_index=False)
        .tail(1)
        .copy()
    )
    segment_df = segment_last_rows.merge(
        segment_stats,
        on=[id_col, "_segment_idx", LABEL_COLUMN],
        how="inner",
    )
    segment_df = segment_df.sort_values([id_col, "segment_start_year", "segment_end_year"]).reset_index(drop=True)

    print(
        "[Preprocess] interest segments ready | "
        f"rows={len(segment_df)} subjects={segment_df[id_col].nunique() if id_col in segment_df.columns else 0} "
        f"interesting_segments={int(segment_df[LABEL_COLUMN].sum()) if LABEL_COLUMN in segment_df.columns else 0}"
    )
    return row_labeled_df.reset_index(drop=True), segment_df


def split_train_test_by_subject(
    df: pd.DataFrame,
    id_col: str,
    label_col: str = LABEL_COLUMN,
    test_size: float = 0.25,
    seed: int = 42,
    max_attempts: int = 50,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    subject_df = (
        df[[id_col, label_col]]
        .groupby(id_col, as_index=False)[label_col]
        .max()
    )
    ids = subject_df[id_col].values
    y_subject = subject_df[label_col].astype(int).values
    tr_ids = te_ids = None
    last_split = None
    for attempt in range(max_attempts):
        rs = int(seed + attempt)
        try:
            tr_try, te_try = train_test_split(
                ids,
                test_size=test_size,
                random_state=rs,
                stratify=y_subject if len(np.unique(y_subject)) > 1 else None,
            )
        except ValueError:
            tr_try, te_try = train_test_split(
                ids,
                test_size=test_size,
                random_state=rs,
            )

        last_split = (tr_try, te_try)
        if len(np.unique(y_subject)) < 2:
            tr_ids, te_ids = tr_try, te_try
            break

        tr_y = subject_df.set_index(id_col).loc[tr_try, label_col].astype(int).values
        te_y = subject_df.set_index(id_col).loc[te_try, label_col].astype(int).values
        if len(np.unique(tr_y)) >= 2 and len(np.unique(te_y)) >= 2:
            tr_ids, te_ids = tr_try, te_try
            break

    if tr_ids is None or te_ids is None:
        tr_ids, te_ids = last_split

    tr_set, te_set = set(tr_ids.tolist()), set(te_ids.tolist())
    row_ids = df[id_col].values
    tr_mask = np.isin(row_ids, list(tr_set))
    te_mask = np.isin(row_ids, list(te_set))
    return tr_mask, te_mask, np.asarray(list(tr_set)), np.asarray(list(te_set))


def prepare_xy_with_ids(
    df: pd.DataFrame,
    id_col: str,
    target_col: str,
    label_col: str = LABEL_COLUMN,
    exclude_columns: Iterable[str] = (),
    exclude_prefixes: Iterable[str] = (),
    meta_columns: Iterable[str] = (),
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray, pd.DataFrame]:
    y = df[label_col].astype(int)
    row_ids = df[id_col]
    target_series = pd.to_numeric(df[target_col], errors="coerce")
    meta_cols = [c for c in meta_columns if c in df.columns]
    meta_df = df[meta_cols].copy() if meta_cols else pd.DataFrame(index=df.index)

    # Always drop identifier/meta supervision columns from model features.
    drop_cols = [id_col, target_col, label_col]
    drop_cols.extend([c for c in exclude_columns if c in df.columns])
    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    if exclude_prefixes:
        pref_cols = drop_by_prefix(feature_df.columns, exclude_prefixes)
        feature_df = feature_df.drop(columns=pref_cols)

    X_df = feature_df.select_dtypes(include=[np.number]).copy()
    if X_df.empty:
        return np.array([]), np.array([]), [], np.array([]), np.array([]), pd.DataFrame()

    valid = ~(y.isna() | row_ids.isna() | target_series.isna())
    X_df = X_df[valid]
    y = y[valid]
    row_ids = row_ids[valid]
    target_series = target_series[valid]
    meta_df = meta_df.loc[valid].reset_index(drop=True)

    return X_df.values, y.values, list(X_df.columns), target_series.values, row_ids.values, meta_df


def preprocess_xy_train_only(
    X: np.ndarray,
    feature_names: List[str],
    train_mask: np.ndarray,
    max_missing_frac: float = 0.95,
    impute_missing: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    if X.size == 0:
        return X, feature_names

    X_proc = np.asarray(X, dtype=float).copy()
    X_train = X_proc[train_mask]
    if X_train.size == 0:
        return np.array([]), []

    miss_frac = np.mean(np.isnan(X_train), axis=0)
    keep = miss_frac <= max_missing_frac
    if not np.any(keep):
        return np.array([]), []

    X_proc = X_proc[:, keep]
    kept_names = [f for f, k in zip(feature_names, keep) if k]

    if impute_missing:
        med = np.nanmedian(X_proc[train_mask], axis=0)
        med = np.where(np.isnan(med), 0.0, med)
        nan_mask = np.isnan(X_proc)
        if nan_mask.any():
            r, c = np.where(nan_mask)
            X_proc[r, c] = med[c]
        return X_proc, kept_names

    finite_cols = np.isfinite(X_proc[train_mask]).all(axis=0)
    if not np.any(finite_cols):
        return np.array([]), []
    return X_proc[:, finite_cols], [f for f, k in zip(kept_names, finite_cols) if k]
