import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.subgroup_pipelines import (
    LABEL_COLUMN,
    label_original_method,
    label_syflow_method,
    prepare_xy,
    rebuild_forest_for_seed,
    run_forest_search,
    save_forest_accuracy_plot,
    save_forest_tree_artifacts,
    save_kde_plot,
    save_outputs,
)


def run_xgboost_baseline(snapshot, target_col, date_col, output_dir, cfg):
    if not cfg or not cfg.get("enabled", False):
        return

    try:
        import xgboost as xgb
    except ImportError as exc:
        raise ImportError("xgboost is not installed in the current environment") from exc

    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_squared_error, r2_score

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = snapshot.copy()
    if date_col in df.columns:
        df = df.sort_values(date_col)

    feature_df = df.select_dtypes(include=[np.number]).copy()
    feature_df = feature_df.drop(columns=[target_col], errors="ignore")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in snapshot")
    y = df[target_col]
    if isinstance(y, pd.DataFrame):
        y = y.iloc[:, 0]
    y = pd.to_numeric(y, errors="coerce")

    valid_mask = ~(feature_df.isna().any(axis=1) | y.isna())
    X = feature_df.loc[valid_mask]
    y = y.loc[valid_mask]

    if len(X) < cfg.get("min_samples", 50):
        return

    params = cfg.get(
        "params",
        {
            "n_estimators": 300,
            "max_depth": 4,
            "learning_rate": 0.05,
            "subsample": 0.9,
            "colsample_bytree": 0.9,
            "objective": "reg:squarederror",
            "random_state": 42,
        },
    )

    n_splits = cfg.get("n_splits", 5)
    tscv = TimeSeriesSplit(n_splits=n_splits)

    fold_metrics = []
    model = None
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = xgb.XGBRegressor(**params)
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
        r2 = float(r2_score(y_test, preds))
        fold_metrics.append(
            {
                "fold": fold,
                "rmse": rmse,
                "r2": r2,
                "n_train": int(len(train_idx)),
                "n_test": int(len(test_idx)),
            }
        )

    metrics_df = pd.DataFrame(fold_metrics)
    metrics_df.to_csv(output_dir / "xgboost_metrics.csv", index=False)

    summary = {
        "rmse_mean": float(metrics_df["rmse"].mean()),
        "rmse_std": float(metrics_df["rmse"].std()),
        "r2_mean": float(metrics_df["r2"].mean()),
        "r2_std": float(metrics_df["r2"].std()),
        "n_splits": int(n_splits),
        "n_samples": int(len(X)),
    }
    (output_dir / "xgboost_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    if model is not None:
        importance = model.get_booster().get_score(importance_type="gain")
        if importance:
            imp_df = (
                pd.DataFrame(
                    [{"feature": k, "gain": float(v)} for k, v in importance.items()]
                )
                .sort_values("gain", ascending=False)
                .reset_index(drop=True)
            )
            imp_df.to_csv(output_dir / "xgboost_feature_importance.csv", index=False)


def _series_trend(values):
    if len(values) < 2:
        return np.nan
    x = np.arange(len(values), dtype=float)
    y = np.asarray(values, dtype=float)
    slope = np.polyfit(x, y, 1)[0]
    return float(slope)


def _summarize_series(values, window_size):
    values = pd.Series(values).dropna()
    if values.empty:
        return {
            "latest": np.nan,
            "lag1": np.nan,
            "lag3": np.nan,
            "roll_mean": np.nan,
            "roll_std": np.nan,
            "trend": np.nan,
            "delta1": np.nan,
            "delta3": np.nan,
            "cagr_w": np.nan,
            "min_w": np.nan,
            "recovery_w": np.nan,
        }
    latest = float(values.iloc[-1])
    lag1 = float(values.iloc[-2]) if len(values) > 1 else np.nan
    lag3 = float(values.iloc[-4]) if len(values) > 3 else np.nan
    window = values.tail(window_size)
    delta1 = latest - lag1 if not np.isnan(lag1) else np.nan
    delta3 = latest - lag3 if not np.isnan(lag3) else np.nan
    min_w = float(window.min()) if len(window) > 0 else np.nan
    recovery_w = latest - min_w if not np.isnan(min_w) else np.nan
    cagr_w = np.nan
    if len(window) >= 2:
        start = window.iloc[0]
        end = window.iloc[-1]
        if start > 0 and end > 0:
            cagr_w = float((end / start) ** (1 / (len(window) - 1)) - 1)
    return {
        "latest": latest,
        "lag1": lag1,
        "lag3": lag3,
        "roll_mean": float(window.mean()) if not window.empty else np.nan,
        "roll_std": float(window.std()) if len(window) > 1 else np.nan,
        "trend": _series_trend(window.values),
        "delta1": delta1,
        "delta3": delta3,
        "cagr_w": cagr_w,
        "min_w": min_w,
        "recovery_w": recovery_w,
    }


def build_country_snapshot(
    df,
    id_col,
    date_col,
    target_col,
    window_size=5,
    min_history=3,
    max_missing_frac=0.5,
    target_mode="value",
    impute_missing=True,
):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[id_col, date_col])
    df = df.sort_values([id_col, date_col])

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    indicator_cols = [c for c in numeric_cols if c != target_col]
    indicator_cols = [
        c for c in indicator_cols if df[c].isna().mean() <= max_missing_frac
    ]

    rows = []
    for country, group in df.groupby(id_col):
        target_series = group[target_col].dropna()
        if len(target_series) < min_history:
            continue

        latest_idx = target_series.index[-1]
        latest_date = group.loc[latest_idx, date_col]
        latest_target = group.loc[latest_idx, target_col]

        row = {
            id_col: country,
            date_col: latest_date,
            target_col: float(latest_target),
        }

        target_stats = _summarize_series(target_series.values, window_size)
        if target_mode == "trend":
            row[target_col] = target_stats["trend"]
        elif target_mode == "recovery":
            row[target_col] = target_stats["recovery_w"]
        row[f"{target_col}_lag1"] = target_stats["lag1"]
        row[f"{target_col}_lag3"] = target_stats["lag3"]
        row[f"{target_col}_roll_mean"] = target_stats["roll_mean"]
        row[f"{target_col}_roll_std"] = target_stats["roll_std"]
        row[f"{target_col}_trend"] = target_stats["trend"]
        row[f"{target_col}_delta1"] = target_stats["delta1"]
        row[f"{target_col}_delta3"] = target_stats["delta3"]
        row[f"{target_col}_cagr{window_size}"] = target_stats["cagr_w"]
        row[f"{target_col}_min{window_size}"] = target_stats["min_w"]
        row[f"{target_col}_recovery{window_size}"] = target_stats["recovery_w"]

        for col in indicator_cols:
            stats = _summarize_series(group[col].values, window_size)
            row[f"{col}_latest"] = stats["latest"]
            row[f"{col}_lag1"] = stats["lag1"]
            row[f"{col}_lag3"] = stats["lag3"]
            row[f"{col}_roll_mean"] = stats["roll_mean"]
            row[f"{col}_roll_std"] = stats["roll_std"]
            row[f"{col}_trend"] = stats["trend"]
            row[f"{col}_delta1"] = stats["delta1"]
            row[f"{col}_delta3"] = stats["delta3"]
            row[f"{col}_cagr{window_size}"] = stats["cagr_w"]
            row[f"{col}_min{window_size}"] = stats["min_w"]
            row[f"{col}_recovery{window_size}"] = stats["recovery_w"]

        rows.append(row)

    snapshot = pd.DataFrame(rows)
    if snapshot.empty:
        raise ValueError("No rows produced after longitudinal feature engineering")

    missing_frac = snapshot.isna().mean()
    keep_cols = missing_frac[missing_frac <= max_missing_frac].index.tolist()
    snapshot = snapshot[keep_cols]
    if impute_missing:
        numeric_cols = snapshot.select_dtypes(include=[np.number]).columns
        snapshot[numeric_cols] = snapshot[numeric_cols].fillna(snapshot[numeric_cols].median())
    snapshot = snapshot.dropna(subset=[target_col])
    return snapshot


def build_country_panel(
    df,
    id_col,
    date_col,
    target_col,
    window_size=5,
    min_history=3,
    max_missing_frac=0.5,
    stride=1,
    target_mode="value",
    impute_missing=True,
):
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[id_col, date_col])
    df = df.sort_values([id_col, date_col])

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found in dataset")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")

    indicator_cols = [c for c in numeric_cols if c != target_col]
    indicator_cols = [
        c for c in indicator_cols if df[c].isna().mean() <= max_missing_frac
    ]

    rows = []
    for country, group in df.groupby(id_col):
        group = group.sort_values(date_col).reset_index(drop=True)
        target_series = group[target_col]

        for i in range(min_history - 1, len(group), stride):
            if pd.isna(target_series.iloc[i]):
                continue

            row = {
                id_col: country,
                date_col: group.loc[i, date_col],
                target_col: float(target_series.iloc[i]),
            }

            target_stats = _summarize_series(
                target_series.iloc[: i + 1].values, window_size
            )
            if target_mode == "trend":
                row[target_col] = target_stats["trend"]
            elif target_mode == "recovery":
                row[target_col] = target_stats["recovery_w"]
            row[f"{target_col}_lag1"] = target_stats["lag1"]
            row[f"{target_col}_lag3"] = target_stats["lag3"]
            row[f"{target_col}_roll_mean"] = target_stats["roll_mean"]
            row[f"{target_col}_roll_std"] = target_stats["roll_std"]
            row[f"{target_col}_trend"] = target_stats["trend"]
            row[f"{target_col}_delta1"] = target_stats["delta1"]
            row[f"{target_col}_delta3"] = target_stats["delta3"]
            row[f"{target_col}_cagr{window_size}"] = target_stats["cagr_w"]
            row[f"{target_col}_min{window_size}"] = target_stats["min_w"]
            row[f"{target_col}_recovery{window_size}"] = target_stats["recovery_w"]

            for col in indicator_cols:
                stats = _summarize_series(
                    group[col].iloc[: i + 1].values, window_size
                )
                row[f"{col}_latest"] = stats["latest"]
                row[f"{col}_lag1"] = stats["lag1"]
                row[f"{col}_lag3"] = stats["lag3"]
                row[f"{col}_roll_mean"] = stats["roll_mean"]
                row[f"{col}_roll_std"] = stats["roll_std"]
                row[f"{col}_trend"] = stats["trend"]
                row[f"{col}_delta1"] = stats["delta1"]
                row[f"{col}_delta3"] = stats["delta3"]
                row[f"{col}_cagr{window_size}"] = stats["cagr_w"]
                row[f"{col}_min{window_size}"] = stats["min_w"]
                row[f"{col}_recovery{window_size}"] = stats["recovery_w"]

            rows.append(row)

    panel = pd.DataFrame(rows)
    if panel.empty:
        raise ValueError("No rows produced after longitudinal panel feature engineering")

    missing_frac = panel.isna().mean()
    keep_cols = missing_frac[missing_frac <= max_missing_frac].index.tolist()
    panel = panel[keep_cols]
    if impute_missing:
        numeric_cols = panel.select_dtypes(include=[np.number]).columns
        panel[numeric_cols] = panel[numeric_cols].fillna(panel[numeric_cols].median())
    panel = panel.dropna(subset=[target_col])
    return panel


def run_wbd_longitudinal_pipeline(
    csv_path,
    output_dir,
    pipeline_cfg,
    forest_cfg,
    xgb_cfg,
    id_col="country",
    date_col="date",
    target_col="life_expectancy_at_birth",
    window_size=5,
    min_history=3,
    max_missing_frac=0.5,
    mode="snapshot",
    panel_stride=1,
    target_mode="value",
    dataset_name="longitudinal",
    impute_missing=True,
):
    df = pd.read_csv(csv_path)
    target_label = target_col
    display_name = target_col
    if target_mode == "trend":
        target_label = f"{target_col}_trend{window_size}"
        display_name = f"{target_col} trend (last {window_size})"
    elif target_mode == "recovery":
        target_label = f"{target_col}_recovery{window_size}"
        display_name = f"{target_col} recovery (last {window_size})"
    if mode == "panel":
        snapshot = build_country_panel(
            df,
            id_col=id_col,
            date_col=date_col,
            target_col=target_col,
            window_size=window_size,
            min_history=min_history,
            max_missing_frac=max_missing_frac,
            stride=panel_stride,
            target_mode=target_mode,
            impute_missing=impute_missing,
        )
    else:
        snapshot = build_country_snapshot(
            df,
            id_col=id_col,
            date_col=date_col,
            target_col=target_col,
            window_size=window_size,
            min_history=min_history,
            max_missing_frac=max_missing_frac,
            target_mode=target_mode,
            impute_missing=impute_missing,
        )

    if target_mode in {"trend", "recovery"}:
        snapshot = snapshot.rename(columns={target_col: target_label})
        target_col = target_label

    dataset_name = dataset_name or "longitudinal"
    run_xgboost_baseline(
        snapshot,
        target_col=target_col,
        date_col=date_col,
        output_dir=Path(output_dir) / dataset_name / "xgboost",
        cfg=xgb_cfg,
    )
    results = {}
    for pipeline_name, settings in pipeline_cfg.items():
        if pipeline_name == "original":
            labeled_df = label_original_method(
                snapshot,
                target_col,
                high_value_quantile=settings.get("high_value_quantile", 0.94),
            )
        elif pipeline_name == "syflow":
            labeled_df = label_syflow_method(
                snapshot,
                target_col,
                n_seeds=settings.get("n_seeds", 1000),
                beta=settings.get("beta", 0.5),
                lambd_div=settings.get("lambd_div", 2.0),
                top_percentile=settings.get("top_percentile", 0.01),
                max_overlap=settings.get("max_overlap", 0.95),
                min_subgroup_size=settings.get("min_subgroup_size", 50),
                max_subgroup_frac=settings.get("max_subgroup_frac", 0.90),
            )
        else:
            raise ValueError(f"Unknown pipeline '{pipeline_name}'")

        X, y, feature_names = prepare_xy(labeled_df, target_col)
        if X.size == 0 or len(y) == 0:
            raise ValueError(
                f"No usable samples after preprocessing for target '{target_col}'. "
                "Check missingness thresholds or target_mode."
            )
        forest_result = run_forest_search(
            X,
            y,
            n_seeds=forest_cfg.get("n_seeds", 1000),
            accuracy_quantile=forest_cfg.get("accuracy_quantile", 0.99),
        )
        trees, _ = rebuild_forest_for_seed(X, y, forest_result.best_seed)

        out_dir = save_outputs(output_dir, dataset_name, pipeline_name, labeled_df, forest_result)
        save_kde_plot(
            labeled_df,
            target_col,
            LABEL_COLUMN,
            Path(out_dir) / "kde_plot.png",
            title_override=f"KDE Plot of {display_name}",
        )
        save_forest_accuracy_plot(
            forest_result.results_df, Path(out_dir) / "accuracy_vs_questions.png"
        )
        save_forest_tree_artifacts(trees, X, y, feature_names, out_dir)
        results[pipeline_name] = out_dir

    return results


def load_longitudinal_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))
