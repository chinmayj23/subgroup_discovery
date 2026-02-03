import json
from pathlib import Path
import numpy as np
import pandas as pd
import polars as pl

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

def _build_features_polars(df, id_col, date_col, target_col, min_history, window_size):
    """
    High-performance feature engineering using Polars.
    Smartly generates lags based on available history to avoid NaN traps.
    """
    print(f"  [Feature Eng] Starting with {len(df)} rows.")
    
    # 1. Convert to Polars
    pldf = pl.from_pandas(df)
    
    # 2. Robust Date Parsing
    try:
        if pldf[date_col].dtype in [pl.Int32, pl.Int64, pl.Float64]:
             pldf = pldf.with_columns(pl.col(date_col).cast(pl.Int64))
        else:
             pldf = pldf.with_columns(
                pl.col(date_col).cast(pl.Utf8).str.to_date("%Y-%m-%d", strict=False).alias(date_col)
             )
             pldf = pldf.filter(pl.col(date_col).is_not_null())
    except Exception as e:
        print(f"  [Warning] Date parsing issue: {e}. Proceeding.")

    # 3. Robust Target Casting
    pldf = pldf.with_columns(
        pl.col(target_col).cast(pl.Float64, strict=False)
    )
    pldf = pldf.filter(pl.col(target_col).is_not_null())
    
    # Sort
    pldf = pldf.sort([id_col, date_col])

    numeric_cols = [
        c for c in pldf.columns 
        if c not in [id_col, date_col] and pldf[c].dtype in [pl.Float64, pl.Int64, pl.Int32]
    ]

    # --- PART A: Labeling Features ---
    labeling_ops = [
        pl.col(target_col)
          .rolling_mean(window_size=window_size, min_periods=1)
          .over(id_col)
          .alias(f"labeling_roll_mean_{target_col}"),
        pl.col(target_col).alias(f"target_raw_{target_col}")
    ]

    # --- PART B: Explanation Features ---
    explanation_ops = []
    
    for col in numeric_cols:
        # Latest
        explanation_ops.append(pl.col(col).alias(f"latest_{col}"))
        
        # Lag 1 (Always needed for Delta)
        explanation_ops.append(pl.col(col).shift(1).over(id_col).alias(f"lag1y_{col}"))
        
        # Delta
        explanation_ops.append(
            (pl.col(col) - pl.col(col).shift(1).over(id_col)).alias(f"delta1y_{col}")
        )

        # Min History Lag (Required)
        if min_history > 1:
             explanation_ops.append(pl.col(col).shift(min_history).over(id_col).alias(f"lag{min_history}y_{col}"))

        # Optional Deep History (Lag 5)
        # ONLY generate if min_history is large enough or we want to risk NaNs.
        # SAFE FIX: Only generate lag5 if min_history >= 5.
        if min_history >= 5:
             explanation_ops.append(pl.col(col).shift(5).over(id_col).alias(f"lag5y_{col}"))
    
    # History Check
    check_col_name = f"lag{min_history}y_{target_col}" if min_history > 1 else f"lag1y_{target_col}"
    
    pldf = pldf.with_columns(labeling_ops + explanation_ops)

    # --- PART C: Filter ---
    # Must have data at 'min_history' ago
    pldf = pldf.filter(pl.col(check_col_name).is_not_null())
    
    print(f"  [Feature Eng] Rows after history check (min_history={min_history}): {len(pldf)}")
    
    if len(pldf) == 0:
        return pd.DataFrame() 

    # Snapshot
    snapshot_pl = pldf.group_by(id_col, maintain_order=True).last()
    
    return snapshot_pl.to_pandas()


def clean_and_impute(df, id_col, date_col, target_col, max_missing_frac=0.5, impute=True):
    """
    Cleans the snapshot DataFrame to prevent '0 samples' error in forest.
    1. Drops columns with > max_missing_frac NaNs.
    2. Fills remaining NaNs with Median (if impute=True).
    3. Drops rows that still have NaNs (if any).
    """
    # Protect meta columns
    meta_cols = [id_col, date_col, target_col, f"target_raw_{target_col}", f"labeling_roll_mean_{target_col}"]
    
    # 1. Drop bad columns
    threshold = len(df) * max_missing_frac
    null_counts = df.isnull().sum()
    drop_cols = null_counts[null_counts > threshold].index.tolist()
    # Don't drop meta columns even if missing (unlikely)
    drop_cols = [c for c in drop_cols if c not in meta_cols]
    
    if drop_cols:
        print(f"  [Cleaning] Dropping {len(drop_cols)} columns with > {max_missing_frac*100}% missing values.")
        df = df.drop(columns=drop_cols)
        
    # 2. Impute
    if impute:
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # Exclude target/meta from imputation just in case (though target shouldn't have NaNs here)
        cols_to_impute = [c for c in numeric_cols if c not in [id_col, date_col]]
        
        # Fill with median
        df[cols_to_impute] = df[cols_to_impute].fillna(df[cols_to_impute].median())
        
        # Determine remaining NaNs
        remaining_nans = df[cols_to_impute].isna().sum().sum()
        if remaining_nans > 0:
             # Fallback to 0
             df[cols_to_impute] = df[cols_to_impute].fillna(0)
             print(f"  [Cleaning] Imputed missing values with Median/Zero.")
    
    # 3. Final Drop
    # prepare_xy is strict, so we must ensure no NaNs remain.
    # If any remain (non-numeric?), drop rows.
    before_len = len(df)
    df = df.dropna()
    if len(df) < before_len:
         print(f"  [Cleaning] Dropped {before_len - len(df)} rows containing residual NaNs.")
         
    return df


def run_xgboost_baseline(snapshot, target_col, date_col, output_dir, cfg):
    if not cfg or not cfg.get("enabled", False):
        return
    try:
        import xgboost as xgb
        from sklearn.model_selection import KFold
        from sklearn.metrics import mean_squared_error, r2_score
    except ImportError:
        return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X = snapshot.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
    if date_col in X.columns:
        X = X.drop(columns=[date_col])
    y = snapshot[target_col]

    if len(X) < cfg.get("min_samples", 50):
        print("Not enough samples for XGBoost.")
        return

    model = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    metrics = []
    
    for train_idx, test_idx in kf.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        metrics.append({
            "rmse": np.sqrt(mean_squared_error(y_test, preds)),
            "r2": r2_score(y_test, preds)
        })

    pd.DataFrame(metrics).to_csv(output_dir / "xgboost_metrics.csv", index=False)


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
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: File not found at {csv_path}")
        return

    # 1. Fast Feature Engineering
    print("Generating features with Polars (Fast Mode)...")
    snapshot = _build_features_polars(
        df, 
        id_col=id_col, 
        date_col=date_col, 
        target_col=target_col, 
        min_history=min_history,
        window_size=window_size
    )
    
    if snapshot.empty:
        print("CRITICAL ERROR: Snapshot is empty. Check min_history.")
        return

    # 2. Impute and Clean (CRITICAL FIX FOR 0 SAMPLES ERROR)
    print("Cleaning and Imputing Snapshot...")
    snapshot = clean_and_impute(
        snapshot, 
        id_col=id_col, 
        date_col=date_col, 
        target_col=target_col,
        max_missing_frac=max_missing_frac,
        impute=impute_missing
    )
    
    if snapshot.empty:
        print("CRITICAL ERROR: Snapshot is empty after cleaning. Data too sparse.")
        return

    # Setup Targets
    raw_target_col = f"target_raw_{target_col}"
    
    if target_mode == "trend":
        lag_col = f"lag{min_history}y_{target_col}"
        latest_col = f"latest_{target_col}"
        if lag_col in snapshot.columns:
             snapshot[f"{target_col}_slope"] = (snapshot[latest_col] - snapshot[lag_col]) / float(min_history)
        else:
             snapshot[f"{target_col}_slope"] = snapshot[latest_col]
        active_target = f"{target_col}_slope"
    else:
        active_target = raw_target_col

    if active_target not in snapshot.columns:
        print(f"Error: Target {active_target} not found.")
        return

    snapshot = snapshot.dropna(subset=[active_target])
    print(f"Snapshot Ready: {snapshot.shape[0]} rows. Target: {active_target}")

    # 3. Baselines
    run_xgboost_baseline(
        snapshot, 
        target_col=active_target, 
        date_col=date_col, 
        output_dir=Path(output_dir) / dataset_name / "xgboost", 
        cfg=xgb_cfg
    )

    results = {}
    
    # 4. Pipelines
    for pipeline_name, settings in pipeline_cfg.items():
        print(f"Running Pipeline: {pipeline_name}...")
        
        if pipeline_name == "original":
            labeled_df = label_original_method(
                snapshot,
                active_target,
                # In config, set this to 0.10 to find LOW values. 
                # Set to 0.90 to find HIGH values.
                high_value_quantile=settings.get("high_value_quantile", 0.94)
            )
        elif pipeline_name == "syflow":
            labeled_df = label_syflow_method(
                snapshot,
                active_target,
                n_seeds=settings.get("n_seeds", 1000),
                beta=settings.get("beta", 0.5),
                lambd_div=settings.get("lambd_div", 2.0),
                top_percentile=settings.get("top_percentile", 0.01),
                max_overlap=settings.get("max_overlap", 0.95),
                min_subgroup_size=settings.get("min_subgroup_size", 50),
                max_subgroup_frac=settings.get("max_subgroup_frac", 0.90),
            )
        else:
            continue

        # Explanation (Forest Search)
        X_full, y, feature_names = prepare_xy(labeled_df, active_target)
        
        # Filter "Right Questions"
        allowed_indices = []
        allowed_names = []
        for idx, fname in enumerate(feature_names):
            if fname.startswith("labeling_"): continue 
            if fname.startswith("target_raw_"): continue
            
            if fname.startswith(("latest_", "lag", "delta")) or "_" not in fname:
                allowed_indices.append(idx)
                allowed_names.append(fname)
        
        if not allowed_indices:
            print("WARNING: No allowed explanation features found! Using all.")
            X_expl = X_full
            names_expl = feature_names
        else:
            X_expl = X_full[:, allowed_indices]
            names_expl = allowed_names
            
        if X_expl.shape[0] == 0:
             print("Error: X_expl is empty after filtering. Skipping forest.")
             continue

        forest_result = run_forest_search(
            X_expl,
            y,
            n_seeds=forest_cfg.get("n_seeds", 100), 
            accuracy_quantile=forest_cfg.get("accuracy_quantile", 0.99),
        )
        
        if forest_result.best_row is None:
             print(f"Skipping {pipeline_name}: No good forest found.")
             continue

        trees, _ = rebuild_forest_for_seed(X_expl, y, forest_result.best_seed)

        out_dir = save_outputs(output_dir, dataset_name, pipeline_name, labeled_df, forest_result)
        
        save_kde_plot(
            labeled_df,
            active_target,
            LABEL_COLUMN,
            Path(out_dir) / "kde_plot.png",
            title_override=f"KDE Plot of {active_target}"
        )
        save_forest_accuracy_plot(
            forest_result.results_df, Path(out_dir) / "accuracy_vs_questions.png"
        )
        save_forest_tree_artifacts(trees, X_expl, y, names_expl, out_dir)
        results[pipeline_name] = out_dir

    return results

def load_longitudinal_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))