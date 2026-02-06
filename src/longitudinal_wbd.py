import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import polars as pl
import seaborn as sns
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from scipy.cluster.hierarchy import linkage, fcluster

# Sklearn imports
from sklearn.tree import DecisionTreeClassifier, _tree
from sklearn.model_selection import train_test_split

# Import shared utilities that DON'T need changing
from src.subgroup_pipelines import (
    LABEL_COLUMN,
    label_syflow_method,  # SyFlow is generic, can keep shared
    save_outputs,
    save_kde_plot,
    save_forest_accuracy_plot,
    save_forest_tree_artifacts,
    prune_pure_subtrees,
    count_questions,
)


# --- 1. LOCAL LOGIC (Decoupled from Static Pipelines) ---

@dataclass
class ForestSearchResult:
    results_df: pd.DataFrame
    best_seed: int
    best_row: pd.Series


def build_and_evaluate_forest_with_split(X, y, seed: int):
    """
    Local version of forest builder that includes TRAIN/TEST SPLIT.
    This fixes the 'Accuracy 1.0' overfitting issue seen in longitudinal data.
    """
    # Split 80/20
    if len(X) < 10:
        X_train, X_val, y_train, y_val = X, X, y, y
    else:
        try:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=seed, stratify=y
            )
        except ValueError:
            X_train, X_val, y_train, y_val = train_test_split(
                X, y, test_size=0.2, random_state=seed
            )

    rng = np.random.RandomState(seed)
    max_n_trees = rng.randint(1, 11)
    max_questions = rng.randint(2, 11)

    trees = []
    tree_question_counts = []

    # Train on X_train
    for _ in range(max_n_trees):
        n_train = X_train.shape[0]
        bootstrap_idx = rng.choice(n_train, size=n_train, replace=True)
        X_boot = X_train[bootstrap_idx]
        y_boot = y_train[bootstrap_idx]

        clf = DecisionTreeClassifier(
            max_leaf_nodes=max_questions + 1,
            random_state=rng.randint(0, 1_000_000),
        )
        clf.fit(X_boot, y_boot)
        prune_pure_subtrees(clf)

        trees.append(clf)
        tree_question_counts.append(count_questions(clf))

    # Evaluate on X_val (Held-out data)
    if len(trees) == 0:
        return {"seed": seed, "forest_accuracy": 0, "n_trees": 0, "total_questions": 0}

    pred_matrix = np.vstack([clf.predict(X_val).astype(bool) for clf in trees])
    forest_pred = np.any(pred_matrix, axis=0).astype(int)

    forest_accuracy = (forest_pred == y_val).mean()
    total_questions = int(np.sum(tree_question_counts))

    return {
        "seed": seed,
        "forest_accuracy": forest_accuracy,
        "n_trees": len(trees),
        "total_questions": total_questions,
    }


def run_forest_search_local(X, y, n_seeds=1000, accuracy_quantile=0.99) -> ForestSearchResult:
    """
    Local runner calling the split-enabled forest builder.
    """
    results = Parallel(n_jobs=-1)(
        delayed(build_and_evaluate_forest_with_split)(X, y, s) for s in range(n_seeds)
    )

    results_df = pd.DataFrame(results)
    acc_threshold = results_df["forest_accuracy"].quantile(accuracy_quantile)
    top_df = results_df[results_df["forest_accuracy"] >= acc_threshold]
    if top_df.empty:
        top_df = results_df

    top_sorted = top_df.sort_values(
        by=["total_questions", "forest_accuracy", "seed"],
        ascending=[True, False, True],
    )

    best_row = top_sorted.iloc[0]
    best_seed = int(best_row["seed"])

    return ForestSearchResult(results_df=results_df, best_seed=best_seed, best_row=best_row)


def rebuild_forest_structure(X, y, seed: int):
    """
    Rebuilds forest structure for visualization/saving.
    """
    rng = np.random.RandomState(seed)
    max_n_trees = rng.randint(1, 11)
    max_questions = rng.randint(2, 11)
    trees = []
    
    # We retrain on full data for the artifacts/visualization
    for _ in range(max_n_trees):
        idx = rng.choice(len(X), size=len(X), replace=True)
        clf = DecisionTreeClassifier(
            max_leaf_nodes=max_questions + 1,
            random_state=rng.randint(0, 1_000_000),
        )
        clf.fit(X[idx], y[idx])
        prune_pure_subtrees(clf)
        trees.append(clf)
    
    return trees


def label_original_method_two_tailed(df, target, high_value_quantile=0.94):
    """
    Local 'Original' Method improved for Longitudinal Data.
    It checks BOTH tails (High and Low) instead of just High.
    """
    y_col = df[target]
    if isinstance(y_col, pd.DataFrame):
        y_col = y_col.iloc[:, 0]
    y_col = pd.to_numeric(y_col, errors="coerce")
    y = y_col.values.reshape(-1, 1)

    # 1. Clustering (Same as shared pipeline)
    try:
        Z = linkage(y, method="single")
        heights = Z[:, 2]
        diffs = np.diff(heights)
        threshold = heights[np.argmax(diffs)] if len(diffs) > 0 else 0
        clusters = fcluster(Z, t=threshold, criterion="distance")
        unique, counts = np.unique(clusters, return_counts=True)
        largest = unique[np.argmax(counts)]
        interesting_mask = clusters != largest
    except:
        interesting_mask = np.zeros(len(y), dtype=bool)

    # 2. Outliers (Two-Tailed)
    upper = y_col.quantile(high_value_quantile)
    lower = y_col.quantile(1.0 - high_value_quantile)
    
    interesting_mask |= (y_col >= upper)
    interesting_mask |= (y_col <= lower)

    labeled_df = df.copy()
    labeled_df[LABEL_COLUMN] = interesting_mask
    return labeled_df


# --- 2. PREPROCESSING (Polars & Supervisor Logic) ---

def _build_features_polars(df, id_col, date_col, target_col, min_history, window_size):
    print(f"  [Feature Eng] Starting with {len(df)} rows.")
    pldf = pl.from_pandas(df)

    # Date Parsing
    try:
        if pldf[date_col].dtype in [pl.Int32, pl.Int64, pl.Float64]:
            pldf = pldf.with_columns(pl.col(date_col).cast(pl.Int64))
        else:
            pldf = pldf.with_columns(
                pl.col(date_col).cast(pl.Utf8).str.to_date("%Y-%m-%d", strict=False).alias(date_col)
            )
            pldf = pldf.filter(pl.col(date_col).is_not_null())
    except Exception:
        pass

    # Target
    pldf = pldf.with_columns(pl.col(target_col).cast(pl.Float64, strict=False))
    pldf = pldf.filter(pl.col(target_col).is_not_null())
    pldf = pldf.sort([id_col, date_col])

    numeric_cols = [
        c for c in pldf.columns
        if c not in [id_col, date_col] and pldf[c].dtype in [pl.Float64, pl.Int64, pl.Int32]
    ]

    # A. Labeling Features (Trends) - Used to FIND the subgroup
    labeling_ops = [
        pl.col(target_col)
        .rolling_mean(window_size=window_size, min_periods=1)
        .over(id_col)
        .alias(f"labeling_roll_mean_{target_col}"),
        
        pl.col(target_col).alias(f"target_raw_{target_col}"),
    ]

    # B. Explanation Features (Lags) - Used to EXPLAIN the subgroup
    explanation_ops = []
    for col in numeric_cols:
        explanation_ops.append(pl.col(col).alias(f"latest_{col}"))
        explanation_ops.append(pl.col(col).shift(1).over(id_col).alias(f"lag1y_{col}"))
        explanation_ops.append(
            (pl.col(col) - pl.col(col).shift(1).over(id_col)).alias(f"delta1y_{col}")
        )
        if min_history > 1:
            explanation_ops.append(
                pl.col(col).shift(min_history).over(id_col).alias(f"lag{min_history}y_{col}")
            )
        if min_history >= 5:
            explanation_ops.append(
                pl.col(col).shift(5).over(id_col).alias(f"lag5y_{col}")
            )

    pldf = pldf.with_columns(labeling_ops + explanation_ops)

    check_col = f"lag{min_history}y_{target_col}" if min_history > 1 else f"lag1y_{target_col}"
    pldf = pldf.filter(pl.col(check_col).is_not_null())

    print(f"  [Feature Eng] Rows after history check: {len(pldf)}")
    if len(pldf) == 0:
        return pd.DataFrame()

    return pldf.group_by(id_col, maintain_order=True).last().to_pandas()


def clean_and_impute(df, id_col, date_col, target_col, max_missing_frac=0.5, impute=True):
    threshold = len(df) * max_missing_frac
    null_counts = df.isnull().sum()
    drop_cols = null_counts[null_counts > threshold].index.tolist()
    
    meta = [id_col, date_col, target_col, f"target_raw_{target_col}"]
    drop_cols = [c for c in drop_cols if c not in meta]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    if impute:
        numeric = df.select_dtypes(include=[np.number]).columns
        cols = [c for c in numeric if c not in meta]
        df[cols] = df[cols].fillna(df[cols].median()).fillna(0)

    return df.dropna()


def prepare_xy(df, target, label_column=LABEL_COLUMN):
    if label_column not in df.columns: raise ValueError(f"Label {label_column} not found")
    if target not in df.columns: raise ValueError(f"Target {target} not found")

    y = df[label_column].astype(int)
    drop = [target, label_column]
    feats = df.drop(columns=[c for c in drop if c in df.columns])
    X = feats.select_dtypes(include=[np.number])
    
    if X.empty: return np.array([]), np.array([]), []
    
    valid = ~(X.isna().any(axis=1) | y.isna())
    return X[valid].values, y[valid].values, list(X.columns)


def run_xgboost_baseline(snapshot, target_col, date_col, output_dir, cfg):
    if not cfg or not cfg.get("enabled", False): return
    try:
        import xgboost as xgb
        from sklearn.model_selection import KFold
        from sklearn.metrics import mean_squared_error, r2_score
    except ImportError: return

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X = snapshot.select_dtypes(include=[np.number]).drop(columns=[target_col], errors='ignore')
    if date_col in X.columns: X = X.drop(columns=[date_col])
    y = snapshot[target_col]

    if len(X) < cfg.get("min_samples", 50): return

    model = xgb.XGBRegressor(n_estimators=100, max_depth=3, random_state=42)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    metrics = []
    
    for train_idx, test_idx in kf.split(X):
        model.fit(X.iloc[train_idx], y.iloc[train_idx])
        preds = model.predict(X.iloc[test_idx])
        metrics.append({
            "rmse": np.sqrt(mean_squared_error(y.iloc[test_idx], preds)),
            "r2": r2_score(y.iloc[test_idx], preds)
        })

    pd.DataFrame(metrics).to_csv(output_dir / "xgboost_metrics.csv", index=False)


# --- 3. MAIN PIPELINE ---

def run_wbd_longitudinal_pipeline(
    csv_path, output_dir, pipeline_cfg, forest_cfg, xgb_cfg,
    id_col="country", date_col="date", target_col="life_expectancy_at_birth",
    window_size=5, min_history=3, max_missing_frac=0.5, mode="snapshot",
    panel_stride=1, target_mode="value", dataset_name="longitudinal", impute_missing=True,
):
    print(f"Loading data from {csv_path}...")
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        return

    # 1. Feature Engineering
    snapshot = _build_features_polars(
        df, id_col, date_col, target_col, min_history, window_size
    )
    if snapshot.empty:
        print("Error: Snapshot empty. Check min_history.")
        return

    # 2. Imputation
    snapshot = clean_and_impute(
        snapshot, id_col, date_col, target_col, max_missing_frac, impute_missing
    )
    if snapshot.empty:
        print("Error: Snapshot empty after cleaning.")
        return

    # 3. Target
    raw_target = f"target_raw_{target_col}"
    active_target = raw_target
    if target_mode == "trend":
        lag_col = f"lag{min_history}y_{target_col}"
        cur_col = f"latest_{target_col}"
        if lag_col in snapshot.columns:
            snapshot[f"{target_col}_slope"] = (snapshot[cur_col] - snapshot[lag_col]) / min_history
            active_target = f"{target_col}_slope"

    snapshot = snapshot.dropna(subset=[active_target])
    print(f"Snapshot Ready: {len(snapshot)} rows. Target: {active_target}")

    # 4. Baseline
    run_xgboost_baseline(
        snapshot, active_target, date_col, Path(output_dir) / dataset_name / "xgboost", xgb_cfg
    )

    # 5. Pipelines
    results = {}
    for pipeline_name, settings in pipeline_cfg.items():
        print(f"Running {pipeline_name}...")
        
        # Use LOCAL Two-Tailed Method
        if pipeline_name == "original":
            labeled_df = label_original_method_two_tailed(
                snapshot, active_target,
                high_value_quantile=settings.get("high_value_quantile", 0.94)
            )
        # Use Shared SyFlow (Generic)
        elif pipeline_name == "syflow":
            labeled_df = label_syflow_method(
                snapshot, active_target,
                n_seeds=settings.get("n_seeds", 1000),
                beta=settings.get("beta", 0.5),
                lambd_div=settings.get("lambd_div", 2.0),
                top_percentile=settings.get("top_percentile", 0.01),
                max_overlap=settings.get("max_overlap", 0.95),
                min_subgroup_size=settings.get("min_subgroup_size", 50),
                max_subgroup_frac=settings.get("max_subgroup_frac", 0.90)
            )
        else:
            continue

        # Prepare Features
        X_full, y, feature_names = prepare_xy(labeled_df, active_target)
        
        # FEATURE FILTERING (Supervisor's Logic)
        allowed_indices = []
        allowed_names = []
        for idx, fname in enumerate(feature_names):
            # Exclude labeling/trend features
            if fname.startswith(("labeling_", "target_raw_")): continue
            # Include explanation features (lags, raw, deltas)
            if fname.startswith(("latest_", "lag", "delta")) or "_" not in fname:
                allowed_indices.append(idx)
                allowed_names.append(fname)
        
        if not allowed_indices:
            print("Warning: No allowed features. Using all.")
            X_expl, names_expl = X_full, feature_names
        else:
            X_expl, names_expl = X_full[:, allowed_indices], allowed_names

        if X_expl.shape[0] == 0: continue

        # Use LOCAL Forest Search (with Split)
        forest_result = run_forest_search_local(
            X_expl, y,
            n_seeds=forest_cfg.get("n_seeds", 1000),
            accuracy_quantile=forest_cfg.get("accuracy_quantile", 0.99)
        )
        
        trees = rebuild_forest_structure(X_expl, y, forest_result.best_seed)
        
        out_dir = save_outputs(output_dir, dataset_name, pipeline_name, labeled_df, forest_result)
        save_kde_plot(labeled_df, active_target, LABEL_COLUMN, Path(out_dir) / "kde_plot.png")
        save_forest_accuracy_plot(forest_result.results_df, Path(out_dir) / "accuracy_vs_questions.png")
        save_forest_tree_artifacts(trees, X_expl, y, names_expl, out_dir)
        results[pipeline_name] = out_dir

    return results

def load_longitudinal_config(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))