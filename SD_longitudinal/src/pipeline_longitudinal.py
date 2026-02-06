import json
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

from .common import LABEL_COLUMN, ensure_dir, load_json, prepare_xy, save_json
from .forest import rebuild_forest, run_forest_search
from .labeling import label_original_method, label_syflow_method
from .longitudinal_features import build_longitudinal_features
from .plotting import save_forest_accuracy_plot, save_forest_tree_artifacts, save_kde_plot


def clean_and_impute(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    label_target_cols: List[str],
    max_missing_frac: float = 0.5,
    impute: bool = True,
) -> pd.DataFrame:
    keep_cols = set([id_col, date_col] + label_target_cols)
    missing = df.isna().mean()
    drop_cols = [c for c, frac in missing.items() if frac > max_missing_frac and c not in keep_cols]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    if impute:
        numeric = df.select_dtypes(include=[np.number]).columns
        cols = [c for c in numeric if c not in keep_cols]
        if cols:
            df[cols] = df[cols].fillna(df[cols].median()).fillna(0)

    return df.dropna()


def _save_outputs(
    base_dir: Path,
    dataset_name: str,
    pipeline_name: str,
    labeled_df: pd.DataFrame,
    forest_result,
    question_map: Dict[str, str],
    feature_base_map: Dict[str, str],
) -> Path:
    out_dir = ensure_dir(base_dir / dataset_name / pipeline_name)
    labeled_df.to_csv(out_dir / "labeled.csv", index=False)

    if forest_result is not None:
        forest_result.results_df.to_csv(out_dir / "forest_results.csv", index=False)
        save_json(out_dir / "best_forest.json", forest_result.best_row.to_dict())

    if question_map:
        q_df = pd.DataFrame(
            {
                "feature": list(question_map.keys()),
                "question": list(question_map.values()),
                "base_indicator": [feature_base_map.get(f, "") for f in question_map.keys()],
            }
        )
        q_df.to_csv(out_dir / "question_map.csv", index=False)

    summary = {
        "rows": int(len(labeled_df)),
        "interesting": int(labeled_df[LABEL_COLUMN].sum()),
        "non_interesting": int((~labeled_df[LABEL_COLUMN]).sum()),
    }
    save_json(out_dir / "summary.json", summary)

    return out_dir


def _select_target_column(target_mode: str, label_window: int) -> str:
    if target_mode == "trend":
        return f"label_target_trend{label_window}"
    if target_mode == "delta":
        return f"label_target_delta{label_window}"
    return "label_target_value"


def run_longitudinal_pipeline(config_path: str, output_dir: str) -> None:
    cfg = load_json(config_path)
    output_base = Path(output_dir)

    pipeline_cfg: Dict = cfg["pipelines"]
    forest_cfg: Dict = cfg.get("forest", {})

    for dataset_cfg in cfg.get("datasets", []):
        if not dataset_cfg.get("enabled", True):
            continue

        df = pd.read_csv(dataset_cfg["path"])
        print(f"[Load] {dataset_cfg['name']}: {len(df)} rows, {len(df.columns)} columns")
        id_col = dataset_cfg.get("id_col", "country")
        date_col = dataset_cfg.get("date_col", "date")
        target_col = dataset_cfg.get("target", "life_expectancy_at_birth")

        question_pack = dataset_cfg.get("question_pack", {})
        lags = question_pack.get("lags", [1, 3, 5, 10])
        windows = question_pack.get("windows", [3, 5, 10])
        label_window = int(dataset_cfg.get("label_window", max(lags)))

        print("[Feature Eng] Building longitudinal features...")
        t0 = time.perf_counter()
        features_df, question_map, feature_base_map = build_longitudinal_features(
            df,
            id_col=id_col,
            date_col=date_col,
            target_col=target_col,
            lags=lags,
            windows=windows,
            label_window=label_window,
            min_history=dataset_cfg.get("min_history", 3),
            mode=dataset_cfg.get("mode", "snapshot"),
            panel_stride=dataset_cfg.get("panel_stride", 1),
            include_pct_change=question_pack.get("include_pct_change", True),
            include_volatility=question_pack.get("include_volatility", False),
            indicator_include=question_pack.get("indicator_include"),
            indicator_exclude=question_pack.get("indicator_exclude"),
        )

        t1 = time.perf_counter()
        print(f"[Feature Eng] Done in {t1 - t0:.1f}s -> {len(features_df)} rows, {len(features_df.columns)} columns")

        label_target_cols = [
            "label_target_value",
            f"label_target_delta{label_window}",
            f"label_target_trend{label_window}",
        ]

        print("[Clean] Dropping high-missing columns and imputing...")
        t2 = time.perf_counter()
        features_df = clean_and_impute(
            features_df,
            id_col=id_col,
            date_col=date_col,
            label_target_cols=label_target_cols,
            max_missing_frac=dataset_cfg.get("max_missing_frac", 0.5),
            impute=dataset_cfg.get("impute_missing", True),
        )
        t3 = time.perf_counter()
        print(f"[Clean] Done in {t3 - t2:.1f}s -> {len(features_df)} rows, {len(features_df.columns)} columns")

        target_modes = dataset_cfg.get("target_modes") or [dataset_cfg.get("target_mode", "value")]

        for target_mode in target_modes:
            active_target = _select_target_column(target_mode, label_window)
            if active_target not in features_df.columns:
                continue

            dataset_name = f"{dataset_cfg['name']}_{target_mode}"
            for pipeline_name, settings in pipeline_cfg.items():
                print(f"[Label] {dataset_name} | {pipeline_name} | target={active_target}")
                t4 = time.perf_counter()
                if pipeline_name == "tail_cluster":
                    print("  [Label] Running TailCluster (single-linkage on target). This can take time on large panels.")
                    labeled_df = label_original_method(
                        features_df,
                        active_target,
                        high_value_quantile=settings.get("high_value_quantile", 0.94),
                        two_tailed=settings.get("two_tailed", True),
                    )
                elif pipeline_name == "timetribes":
                    labeled_df = label_syflow_method(
                        features_df,
                        active_target,
                        n_seeds=settings.get("n_seeds", 1000),
                        beta=settings.get("beta", 0.5),
                        lambd_div=settings.get("lambd_div", 2.0),
                        top_percentile=settings.get("top_percentile", 0.01),
                        max_overlap=settings.get("max_overlap", 0.95),
                        min_subgroup_size=settings.get("min_subgroup_size", 50),
                        max_subgroup_frac=settings.get("max_subgroup_frac", 0.9),
                        progress_every=settings.get("progress_every", 100),
                    )
                else:
                    continue

                t5 = time.perf_counter()
                if LABEL_COLUMN in labeled_df.columns:
                    n_int = int(labeled_df[LABEL_COLUMN].sum())
                    print(f"  [Label] Done in {t5 - t4:.1f}s | interesting={n_int} / {len(labeled_df)}")
                else:
                    print(f"  [Label] Done in {t5 - t4:.1f}s")

                exclude_prefixes = ["label_target_"]
                exclude_cols = [c for c, base in feature_base_map.items() if base == target_col]
                if target_col in labeled_df.columns:
                    exclude_cols.append(target_col)
                if dataset_cfg.get("include_target_features", False):
                    exclude_cols = []

                X, y, feature_names = prepare_xy(
                    labeled_df,
                    active_target,
                    exclude_columns=exclude_cols,
                    exclude_prefixes=exclude_prefixes,
                )
                if X.size == 0:
                    continue

                forest_result = run_forest_search(
                    X,
                    y,
                    n_seeds=forest_cfg.get("n_seeds", 1000),
                    accuracy_quantile=forest_cfg.get("accuracy_quantile", 0.99),
                    min_trees=forest_cfg.get("min_trees", 1),
                    max_trees=forest_cfg.get("max_trees", 12),
                    min_questions=forest_cfg.get("min_questions", 2),
                    max_questions=forest_cfg.get("max_questions", 12),
                    test_size=forest_cfg.get("test_size", 0.2),
                    vote_rule=forest_cfg.get("vote_rule", "any"),
                    eval_mode=forest_cfg.get("eval_mode", "train"),
                    n_jobs=forest_cfg.get("n_jobs", -1),
                )

                trees, _ = rebuild_forest(
                    X,
                    y,
                    forest_result.best_seed,
                    forest_cfg.get("min_trees", 1),
                    forest_cfg.get("max_trees", 12),
                    forest_cfg.get("min_questions", 2),
                    forest_cfg.get("max_questions", 12),
                )

                out_dir = _save_outputs(
                    output_base,
                    dataset_name,
                    pipeline_name,
                    labeled_df,
                    forest_result,
                    question_map,
                    feature_base_map,
                )

                save_kde_plot(
                    labeled_df,
                    active_target,
                    LABEL_COLUMN,
                    out_dir / "kde_plot",
                    title=f"{dataset_name}: {pipeline_name} target distribution",
                )
                save_forest_accuracy_plot(forest_result.results_df, out_dir / "accuracy_vs_questions")
                save_forest_tree_artifacts(trees, X, y, feature_names, out_dir, question_map=question_map)

    print(f"Longitudinal pipeline complete. Outputs in {output_base}")
