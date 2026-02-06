import json
from pathlib import Path
from typing import Dict

import pandas as pd

from .common import LABEL_COLUMN, ensure_dir, load_json, prepare_xy, save_json
from .datasets import load_dataset_from_config
from .forest import rebuild_forest, run_forest_search
from .labeling import label_original_method, label_syflow_method
from .plotting import save_forest_accuracy_plot, save_forest_tree_artifacts, save_kde_plot


def _save_outputs(
    base_dir: Path,
    dataset_name: str,
    pipeline_name: str,
    labeled_df: pd.DataFrame,
    forest_result,
) -> Path:
    out_dir = ensure_dir(base_dir / dataset_name / pipeline_name)
    labeled_df.to_csv(out_dir / "labeled.csv", index=False)

    if forest_result is not None:
        forest_result.results_df.to_csv(out_dir / "forest_results.csv", index=False)
        save_json(out_dir / "best_forest.json", forest_result.best_row.to_dict())

    summary = {
        "rows": int(len(labeled_df)),
        "interesting": int(labeled_df[LABEL_COLUMN].sum()),
        "non_interesting": int((~labeled_df[LABEL_COLUMN]).sum()),
    }
    save_json(out_dir / "summary.json", summary)

    return out_dir


def run_static_pipeline(config_path: str, output_dir: str) -> None:
    cfg = load_json(config_path)
    output_base = Path(output_dir)

    pipeline_cfg: Dict = cfg["pipelines"]
    forest_cfg: Dict = cfg.get("forest", {})

    for dataset_cfg in cfg.get("datasets", []):
        if not dataset_cfg.get("enabled", True):
            continue
        dataset_name = dataset_cfg["name"]
        target = dataset_cfg["target"]

        df = load_dataset_from_config(dataset_cfg)
        if target not in df.columns:
            raise ValueError(f"Target '{target}' not found in dataset {dataset_name}")

        for pipeline_name, settings in pipeline_cfg.items():
            if pipeline_name == "tail_cluster":
                labeled_df = label_original_method(
                    df,
                    target,
                    high_value_quantile=settings.get("high_value_quantile", 0.94),
                    two_tailed=settings.get("two_tailed", False),
                )
            elif pipeline_name == "timetribes":
                labeled_df = label_syflow_method(
                    df,
                    target,
                    n_seeds=settings.get("n_seeds", 1000),
                    beta=settings.get("beta", 0.5),
                    lambd_div=settings.get("lambd_div", 2.0),
                    top_percentile=settings.get("top_percentile", 0.01),
                    max_overlap=settings.get("max_overlap", 0.95),
                    min_subgroup_size=settings.get("min_subgroup_size", 50),
                    max_subgroup_frac=settings.get("max_subgroup_frac", 0.9),
                )
            else:
                continue

            X, y, feature_names = prepare_xy(
                labeled_df,
                target,
                exclude_columns=dataset_cfg.get("drop_columns", []),
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

            out_dir = _save_outputs(output_base, dataset_name, pipeline_name, labeled_df, forest_result)

            save_kde_plot(
                labeled_df,
                target,
                LABEL_COLUMN,
                out_dir / "kde_plot",
                title=f"{dataset_name}: {pipeline_name} target distribution",
            )
            save_forest_accuracy_plot(forest_result.results_df, out_dir / "accuracy_vs_questions")
            save_forest_tree_artifacts(trees, X, y, feature_names, out_dir)

    print(f"Static pipeline complete. Outputs in {output_base}")
