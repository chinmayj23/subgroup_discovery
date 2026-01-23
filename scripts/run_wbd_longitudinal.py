import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.longitudinal_wbd import load_longitudinal_config, run_wbd_longitudinal_pipeline
from src.subgroup_pipelines import load_config


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run longitudinal WBD pipeline with feature engineering."
    )
    parser.add_argument(
        "--config",
        default="configs/wbd_longitudinal.json",
        help="Path to longitudinal config JSON.",
    )
    parser.add_argument(
        "--pipeline-config",
        default="configs/subgroup_pipelines.json",
        help="Path to subgroup pipeline config JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs_subgroups/longitudinal",
        help="Base directory for outputs.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_longitudinal_config(args.config)
    pipeline_cfg = load_config(args.pipeline_config)["pipelines"]
    forest_cfg = load_config(args.pipeline_config).get("forest", {})

    run_wbd_longitudinal_pipeline(
        csv_path=cfg["path"],
        output_dir=args.output_dir,
        pipeline_cfg=pipeline_cfg,
        forest_cfg=forest_cfg,
        xgb_cfg=cfg.get("xgboost", {}),
        id_col=cfg.get("id_col", "country"),
        date_col=cfg.get("date_col", "date"),
        target_col=cfg.get("target", "life_expectancy_at_birth"),
        window_size=cfg.get("window_size", 5),
        min_history=cfg.get("min_history", 3),
        max_missing_frac=cfg.get("max_missing_frac", 0.5),
        mode=cfg.get("mode", "snapshot"),
        panel_stride=cfg.get("panel_stride", 1),
        target_mode=cfg.get("target_mode", "value"),
        dataset_name=cfg.get("dataset_name", "longitudinal"),
        impute_missing=cfg.get("impute_missing", True),
    )


if __name__ == "__main__":
    main()
