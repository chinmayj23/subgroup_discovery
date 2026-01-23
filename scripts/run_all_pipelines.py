import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.longitudinal_wbd import run_wbd_longitudinal_pipeline
from src.subgroup_pipelines import load_config, run_pipelines_for_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run static and longitudinal subgroup pipelines in one run."
    )
    parser.add_argument(
        "--config",
        default="configs/run_all.json",
        help="Path to combined run config JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = json.loads(Path(args.config).read_text(encoding="utf-8"))

    output_dir = cfg.get("output_dir", "outputs_subgroups/run_all")
    static_cfg_path = cfg.get("static_config", "configs/subgroup_pipelines.json")
    static_cfg = load_config(static_cfg_path)
    pipeline_cfg = static_cfg["pipelines"]
    forest_cfg = static_cfg.get("forest", {})

    static_list = cfg.get("static_datasets", [])
    if static_list:
        static_names = set(static_list)
        for dataset_cfg in static_cfg.get("datasets", []):
            if dataset_cfg["name"] not in static_names:
                continue
            run_pipelines_for_dataset(
                dataset_cfg=dataset_cfg,
                pipeline_cfg=pipeline_cfg,
                forest_cfg=forest_cfg,
                output_base_dir=Path(output_dir) / "static",
                run_forest=True,
            )

    for long_cfg in cfg.get("longitudinal_datasets", []) or []:
        target_modes = long_cfg.get("target_modes", ["value"])
        for mode in target_modes:
            dataset_name = f"{long_cfg['name']}_{mode}"
            run_wbd_longitudinal_pipeline(
                csv_path=long_cfg["path"],
                output_dir=Path(output_dir) / "longitudinal",
                pipeline_cfg=pipeline_cfg,
                forest_cfg=forest_cfg,
                xgb_cfg=long_cfg.get("xgboost", {}),
                id_col=long_cfg.get("id_col", "country"),
                date_col=long_cfg.get("date_col", "date"),
                target_col=long_cfg.get("target", "life_expectancy_at_birth"),
                window_size=long_cfg.get("window_size", 3),
                min_history=long_cfg.get("min_history", 2),
                max_missing_frac=long_cfg.get("max_missing_frac", 1.0),
                mode=long_cfg.get("mode", "panel"),
                panel_stride=long_cfg.get("panel_stride", 1),
                target_mode=mode,
                dataset_name=dataset_name,
                impute_missing=long_cfg.get("impute_missing", True),
            )


if __name__ == "__main__":
    main()
