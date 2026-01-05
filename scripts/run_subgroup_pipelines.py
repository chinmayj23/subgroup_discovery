import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.subgroup_pipelines import load_config, run_pipelines_for_dataset


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run subgroup discovery pipelines on configured datasets."
    )
    parser.add_argument(
        "--config",
        default="configs/subgroup_pipelines.json",
        help="Path to pipeline config JSON.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset names to run (space-separated).",
    )
    parser.add_argument(
        "--pipelines",
        nargs="*",
        default=None,
        help="Optional pipeline names to run (space-separated).",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs_subgroups/pipeline_runs",
        help="Base directory for outputs.",
    )
    parser.add_argument(
        "--skip-forest",
        action="store_true",
        help="Skip forest search stage.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_config(args.config)

    all_datasets = cfg["datasets"]
    if args.datasets:
        dataset_cfgs = [d for d in all_datasets if d["name"] in set(args.datasets)]
    else:
        dataset_cfgs = [d for d in all_datasets if d.get("enabled", True)]

    pipeline_cfg = cfg["pipelines"]
    if args.pipelines:
        pipeline_cfg = {k: v for k, v in pipeline_cfg.items() if k in set(args.pipelines)}

    if not dataset_cfgs:
        raise ValueError("No datasets selected. Check --datasets or config.")
    if not pipeline_cfg:
        raise ValueError("No pipelines selected. Check --pipelines or config.")

    output_base_dir = Path(args.output_dir)
    output_base_dir.mkdir(parents=True, exist_ok=True)

    for dataset_cfg in dataset_cfgs:
        print(f"Running dataset: {dataset_cfg['name']}")
        run_pipelines_for_dataset(
            dataset_cfg=dataset_cfg,
            pipeline_cfg=pipeline_cfg,
            forest_cfg=cfg.get("forest", {}),
            output_base_dir=output_base_dir,
            run_forest=not args.skip_forest,
        )

    print(f"Done. Outputs in {output_base_dir}")


if __name__ == "__main__":
    main()
