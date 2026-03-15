import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline_cutoff_snapshot_eval import run_cutoff_snapshot_eval


def parse_args():
    parser = argparse.ArgumentParser(description="Run fixed-row snapshot cut-off evaluation.")
    parser.add_argument(
        "--config",
        default="SD_longitudinal/configs/cutoff_snapshot_eval.json",
        help="Path to cutoff snapshot eval config JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="SD_longitudinal/outputs/cutoff_snapshot_eval",
        help="Output directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_cutoff_snapshot_eval(args.config, args.output_dir)


if __name__ == "__main__":
    main()
