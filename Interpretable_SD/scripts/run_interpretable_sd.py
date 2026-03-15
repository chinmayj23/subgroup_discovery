import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
for p in [REPO_ROOT, SRC_ROOT]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from interpretable_sd.pipeline import run_interpretable_sd_pipeline


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run Interpretable_SD modular longitudinal subgroup discovery."
    )
    parser.add_argument(
        "--config",
        default="Interpretable_SD/configs/world_bank_all_in_one.json",
        help="Path to config JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="Interpretable_SD/outputs/world_bank",
        help="Output directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_interpretable_sd_pipeline(args.config, args.output_dir)


if __name__ == "__main__":
    main()
