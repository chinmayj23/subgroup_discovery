import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline_longitudinal import run_longitudinal_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run longitudinal subgroup pipelines.")
    parser.add_argument(
        "--config",
        default="SD_longitudinal/configs/longitudinal.json",
        help="Path to longitudinal config JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="SD_longitudinal/outputs/longitudinal",
        help="Output directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_longitudinal_pipeline(args.config, args.output_dir)


if __name__ == "__main__":
    main()
