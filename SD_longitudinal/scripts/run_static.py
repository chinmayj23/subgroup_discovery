import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.pipeline_static import run_static_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run static subgroup pipelines.")
    parser.add_argument(
        "--config",
        default="SD_longitudinal/configs/static.json",
        help="Path to static config JSON.",
    )
    parser.add_argument(
        "--output-dir",
        default="SD_longitudinal/outputs/static",
        help="Output directory.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    run_static_pipeline(args.config, args.output_dir)


if __name__ == "__main__":
    main()
