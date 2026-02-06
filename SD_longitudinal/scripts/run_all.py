import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.common import load_json
from src.pipeline_longitudinal import run_longitudinal_pipeline
from src.pipeline_static import run_static_pipeline


def parse_args():
    parser = argparse.ArgumentParser(description="Run static + longitudinal pipelines.")
    parser.add_argument(
        "--config",
        default="SD_longitudinal/configs/run_all.json",
        help="Path to combined run config JSON.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = load_json(args.config)

    output_dir = Path(cfg.get("output_dir", "SD_longitudinal/outputs"))

    static_cfg = cfg.get("static_config", "SD_longitudinal/configs/static.json")
    long_cfg = cfg.get("longitudinal_config", "SD_longitudinal/configs/longitudinal.json")

    run_static_pipeline(static_cfg, str(output_dir / "static"))
    run_longitudinal_pipeline(long_cfg, str(output_dir / "longitudinal"))


if __name__ == "__main__":
    main()
