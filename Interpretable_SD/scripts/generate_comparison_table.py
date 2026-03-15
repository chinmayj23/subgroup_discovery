import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = Path(__file__).resolve().parents[1] / "src"
for p in [REPO_ROOT, SRC_ROOT]:
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from interpretable_sd.comparison import generate_comparison_tables


def parse_args():
    parser = argparse.ArgumentParser(description="Generate comparison tables from Interpretable_SD outputs.")
    parser.add_argument(
        "--output-dir",
        default="Interpretable_SD/outputs/world_bank",
        help="Output directory produced by run_interpretable_sd.py",
    )
    parser.add_argument(
        "--table-name",
        default="comparison",
        help="Base file name for generated tables.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        out = generate_comparison_tables(args.output_dir, table_name=args.table_name)
        print(json.dumps(out, indent=2))
    except Exception as exc:
        print(f"[Comparison] ERROR: {exc}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
