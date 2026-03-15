from pathlib import Path
from typing import Dict, List

import pandas as pd

from interpretable_sd.utils.io import ensure_dir, load_json


def _safe_get(d: Dict, keys: List[str], default=None):
    cur = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def _add_metric_block(
    row: Dict,
    prefix: str,
    metric_block: Dict,
) -> None:
    fields = [
        "accuracy",
        "balanced_accuracy",
        "precision",
        "recall",
        "f1",
        "roc_auc",
        "n_rows",
        "n_positive",
    ]
    for f in fields:
        row[f"{prefix}_{f}"] = metric_block.get(f)


def _collect_method_rows(
    experiment_name: str,
    method_name: str,
    method_dir: Path,
) -> List[Dict]:
    rows: List[Dict] = []

    interestingness_path = method_dir / "interestingness_metrics.json"
    rule_path = method_dir / "rule_module_metrics.json"
    baseline_path = method_dir / "baseline_metrics.json"

    if not interestingness_path.exists():
        return rows

    interest = load_json(str(interestingness_path))
    common = {
        "experiment": experiment_name,
        "interestingness_method": method_name,
        "support": interest.get("support"),
        "n_total": interest.get("n_total"),
        "n_interesting": interest.get("n_interesting"),
        "n_rows": interest.get("n_rows"),
        "n_interesting_rows": interest.get("n_interesting_rows"),
        "n_segments": interest.get("n_segments"),
        "n_interesting_segments": interest.get("n_interesting_segments"),
        "segment_support": interest.get("segment_support"),
        "mean_all": interest.get("mean_all"),
        "mean_interesting": interest.get("mean_interesting"),
        "mean_shift": interest.get("mean_shift"),
        "kl_divergence": interest.get("kl_divergence"),
    }

    if rule_path.exists():
        rf = load_json(str(rule_path))
        rf_row = dict(common)
        rf_row.update(
            {
                "model": "rf_rule_module",
                "n_rules": rf.get("n_rules"),
                "n_trees": rf.get("n_trees"),
                "best_seed": rf.get("best_seed"),
            }
        )
        _add_metric_block(rf_row, "train", _safe_get(rf, ["metrics", "train"], {}) or {})
        _add_metric_block(rf_row, "test", _safe_get(rf, ["metrics", "test"], {}) or {})
        rows.append(rf_row)

    if baseline_path.exists():
        baselines = load_json(str(baseline_path))
        for baseline_name, payload in baselines.items():
            b_row = dict(common)
            b_row.update(
                {
                    "model": baseline_name,
                    "n_rules": payload.get("n_rules"),
                    "n_trees": None,
                    "best_seed": None,
                }
            )
            metric_train = _safe_get(payload, ["metrics", "train"], {}) or {}
            metric_test = _safe_get(payload, ["metrics", "test"], {}) or {}
            _add_metric_block(b_row, "train", metric_train)
            _add_metric_block(b_row, "test", metric_test)
            if baseline_name == "sysurv_style":
                b_row["risk_threshold_quantile"] = _safe_get(payload, ["metrics", "risk_threshold_quantile"])
                b_row["risk_threshold"] = _safe_get(payload, ["metrics", "risk_threshold"])
            rows.append(b_row)

    return rows


def generate_comparison_tables(output_dir: str, table_name: str = "comparison") -> Dict[str, str]:
    out_root = Path(output_dir)
    if not out_root.exists():
        parent = out_root.parent
        siblings = []
        if parent.exists():
            siblings = sorted([p.name for p in parent.iterdir() if p.is_dir()])
        hint = (
            f"Output directory not found: {out_root}. "
            "Run the main pipeline first, e.g. "
            "'python Interpretable_SD/scripts/run_interpretable_sd.py --config "
            "Interpretable_SD/configs/world_bank_full_paper_grade.json'. "
            f"Existing sibling output folders under {parent}: {siblings}"
        )
        raise FileNotFoundError(hint)

    all_rows: List[Dict] = []
    for exp_dir in sorted([p for p in out_root.iterdir() if p.is_dir()]):
        exp_summary = exp_dir / "experiment_summary.json"
        if not exp_summary.exists():
            continue
        exp_name = exp_dir.name
        for method_dir in sorted([p for p in exp_dir.iterdir() if p.is_dir()]):
            if method_dir.name in {"baselines", "rf_rule_module", "trees"}:
                continue
            all_rows.extend(_collect_method_rows(exp_name, method_dir.name, method_dir))

    if not all_rows:
        raise RuntimeError("No method output folders with metrics found.")

    df = pd.DataFrame(all_rows)
    df = df.sort_values(
        by=["experiment", "interestingness_method", "test_balanced_accuracy", "test_accuracy"],
        ascending=[True, True, False, False],
    )

    comparison_dir = ensure_dir(out_root / "comparison")
    csv_path = comparison_dir / f"{table_name}.csv"
    md_path = comparison_dir / f"{table_name}.md"
    leaderboard_path = comparison_dir / f"{table_name}_leaderboard.csv"

    df.to_csv(csv_path, index=False)

    md_cols = [
        "experiment",
        "interestingness_method",
        "model",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_precision",
        "test_recall",
        "test_f1",
        "support",
        "mean_shift",
        "kl_divergence",
        "n_interesting",
        "n_rules",
    ]
    md_df = df[[c for c in md_cols if c in df.columns]].copy()
    for c in md_df.columns:
        if md_df[c].dtype.kind in {"f"}:
            md_df[c] = md_df[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "")
    md_path.write_text(md_df.to_markdown(index=False), encoding="utf-8")

    # Best row per (experiment, interestingness_method) by test balanced accuracy then test accuracy.
    leaderboard = (
        df.sort_values(
            by=["experiment", "interestingness_method", "test_balanced_accuracy", "test_accuracy"],
            ascending=[True, True, False, False],
        )
        .groupby(["experiment", "interestingness_method"], as_index=False)
        .head(1)
    )
    leaderboard.to_csv(leaderboard_path, index=False)

    return {
        "comparison_csv": str(csv_path),
        "comparison_md": str(md_path),
        "leaderboard_csv": str(leaderboard_path),
    }
