import time
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd

from SD_longitudinal.src.common import LABEL_COLUMN
from SD_longitudinal.src.plotting import save_kde_plot
from interpretable_sd.comparison import generate_comparison_tables
from interpretable_sd.baselines import (
    run_sysurv_style_baseline,
    run_tree_paper_style_baseline,
)
from interpretable_sd.interestingness import run_tail_cluster, run_timetribes_windowed
from interpretable_sd.preprocessing import (
    apply_target_transform,
    build_panel_feature_table,
    build_interest_segments,
    extract_year,
    normalize_time_key,
    prepare_xy_with_ids,
    preprocess_xy_train_only,
    split_train_test_by_subject,
)
from interpretable_sd.reporting import save_syflow_like_report
from interpretable_sd.rules import run_rf_rule_module
from interpretable_sd.utils.io import ensure_dir, load_json, save_json
from interpretable_sd.utils.metrics import interestingness_metrics


INTERESTINGNESS_RUNNERS = {
    "tail_cluster": run_tail_cluster,
    "timetribes_windowed": run_timetribes_windowed,
}
def _prepare_rule_matrices(
    labeled_df: pd.DataFrame,
    id_col: str,
    date_col: str,
    target_col: str,
    split_cfg: Dict,
    feature_cfg: Dict,
):
    _, _, train_subjects, test_subjects = split_train_test_by_subject(
        labeled_df,
        id_col=id_col,
        label_col=LABEL_COLUMN,
        test_size=float(split_cfg.get("test_size", 0.25)),
        seed=int(split_cfg.get("seed", 42)),
    )

    non_feature_cols = {
        "sample_id",
        "segment_id",
        "segment_start_date",
        "segment_end_date",
        "segment_start_year",
        "segment_end_year",
        "segment_length",
        "window_spec",
        "window_length",
        "window_start_year",
        "window_end_year",
        "window_target_mean",
        "window_target_delta",
        "window_target_trend",
        "window_target_std",
        "_year",
        date_col,
    }
    X, y, feature_names, target_values, row_ids, meta_df = prepare_xy_with_ids(
        labeled_df,
        id_col=id_col,
        target_col=target_col,
        label_col=LABEL_COLUMN,
        exclude_columns=[c for c in non_feature_cols if c in labeled_df.columns],
        exclude_prefixes=["label_target_", "window_", "segment_"],
        meta_columns=[
            "segment_id",
            "segment_start_year",
            "segment_end_year",
            "segment_length",
        ],
    )
    if X.size == 0:
        return None

    train_mask = np.isin(row_ids, list(train_subjects))
    test_mask = np.isin(row_ids, list(test_subjects))
    if train_mask.sum() == 0 or test_mask.sum() == 0:
        return None

    X_proc, feature_names = preprocess_xy_train_only(
        X,
        feature_names,
        train_mask=train_mask,
        max_missing_frac=float(feature_cfg.get("max_missing_frac", 0.95)),
        impute_missing=bool(feature_cfg.get("impute_missing", True)),
    )
    if X_proc.size == 0:
        return None

    return {
        "X_train": X_proc[train_mask],
        "y_train": y[train_mask],
        "X_test": X_proc[test_mask],
        "y_test": y[test_mask],
        "feature_names": feature_names,
        "target_train": target_values[train_mask],
        "target_test": target_values[test_mask],
        "train_subjects": [str(x) for x in train_subjects],
        "test_subjects": [str(x) for x in test_subjects],
        "meta_train": meta_df.iloc[train_mask].reset_index(drop=True),
        "meta_test": meta_df.iloc[test_mask].reset_index(drop=True),
    }


def _run_one_interestingness_method(
    method_name: str,
    method_cfg: Dict,
    raw_interest_df: pd.DataFrame,
    work_df: pd.DataFrame,
    target_col: str,
    id_col: str,
    date_col: str,
    feature_cfg: Dict,
    exp_dir: Path,
    split_cfg: Dict,
    rf_cfg: Dict,
    baseline_cfg: Dict,
    segment_cfg: Dict,
    out_dir: Path,
) -> Dict:
    if method_name not in INTERESTINGNESS_RUNNERS:
        raise ValueError(f"Unknown interestingness method: {method_name}")

    run_fun = INTERESTINGNESS_RUNNERS[method_name]
    print(f"  [Method] {method_name} starting...")
    t0 = time.perf_counter()
    output = run_fun(raw_interest_df, target_col=target_col, cfg=method_cfg)
    raw_labeled_df = output["labeled_df"].copy()
    raw_metrics = output["metrics"]
    print(
        f"  [Method] {method_name} labeled rows={len(raw_labeled_df)} "
        f"in {time.perf_counter() - t0:.1f}s"
    )

    out_dir = ensure_dir(out_dir)
    raw_labeled_df.to_csv(out_dir / "raw_rows_labeled.csv", index=False)
    save_json(out_dir / "interestingness_metrics_raw.json", raw_metrics)

    panel_df, question_map, feature_base_map = build_panel_feature_table(
        work_df,
        id_col=id_col,
        date_col=date_col,
        target_col=target_col,
        feature_cfg=feature_cfg,
    )
    question_map_path = exp_dir / "question_map.json"
    if not question_map_path.exists():
        question_map_path.write_text(
            pd.DataFrame(
                {
                    "feature": list(question_map.keys()),
                    "question": list(question_map.values()),
                    "base_indicator": [feature_base_map.get(k, "") for k in question_map.keys()],
                }
            ).to_json(orient="records", indent=2),
            encoding="utf-8",
        )

    row_labeled_df, segment_df = build_interest_segments(
        raw_labeled_df,
        panel_df=panel_df,
        id_col=id_col,
        date_col=date_col,
        target_col=target_col,
        target_aggregation=str(segment_cfg.get("target_aggregation", "mean")),
        min_interesting_run_length=int(segment_cfg.get("min_interesting_run_length", 1)),
    )
    if row_labeled_df.empty or segment_df.empty:
        print(f"  [Method] {method_name} no usable segments after segmentation.")
        save_json(out_dir / "rule_module_metrics.json", {"error": "No usable segments after segmentation"})
        return {
            "interestingness": {"error": "No usable segments after segmentation"},
            "rules": {"error": "No usable segments after segmentation"},
            "baselines": {},
        }

    metrics_interest = interestingness_metrics(
        pd.to_numeric(row_labeled_df[target_col], errors="coerce").values,
        row_labeled_df[LABEL_COLUMN].astype(int).values,
    )
    metrics_interest["n_rows"] = int(len(row_labeled_df))
    metrics_interest["n_interesting_rows"] = int(row_labeled_df[LABEL_COLUMN].sum())
    metrics_interest["n_segments"] = int(len(segment_df))
    metrics_interest["n_interesting_segments"] = int(segment_df[LABEL_COLUMN].sum())
    metrics_interest["segment_support"] = (
        float(segment_df[LABEL_COLUMN].mean()) if len(segment_df) else 0.0
    )

    row_labeled_df.to_csv(out_dir / "row_labels.csv", index=False)
    segment_df.to_csv(out_dir / "segment_samples.csv", index=False)
    segment_df.to_csv(out_dir / "samples_labeled.csv", index=False)
    save_json(out_dir / "interestingness_metrics.json", metrics_interest)
    save_kde_plot(
        row_labeled_df,
        target_col,
        LABEL_COLUMN,
        out_dir / "interestingness_kde",
        title=f"{method_name}: target distribution (original vs interesting vs non-interesting)",
    )
    print(f"  [Method] {method_name} saved interestingness KDE plot.")

    matrix_pack = _prepare_rule_matrices(
        segment_df,
        id_col=id_col,
        date_col=date_col,
        target_col="segment_target_score",
        split_cfg=split_cfg,
        feature_cfg=rf_cfg,
    )
    if matrix_pack is None:
        print(f"  [Method] {method_name} no usable train/test matrices after preprocessing.")
        save_json(out_dir / "rule_module_metrics.json", {"error": "No usable train/test matrices"})
        return {
            "interestingness": metrics_interest,
            "rules": {"error": "No usable train/test matrices"},
            "baselines": {},
        }

    save_json(
        out_dir / "subject_split.json",
        {
            "n_train_subjects": int(len(matrix_pack["train_subjects"])),
            "n_test_subjects": int(len(matrix_pack["test_subjects"])),
            "train_subjects": matrix_pack["train_subjects"],
            "test_subjects": matrix_pack["test_subjects"],
        },
    )

    rf_out = run_rf_rule_module(
        matrix_pack["X_train"],
        matrix_pack["y_train"],
        matrix_pack["X_test"],
        matrix_pack["y_test"],
        matrix_pack["feature_names"],
        matrix_pack["target_train"],
        matrix_pack["meta_train"],
        out_dir / "rf_rule_module",
        rf_cfg,
    )
    save_json(out_dir / "rule_module_metrics.json", rf_out)

    baseline_results = {}
    if baseline_cfg.get("sysurv_style", {}).get("enabled", True):
        sys_out = run_sysurv_style_baseline(
            matrix_pack["X_train"],
            matrix_pack["y_train"],
            matrix_pack["X_test"],
            matrix_pack["y_test"],
            matrix_pack["feature_names"],
            matrix_pack["target_train"],
            matrix_pack["meta_train"],
            out_dir / "baselines" / "sysurv_style",
            baseline_cfg.get("sysurv_style", {}),
        )
        baseline_results["sysurv_style"] = sys_out

    if baseline_cfg.get("tree_paper_style", {}).get("enabled", True):
        tree_out = run_tree_paper_style_baseline(
            matrix_pack["X_train"],
            matrix_pack["y_train"],
            matrix_pack["X_test"],
            matrix_pack["y_test"],
            matrix_pack["feature_names"],
            matrix_pack["target_train"],
            matrix_pack["meta_train"],
            out_dir / "baselines" / "tree_paper_style",
            baseline_cfg.get("tree_paper_style", {}),
        )
        baseline_results["tree_paper_style"] = tree_out

    save_json(out_dir / "baseline_metrics.json", baseline_results)

    save_syflow_like_report(
        out_dir / "syflow_like_report.json",
        method_name=method_name,
        labeled_df=segment_df,
        target_col="segment_target_score",
        interestingness_metrics_payload=metrics_interest,
        rule_metrics_payload=rf_out,
        baseline_payload=baseline_results,
    )

    return {
        "interestingness": metrics_interest,
        "rules": rf_out,
        "baselines": baseline_results,
    }


def run_interpretable_sd_pipeline(config_path: str, output_dir: str = None) -> None:
    cfg = load_json(config_path)
    data_cfg = cfg["data"]
    experiments = cfg.get("experiments", [])
    if not experiments:
        raise ValueError("Config must contain at least one experiment.")

    out_root = Path(output_dir or cfg.get("output_dir", "Interpretable_SD/outputs"))
    ensure_dir(out_root)

    df = pd.read_csv(data_cfg["path"])
    id_col = data_cfg.get("id_col", "country")
    date_col = data_cfg.get("date_col", "date")
    print(f"[Load] {data_cfg.get('name', 'dataset')} rows={len(df)} cols={len(df.columns)}")

    summary_records = []
    for exp_cfg in experiments:
        if not exp_cfg.get("enabled", True):
            continue
        exp_name = exp_cfg["name"]
        print(f"[Experiment] {exp_name}")
        t0 = time.perf_counter()
        exp_dir = ensure_dir(out_root / exp_name)

        work_df, target_col = apply_target_transform(df, exp_cfg["target"])
        print(f"  [Experiment] target={target_col}")
        work_df = work_df.dropna(subset=[id_col, date_col, target_col]).copy()
        work_df["_year"] = extract_year(work_df[date_col])
        work_df["_time_key"] = normalize_time_key(work_df[date_col])
        raw_interest_df = work_df[[id_col, date_col, target_col, "_year", "_time_key"]].copy()
        feature_cfg = dict(cfg.get("defaults", {}).get("feature_engineering", {}))
        feature_cfg.update(exp_cfg.get("feature_engineering", {}))

        segment_cfg = dict(cfg.get("defaults", {}).get("segmentation", {}))
        segment_cfg.update(exp_cfg.get("segmentation", {}))

        methods_cfg = exp_cfg.get("interestingness", {})
        methods = methods_cfg.get("methods", ["tail_cluster", "timetribes_windowed"])
        split_cfg = dict(cfg.get("defaults", {}).get("split", {}))
        split_cfg.update(exp_cfg.get("split", {}))
        rf_cfg = dict(cfg.get("defaults", {}).get("rf_rule_module", {}))
        rf_cfg.update(exp_cfg.get("rf_rule_module", {}))
        baseline_cfg = dict(cfg.get("defaults", {}).get("baselines", {}))
        baseline_cfg.update(exp_cfg.get("baselines", {}))

        exp_summary = {}
        for method_name in methods:
            m_cfg = dict(methods_cfg.get(method_name, {}))
            method_out = _run_one_interestingness_method(
                method_name=method_name,
                method_cfg=m_cfg,
                raw_interest_df=raw_interest_df,
                work_df=work_df,
                target_col=target_col,
                id_col=id_col,
                date_col=date_col,
                feature_cfg=feature_cfg,
                exp_dir=exp_dir,
                split_cfg=split_cfg,
                rf_cfg=rf_cfg,
                baseline_cfg=baseline_cfg,
                segment_cfg=segment_cfg,
                out_dir=exp_dir / method_name,
            )
            exp_summary[method_name] = method_out
            print(f"  [Method] {method_name} complete.")

        elapsed = time.perf_counter() - t0
        print(f"[Experiment] {exp_name} done in {elapsed:.1f}s")
        save_json(
            exp_dir / "experiment_summary.json",
            {
                "experiment": exp_name,
                "target_col": target_col,
                "n_rows": int(len(raw_interest_df)),
                "elapsed_sec": float(elapsed),
                "methods": exp_summary,
            },
        )
        summary_records.append(
            {
                "experiment": exp_name,
                "target_col": target_col,
                "n_rows": int(len(raw_interest_df)),
                "elapsed_sec": float(elapsed),
            }
        )

    if summary_records:
        pd.DataFrame(summary_records).to_csv(out_root / "run_summary.csv", index=False)
        save_json(out_root / "run_summary.json", {"experiments": summary_records})

    comparison_cfg = cfg.get("comparison_table", {})
    if bool(comparison_cfg.get("enabled", True)):
        table_name = str(comparison_cfg.get("table_name", "comparison"))
        try:
            out_paths = generate_comparison_tables(str(out_root), table_name=table_name)
            save_json(out_root / "comparison" / "comparison_artifacts.json", out_paths)
        except Exception as exc:
            save_json(
                out_root / "comparison" / "comparison_artifacts.json",
                {"error": f"Could not generate comparison tables: {exc}"},
            )
    print(f"Interpretable_SD pipeline complete. Outputs in {out_root}")
