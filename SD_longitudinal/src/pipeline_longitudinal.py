import time
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .common import LABEL_COLUMN, drop_by_prefix, ensure_dir, load_json, prepare_xy_with_target, save_json
from .forest import forest_predict, rebuild_forest, run_forest_search
from .labeling import label_original_method, label_syflow_method
from .longitudinal_features import build_longitudinal_features
from .plotting import (
    save_cutoff_accuracy_plot,
    save_forest_accuracy_plot,
    save_forest_tree_artifacts,
    save_kde_plot,
)


def clean_and_impute(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    label_target_cols: List[str],
    max_missing_frac: float = 0.5,
    impute: bool = True,
) -> pd.DataFrame:
    keep_cols = set([id_col, date_col] + label_target_cols)
    missing = df.isna().mean()
    drop_cols = [c for c, frac in missing.items() if frac > max_missing_frac and c not in keep_cols]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    if impute:
        numeric = df.select_dtypes(include=[np.number]).columns
        cols = [c for c in numeric if c not in keep_cols]
        if cols:
            df[cols] = df[cols].fillna(df[cols].median()).fillna(0)

    return df.dropna()


def _save_outputs(
    base_dir: Path,
    dataset_name: str,
    pipeline_name: str,
    labeled_df: pd.DataFrame,
    forest_result,
    question_map: Dict[str, str],
    feature_base_map: Dict[str, str],
) -> Path:
    out_dir = ensure_dir(base_dir / dataset_name / pipeline_name)
    labeled_df.to_csv(out_dir / "labeled.csv", index=False)

    if forest_result is not None:
        forest_result.results_df.to_csv(out_dir / "forest_results.csv", index=False)
        save_json(out_dir / "best_forest.json", forest_result.best_row.to_dict())

    if question_map:
        q_df = pd.DataFrame(
            {
                "feature": list(question_map.keys()),
                "question": list(question_map.values()),
                "base_indicator": [feature_base_map.get(f, "") for f in question_map.keys()],
            }
        )
        q_df.to_csv(out_dir / "question_map.csv", index=False)

    summary = {
        "rows": int(len(labeled_df)),
        "interesting": int(labeled_df[LABEL_COLUMN].sum()),
        "non_interesting": int((~labeled_df[LABEL_COLUMN]).sum()),
    }
    save_json(out_dir / "summary.json", summary)

    return out_dir


def _select_target_column(target_mode: str, label_window: int) -> str:
    if target_mode == "trend":
        return f"label_target_trend{label_window}"
    if target_mode == "delta":
        return f"label_target_delta{label_window}"
    return "label_target_value"


def _extract_year(series: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(series):
        years = pd.to_numeric(series, errors="coerce")
    else:
        parsed = pd.to_datetime(series, errors="coerce")
        years = parsed.dt.year
        if years.notna().mean() < 0.8:
            fallback = pd.to_numeric(series.astype(str).str.extract(r"(\d{4})")[0], errors="coerce")
            years = years.fillna(fallback)
    return years.astype("Int64")


def _prepare_xy_with_target_and_ids(
    df: pd.DataFrame,
    id_col: str,
    target: str,
    label_column: str = LABEL_COLUMN,
    exclude_columns: Iterable[str] = (),
    exclude_prefixes: Iterable[str] = (),
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray, np.ndarray]:
    if label_column not in df.columns:
        raise ValueError(f"Label column '{label_column}' not found")
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' not found")

    row_ids = df[id_col]
    y = df[label_column].astype(int)
    target_series = pd.to_numeric(df[target], errors="coerce")

    drop_cols = [target, label_column]
    drop_cols.extend([c for c in exclude_columns if c in df.columns])

    feature_df = df.drop(columns=[c for c in drop_cols if c in df.columns])
    if exclude_prefixes:
        prefix_cols = drop_by_prefix(feature_df.columns, exclude_prefixes)
        feature_df = feature_df.drop(columns=prefix_cols)

    numeric_df = feature_df.select_dtypes(include=[np.number]).copy()
    if numeric_df.empty:
        return np.array([]), np.array([]), [], np.array([]), np.array([])

    valid_mask = ~(numeric_df.isna().any(axis=1) | y.isna() | target_series.isna() | row_ids.isna())
    numeric_df = numeric_df[valid_mask]
    y = y[valid_mask]
    target_series = target_series[valid_mask]
    row_ids = row_ids[valid_mask]

    return (
        numeric_df.values,
        y.values,
        list(numeric_df.columns),
        target_series.values,
        row_ids.values,
    )


def _run_standard_longitudinal_for_dataset(
    df: pd.DataFrame,
    dataset_cfg: Dict,
    pipeline_cfg: Dict,
    forest_cfg: Dict,
    output_base: Path,
) -> None:
    id_col = dataset_cfg.get("id_col", "country")
    date_col = dataset_cfg.get("date_col", "date")
    target_col = dataset_cfg.get("target", "life_expectancy_at_birth")

    question_pack = dataset_cfg.get("question_pack", {})
    lags = question_pack.get("lags", [1, 3, 5, 10])
    windows = question_pack.get("windows", [3, 5, 10])
    label_window = int(dataset_cfg.get("label_window", max(lags) if lags else 1))

    print("[Feature Eng] Building longitudinal features...")
    t0 = time.perf_counter()
    features_df, question_map, feature_base_map = build_longitudinal_features(
        df,
        id_col=id_col,
        date_col=date_col,
        target_col=target_col,
        lags=lags,
        windows=windows,
        label_window=label_window,
        min_history=dataset_cfg.get("min_history", 3),
        mode=dataset_cfg.get("mode", "snapshot"),
        panel_stride=dataset_cfg.get("panel_stride", 1),
        include_pct_change=question_pack.get("include_pct_change", True),
        include_volatility=question_pack.get("include_volatility", False),
        indicator_include=question_pack.get("indicator_include"),
        indicator_exclude=question_pack.get("indicator_exclude"),
    )
    t1 = time.perf_counter()
    print(f"[Feature Eng] Done in {t1 - t0:.1f}s -> {len(features_df)} rows, {len(features_df.columns)} columns")

    label_target_cols = [
        "label_target_value",
        f"label_target_delta{label_window}",
        f"label_target_trend{label_window}",
    ]

    print("[Clean] Dropping high-missing columns and imputing...")
    t2 = time.perf_counter()
    features_df = clean_and_impute(
        features_df,
        id_col=id_col,
        date_col=date_col,
        label_target_cols=label_target_cols,
        max_missing_frac=dataset_cfg.get("max_missing_frac", 0.5),
        impute=dataset_cfg.get("impute_missing", True),
    )
    t3 = time.perf_counter()
    print(f"[Clean] Done in {t3 - t2:.1f}s -> {len(features_df)} rows, {len(features_df.columns)} columns")

    target_modes = dataset_cfg.get("target_modes") or [dataset_cfg.get("target_mode", "value")]
    for target_mode in target_modes:
        active_target = _select_target_column(target_mode, label_window)
        if active_target not in features_df.columns:
            continue

        dataset_name = f"{dataset_cfg['name']}_{target_mode}"
        for pipeline_name, settings in pipeline_cfg.items():
            print(f"[Label] {dataset_name} | {pipeline_name} | target={active_target}")
            t4 = time.perf_counter()
            if pipeline_name == "tail_cluster":
                print("  [Label] Running TailCluster (single-linkage on target). This can take time on large panels.")
                labeled_df = label_original_method(
                    features_df,
                    active_target,
                    high_value_quantile=settings.get("high_value_quantile", 0.94),
                    two_tailed=settings.get("two_tailed", True),
                )
            elif pipeline_name == "timetribes":
                labeled_df = label_syflow_method(
                    features_df,
                    active_target,
                    n_seeds=settings.get("n_seeds", 1000),
                    beta=settings.get("beta", 0.5),
                    lambd_div=settings.get("lambd_div", 2.0),
                    top_percentile=settings.get("top_percentile", 0.01),
                    max_overlap=settings.get("max_overlap", 0.95),
                    min_subgroup_size=settings.get("min_subgroup_size", 50),
                    max_subgroup_frac=settings.get("max_subgroup_frac", 0.9),
                    progress_every=settings.get("progress_every", 100),
                )
            else:
                continue

            t5 = time.perf_counter()
            if LABEL_COLUMN in labeled_df.columns:
                n_int = int(labeled_df[LABEL_COLUMN].sum())
                print(f"  [Label] Done in {t5 - t4:.1f}s | interesting={n_int} / {len(labeled_df)}")
            else:
                print(f"  [Label] Done in {t5 - t4:.1f}s")

            exclude_prefixes = ["label_target_"]
            exclude_cols = [c for c, base in feature_base_map.items() if base == target_col]
            if target_col in labeled_df.columns:
                exclude_cols.append(target_col)
            if dataset_cfg.get("include_target_features", False):
                exclude_cols = []

            X, y, feature_names, target_values = prepare_xy_with_target(
                labeled_df,
                active_target,
                exclude_columns=exclude_cols,
                exclude_prefixes=exclude_prefixes,
            )
            if X.size == 0:
                continue

            forest_result = run_forest_search(
                X,
                y,
                n_seeds=forest_cfg.get("n_seeds", 1000),
                accuracy_quantile=forest_cfg.get("accuracy_quantile", 0.99),
                min_trees=forest_cfg.get("min_trees", 1),
                max_trees=forest_cfg.get("max_trees", 12),
                min_questions=forest_cfg.get("min_questions", 2),
                max_questions=forest_cfg.get("max_questions", 12),
                test_size=forest_cfg.get("test_size", 0.2),
                vote_rule=forest_cfg.get("vote_rule", "any"),
                eval_mode=forest_cfg.get("eval_mode", "train"),
                n_jobs=forest_cfg.get("n_jobs", -1),
            )

            trees, _ = rebuild_forest(
                X,
                y,
                forest_result.best_seed,
                forest_cfg.get("min_trees", 1),
                forest_cfg.get("max_trees", 12),
                forest_cfg.get("min_questions", 2),
                forest_cfg.get("max_questions", 12),
            )

            out_dir = _save_outputs(
                output_base,
                dataset_name,
                pipeline_name,
                labeled_df,
                forest_result,
                question_map,
                feature_base_map,
            )

            save_kde_plot(
                labeled_df,
                active_target,
                LABEL_COLUMN,
                out_dir / "kde_plot",
                title=f"{dataset_name}: {pipeline_name} target distribution",
            )
            save_forest_accuracy_plot(
                forest_result.results_df,
                out_dir / "accuracy_vs_questions",
                best_row=forest_result.best_row,
            )
            save_forest_tree_artifacts(
                trees,
                X,
                y,
                feature_names,
                out_dir,
                question_map=question_map,
                target_values=target_values,
                target_name=active_target,
            )


def _run_cutoff_growth_eval_for_dataset(
    df: pd.DataFrame,
    dataset_cfg: Dict,
    forest_cfg: Dict,
    output_base: Path,
) -> None:
    eval_cfg = dataset_cfg.get("cutoff_growth_eval", {})
    if not eval_cfg.get("enabled", False):
        return

    dataset_name = dataset_cfg["name"]
    id_col = dataset_cfg.get("id_col", "country")
    date_col = dataset_cfg.get("date_col", "date")

    gdp_col = eval_cfg.get("gdp_col", "GDP_current_US")
    population_col = eval_cfg.get("population_col", "population")
    feature_target_col = eval_cfg.get("feature_target_col", "gdp_per_capita")

    start_year = int(eval_cfg.get("growth_start_year", 2010))
    end_year = int(eval_cfg.get("growth_end_year", 2020))
    cutoff_years = sorted({int(y) for y in eval_cfg.get("cutoff_years", [1980, 1990, 2000, 2010])})
    interesting_quantile = float(eval_cfg.get("interesting_quantile", 0.9))
    min_growth = eval_cfg.get("min_growth")
    if min_growth is not None:
        min_growth = float(min_growth)

    split_seed = int(eval_cfg.get("split_seed", 42))
    test_size = float(eval_cfg.get("test_size", forest_cfg.get("test_size", 0.2)))
    eval_name = eval_cfg.get("name", f"gdp_growth_{start_year}_{end_year}")
    split_country_pool = str(eval_cfg.get("split_country_pool", "all_labeled")).strip().lower()
    snapshot_constant_rows = bool(eval_cfg.get("snapshot_constant_rows", True))
    strict_country_split = bool(eval_cfg.get("strict_country_split", True))
    min_train_rows = int(eval_cfg.get("min_train_rows", 10))
    min_test_rows = int(eval_cfg.get("min_test_rows", 5))
    min_train_positives = int(eval_cfg.get("min_train_positives", 1))
    min_test_positives = int(eval_cfg.get("min_test_positives", 1))

    required = [id_col, date_col, gdp_col, population_col]
    missing = [c for c in required if c not in df.columns]
    if missing:
        print(f"[Cutoff Eval] Skip {dataset_name}: missing columns {missing}")
        return

    work_df = df.copy()
    work_df["_year"] = _extract_year(work_df[date_col])
    work_df[gdp_col] = pd.to_numeric(work_df[gdp_col], errors="coerce")
    work_df[population_col] = pd.to_numeric(work_df[population_col], errors="coerce")
    work_df[feature_target_col] = work_df[gdp_col] / (work_df[population_col] + 1e-9)
    work_df.loc[~np.isfinite(work_df[feature_target_col]), feature_target_col] = np.nan

    start_col = f"gdp_per_capita_{start_year}"
    end_col = f"gdp_per_capita_{end_year}"
    growth_col = f"gdp_per_capita_growth_{start_year}_{end_year}"

    growth_df = (
        work_df[[id_col, "_year", feature_target_col]]
        .dropna(subset=[id_col, "_year", feature_target_col])
        .query("_year == @start_year or _year == @end_year")
    )
    if growth_df.empty:
        print(f"[Cutoff Eval] Skip {dataset_name}: no data for years {start_year} and {end_year}")
        return

    pivot = growth_df.pivot_table(index=id_col, columns="_year", values=feature_target_col, aggfunc="last")
    if start_year not in pivot.columns or end_year not in pivot.columns:
        print(f"[Cutoff Eval] Skip {dataset_name}: insufficient country coverage for label years")
        return

    country_growth = (
        pivot[[start_year, end_year]]
        .rename(columns={start_year: start_col, end_year: end_col})
        .reset_index()
        .dropna()
    )
    country_growth[growth_col] = country_growth[end_col] - country_growth[start_col]
    threshold = float(country_growth[growth_col].quantile(interesting_quantile))
    if min_growth is not None:
        threshold = max(threshold, min_growth)
    country_growth[LABEL_COLUMN] = country_growth[growth_col] >= threshold
    country_growth = country_growth.sort_values(id_col).reset_index(drop=True)

    if country_growth[LABEL_COLUMN].nunique() < 2:
        print(f"[Cutoff Eval] Skip {dataset_name}: growth labels have a single class")
        return

    if split_country_pool not in {"all_labeled", "available_all_cutoffs"}:
        raise ValueError(
            "cutoff_growth_eval.split_country_pool must be one of "
            "{'all_labeled', 'available_all_cutoffs'}"
        )

    split_pool_df = country_growth.copy()
    if split_country_pool == "available_all_cutoffs":
        available_sets = []
        valid_df = work_df[work_df[id_col].notna() & work_df["_year"].notna() & work_df[feature_target_col].notna()]
        for cutoff_year in cutoff_years:
            ids_at_cutoff = set(valid_df.loc[valid_df["_year"] <= cutoff_year, id_col].astype(str).unique().tolist())
            available_sets.append(ids_at_cutoff)
        if available_sets:
            eligible_ids = set.intersection(*available_sets)
            split_pool_df = split_pool_df[split_pool_df[id_col].astype(str).isin(eligible_ids)].copy()
        if split_pool_df.empty:
            print("[Cutoff Eval] No countries available across all cut-offs for fixed split pool.")
            return

    label_ids = split_pool_df[id_col].values
    label_y = split_pool_df[LABEL_COLUMN].astype(int).values
    try:
        train_ids, test_ids = train_test_split(
            label_ids,
            test_size=test_size,
            random_state=split_seed,
            stratify=label_y,
        )
    except ValueError:
        train_ids, test_ids = train_test_split(
            label_ids,
            test_size=test_size,
            random_state=split_seed,
        )
    train_ids = set(train_ids.tolist())
    test_ids = set(test_ids.tolist())

    eval_dir = ensure_dir(output_base / f"{dataset_name}_{eval_name}")
    country_growth.to_csv(eval_dir / "country_growth_labels.csv", index=False)
    split_pool_df.to_csv(eval_dir / "split_pool_countries.csv", index=False)
    save_json(
        eval_dir / "label_definition.json",
        {
            "growth_start_year": start_year,
            "growth_end_year": end_year,
            "interesting_quantile": interesting_quantile,
            "threshold": threshold,
            "n_countries": int(len(country_growth)),
            "n_interesting": int(country_growth[LABEL_COLUMN].sum()),
            "n_non_interesting": int((~country_growth[LABEL_COLUMN]).sum()),
            "split_pool_type": split_country_pool,
            "split_pool_countries": int(len(split_pool_df)),
            "split_pool_interesting": int(split_pool_df[LABEL_COLUMN].sum()),
            "split_pool_non_interesting": int((~split_pool_df[LABEL_COLUMN]).sum()),
            "cutoff_years": cutoff_years,
            "split_seed": split_seed,
            "test_size": test_size,
            "strict_country_split": strict_country_split,
            "snapshot_constant_rows": snapshot_constant_rows,
            "min_train_rows": min_train_rows,
            "min_test_rows": min_test_rows,
            "min_train_positives": min_train_positives,
            "min_test_positives": min_test_positives,
        },
    )

    split_assign_df = split_pool_df[[id_col, LABEL_COLUMN]].copy()
    split_assign_df["split"] = np.where(split_assign_df[id_col].isin(train_ids), "train", "test")
    split_assign_df = split_assign_df.sort_values(by=["split", id_col]).reset_index(drop=True)
    split_assign_df.to_csv(eval_dir / "country_split.csv", index=False)
    split_pool_ids = set(split_pool_df[id_col].tolist())
    split_pool_ids_as_str = set(split_pool_df[id_col].astype(str).tolist())

    train_country_count = int((split_assign_df["split"] == "train").sum())
    test_country_count = int((split_assign_df["split"] == "test").sum())
    train_country_pos = int(split_assign_df.loc[split_assign_df["split"] == "train", LABEL_COLUMN].sum())
    test_country_pos = int(split_assign_df.loc[split_assign_df["split"] == "test", LABEL_COLUMN].sum())
    print(
        "[Cutoff Eval] Fixed country split | "
        f"train={train_country_count} (pos={train_country_pos}, neg={train_country_count-train_country_pos}) | "
        f"test={test_country_count} (pos={test_country_pos}, neg={test_country_count-test_country_pos})"
    )

    base_q = dataset_cfg.get("question_pack", {})
    eval_q = eval_cfg.get("question_pack", base_q)
    lags = eval_q.get("lags", [1, 3, 5, 10])
    windows = eval_q.get("windows", [3, 5, 10])
    label_window = int(eval_cfg.get("label_window", dataset_cfg.get("label_window", max(lags) if lags else 1)))
    min_history = int(eval_cfg.get("min_history", dataset_cfg.get("min_history", 3)))
    mode = eval_cfg.get("mode", "snapshot")
    panel_stride = int(eval_cfg.get("panel_stride", dataset_cfg.get("panel_stride", 1)))
    include_pct_change = bool(eval_q.get("include_pct_change", True))
    include_volatility = bool(eval_q.get("include_volatility", False))
    indicator_include = eval_q.get("indicator_include")
    indicator_exclude = eval_q.get("indicator_exclude")
    max_missing_frac = float(eval_cfg.get("max_missing_frac", dataset_cfg.get("max_missing_frac", 0.5)))
    impute_missing = bool(eval_cfg.get("impute_missing", dataset_cfg.get("impute_missing", True)))
    enforce_min_history = bool(eval_cfg.get("enforce_min_history", True))
    include_target_features = bool(
        eval_cfg.get("include_target_features", dataset_cfg.get("include_target_features", False))
    )

    eval_forest_cfg = dict(forest_cfg)
    eval_forest_cfg.update(eval_cfg.get("forest_overrides", {}))
    vote_rule = eval_forest_cfg.get("vote_rule", "any")

    cutoff_records = []

    for cutoff_year in cutoff_years:
        print(f"[Cutoff Eval] {dataset_name} | cutoff={cutoff_year}")
        cutoff_df = work_df[work_df["_year"].notna() & (work_df["_year"] <= cutoff_year)].copy()
        if cutoff_df.empty:
            print("  [Cutoff Eval] No rows available at this cut-off.")
            continue

        t0 = time.perf_counter()
        features_df, question_map, feature_base_map = build_longitudinal_features(
            cutoff_df,
            id_col=id_col,
            date_col=date_col,
            target_col=feature_target_col,
            lags=lags,
            windows=windows,
            label_window=label_window,
            min_history=min_history,
            mode=mode,
            panel_stride=panel_stride,
            include_pct_change=include_pct_change,
            include_volatility=include_volatility,
            indicator_include=indicator_include,
            indicator_exclude=indicator_exclude,
            enforce_min_history=enforce_min_history,
        )
        t1 = time.perf_counter()
        print(
            f"  [Cutoff Eval] Features done in {t1 - t0:.1f}s -> "
            f"{len(features_df)} rows, {len(features_df.columns)} columns"
        )

        # Keep only the supervised growth label as required in this evaluation mode.
        # This avoids dropping rows due to NaNs in auxiliary label_target_* columns.
        label_target_cols = [growth_col]
        features_df = clean_and_impute(
            features_df,
            id_col=id_col,
            date_col=date_col,
            label_target_cols=label_target_cols,
            max_missing_frac=max_missing_frac,
            impute=impute_missing,
        )

        labeled_df = features_df.merge(
            country_growth[[id_col, growth_col, LABEL_COLUMN]],
            on=id_col,
            how="inner",
        )
        labeled_df = labeled_df[labeled_df[id_col].isin(split_pool_ids)].copy()
        if labeled_df.empty:
            print("  [Cutoff Eval] No countries remain after merging growth labels.")
            continue

        if mode == "snapshot" and snapshot_constant_rows:
            present_ids = set(labeled_df[id_col].astype(str).unique().tolist())
            if present_ids != split_pool_ids_as_str:
                print(
                    "  [Cutoff Eval] Snapshot country rows are not constant for this cut-off; "
                    "skipping due to snapshot_constant_rows=true."
                )
                continue

        if labeled_df[LABEL_COLUMN].nunique() < 2:
            print("  [Cutoff Eval] Single label class after merge; skipping cut-off.")
            continue

        exclude_prefixes = ["label_target_"]
        exclude_cols = [c for c, base in feature_base_map.items() if base == feature_target_col]
        if feature_target_col in labeled_df.columns:
            exclude_cols.append(feature_target_col)
        if include_target_features:
            exclude_cols = []

        X, y, feature_names, target_values, row_ids = _prepare_xy_with_target_and_ids(
            labeled_df,
            id_col=id_col,
            target=growth_col,
            exclude_columns=exclude_cols,
            exclude_prefixes=exclude_prefixes,
        )
        if X.size == 0:
            print("  [Cutoff Eval] No numeric training features after filtering.")
            continue

        train_mask = np.isin(row_ids, list(train_ids))
        test_mask = np.isin(row_ids, list(test_ids))
        train_rows = int(train_mask.sum())
        test_rows = int(test_mask.sum())
        train_pos = int(y[train_mask].sum()) if train_rows > 0 else 0
        test_pos = int(y[test_mask].sum()) if test_rows > 0 else 0
        split_ok = (
            train_rows >= min_train_rows
            and test_rows >= min_test_rows
            and train_pos >= min_train_positives
            and test_pos >= min_test_positives
            and (train_rows - train_pos) >= min_train_positives
            and (test_rows - test_pos) >= min_test_positives
        )

        if not split_ok and not strict_country_split:
            idx = np.arange(len(y))
            try:
                train_idx, test_idx = train_test_split(
                    idx,
                    test_size=test_size,
                    random_state=split_seed,
                    stratify=y,
                )
            except ValueError:
                train_idx, test_idx = train_test_split(
                    idx,
                    test_size=test_size,
                    random_state=split_seed,
                )
            train_mask = np.isin(idx, train_idx)
            test_mask = np.isin(idx, test_idx)
            train_rows = int(train_mask.sum())
            test_rows = int(test_mask.sum())
            train_pos = int(y[train_mask].sum()) if train_rows > 0 else 0
            test_pos = int(y[test_mask].sum()) if test_rows > 0 else 0
            split_ok = (
                train_rows >= min_train_rows
                and test_rows >= min_test_rows
                and train_pos >= min_train_positives
                and test_pos >= min_test_positives
                and (train_rows - train_pos) >= min_train_positives
                and (test_rows - test_pos) >= min_test_positives
            )

        if not split_ok:
            print("  [Cutoff Eval] Could not create a valid train/test split; skipping cut-off.")
            continue

        X_train, y_train = X[train_mask], y[train_mask]
        X_test, y_test = X[test_mask], y[test_mask]
        target_train = target_values[train_mask]
        train_coverage = train_rows / max(1, len(train_ids))
        test_coverage = test_rows / max(1, len(test_ids))

        forest_result = run_forest_search(
            X_train,
            y_train,
            n_seeds=eval_forest_cfg.get("n_seeds", 1000),
            accuracy_quantile=eval_forest_cfg.get("accuracy_quantile", 0.99),
            min_trees=eval_forest_cfg.get("min_trees", 1),
            max_trees=eval_forest_cfg.get("max_trees", 12),
            min_questions=eval_forest_cfg.get("min_questions", 2),
            max_questions=eval_forest_cfg.get("max_questions", 12),
            test_size=eval_forest_cfg.get("test_size", 0.2),
            vote_rule=vote_rule,
            eval_mode="train",
            n_jobs=eval_forest_cfg.get("n_jobs", -1),
        )

        trees, _ = rebuild_forest(
            X_train,
            y_train,
            forest_result.best_seed,
            eval_forest_cfg.get("min_trees", 1),
            eval_forest_cfg.get("max_trees", 12),
            eval_forest_cfg.get("min_questions", 2),
            eval_forest_cfg.get("max_questions", 12),
        )

        train_pred = forest_predict(trees, X_train, vote_rule=vote_rule)
        test_pred = forest_predict(trees, X_test, vote_rule=vote_rule)
        train_accuracy = float((train_pred == y_train).mean())
        test_accuracy = float((test_pred == y_test).mean())
        baseline_test_accuracy = float(max(y_test.mean(), 1.0 - y_test.mean()))

        out_dir = ensure_dir(eval_dir / f"cutoff_{cutoff_year}")
        labeled_df.to_csv(out_dir / "labeled.csv", index=False)
        forest_result.results_df.to_csv(out_dir / "forest_results.csv", index=False)
        save_json(out_dir / "best_forest.json", forest_result.best_row.to_dict())

        if question_map:
            q_df = pd.DataFrame(
                {
                    "feature": list(question_map.keys()),
                    "question": list(question_map.values()),
                    "base_indicator": [feature_base_map.get(f, "") for f in question_map.keys()],
                }
            )
            q_df.to_csv(out_dir / "question_map.csv", index=False)

        summary = {
            "cutoff_year": int(cutoff_year),
            "rows": int(len(labeled_df)),
            "countries": int(pd.Series(labeled_df[id_col]).nunique()),
            "interesting": int(labeled_df[LABEL_COLUMN].sum()),
            "non_interesting": int((~labeled_df[LABEL_COLUMN]).sum()),
            "label_countries_total": int(len(label_ids)),
            "split_train_countries_total": int(len(train_ids)),
            "split_test_countries_total": int(len(test_ids)),
            "train_rows": train_rows,
            "test_rows": test_rows,
            "train_coverage": float(train_coverage),
            "test_coverage": float(test_coverage),
            "train_positive": train_pos,
            "test_positive": test_pos,
            "train_negative": int(train_rows - train_pos),
            "test_negative": int(test_rows - test_pos),
            "train_accuracy": train_accuracy,
            "test_accuracy": test_accuracy,
            "baseline_test_accuracy": baseline_test_accuracy,
            "label_threshold": threshold,
            "strict_country_split": strict_country_split,
        }
        save_json(out_dir / "summary.json", summary)

        save_kde_plot(
            labeled_df,
            growth_col,
            LABEL_COLUMN,
            out_dir / "kde_plot",
            title=f"{dataset_name} cutoff {cutoff_year}: GDP per capita growth ({start_year}-{end_year})",
        )
        save_forest_accuracy_plot(
            forest_result.results_df,
            out_dir / "accuracy_vs_questions",
            best_row=forest_result.best_row,
        )
        save_forest_tree_artifacts(
            trees,
            X_train,
            y_train,
            feature_names,
            out_dir,
            question_map=question_map,
            target_values=target_train,
            target_name=growth_col,
        )

        cutoff_records.append(summary)
        print(
            f"  [Cutoff Eval] available train/test={train_rows}/{test_rows} "
            f"(coverage={train_coverage:.3f}/{test_coverage:.3f})"
        )
        print(
            f"  [Cutoff Eval] train_acc={train_accuracy:.4f} | test_acc={test_accuracy:.4f} "
            f"| baseline={baseline_test_accuracy:.4f}"
        )

    if not cutoff_records:
        print(f"[Cutoff Eval] No successful cut-off runs for {dataset_name}")
        return

    results_df = pd.DataFrame(cutoff_records).sort_values("cutoff_year")
    results_df.to_csv(eval_dir / "cutoff_results.csv", index=False)
    save_json(eval_dir / "cutoff_results.json", {"results": results_df.to_dict(orient="records")})
    save_cutoff_accuracy_plot(
        results_df,
        eval_dir / "accuracy_vs_cutoff",
        title=f"{dataset_name}: test accuracy by cut-off year",
    )


def run_longitudinal_pipeline(config_path: str, output_dir: str) -> None:
    cfg = load_json(config_path)
    output_base = Path(output_dir)

    pipeline_cfg: Dict = cfg.get("pipelines", {})
    forest_cfg: Dict = cfg.get("forest", {})

    for dataset_cfg in cfg.get("datasets", []):
        if not dataset_cfg.get("enabled", True):
            continue

        df = pd.read_csv(dataset_cfg["path"])
        print(f"[Load] {dataset_cfg['name']}: {len(df)} rows, {len(df.columns)} columns")

        if dataset_cfg.get("run_standard_pipelines", True):
            _run_standard_longitudinal_for_dataset(
                df=df,
                dataset_cfg=dataset_cfg,
                pipeline_cfg=pipeline_cfg,
                forest_cfg=forest_cfg,
                output_base=output_base,
            )

        if dataset_cfg.get("cutoff_growth_eval", {}).get("enabled", False):
            _run_cutoff_growth_eval_for_dataset(
                df=df,
                dataset_cfg=dataset_cfg,
                forest_cfg=forest_cfg,
                output_base=output_base,
            )

    print(f"Longitudinal pipeline complete. Outputs in {output_base}")
