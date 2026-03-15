import time
from pathlib import Path
from typing import Dict, Iterable, List, Set, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from .common import LABEL_COLUMN, drop_by_prefix, ensure_dir, load_json, save_json
from .forest import forest_predict, rebuild_forest, run_forest_search
from .longitudinal_features import build_longitudinal_features
from .plotting import (
    save_cutoff_accuracy_plot,
    save_forest_accuracy_plot,
    save_forest_tree_artifacts,
    save_kde_plot,
)


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

    # Keep NaNs in numeric features for train-only preprocessing later.
    valid_mask = ~(y.isna() | target_series.isna() | row_ids.isna())
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


def _preprocess_xy_train_only(
    X: np.ndarray,
    feature_names: List[str],
    train_mask: np.ndarray,
    max_missing_frac: float = 0.95,
    impute_missing: bool = True,
) -> Tuple[np.ndarray, List[str]]:
    if X.size == 0:
        return X, feature_names

    X_proc = np.asarray(X, dtype=float).copy()
    X_train = X_proc[train_mask]
    if X_train.size == 0:
        return np.array([]), []

    missing_frac_train = np.mean(np.isnan(X_train), axis=0)
    keep_mask = missing_frac_train <= max_missing_frac
    if not np.any(keep_mask):
        return np.array([]), []

    X_proc = X_proc[:, keep_mask]
    kept_feature_names = [f for f, k in zip(feature_names, keep_mask) if k]

    if impute_missing:
        train_medians = np.nanmedian(X_proc[train_mask], axis=0)
        train_medians = np.where(np.isnan(train_medians), 0.0, train_medians)
        nan_mask = np.isnan(X_proc)
        if nan_mask.any():
            rows, cols = np.where(nan_mask)
            X_proc[rows, cols] = train_medians[cols]
    else:
        # If not imputing, remove any remaining NaN-containing columns (still train-only decision).
        finite_cols = np.isfinite(X_proc[train_mask]).all(axis=0)
        if not np.any(finite_cols):
            return np.array([]), []
        X_proc = X_proc[:, finite_cols]
        kept_feature_names = [f for f, k in zip(kept_feature_names, finite_cols) if k]

    return X_proc, kept_feature_names


def _clean_and_impute(
    df: pd.DataFrame,
    id_col: str,
    date_col: str,
    protected_cols: List[str],
    max_missing_frac: float = 0.95,
    impute: bool = True,
) -> pd.DataFrame:
    keep_cols = set([id_col, date_col] + protected_cols)
    missing = df.isna().mean()
    drop_cols = [c for c, frac in missing.items() if frac > max_missing_frac and c not in keep_cols]
    if drop_cols:
        df = df.drop(columns=drop_cols)

    if impute:
        numeric = df.select_dtypes(include=[np.number]).columns
        cols = [c for c in numeric if c not in keep_cols]
        if cols:
            df[cols] = df[cols].fillna(df[cols].median()).fillna(0)

    return df.dropna(subset=[id_col, date_col])


def _build_growth_labels(
    work_df: pd.DataFrame,
    id_col: str,
    target_col: str,
    start_year: int,
    end_year: int,
    interesting_quantile: float,
    min_growth: float = None,
) -> Tuple[pd.DataFrame, str, float]:
    growth_col = f"gdp_per_capita_growth_{start_year}_{end_year}"
    start_col = f"gdp_per_capita_{start_year}"
    end_col = f"gdp_per_capita_{end_year}"

    growth_df = (
        work_df[[id_col, "_year", target_col]]
        .dropna(subset=[id_col, "_year", target_col])
        .query("_year == @start_year or _year == @end_year")
    )
    if growth_df.empty:
        raise ValueError("No data for growth label years.")

    pivot = growth_df.pivot_table(index=id_col, columns="_year", values=target_col, aggfunc="last")
    if start_year not in pivot.columns or end_year not in pivot.columns:
        raise ValueError("Insufficient coverage for growth label years.")

    out = (
        pivot[[start_year, end_year]]
        .rename(columns={start_year: start_col, end_year: end_col})
        .reset_index()
        .dropna()
    )
    out[growth_col] = out[end_col] - out[start_col]
    threshold = float(out[growth_col].quantile(interesting_quantile))
    if min_growth is not None:
        threshold = max(threshold, float(min_growth))
    out[LABEL_COLUMN] = out[growth_col] >= threshold
    out = out.sort_values(id_col).reset_index(drop=True)
    return out, growth_col, threshold


def run_cutoff_snapshot_eval(config_path: str, output_dir: str) -> None:
    cfg = load_json(config_path)
    output_base = Path(output_dir)

    forest_defaults: Dict = cfg.get("forest", {})

    for dataset_cfg in cfg.get("datasets", []):
        if not dataset_cfg.get("enabled", True):
            continue

        dataset_name = dataset_cfg["name"]
        id_col = dataset_cfg.get("id_col", "country")
        date_col = dataset_cfg.get("date_col", "date")
        gdp_col = dataset_cfg.get("gdp_col", "GDP_current_US")
        population_col = dataset_cfg.get("population_col", "population")
        feature_target_col = dataset_cfg.get("feature_target_col", "gdp_per_capita")

        start_year = int(dataset_cfg.get("growth_start_year", 2010))
        end_year = int(dataset_cfg.get("growth_end_year", 2020))
        interesting_quantile = float(dataset_cfg.get("interesting_quantile", 0.9))
        min_growth = dataset_cfg.get("min_growth")
        cutoff_years = sorted({int(y) for y in dataset_cfg.get("cutoff_years", [1980, 1990, 2000, 2010])})
        eval_name = dataset_cfg.get("eval_name", f"gdp_growth_{start_year}_{end_year}_snapshot")

        split_seed = int(dataset_cfg.get("split_seed", 42))
        test_size = float(dataset_cfg.get("test_size", 0.25))
        split_country_pool = str(dataset_cfg.get("split_country_pool", "available_all_cutoffs")).strip().lower()
        if split_country_pool not in {"all_labeled", "available_all_cutoffs"}:
            raise ValueError("split_country_pool must be 'all_labeled' or 'available_all_cutoffs'")

        snapshot_constant_rows = bool(dataset_cfg.get("snapshot_constant_rows", True))
        max_missing_frac = float(dataset_cfg.get("max_missing_frac", 0.95))
        impute_missing = bool(dataset_cfg.get("impute_missing", True))
        enforce_min_history = bool(dataset_cfg.get("enforce_min_history", False))
        include_target_features = bool(dataset_cfg.get("include_target_features", False))
        label_window = int(dataset_cfg.get("label_window", 5))
        min_history = int(dataset_cfg.get("min_history", 0))

        q_cfg = dataset_cfg.get("question_pack", {})
        lags_all = sorted({int(x) for x in q_cfg.get("lags", [5, 7, 10, 15, 20, 30, 40])})
        windows_all = sorted({int(x) for x in q_cfg.get("windows", [3, 5, 7, 10, 15, 20, 30, 40])})
        include_pct_change = bool(q_cfg.get("include_pct_change", True))
        include_volatility = bool(q_cfg.get("include_volatility", True))
        indicator_include = q_cfg.get("indicator_include")
        indicator_exclude = q_cfg.get("indicator_exclude")
        expand_features_with_cutoff = bool(dataset_cfg.get("expand_features_with_cutoff", True))

        print(f"[Load] {dataset_name}")
        df = pd.read_csv(dataset_cfg["path"])
        print(f"[Load] rows={len(df)} cols={len(df.columns)}")

        required = [id_col, date_col, gdp_col, population_col]
        missing = [c for c in required if c not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns for {dataset_name}: {missing}")

        work_df = df.copy()
        work_df["_year"] = _extract_year(work_df[date_col])
        work_df[gdp_col] = pd.to_numeric(work_df[gdp_col], errors="coerce")
        work_df[population_col] = pd.to_numeric(work_df[population_col], errors="coerce")
        work_df[feature_target_col] = work_df[gdp_col] / (work_df[population_col] + 1e-9)
        work_df.loc[~np.isfinite(work_df[feature_target_col]), feature_target_col] = np.nan

        country_growth, growth_col, threshold = _build_growth_labels(
            work_df=work_df,
            id_col=id_col,
            target_col=feature_target_col,
            start_year=start_year,
            end_year=end_year,
            interesting_quantile=interesting_quantile,
            min_growth=min_growth,
        )
        if country_growth[LABEL_COLUMN].nunique() < 2:
            raise ValueError(f"{dataset_name}: growth labels have only one class.")

        split_pool_df = country_growth.copy()
        if split_country_pool == "available_all_cutoffs":
            valid_df = work_df[work_df[id_col].notna() & work_df["_year"].notna() & work_df[feature_target_col].notna()]
            available_sets: List[Set[str]] = []
            for cutoff_year in cutoff_years:
                ids = set(valid_df.loc[valid_df["_year"] <= cutoff_year, id_col].astype(str).unique().tolist())
                available_sets.append(ids)
            eligible_ids = set.intersection(*available_sets) if available_sets else set()
            split_pool_df = split_pool_df[split_pool_df[id_col].astype(str).isin(eligible_ids)].copy()

        if split_pool_df.empty:
            raise ValueError(f"{dataset_name}: split pool is empty.")
        if split_pool_df[LABEL_COLUMN].nunique() < 2:
            raise ValueError(f"{dataset_name}: split pool has one class.")

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
        split_pool_ids = set(split_pool_df[id_col].tolist())
        split_pool_ids_as_str = set(split_pool_df[id_col].astype(str).tolist())

        eval_dir = ensure_dir(output_base / f"{dataset_name}_{eval_name}")
        country_growth.to_csv(eval_dir / "country_growth_labels.csv", index=False)
        split_pool_df.to_csv(eval_dir / "split_pool_countries.csv", index=False)

        split_assign = split_pool_df[[id_col, LABEL_COLUMN]].copy()
        split_assign["split"] = np.where(split_assign[id_col].isin(train_ids), "train", "test")
        split_assign = split_assign.sort_values(by=["split", id_col]).reset_index(drop=True)
        split_assign.to_csv(eval_dir / "country_split.csv", index=False)

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
                "split_seed": split_seed,
                "test_size": test_size,
                "snapshot_constant_rows": snapshot_constant_rows,
                "cutoff_years": cutoff_years,
            },
        )

        forest_cfg = dict(forest_defaults)
        forest_cfg.update(dataset_cfg.get("forest_overrides", {}))
        vote_rule = forest_cfg.get("vote_rule", "any")
        eval_mode = str(forest_cfg.get("eval_mode", "holdout"))
        max_depth = forest_cfg.get("max_depth")
        if max_depth is not None:
            max_depth = int(max_depth)
        min_year = int(work_df["_year"].dropna().min())

        print(
            "[Cutoff Snapshot] Fixed split | "
            f"train={len(train_ids)} test={len(test_ids)} pool={len(split_pool_ids)}"
        )

        cutoff_records = []
        for cutoff_year in cutoff_years:
            print(f"[Cutoff Snapshot] {dataset_name} | cutoff={cutoff_year}")
            cutoff_df = work_df[
                work_df["_year"].notna()
                & (work_df["_year"] <= cutoff_year)
                & (work_df[id_col].isin(split_pool_ids))
            ].copy()
            if cutoff_df.empty:
                print("  [Cutoff Snapshot] No rows at this cut-off.")
                continue

            if expand_features_with_cutoff:
                span = max(1, int(cutoff_year - min_year))
                active_lags = [x for x in lags_all if x <= span] or [min(lags_all)]
                active_windows = [x for x in windows_all if x <= span] or [min(windows_all)]
            else:
                active_lags = list(lags_all)
                active_windows = list(windows_all)

            t0 = time.perf_counter()
            features_df, question_map, feature_base_map = build_longitudinal_features(
                cutoff_df,
                id_col=id_col,
                date_col=date_col,
                target_col=feature_target_col,
                lags=active_lags,
                windows=active_windows,
                label_window=label_window,
                min_history=min_history,
                mode="snapshot",
                panel_stride=1,
                include_pct_change=include_pct_change,
                include_volatility=include_volatility,
                indicator_include=indicator_include,
                indicator_exclude=indicator_exclude,
                enforce_min_history=enforce_min_history,
            )
            t1 = time.perf_counter()
            print(
                f"  [Cutoff Snapshot] Features in {t1 - t0:.1f}s -> "
                f"{len(features_df)} rows, {len(features_df.columns)} cols"
            )

            labeled_df = features_df.merge(
                split_pool_df[[id_col, growth_col, LABEL_COLUMN]],
                on=id_col,
                how="inner",
            )
            labeled_df = labeled_df[labeled_df[id_col].isin(split_pool_ids)].copy()

            if labeled_df.empty:
                print("  [Cutoff Snapshot] No rows after merge.")
                continue

            if snapshot_constant_rows:
                present_ids = set(labeled_df[id_col].astype(str).unique().tolist())
                if present_ids != split_pool_ids_as_str:
                    print("  [Cutoff Snapshot] Missing countries for this cut-off; skipping.")
                    continue

            if labeled_df[LABEL_COLUMN].nunique() < 2:
                print("  [Cutoff Snapshot] Single class labels after merge; skipping.")
                continue

            # Guardrail: ensure no post-cutoff timestamps are present.
            max_obs_year = int(_extract_year(labeled_df[date_col]).dropna().max())
            if max_obs_year > cutoff_year:
                raise RuntimeError(
                    f"Cut-off leakage detected: found year {max_obs_year} for cutoff {cutoff_year}."
                )

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
                print("  [Cutoff Snapshot] No numeric features.")
                continue

            train_mask = np.isin(row_ids, list(train_ids))
            test_mask = np.isin(row_ids, list(test_ids))
            if train_mask.sum() == 0 or test_mask.sum() == 0:
                print("  [Cutoff Snapshot] Empty train/test after fixed split.")
                continue

            X, feature_names = _preprocess_xy_train_only(
                X,
                feature_names,
                train_mask=train_mask,
                max_missing_frac=max_missing_frac,
                impute_missing=impute_missing,
            )
            if X.size == 0:
                print("  [Cutoff Snapshot] No usable features after train-only preprocessing.")
                continue

            X_train, y_train = X[train_mask], y[train_mask]
            X_test, y_test = X[test_mask], y[test_mask]
            target_train = target_values[train_mask]

            if len(np.unique(y_train)) < 2 or len(np.unique(y_test)) < 2:
                print("  [Cutoff Snapshot] Train/test class collapse; skipping.")
                continue

            forest_result = run_forest_search(
                X_train,
                y_train,
                n_seeds=forest_cfg.get("n_seeds", 500),
                accuracy_quantile=forest_cfg.get("accuracy_quantile", 0.99),
                min_trees=forest_cfg.get("min_trees", 1),
                max_trees=forest_cfg.get("max_trees", 10),
                min_questions=forest_cfg.get("min_questions", 2),
                max_questions=forest_cfg.get("max_questions", 10),
                test_size=forest_cfg.get("test_size", 0.2),
                vote_rule=vote_rule,
                eval_mode=eval_mode,
                n_jobs=forest_cfg.get("n_jobs", 1),
                max_depth=max_depth,
            )

            trees, _ = rebuild_forest(
                X_train,
                y_train,
                forest_result.best_seed,
                forest_cfg.get("min_trees", 1),
                forest_cfg.get("max_trees", 10),
                forest_cfg.get("min_questions", 2),
                forest_cfg.get("max_questions", 10),
                max_depth=max_depth,
            )

            train_pred = forest_predict(trees, X_train, vote_rule=vote_rule)
            test_pred = forest_predict(trees, X_test, vote_rule=vote_rule)
            train_acc = float((train_pred == y_train).mean())
            test_acc = float((test_pred == y_test).mean())
            baseline_test_acc = float(max(y_test.mean(), 1.0 - y_test.mean()))

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
                "split_train_countries_total": int(len(train_ids)),
                "split_test_countries_total": int(len(test_ids)),
                "train_rows": int(train_mask.sum()),
                "test_rows": int(test_mask.sum()),
                "train_accuracy": train_acc,
                "test_accuracy": test_acc,
                "baseline_test_accuracy": baseline_test_acc,
                "label_threshold": threshold,
                "n_features": int(X.shape[1]),
                "active_lags": active_lags,
                "active_windows": active_windows,
                "forest_eval_mode": eval_mode,
                "forest_max_depth": max_depth,
            }
            save_json(out_dir / "summary.json", summary)

            save_kde_plot(
                labeled_df,
                growth_col,
                LABEL_COLUMN,
                out_dir / "kde_plot",
                title=f"{dataset_name} snapshot cutoff {cutoff_year}: growth {start_year}-{end_year}",
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
                f"  [Cutoff Snapshot] train/test rows={int(train_mask.sum())}/{int(test_mask.sum())} "
                f"| train_acc={train_acc:.4f} test_acc={test_acc:.4f}"
            )

        if not cutoff_records:
            print(f"[Cutoff Snapshot] No successful cut-off runs for {dataset_name}")
            continue

        results_df = pd.DataFrame(cutoff_records).sort_values("cutoff_year")
        results_df.to_csv(eval_dir / "cutoff_results.csv", index=False)
        save_json(eval_dir / "cutoff_results.json", {"results": results_df.to_dict(orient="records")})
        save_cutoff_accuracy_plot(
            results_df,
            eval_dir / "accuracy_vs_cutoff",
            title=f"{dataset_name}: snapshot cutoff test accuracy",
        )

    print(f"Cutoff snapshot evaluation complete. Outputs in {output_base}")
