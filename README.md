# Subgroup Discovery Pipelines

This repo implements an independent, plug-and-play pipeline for interpretable subgroup discovery with a primary focus on longitudinal data. The goal is to label "interesting" subgroups on a target column and then train a compact decision-forest explanation.

## What the methods do

1) Original Method (clustering + high-value override)
- Uses single-linkage hierarchical clustering on the target column only.
- Marks the largest cluster as "normal" and the rest as "interesting".
- Forces all target values above a specified quantile (default 0.94) to be interesting.
- Output: a boolean `is_interesting_subgroup` label for each row.

2) KL-Based Method (bootstrap + cluster ranges + KL + diversity)
- Bootstraps the target values, clusters each bootstrap sample, and turns each cluster into a target range.
- Applies those ranges to the full dataset to create candidate subgroup masks.
- Scores candidates by size-adjusted KL divergence vs the population and selects a diverse set via a greedy objective.
- This objective is inspired by Syflow (KL-regularized subgroup discovery).
- Output: a boolean `is_interesting_subgroup` label for each row.

3) Forest Explanation (shared)
- Trains many small constrained decision trees (randomized per seed).
- Selects the best forest based on high accuracy and minimal questions.
- Exports all trees in the best forest as JSON + PNG for interpretation.

## Datasets used (static support is optional)

Configured in `configs/subgroup_pipelines.json`.

Enabled by default:
- california_housing
  - Source: local CSV `california_housing.csv` (California Housing from sklearn).
  - Target: `MedHouseVal`
  - Expected subgroups: very high values in coastal/urban areas; very low values in rural areas.
- insurance
  - Source: `data/insurance/insurance.csv` (Kaggle "Medical Cost Personal Datasets").
  - Target: `charges`
  - Expected subgroups: high charges for smokers; higher charges for older age and higher BMI.
- diabetes
  - Source: `outputs_subgroups/diabetes_with_subgroups.csv` (local derived file).
  - Target: `target`
  - Expected subgroups: high target values aligned with BMI and serum markers.
- heart
  - Source: `syflow_env/Lib/site-packages/statsmodels/datasets/heart/heart.csv` (statsmodels).
  - Target: `survival`
  - Expected subgroups: low survival at older ages.
- engel
  - Source: `syflow_env/Lib/site-packages/statsmodels/datasets/engel/engel.csv` (statsmodels).
  - Target: `foodexp`
  - Expected subgroups: unusually high or low food expenditure relative to income.

Optional datasets (disabled by default in config):
- sklearn_diabetes (sklearn package)
- sklearn_wine (sklearn package)
- sklearn_california_housing (sklearn package)
- statsmodels_longley (statsmodels package)
- openml_abalone (OpenML; requires network)

## Longitudinal pipeline (panel data)

Configured in `configs/wbd_longitudinal.json` and run via `scripts/run_wbd_longitudinal.py`.
It accepts any CSV with an entity column, a date column, and a target column.

Temporal feature engineering (per entity/time):
- Lags: 1 and 3
- Rolling mean and standard deviation (window size configurable)
- Local trend slope over the window
- Deltas: 1-step and 3-step
- Recovery: current value minus window minimum
- CAGR over the window (positive-valued series only)

Target options:
- `target_mode = "value"` uses the raw target value.
- `target_mode = "trend"` uses the recent trend slope as the target (extreme trends).
- `target_mode = "recovery"` uses recovery over the window (current minus window minimum).

This project emphasizes interpretable subgroup discovery for longitudinal data: target-only subgroup labels are explained via compact decision-forest rules over temporal features (lags, rolling stats, trend, deltas, recovery, CAGR).

## Running longitudinal

Run with the config:
```bash
python scripts/run_wbd_longitudinal.py
```

To use another CSV, update:
`path`, `dataset_name`, `id_col`, `date_col`, and `target` in `configs/wbd_longitudinal.json`.
You can also set `impute_missing` to keep more rows and `max_missing_frac` to control column filtering.

## Run static + longitudinal together

Use `configs/run_all.json` to run multiple static and longitudinal datasets in one run:
```bash
python scripts/run_all_pipelines.py
```
Set `static_datasets` or `longitudinal_datasets` to `null` to skip that section.

## Outputs

For each dataset and pipeline, results are written to:
`outputs_subgroups/pipeline_runs/<dataset>/<pipeline>/`

Typical files:
- `labeled.csv` with `is_interesting_subgroup`
- `forest_results.csv` and `best_forest.json`
- `kde_plot.png`
- `tree_XX_acc_*.json` and `tree_XX_acc_*.png`

## Running (recommended)

Use the unified runner to execute longitudinal and (optionally) static datasets in one run:
```bash
python scripts/run_all_pipelines.py
```
Configure datasets in `configs/run_all.json`. Set `static_datasets` to `null` if you only want longitudinal data.

## Static-only runner (optional)

If you only need static datasets, you can still use:
```bash
python scripts/run_subgroup_pipelines.py
```
