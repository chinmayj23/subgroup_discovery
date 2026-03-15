# Interpretable_SD - Detailed Implementation Notes

## 1. Goal

Build a generic pipeline for interpretable subgroup discovery in longitudinal data with:

1. a modular interestingness module,
2. a modular rule-learning module,
3. comparable baselines,
4. human-readable outputs.

The current experiments use World Bank indicators, but the implementation is meant to be reusable for any dataset with:

1. a subject identifier,
2. a time variable,
3. one target variable,
4. additional numeric covariates.

## 2. Core Design Decision

The project now follows this explicit order:

1. compute interestingness on the raw longitudinal target values,
2. convert row-level labels into contiguous same-label subject segments,
3. engineer temporal covariates,
4. learn interpretable rules on segment rows.

This is different from the earlier fixed-window approach. The sample unit is no longer predetermined. Instead, the sample unit is defined **after** the interestingness step by the observed stretches of interestingness and non-interestingness within each subject.

## 3. Main Pipeline

File:

`Interpretable_SD/src/interpretable_sd/pipeline.py`

Per experiment, the pipeline now does:

1. load the dataset and apply the target transform,
2. drop rows with missing subject, date, or target,
3. build the panel feature table,
4. run each interestingness method on raw rows,
5. post-process the raw row labels into contiguous segments,
6. build train/test matrices from segment rows,
7. run the RF rule module,
8. run the comparable baselines,
9. write reports and comparison tables.

The important conceptual split is:

1. **interestingness** is row-level and target-only,
2. **rule learning** is segment-level and covariate-based.

## 4. Raw-row Interestingness

Files:

1. `Interpretable_SD/src/interpretable_sd/interestingness/tail_cluster.py`
2. `Interpretable_SD/src/interpretable_sd/interestingness/timetribes_windowed.py`

Although the module name still contains `windowed`, the current pipeline uses `TimeTribes` on raw subject-time rows.

### 4.1 TailCluster

TailCluster receives the raw target column and labels rows directly from the target distribution.

It is still target-only and does not use covariate features.

### 4.2 TimeTribes

TimeTribes also receives the raw target column and labels rows directly from that target distribution.

It uses:

1. bootstrap candidate generation,
2. divergence-based scoring,
3. overlap control,
4. subgroup size constraints.

Again, it does not use covariate feature engineering for the interestingness decision.

## 5. Segment Construction

File:

`Interpretable_SD/src/interpretable_sd/preprocessing.py`

Main function:

`build_interest_segments(...)`

### 5.1 What it does

For each subject:

1. sort rows by time,
2. take the boolean interestingness labels from the raw interestingness module,
3. optionally suppress very short positive runs with `min_interesting_run_length`,
4. assign a segment id every time the label changes,
5. aggregate each contiguous run into one segment row.

### 5.2 Segment metadata

Each segment row stores:

1. `segment_id`
2. `segment_start_date`
3. `segment_end_date`
4. `segment_start_year`
5. `segment_end_year`
6. `segment_length`
7. `segment_target_mean`
8. `segment_target_delta`
9. `segment_target_trend`
10. `segment_target_std`
11. `segment_target_score`

### 5.3 Why this matters

This gives the rule learner a much more meaningful sample unit:

1. a subject with no interesting years becomes one non-interesting segment,
2. a subject with one interesting block becomes three segments at most: before, during, after,
3. a subject with multiple interesting blocks yields multiple interesting and non-interesting segments.

This structure is much closer to the intended longitudinal subgroup-discovery problem than fixed windows chosen before labeling.

## 6. Temporal Feature Processing

File:

`Interpretable_SD/src/interpretable_sd/preprocessing.py`

Supporting function:

`build_panel_feature_table(...)`

This uses `build_longitudinal_features(...)` to produce row-aligned temporal features:

1. `latest_*`
2. `lag*k_*`
3. `delta*k_*`
4. `pct_change*k_*`
5. `trend*k_*`
6. `vol*k_*`

These features are built in panel mode, so they remain aligned to individual subject-time rows.

After that, the row labels from the interestingness module are merged back onto the feature table and collapsed into segment rows.

Implementation detail:

1. the segment row keeps the feature values from the final row of that contiguous segment,
2. the segment metadata separately records the span start, span end, and segment-level target summaries.

This is a pragmatic compromise:

1. the covariate features still summarize recent longitudinal history,
2. the sample unit is now a segment rather than an arbitrary fixed window.

## 7. Train/Test Split

File:

`Interpretable_SD/src/interpretable_sd/preprocessing.py`

Function:

`split_train_test_by_subject(...)`

Important behavior:

1. split is always done by subject,
2. all segments from the same subject remain in the same split,
3. the function retries several seeds to preserve both classes in train and test when possible.

This prevents leakage from having some segments of a subject in train and others in test.

## 8. Rule-learning Matrix Construction

File:

`Interpretable_SD/src/interpretable_sd/pipeline.py`

Function:

`_prepare_rule_matrices(...)`

This stage:

1. removes metadata columns from prediction,
2. removes target-derived segment fields from prediction,
3. keeps only numeric covariates,
4. preprocesses missingness using train-only information,
5. carries segment span metadata forward for reporting.

The following are explicitly excluded from the predictive feature set:

1. identifiers,
2. dates,
3. `segment_*` metadata,
4. `label_target_*` helpers,
5. `window_*` legacy metadata.

This is important because segment start/end years are used only for interpretation, not as predictive covariates.

## 9. RF Rule Module

File:

`Interpretable_SD/src/interpretable_sd/rules/rf_module.py`

The RF module:

1. searches many small forests,
2. keeps compact forests near the top of the accuracy distribution,
3. rebuilds the chosen forest on the full training split,
4. evaluates on held-out subjects,
5. exports tree and rule artifacts.

### 9.1 Output artifacts

1. `forest_search_results.csv`
2. `best_forest.json`
3. `trees/*.json`
4. `trees/*.png`
5. `rules.json`
6. `rules_natural_language.txt`

### 9.2 New segment-aware behavior

Leaf summaries now include:

1. mean target value,
2. typical span start year,
3. typical span end year,
4. average span length.

So positive rule text can say not only what covariate pattern is interesting, but also when that interesting span tends to occur.

## 10. Baselines

### 10.1 Sysurv-style baseline

File:

`Interpretable_SD/src/interpretable_sd/baselines/sysurv_style.py`

This baseline was adapted for compatibility:

1. raw survival framing was replaced by fixed-label interesting/not-interesting prediction,
2. it uses the same segment-level training matrix,
3. it produces the same kinds of outputs as the RF module,
4. it includes single-class and threshold-collapse fallbacks.

It still follows a two-stage pattern:

1. nonlinear scorer,
2. shallow naming tree.

### 10.2 Tree-paper-style baseline

File:

`Interpretable_SD/src/interpretable_sd/baselines/tree_paper_style.py`

This baseline:

1. fits a single decision tree,
2. selects the pruning level by validation accuracy and compactness,
3. evaluates on held-out test subjects,
4. exports the same simplified tree and rule artifacts.

It also receives the same segment metadata for positive-leaf reporting.

## 11. Reporting

Files:

1. `Interpretable_SD/src/interpretable_sd/reporting.py`
2. `Interpretable_SD/src/interpretable_sd/comparison.py`

### 11.1 Per-method outputs

Each interestingness-method folder now contains:

1. `raw_rows_labeled.csv`
2. `row_labels.csv`
3. `segment_samples.csv`
4. `interestingness_metrics_raw.json`
5. `interestingness_metrics.json`
6. `interestingness_kde.*`
7. `rule_module_metrics.json`
8. `baseline_metrics.json`
9. `syflow_like_report.json`

### 11.2 Comparison table

The comparison generator now combines:

1. raw-row interestingness metrics,
2. segment counts,
3. train/test rule-learning metrics.

It exports:

1. full CSV,
2. markdown table,
3. leaderboard CSV.

## 12. Config Structure

Main config files:

1. `Interpretable_SD/configs/world_bank_fast_debug.json`
2. `Interpretable_SD/configs/world_bank_full_paper_grade.json`
3. `Interpretable_SD/configs/world_bank_all_in_one.json`

Important blocks:

1. `data`
2. `defaults.feature_engineering`
3. `defaults.segmentation`
4. `defaults.split`
5. `defaults.rf_rule_module`
6. `defaults.baselines`
7. `experiments`

`defaults.segmentation` is now the key block for the segment-first flow:

1. `target_aggregation`
2. `min_interesting_run_length`

`target_aggregation` determines the segment-level target summary used in reporting and leaf summaries.

`min_interesting_run_length` allows slight suppression of isolated positive spikes before segment rows are formed.

## 13. Run Commands

Full paper-grade:

```bash
python Interpretable_SD/scripts/run_interpretable_sd.py --config Interpretable_SD/configs/world_bank_full_paper_grade.json --output-dir Interpretable_SD/outputs/world_bank_full_paper_grade
```

Fast debug:

```bash
python Interpretable_SD/scripts/run_interpretable_sd.py --config Interpretable_SD/configs/world_bank_fast_debug.json --output-dir Interpretable_SD/outputs/world_bank_fast_debug
```

Comparison table only:

```bash
python Interpretable_SD/scripts/generate_comparison_table.py --output-dir Interpretable_SD/outputs/world_bank_full_paper_grade --table-name comparison_full_paper_grade
```

## 14. Notes On Reusability

To adapt the pipeline to another dataset such as CMIE:

1. change `data.path`, `id_col`, and `date_col`,
2. change the experiment target transform,
3. adjust lags/windows for the new temporal granularity,
4. leave the interestingness, segmentation, RF, and baseline modules unchanged.

The code is structured so that the dataset-specific choices live mainly in config, while the subgroup-discovery logic remains generic.
