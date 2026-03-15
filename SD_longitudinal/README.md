# SD_longitudinal

A clean, minimal codebase for **subgroup discovery on static and longitudinal data**. Subgroup discovery is kept fixed (target‑only candidate generation). Interpretability comes from **asking the right temporal questions** via longitudinal preprocessing and then fitting a compact question forest.

**Overview**

1. **Module 1: Subgroup discovery (target‑only).**
`tail_cluster`: single‑linkage clustering on the target with tail detection.  
`TimeTribes`: bootstrapped clustering with a KL‑divergence + diversity objective inspired by **SyFlow**.

2. **Module 2: Interpretable rules (question forest).**
Train many small forests with constrained size, then select the best forest using **fewest questions** and **highest accuracy** (matches the notebook logic).

**Motivation: Subgroup Discovery For Longitudinal Data**

Static subgroup discovery highlights unusual target distributions, but it can miss *why* change happens over time. Longitudinal subgroup discovery shifts the focus from “who is high/low today” to **“who changed, when, and what preceded the change.”** The preprocessing step explicitly encodes past context (lags), trajectories (deltas/trends), and stability (volatility), so the rules become temporal statements that can be interpreted causally or policy‑relevantly.

**Algorithms (Detailed)**

TailCluster (target‑only):
1. Take the target vector for the dataset (current level, trend, or delta depending on `target_modes`).
2. Run single‑linkage hierarchical clustering on the target values.
3. Identify the largest cluster; label all other clusters as “interesting.”
4. Optionally include explicit tails: values above `high_value_quantile` (and below the symmetric tail if `two_tailed=true`).
5. Output a binary label `is_interesting_subgroup`.

TimeTribes (target‑only, SyFlow‑inspired):
1. Bootstrap the target values for each seed.
2. For each seed, single‑linkage cluster the bootstrapped target and create candidate target ranges.
3. Map each candidate range back onto the original data and compute its size and KL divergence vs the full target distribution.
4. Score candidates by **size‑weighted KL** and add a diversity bonus to reduce overlap between selected candidates.
5. Select the top‑scoring candidates and label their union as “interesting.”

Question Forest (interpretable rules):
1. Train small decision trees on the binary label using engineered features.
2. Each tree is capped by a maximum number of **questions** (internal split nodes). This uses `max_leaf_nodes = max_questions + 1`.
3. A forest prediction is the **OR** of all trees (any tree can trigger “interesting”).
4. For each seed, compute forest accuracy and total questions (sum of splits across trees).
5. Select the best forest from the top accuracy quantile, then minimize total questions.

**Longitudinal Preprocessing And Question Space**

This is where “asking the right questions” is enforced. The model never invents questions; it only asks questions defined by the feature engineering below.

1. Parse and sort by `(id, date)`; cast numerics and drop null targets.
2. Build temporal features for each indicator:
`latest_*` (current value), `lagNy_*` (value N years ago), `deltaNy_*` (change over N years), `pct_changeNy_*` (percent change over N years), `trendNy_*` (average yearly change over N years), and optional `volNy_*` (volatility over N years).
3. Build label targets from the selected target column:
`label_target_value` (current level), `label_target_deltaW` (change over W years), `label_target_trendW` (average yearly change over W years).
4. History gating: drop rows without at least `min_history` years of target history.
5. Snapshot vs panel:
`snapshot` keeps the latest row per entity; `panel` keeps multiple time points (optionally downsampled by `panel_stride`).
6. Question map: features are mapped into human‑readable questions like:
“Change in GDP over last 7 years <= threshold?” or “Health expenditure 3 years ago > threshold?”

You control the question space directly with `question_pack.lags`, `question_pack.windows`, `include_pct_change`, and `include_volatility`.

**Quickstart**

```bash
python SD_longitudinal/scripts/run_static.py --config SD_longitudinal/configs/static.json
python SD_longitudinal/scripts/run_longitudinal.py --config SD_longitudinal/configs/longitudinal.json
python SD_longitudinal/scripts/run_longitudinal.py --config SD_longitudinal/configs/longitudinal_gdp_cutoff_eval.json
python SD_longitudinal/scripts/run_cutoff_snapshot_eval.py --config SD_longitudinal/configs/cutoff_snapshot_eval.json
python SD_longitudinal/scripts/run_all.py --config SD_longitudinal/configs/run_all.json
```

`run_cutoff_snapshot_eval.py` enforces: fixed country split, cutoff-time filtering before feature engineering, and train-only preprocessing (missing-column filtering + median imputation fit on train countries only).

**Configs And Hyperparameters**

Static config: `SD_longitudinal/configs/static.json`

`datasets.name`: dataset identifier used in output paths.  
`datasets.path`: CSV path (for file‑backed datasets).  
`datasets.source`: dataset source (`sklearn` supported).  
`datasets.dataset`: sklearn dataset name (e.g., `california_housing`).  
`datasets.target`: regression target used for subgroup labeling.  
`datasets.enabled`: toggle dataset on/off.  
`datasets.drop_columns`: columns to drop before modeling.  
`datasets.sep`: CSV delimiter.

`pipelines.tail_cluster.high_value_quantile`: tail cutoff for “interesting” points in TailCluster. Higher = more extreme.  
`pipelines.tail_cluster.two_tailed`: whether to include low‑tail outliers as interesting.  
`pipelines.timetribes.n_seeds`: number of bootstrap seeds for subgroup candidates. Higher = more coverage, slower.  
`pipelines.timetribes.beta`: size penalty exponent for candidate quality.  
`pipelines.timetribes.lambd_div`: diversity weight to reduce overlap between selected subgroups (SyFlow‑style).  
`pipelines.timetribes.top_percentile`: fraction of candidates to select as final subgroups. Lower = smaller, more selective.  
`pipelines.timetribes.max_overlap`: maximum Jaccard overlap allowed between selected candidates.  
`pipelines.timetribes.min_subgroup_size`: minimum subgroup size allowed.  
`pipelines.timetribes.max_subgroup_frac`: maximum subgroup fraction allowed.

`forest.n_seeds`: number of random forests to sample.  
`forest.accuracy_quantile`: keep only top quantile of forests by accuracy before selecting.  
`forest.min_trees` / `forest.max_trees`: forest size range (not depth).  
`forest.min_questions` / `forest.max_questions`: per‑tree question limit (internal split nodes).  
`forest.test_size`: holdout size when `eval_mode=holdout`.  
`forest.vote_rule`: `any` (OR across trees, matches notebook) or `majority`.  
`forest.eval_mode`: `train` (match notebook) or `holdout`.  
`forest.n_jobs`: parallelism for search (`-1` = all cores).

Longitudinal config: `SD_longitudinal/configs/longitudinal.json`

`datasets.id_col`: entity identifier (e.g., country).  
`datasets.date_col`: time column.  
`datasets.target`: longitudinal target for labeling.  
`datasets.target_modes`: which targets to run: `value`, `trend`, `delta`.  
`datasets.label_window`: years for `trend`/`delta` targets.  
`datasets.min_history`: minimum history length required to keep a row.  
`datasets.mode`: `snapshot` (latest per entity) or `panel` (many time points).  
`datasets.panel_stride`: keep every Nth time step in panel mode.  
`datasets.max_missing_frac`: drop columns with higher missingness than this fraction.  
`datasets.impute_missing`: median‑impute remaining numeric gaps.  
`datasets.include_target_features`: allow target‑derived features in rules (usually false to avoid leakage).
`datasets.run_standard_pipelines`: if false, skip TailCluster/TimeTribes standard runs for this dataset.

`datasets.cutoff_growth_eval.enabled`: enables supervised cut‑off evaluation for future GDP per capita growth labels.  
`datasets.cutoff_growth_eval.gdp_col` / `population_col`: columns used to compute GDP per capita.  
`datasets.cutoff_growth_eval.growth_start_year` / `growth_end_year`: years used to define growth labels.  
`datasets.cutoff_growth_eval.interesting_quantile`: countries above this growth quantile are labeled interesting.  
`datasets.cutoff_growth_eval.cutoff_years`: historical cut‑offs to evaluate (uses only data up to each year).  
`datasets.cutoff_growth_eval.test_size` / `split_seed`: fixed country‑level split for comparable test accuracy across cut‑offs.  
`datasets.cutoff_growth_eval.split_country_pool`: `all_labeled` or `available_all_cutoffs` to define which countries are eligible for the fixed split.  
`datasets.cutoff_growth_eval.snapshot_constant_rows`: when `mode=snapshot`, require the same country rows at every cut‑off (skip cut‑offs that violate this).  
`datasets.cutoff_growth_eval.enforce_min_history`: if false, do not filter rows by required lag history before modeling.  
`datasets.cutoff_growth_eval.strict_country_split`: if true, never re-split per cut‑off; skip cut‑offs with insufficient split coverage.  
`datasets.cutoff_growth_eval.min_train_rows` / `min_test_rows`: minimum available rows required for the fixed split at each cut‑off.  
`datasets.cutoff_growth_eval.min_train_positives` / `min_test_positives`: minimum positive examples required in train/test per cut‑off.  
`datasets.cutoff_growth_eval.forest_overrides`: optional forest search overrides only for this evaluation block.

`question_pack.lags`: lag windows for `lagNy`, `deltaNy`, `pct_changeNy`.  
`question_pack.windows`: windows for `trendNy` and optional `volNy`.  
`question_pack.include_pct_change`: include percent change features.  
`question_pack.include_volatility`: include rolling volatility features.  
`question_pack.indicator_include`: optional whitelist of indicators.  
`question_pack.indicator_exclude`: optional blacklist of indicators.

The longitudinal preprocessing choices directly define which questions the forest is allowed to ask.

**Recommended Settings (By Dataset Size And Expectation Type)**

Dataset size:

1. **Small datasets (<5k rows)**: increase coverage.  
Use `timetribes.n_seeds` 1000, `top_percentile` 0.01, `min_subgroup_size` 10–25.

2. **Medium datasets (5k–20k rows)**: balance coverage vs runtime.  
Use `timetribes.n_seeds` 300–700, `top_percentile` 0.005–0.01, `min_subgroup_size` 20–50.  
Why: still enough candidates, but avoids heavy O(n^2) clustering in TimeTribes.

3. **Large datasets (>20k rows)**: reduce candidates or disable TimeTribes.  
Use `timetribes.n_seeds` 50–200 or run only `tail_cluster`.

Expectation type (what you want “interesting” to mean):

1. **Level‑based outliers (high/low levels)**  
`target_modes`: include `value`  
`tail_cluster.high_value_quantile`: 0.95–0.99

2. **Rapid improvement / decline**  
`target_modes`: include `trend` and/or `delta`  
`label_window`: 5–10 years

3. **Short‑term shocks**  
`target_modes`: include `delta`  
`label_window`: 2–5 years

4. **Long‑run structural shifts**  
`target_modes`: include `trend`  
`label_window`: 10–15 years

5. **Highly noisy indicators**  
`question_pack.include_volatility`: true  
`question_pack.windows`: include smaller windows (2–5)

Question space (what questions the forest can ask):

1. **Broader temporal context**  
Add more values to `question_pack.lags` (e.g., 1, 2, 3, 5, 7, 10).

2. **Focused indicators**  
Use `question_pack.indicator_include` or `indicator_exclude`.

3. **Avoid target leakage**  
Keep `include_target_features` = false unless you explicitly want target history in rules.

**Results: World Bank (WBD)**

WBD Value (`wbd_value`):

TailCluster summary: rows=13,281; interesting=267; non‑interesting=13,014.  
Best forest: accuracy=0.9934; n_trees=2; total_questions=17; max_questions=9.

Plots and trees:
`SD_longitudinal/outputs/longitudinal/wbd_value/tail_cluster/accuracy_vs_questions.png`  
`SD_longitudinal/outputs/longitudinal/wbd_value/tail_cluster/kde_plot.png`  
`SD_longitudinal/outputs/longitudinal/wbd_value/tail_cluster/tree_00_acc_*.png`  
`SD_longitudinal/outputs/longitudinal/wbd_value/tail_cluster/tree_01_acc_*.png`

TimeTribes summary: rows=13,281; interesting=33; non‑interesting=13,248.  
Best forest: accuracy=1.0000; n_trees=2; total_questions=8; max_questions=6.

Plots and trees:
`SD_longitudinal/outputs/longitudinal/wbd_value/timetribes/accuracy_vs_questions.png`  
`SD_longitudinal/outputs/longitudinal/wbd_value/timetribes/kde_plot.png`  
`SD_longitudinal/outputs/longitudinal/wbd_value/timetribes/tree_00_acc_*.png`  
`SD_longitudinal/outputs/longitudinal/wbd_value/timetribes/tree_01_acc_*.png`

WBD Delta (`wbd_delta`):

TailCluster summary: rows=13,281; interesting=266; non‑interesting=13,015.  
Best forest: accuracy=0.9960; n_trees=1; total_questions=9; max_questions=9.

Plots and trees:
`SD_longitudinal/outputs/longitudinal/wbd_delta/tail_cluster/accuracy_vs_questions.png`  
`SD_longitudinal/outputs/longitudinal/wbd_delta/tail_cluster/kde_plot.png`  
`SD_longitudinal/outputs/longitudinal/wbd_delta/tail_cluster/tree_00_acc_*.png`

TimeTribes summary: rows=13,281; interesting=12; non‑interesting=13,269.  
Best forest: accuracy=0.9999; n_trees=1; total_questions=4; max_questions=6.

Plots and trees:
`SD_longitudinal/outputs/longitudinal/wbd_delta/timetribes/accuracy_vs_questions.png`  
`SD_longitudinal/outputs/longitudinal/wbd_delta/timetribes/kde_plot.png`  
`SD_longitudinal/outputs/longitudinal/wbd_delta/timetribes/tree_00_acc_*.png`

**How To Read rules.json**

Each `tree_XX_rules.json` contains a list of rules (one per leaf that predicts “interesting”). Each rule includes:

`leaf_id`: leaf node ID from the tree.  
`prediction`: always 1 (interesting).  
`n_samples`: number of samples reaching that leaf.  
`rule`: a list of question strings that define the path to that leaf.  
`target_summary`: summary statistics of the target for samples in that leaf.

`target_summary` contains:

`target_mean`, `target_median`, `target_min`, `target_max`, `n_samples`, and `target_position` (high/low relative to the global median). This lets you see whether each subgroup corresponds to higher‑target or lower‑target outcomes.

**Outputs**

Outputs are written under:

```
SD_longitudinal/outputs/{static|longitudinal}/{dataset}/{pipeline}/
```

Each run includes:

`labeled.csv`, `forest_results.csv`, `best_forest.json`, `summary.json`, `question_map.csv`, `kde_plot.*`, `accuracy_vs_questions.*`, `tree_XX_acc_*.*`, `tree_XX_rules.json`

Tree plots now use compact labels: internal nodes show only the split question, and leaf nodes show only mean target value and predicted class.

**Dependencies**

`numpy`, `pandas`, `polars`, `scikit-learn`, `scipy`, `seaborn`, `matplotlib`  
Optional: `joblib` for parallel forest search.

The California housing dataset uses `sklearn.datasets.fetch_california_housing`, which may download data on first use.
