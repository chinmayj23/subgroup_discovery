# SD_longitudinal

A clean, minimal codebase for **subgroup discovery on static and longitudinal data**. Subgroup discovery is kept fixed (target-only candidate generation). Interpretability comes from **asking the right temporal questions** via longitudinal preprocessing and then fitting a compact question forest.

**Overview**

1. **Module 1: Subgroup discovery (target-only).**
`tail_cluster`: single-linkage clustering on the target with tail detection.  
`TimeTribes`: bootstrapped clustering with a KL‑divergence + diversity objective inspired by **SyFlow**.

2. **Module 2: Interpretable rules (question forest).**
Train many small forests with constrained size, then select the best forest using **fewest questions** and **highest accuracy** (matches the notebook logic).

**Motivation: Subgroup Discovery For Longitudinal Data**

Static subgroup discovery highlights unusual target distributions, but it can miss *why* change happens *over time*. Longitudinal subgroup discovery shifts the focus from “who is high/low today” to **“who changed, when, and what preceded the change.”** The preprocessing step explicitly encodes past context (lags), trajectories (deltas/trends), and stability (volatility), so the rules become temporal statements that can be interpreted causally or policy‑relevantly.

**Longitudinal Preprocessing And Questions**

This is where “asking the right questions” is enforced. The model never invents questions; it only asks questions defined by the feature engineering below.

1. **Parse and sort**: parse dates, cast numerics, and sort by `(id, date)`.
2. **Build temporal features** for each indicator:
`latest_*` (current value), `lagNy_*` (value N years ago), `deltaNy_*` (change over N years), `pct_changeNy_*` (percent change over N years), `trendNy_*` (average yearly change over N years), and optional `volNy_*` (volatility over N years).
3. **Build label targets** from the selected target column:
`label_target_value` (current level), `label_target_deltaW` (change over W years), `label_target_trendW` (average yearly change over W years).
4. **History gating**: drop rows that don’t have at least `min_history` years of target history, so questions are about real past context.
5. **Snapshot vs panel**:
`snapshot` keeps the latest row per entity; `panel` keeps multiple time points (optionally downsampled by `panel_stride`).
6. **Question map**: features are mapped into human‑readable questions like:
“Change in GDP over last 7 years <= threshold?” or “Health expenditure 3 years ago > threshold?”

You control the question space directly with `question_pack.lags`, `question_pack.windows`, `include_pct_change`, and `include_volatility`.

**Quickstart**

```bash
python SD_longitudinal/scripts/run_static.py --config SD_longitudinal/configs/static.json
python SD_longitudinal/scripts/run_longitudinal.py --config SD_longitudinal/configs/longitudinal.json
python SD_longitudinal/scripts/run_all.py --config SD_longitudinal/configs/run_all.json
```

**Configs And Hyperparameters**

Static config: `SD_longitudinal/configs/static.json`

`datasets.name`: dataset identifier used in output paths.  
`datasets.path`: CSV path (for file-backed datasets).  
`datasets.source`: dataset source (`sklearn` supported).  
`datasets.dataset`: sklearn dataset name (e.g., `california_housing`).  
`datasets.target`: regression target used for subgroup labeling.  
`datasets.enabled`: toggle dataset on/off.  
`datasets.drop_columns`: columns to drop before modeling.  
`datasets.sep`: CSV delimiter.

`pipelines.tail_cluster.high_value_quantile`: tail cutoff for “interesting” points in the TailCluster method. Higher = more extreme.  
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

`question_pack.lags`: lag windows for `lagNy`, `deltaNy`, `pct_changeNy`.  
`question_pack.windows`: windows for `trendNy` and optional `volNy`.  
`question_pack.include_pct_change`: include percent change features.  
`question_pack.include_volatility`: include rolling volatility features.  
`question_pack.indicator_include`: optional whitelist of indicators.  
`question_pack.indicator_exclude`: optional blacklist of indicators.

The longitudinal preprocessing choices directly define which questions the forest is allowed to ask.

**Recommended Settings (By Dataset Size And Expectation Type)**

Use these as starting points, then tighten or relax based on signal and runtime.

Dataset size:

1. **Small datasets (<5k rows)**: increase coverage.  
Use `timetribes.n_seeds` 1000, `top_percentile` 0.01, `min_subgroup_size` 10–25.  
Why: small datasets benefit from more candidate sampling without extreme runtime.

2. **Medium datasets (5k–20k rows)**: balance coverage vs runtime.  
Use `timetribes.n_seeds` 300–700, `top_percentile` 0.005–0.01, `min_subgroup_size` 20–50.  
Why: still enough candidates, but avoids heavy O(n^2) clustering in TimeTribes.

3. **Large datasets (>20k rows)**: reduce candidates or disable TimeTribes.  
Use `timetribes.n_seeds` 50–200 or run only `tail_cluster`.  
Why: hierarchical clustering is the main bottleneck at scale.

Expectation type (what you want “interesting” to mean):

1. **Level‑based outliers (high/low levels)**  
`target_modes`: include `value`  
`tail_cluster.high_value_quantile`: 0.95–0.99  
Why: captures persistent high/low levels without requiring change.

2. **Rapid improvement / decline**  
`target_modes`: include `trend` and/or `delta`  
`label_window`: 5–10 years  
Why: focuses labeling on sustained change rather than absolute level.

3. **Short‑term shocks**  
`target_modes`: include `delta`  
`label_window`: 2–5 years  
Why: shorter windows isolate sudden changes.

4. **Long‑run structural shifts**  
`target_modes`: include `trend`  
`label_window`: 10–15 years  
Why: longer windows capture gradual regime changes.

5. **Highly noisy indicators**  
`question_pack.include_volatility`: true  
`question_pack.windows`: include smaller windows (2–5)  
Why: allows questions about stability vs volatility rather than only level.

Question space (what questions the forest can ask):

1. **Broader temporal context**  
Add more values to `question_pack.lags` (e.g., 1, 2, 3, 5, 7, 10).  
Why: gives the forest multiple past horizons.

2. **Focused indicators**  
Use `question_pack.indicator_include` or `indicator_exclude`.  
Why: limits the search to domain‑relevant signals and avoids spurious rules.

3. **Avoid target leakage**  
Keep `include_target_features` = false unless you explicitly want target history in rules.

**Outputs**

Outputs are written under:

```
SD_longitudinal/outputs/{static|longitudinal}/{dataset}/{pipeline}/
```

Each run includes:

`labeled.csv`, `forest_results.csv`, `best_forest.json`, `summary.json`, `question_map.csv`, `kde_plot.*`, `accuracy_vs_questions.*`, `tree_XX_acc_*.*`, `tree_XX_rules.json`

**Dependencies**

`numpy`, `pandas`, `polars`, `scikit-learn`, `scipy`, `seaborn`, `matplotlib`  
Optional: `joblib` for parallel forest search.

The California housing dataset uses `sklearn.datasets.fetch_california_housing`, which may download data on first use.
