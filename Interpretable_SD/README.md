# Interpretable_SD

Interpretable subgroup discovery for longitudinal data.

## Motivation

Longitudinal data describe the same subject repeatedly over time. Examples include countries observed across years, patients followed through clinical visits, households revisited in panel surveys, or firms tracked through financial reports.

This setting is different from ordinary static classification. In a static dataset, the natural question is whether a subject belongs to an interesting subgroup. In a longitudinal dataset, that is often too coarse. A subject may be ordinary for most of the observation period and become unusual only during a specific interval. For many scientific, policy, and monitoring tasks, that interval is the real object of interest.

This repository studies the following problem:

1. identify which subject-time observations are unusual in a target variable,
2. convert those observations into meaningful subject-level spans,
3. explain those spans with compact, human-readable rules.

The goal is therefore not only prediction. The goal is a complete interpretable subgroup discovery workflow for longitudinal data.

## Method Overview

The pipeline has two modules and one linking step.

1. **Interestingness module**
   Assign row-level interesting or non-interesting labels from the target variable alone.
2. **Span construction**
   Merge consecutive rows with the same label within each subject into one longitudinal span.
3. **Rule-learning module**
   Learn interpretable rules that explain why a span is interesting.

This separation is important. Interestingness is defined from the target distribution. Explanation is learned afterwards from temporal covariates. The pipeline is therefore designed to recover meaningful longitudinal subgroups first, and only then explain them.

## Pipeline

For each experiment, the workflow is:

1. load a longitudinal dataset with subject id, time variable, target, and covariates,
2. apply an interestingness method to the raw subject-time target values,
3. group consecutive equal labels within each subject into spans,
4. compute temporal covariates from the longitudinal history,
5. split subjects into train and test sets,
6. fit interpretable rule models on the span rows,
7. report subgroup-quality metrics, predictive metrics, trees, and natural-language rules.

The train/test split is done at the subject level, so all spans from the same subject remain in the same split.

## Interestingness Methods

### TailCluster

TailCluster is a target-only method. It marks observations as interesting when they belong to detached target clusters or to extreme tails of the target distribution.

It is useful when the subgroup of interest is expected to be rare and extreme.

### TimeTribes

TimeTribes is also target-only, but it is not restricted to simple tails. It searches for target ranges that are distributionally distinct, using bootstrap candidate generation, divergence-based scoring, and overlap control.

It is useful when the interesting subgroup is unusual in distributional shape rather than only in absolute magnitude.

## Span Construction

After row-level interestingness is computed, the labels are converted into subject-level spans.

For each subject:

1. sort rows by time,
2. keep the row-level interestingness labels,
3. merge consecutive rows with the same label,
4. store one span row with start time, end time, span length, and target summaries.

This gives a more natural unit for longitudinal subgroup discovery than isolated rows or a fixed window chosen in advance.

## Rule-learning Methods

### RF Rule Learner

The main explanation method is a random-forest-based rule learner.

It uses the temporal covariates on the span rows and performs a multi-seed search over compact forests. The selected forest is exported as:

1. simplified tree plots,
2. JSON rule files,
3. natural-language rules,
4. accuracy-versus-questions plots.

The leaves report whether a span is interesting, the mean target value of positive leaves, and typical time spans captured by the leaf.

### Sysurv-style Baseline

This baseline adapts the idea of a nonlinear scorer followed by a shallow explanatory tree to the present fixed-label span-classification setting.

The adaptation is necessary because the original survival setting predicts time-to-event, whereas the present task predicts interesting versus non-interesting spans.

### Tree-paper-style Baseline

This baseline uses a single decision tree with pruning-path selection. It is included to test whether a single compact tree is sufficient once interesting spans have already been defined.

This is also an adaptation to the present fixed-label span-classification task. The original tree-based survival setting is not used directly here.

## Temporal Features

The temporal feature engineering is shared across methods. It uses the same longitudinal feature families throughout:

1. `latest_*`
2. `lag*k_*`
3. `delta*k_*`
4. `pct_change*k_*`
5. `trend*k_*`
6. `vol*k_*`

These features are computed from the longitudinal history and aligned to the span rows used by the rule-learning stage.

## Evaluation

Two groups of metrics are reported.

### Interestingness Metrics

These measure how different the discovered subgroup is from the overall target distribution.

1. `support`: fraction of raw subject-time rows labeled interesting,
2. `mean_all`: target mean over all raw rows,
3. `mean_interesting`: target mean inside the interesting subgroup,
4. `mean_shift`: difference between subgroup mean and global mean,
5. `kl_divergence`: distributional separation between subgroup and overall target distribution,
6. `n_segments`, `n_interesting_segments`, `segment_support`: span-level summary statistics after span construction.

### Rule-learning Metrics

These measure how well the rule models recover the interesting spans on held-out subjects.

1. `accuracy`
2. `balanced_accuracy`
3. `precision`
4. `recall`
5. `f1`
6. `roc_auc`

## Example Study

The repository includes an illustrative experiment on World Bank indicators. In that experiment:

1. each country is a subject,
2. each year is a time point,
3. interestingness is computed on raw country-year target values,
4. spans are formed within each country,
5. rules are learned on the resulting country-span rows.

The same code is intended to be reusable for other longitudinal datasets with the same basic structure.

## Outputs

The pipeline produces three main result types.

### Interestingness Plots

These compare the target distribution of the interesting subgroup against the overall distribution.

TailCluster KDE:

![TailCluster KDE](outputs/world_bank_full_paper_grade/life_expectancy_spectrum/tail_cluster/interestingness_kde.png)

TimeTribes KDE:

![TimeTribes KDE](outputs/world_bank_full_paper_grade/life_expectancy_spectrum/timetribes_windowed/interestingness_kde.png)

### Trees

Tree plots are intentionally simplified:

1. internal nodes show only the question,
2. leaves show whether the span is interesting,
3. positive leaves also show mean target value.

Representative `TimeTribes + RF` tree:

![TimeTribes RF Tree](outputs/world_bank_full_paper_grade/life_expectancy_spectrum/timetribes_windowed/rf_rule_module/trees/tree_00_acc_0.png)

### Natural-language Rules

Rules are exported in directly readable form. A positive rule reports:

1. the covariate conditions,
2. the mean target value of the leaf,
3. the typical years covered by the interesting spans in that leaf,
4. the average span length.

A typical rule has the form:

```text
IF <conditions>, THEN interesting (mean target=..., typical span=YYYY-YYYY, avg span length=...).
```

## Code Structure

The main files are:

1. `Interpretable_SD/src/interpretable_sd/pipeline.py`
   Main experiment pipeline.
2. `Interpretable_SD/src/interpretable_sd/preprocessing.py`
   Span construction, subject split, and matrix preparation.
3. `Interpretable_SD/src/interpretable_sd/interestingness/tail_cluster.py`
   TailCluster interestingness method.
4. `Interpretable_SD/src/interpretable_sd/interestingness/timetribes_windowed.py`
   TimeTribes interestingness method.
5. `Interpretable_SD/src/interpretable_sd/rules/rf_module.py`
   RF rule learner.
6. `Interpretable_SD/src/interpretable_sd/baselines/`
   Comparable baseline methods.
7. `Interpretable_SD/src/interpretable_sd/comparison.py`
   Comparison-table generation.

## Running The Code

Full run:

```bash
python Interpretable_SD/scripts/run_interpretable_sd.py --config Interpretable_SD/configs/world_bank_full_paper_grade.json --output-dir Interpretable_SD/outputs/world_bank_full_paper_grade
```

Reduced-compute run:

```bash
python Interpretable_SD/scripts/run_interpretable_sd.py --config Interpretable_SD/configs/world_bank_fast_debug.json --output-dir Interpretable_SD/outputs/world_bank_fast_debug
```

Generate comparison tables:

```bash
python Interpretable_SD/scripts/generate_comparison_table.py --output-dir Interpretable_SD/outputs/world_bank_full_paper_grade --table-name comparison_full_paper_grade
```

## Configuration

The principal configuration files are:

1. `Interpretable_SD/configs/world_bank_full_paper_grade.json`
2. `Interpretable_SD/configs/world_bank_fast_debug.json`

The most important configuration blocks are:

1. `data`
2. `defaults.feature_engineering`
3. `defaults.segmentation`
4. `defaults.split`
5. `defaults.rf_rule_module`
6. `defaults.baselines`
7. `experiments`

## Additional Documentation

Detailed implementation notes are in:

`Interpretable_SD/docs/IMPLEMENTATION_DETAILS.md`

## Conclusion

This repository provides a compact framework for interpretable subgroup discovery in longitudinal data.

Its central idea is simple:

1. detect unusual subject-time behavior from the target distribution,
2. convert those labels into meaningful subject-level spans,
3. explain those spans with compact temporal rules.

The result is a workflow that produces both quantitative subgroup evidence and directly readable explanations. This makes it suitable not only for benchmark experiments, but also for real longitudinal studies where the aim is to understand when unusual behavior occurs and how it can be described in a form that domain experts can use.
