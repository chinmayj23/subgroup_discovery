# Interpretable_SD

Interpretable subgroup discovery for longitudinal data.

## Motivation

Longitudinal data describe the same subject repeatedly over time. Examples include countries observed across years, patients followed through clinical visits, households revisited in panel surveys, or firms tracked through financial reports.

This setting is different from ordinary static classification. In a static dataset, the natural question is whether a subject belongs to an interesting subgroup. In a longitudinal dataset, that question is often too coarse. A subject may look ordinary for most of the observation period and become unusual only during a short interval. For many scientific, policy, and monitoring tasks, that interval is the real object of interest.

That is the practical motivation for this repository. Analysts often do not only want to know which subjects are unusual. They want to know when the unusual behavior starts, how long it lasts, and whether it can be described in a form that another researcher, policymaker, or domain expert can read without reverse-engineering a complex model.

This repository studies the following problem:

1. identify which subject-time observations are unusual in a target variable,
2. convert those observations into meaningful subject-level spans,
3. explain those spans with compact, human-readable rules.

The goal is therefore not only prediction. The goal is a complete interpretable subgroup discovery workflow for longitudinal data: first identify unusual target behavior, then organize it into meaningful spans, and finally explain those spans with compact rules.

## Method Overview

The pipeline has two modules and one linking step.

1. **Interestingness module**
   Assign row-level interesting or non-interesting labels from the target variable alone.
2. **Temporal feature processing**
   Compute subject-time covariates from the longitudinal history.
3. **Span construction**
   Merge consecutive rows with the same label within each subject into one longitudinal span.
4. **Rule-learning module**
   Learn interpretable rules that explain why a span is interesting.

This separation is important. Interestingness is defined from the target distribution. Explanation is learned afterwards from temporal covariates. The pipeline is therefore designed to recover meaningful longitudinal subgroups first, and only then explain them.

## Pipeline

For each experiment, the workflow is:

1. load a longitudinal dataset with subject id, time variable, target, and covariates,
2. apply an interestingness method to the raw subject-time target values,
3. compute temporal covariates on the original subject-time panel,
4. group consecutive equal labels within each subject into spans,
5. split subjects into train and test sets,
6. fit interpretable rule models on the span rows,
7. report subgroup-quality metrics, predictive metrics, trees, and natural-language rules.

The train/test split is done at the subject level, so all spans from the same subject remain in the same split.

## Interestingness Methods

### TailCluster

TailCluster is a target-only method. It marks observations as interesting when they belong to detached target clusters or to extreme tails of the target distribution.

It is useful when the subgroup of interest is expected to be rare and extreme.

### TimeTribes

TimeTribes is also target-only, but it is not restricted to simple tails. It searches for target ranges that are distributionally distinct, using bootstrap candidate generation, divergence-based scoring, and overlap control. In the current implementation, candidate intervals are drawn both from bootstrap clustering and from small bootstrap quantile windows. This does not change the objective; it only gives the selection stage a healthier set of target-only candidates when one-dimensional clustering alone is too brittle.

It is useful when the interesting subgroup is unusual in distributional shape rather than only in absolute magnitude. In the supplied configurations, no minimum subgroup size is imposed, while a moderate upper subgroup-fraction cap prevents the method from declaring nearly the whole dataset interesting.

## Span Construction

After row-level interestingness is computed, the labels are converted into subject-level spans.

For each subject:

1. sort rows by time,
2. keep the row-level interestingness labels,
3. merge consecutive rows with the same label,
4. store one span row with start time, end time, span length, and target summaries.

This gives a more natural unit for longitudinal subgroup discovery than isolated rows or a fixed window chosen in advance. If a subject has only one interesting year, that year is kept and becomes a one-year interesting span. If several interesting years are consecutive, they are grouped into one longer interesting span. The implementation assumes one row per `(subject, time)` pair; if that is violated, the code raises an error rather than silently building ambiguous spans.

## Rule-learning Method

### RF Rule Learner

The main explanation method is a random-forest-based rule learner.

It uses the temporal covariates on the span rows and performs a multi-seed search over compact forests. The selected forest is exported as:

1. simplified tree plots,
2. JSON rule files,
3. natural-language rules,
4. accuracy-versus-questions plots.

The leaves report whether a span is interesting, the mean target value of positive leaves, and the span years captured by the leaf.

## Temporal Features

The temporal feature engineering is shared across methods. It uses the same longitudinal feature families throughout:

1. `latest_*`
2. `lag*k_*`
3. `delta*k_*`
4. `pct_change*k_*`
5. `trend*k_*`
6. `vol*k_*`

These features are computed from the longitudinal history on the original subject-time panel and are then aligned to the span rows used by the rule-learning stage.

For each span, the attached temporal covariates come from the final row of that span, while the span metadata separately record the start year, end year, length, and span-level target summaries. In the rule-learning stage, temporal features derived from the target variable itself are excluded from prediction; the rules are learned from the other longitudinal covariates.

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
3. positive leaves also show mean target value,
4. positive leaves list the year spans observed in that leaf, with multiple spans shown when present.

Representative `TimeTribes + RF` tree:

![TimeTribes RF Tree](outputs/world_bank_full_paper_grade/life_expectancy_spectrum/timetribes_windowed/rf_rule_module/trees/tree_00_acc_0.png)

### Natural-language Rules

Rules are exported in directly readable form. A positive rule reports:

1. the covariate conditions,
2. the mean target value of the leaf,
3. the year spans covered by the interesting segments in that leaf,
4. the average span length.

A typical rule has the form:

```text
IF <conditions>, THEN interesting (mean target=..., spans=YYYY-YYYY, YYYY-YYYY, avg span length=...).
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
6. `Interpretable_SD/src/interpretable_sd/comparison.py`
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
6. `experiments`

The supplied configurations keep isolated interesting years (`min_interesting_run_length = 1`). For `TimeTribes`, the subgroup-size filters are controlled through `min_subgroup_size` and `max_subgroup_frac`; in the main configs the lower bound is left `null`, while the upper bound is set to a moderate fraction so the interesting subgroup does not collapse to almost the entire dataset.

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
