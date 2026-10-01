# Sample-Efficiency Experiment Protocol

Status: `v1_defined_not_instantiated`. This protocol is ready to instantiate only
after real CFD data and representative-model selection are available.

This document defines sample-efficiency methodology only. Metric formulas remain in
`METRIC_DEFINITIONS.md`; stochastic repeats remain in
`STOCHASTIC_REPEAT_PROTOCOL.md`.

## Purpose and placement

The experiment asks how representative-model performance changes as the available
number of CFD training operating conditions increases. It follows, and is not part
of, the 35-model unified screening:

```text
Unified Accuracy Benchmark
-> Within-Family Screening
-> Representative Model Comparison
-> Sample-Efficiency Analysis
```

Only selected representative models participate. The protocol does not authorize
running all 35 implemented models at every fraction.

## Sample unit and fraction denominator

The sample unit is one complete CFD operating condition, normally identified by
`(n,w)` or by a validated condition/state ID representing that same profile. Every
xi radial point belonging to one condition is included or excluded together.
Point-level `(n,w,xi)` sampling is prohibited.

The v1 training fractions are `0.25`, `0.50`, `0.75`, and `1.00`. Their denominator
is the complete **training-condition set after the formal condition-level
train/validation/test split**, not every condition in the database. If that complete
set contains `N_train_full` conditions, each fraction targets that fraction of
`N_train_full`.

The 100% subset contains all and only the formal training conditions. It never
includes validation or test conditions.

Concrete integer counts are not yet frozen. The v1 `count_rounding_policy` is
`to_be_finalized_when_real_condition_count_is_known`; one common rule must be frozen
before the real benchmark and applied equally to every model. Synthetic-data counts
must not determine that rule.

## Fixed evaluation sets and nested training subsets

Validation and test conditions stay unchanged across every training fraction. This
keeps both the evaluation population and each NRMSE_range denominator fixed while
training data grow.

Training subsets must satisfy:

```text
25% ⊂ 50% ⊂ 75% ⊂ 100%
```

For each case, begin with its complete training-condition universe, generate one
condition order using `subset_selection_seed = 20261001`, and take successively
larger prefixes after the integer counts are frozen. Fractions must not be sampled
independently. Every representative model for the same component, stage, section,
target, and fraction uses exactly the same train, validation, and test conditions.
Future manifests must record hashes for all three condition sets.

Different cases must build subsets from their own condition universes. Cases with
identical complete condition sets may reuse one split/subset manifest; cases with
different universes must not share incompatible condition indices.

## Three separate seeds

- `split_seed` determines the complete train/validation/test condition split and is
  inherited as `42` from the formal benchmark protocol.
- `subset_selection_seed` determines only the fixed order within the complete
  training set and is `20261001`.
- `model_seed` controls only model training/search randomness and is inherited from
  the stochastic repeat protocol.

Changing a model seed must never change a condition subset. No model may resample
its own training conditions.

## Metrics, repeats, and recording

The formal metrics are inherited without redefinition: RMSE, MAE, MAPE, R2, and
NRMSE_range. Because validation/test sets remain fixed, their NRMSE_range
denominators remain fixed across fractions.

This is representative/deep analysis, so each fraction uses:

- deterministic models: 1 run;
- stochastic models: 5 runs with `[42, 2026, 3407, 1234, 5678]`;
- search-stochastic models: the representative-comparison rule, currently 3 PySR
  runs with `[42, 2026, 3407]`, or the uniformly frozen 5-run extension allowed by
  protocol v1.

The same model-seed set is used at every fraction. Stochastic results report mean
and sample standard deviation for all five formal metrics; best-run reporting is
prohibited. Deterministic models record one result.

Future records must identify the experiment/case/model, fraction and realized
condition count, train/validation/test condition hashes, all three seed roles,
repeat index, five metrics, and status. Phase 5E defines that schema but creates no
real records or subset manifests.

## Deferred work

No convergence threshold, sufficient-sample criterion, automatic best fraction,
active learning, optimal design, Latin-hypercube resampling, uncertainty-driven
sampling, adaptive CFD selection, model artifact, prediction file, learning curve,
or plot is part of Phase 5E. Those require real data and a later authorized stage.
