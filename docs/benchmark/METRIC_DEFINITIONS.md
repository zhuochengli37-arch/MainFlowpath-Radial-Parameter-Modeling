# Formal Benchmark Metric Definitions

Status: `provisional_v1` — frozen for implementation before the real CFD benchmark.

This document defines metrics only. Experiment flow is in
`BENCHMARK_EXPERIMENT_PROTOCOL.md`; repeat and seed rules are in
`STOCHASTIC_REPEAT_PROTOCOL.md`.

## Formal metrics

For an evaluation set of size \(N\), with truth \(y_i\) and prediction
\(\hat y_i\):

- **RMSE**: \(\sqrt{N^{-1}\sum_i(\hat y_i-y_i)^2}\). This is the primary
  absolute-error metric.
- **MAE**: \(N^{-1}\sum_i|\hat y_i-y_i|\). This is the supporting
  absolute-error metric and is less sensitive than RMSE to large errors.
- **MAPE**: \(N^{-1}\sum_i |(y_i-\hat y_i)/\max(|y_i|,10^{-6})|\), matching
  the existing metric pipeline. It is stored as a ratio and remains supporting
  relative-error evidence.
- **R2**: the coefficient of determination. It describes fit to the variation in
  the evaluation target and is supporting evidence.
- **NRMSE_range**:

  \[
  \mathrm{NRMSE}_{range}=
  \frac{\mathrm{RMSE}}{\max(y_{eval})-\min(y_{eval})}.
  \]

  This is the primary scale-normalized metric for comparison across cases.

NRMSE_range complements rather than replaces RMSE. RMSE and MAE retain the
physical error scale; NRMSE_range supports comparisons across stage, section,
and target; MAPE and R2 remain supporting information.

## Evaluation-set denominator

- Validation: `y_eval = y_validation`, so the denominator is
  `max(y_val) - min(y_val)`.
- Test: `y_eval = y_test`, so the denominator is
  `max(y_test) - min(y_test)`.

All models in one case use the same fixed validation and test sets, so their
denominators are identical. A later sample-efficiency study must also keep those
sets fixed; then changing the training-condition count does not change the
normalization scale. NRMSE_range therefore states the fraction of the observed
evaluation-target range represented by RMSE.

Raw RMSE values from unlike targets, stages, or sections must never be averaged
as though they shared one scale.

## Storage and undefined values

NRMSE_range is stored as a unitless ratio. Store `0.032`, not `3.2` or
`"3.2%"`; percentage conversion belongs only in presentation output.

Let `target_range = max(y_eval) - min(y_eval)` and
`reference_scale = max(1.0, max(abs(y_eval)))`. The provisional v1
near-constant test is:

```text
target_range <= absolute_tolerance + relative_tolerance * reference_scale
absolute_tolerance = 0.0
relative_tolerance = 1e-12
```

When that test is true, NRMSE_range is undefined/NA and the result must record
`constant_or_near_constant_target`. The denominator must not be clamped with an
epsilon. Other metrics retain whatever values their own documented rules permit.
The tolerance is explicit protocol configuration and must not be hidden inside a
metric function.

No NRMSE_mean, NRMSE_std, CVRMSE, or SMAPE variant is part of v1.
