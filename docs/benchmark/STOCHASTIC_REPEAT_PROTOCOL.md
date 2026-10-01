# Formal Benchmark Stochastic Repeat Protocol

Status: `v1` — Phase 5D preflight closed. It must not be changed in response to
benchmark results.

This document defines randomness, seeds, repeats, and repeat aggregation only.
Metric definitions are in `METRIC_DEFINITIONS.md`.

## Two independent sources of randomness

Data-split randomness selects complete `(n,w)` conditions for train, validation,
and test. Model randomness includes initialization, sampling, bagging, stochastic
optimization or preprocessing, and randomized search.

Within one repeat group, the condition-level split is fixed and only the model or
search seed changes. Changing split seed and model seed together is prohibited,
because their effects could not be separated.

The formal split seed is `42`. A complete `(n,w)` condition, including every xi
point, is the minimum split unit. Point-level xi random splitting is prohibited.
Phase 5C did not create a split; Phase 5D reused the saved Phase 4B/4C split without
repartitioning.

## Stochasticity metadata

`stochasticity_type` is experiment metadata, not a paper family:

- `deterministic`: training is intrinsically fixed for identical data,
  parameters, environment, and seed configuration.
- `stochastic`: initialization, sampling, bagging, random optimization, or other
  effective training randomness is present.
- `search_stochastic`: the method performs a randomized search; PySR is the
  current example.

The type is determined from the actual configured implementation, not from its
paper family or model name. The machine-readable protocol records the current
per-model decision and evidence. Phase 5D verified that TabPFN and TabICLv2 are
both `stochastic` under the current formal configurations: seeds 42, 2026, and 3407
reached the official ensemble/preprocessing implementation and changed validation
and test predictions while data, split, checkpoint, and all other configuration
remained fixed. Detailed pairwise evidence is in
`data/output/benchmark/preflight/foundation_model_stochasticity_check.json`.

## Repeats and fixed seeds

Initial screening uses:

- deterministic: 1 run, seed `42` where an API requires one;
- stochastic: 3 runs using `[42, 2026, 3407]`;
- search_stochastic: 3 runs using `[42, 2026, 3407]`.

Representative comparison uses:

- deterministic: 1 run;
- stochastic: 5 runs using `[42, 2026, 3407, 1234, 5678]`;
- search_stochastic: 3 runs using `[42, 2026, 3407]` by default.

PySR may be extended uniformly to 5 runs with additional seeds `[1234, 5678]` only
if it passes screening, becomes a primary paper representative, compute resources
permit, and the decision is frozen before Representative Model Comparison starts.
Once that comparison starts, its repeat count cannot change. Seeds are never
generated at run time, and result-driven changes are prohibited.

## Aggregation

For stochastic and search-stochastic models, report arithmetic `mean` and sample
`std` across all scheduled runs for RMSE, MAE, MAPE, R2, and NRMSE_range. Every
scheduled run remains recorded. Choosing the best run as the model result is
prohibited. Median and IQR are deferred unless a later protocol revision explicitly
adds them before the benchmark.

## PySR

PySR is `search_stochastic`. Protocol v1 uses 3 runs with seeds 42, 2026, and 3407
for both initial screening and, by default, representative comparison. Across
repeats, keep operators, maxsize, maxdepth, search budget, timeout, complexity rule,
and selection rule identical; change only the search seed. Record each run's
expression and metrics. Benchmark performance is the multi-run mean/std, never the
lowest-error expression selected after inspecting several runs. Symbolic-expression
analysis may be reported separately. Phase 5D did not run PySR.
