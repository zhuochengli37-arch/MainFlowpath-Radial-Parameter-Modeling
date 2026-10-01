# Formal Benchmark Experiment Protocol

Status: `provisional_v1` — protocol definition only; no benchmark has run.

## Scope and existing governance

The formal benchmark uses real CFD data and the Phase 5B five-stage screening
workflow in `data/output/benchmark/selection/selection_protocol.json`. This document
does not replace that workflow or the frozen paper taxonomy in
`data/output/benchmark/config/model_taxonomy.csv`.

Detailed rules are intentionally separated:

- metrics: `METRIC_DEFINITIONS.md` and
  `data/output/benchmark/config/metric_definitions.json`;
- model randomness and repeats: `STOCHASTIC_REPEAT_PROTOCOL.md` and
  `data/output/benchmark/config/stochastic_repeat_protocol.json`;
- representative-model sample efficiency: `SAMPLE_EFFICIENCY_PROTOCOL.md` and
  `data/output/benchmark/config/sample_efficiency_protocol.json`;
- overall machine-readable links:
  `data/output/benchmark/config/benchmark_protocol.json`.

Phase 5C supplies metric and repeat rules to Phase 5B Stage 2 (Unified Accuracy
Benchmark) and Stage 4 (Representative Model Comparison). It does not select models.

## Formal flow

1. Freeze the real CFD dataset identity and complete condition inventory.
2. Create one condition-level split with split seed `42`; all xi points belonging
   to one `(n,w)` condition stay in exactly one subset.
3. Freeze model configurations, unresolved stochasticity decisions, metric-pipeline
   details, hardware policy, and the 3-or-5 PySR representative-repeat decision.
4. Run engineering eligibility, then the shared Stage 2 accuracy benchmark using
   the formal five metrics and the prescribed repeats.
5. Perform within-family screening under the existing Phase 5B taxonomy.
6. Run Stage 4 representative comparisons with their prescribed repeats and report
   mean/std rather than a best run.
7. After representative selection, instantiate the separate sample-efficiency
   protocol when the real CFD condition count is known.
8. Preserve split hashes, dataset identity, seeds, configurations, individual-run
   records, aggregated metrics, runtime, and failure status.

Validation and test evaluation use their own fixed target ranges for NRMSE_range.
Raw RMSE must not be averaged across unlike target scales. Prediction arrays and
large model artifacts are outside the lightweight result-management contract.

## Sample-efficiency study

Phase 5E defines this representative-model study in
`SAMPLE_EFFICIENCY_PROTOCOL.md`. Its nested 25%/50%/75%/100% training-condition
subsets keep validation and test fixed. Concrete condition counts and subset
manifests wait for the real CFD condition inventory; no sample-efficiency run has
started.
