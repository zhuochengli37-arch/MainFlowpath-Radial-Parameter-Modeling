from __future__ import annotations

import argparse
import json

from project1.experiments.benchmark_1d_runner import run_benchmark_1d
from project1.experiments.benchmark_data import (
    AnySample,
    FlexSample,
    Sample,
    SCHEMA_CONFIG,
    all_targets,
    load_predict_samples,
    load_samples,
    resolve_1d_input_files,
    schema_input_type,
)
from project1.modeling.custom_models import (
    EndpointRegression,
    HierarchicalWorklineRegressor,
    LOWESSLikeRegressor,
    MODEL_CATEGORY,
    PhysicsClampedRegressor,
    PiecewiseRidgeRegressor,
    RBFRegressor,
)
from project1.experiments.benchmark_partitioning import (
    cv_splits,
    metrics,
    partition_samples,
    prepare_model_for_fold,
    resolve_kfold_splits,
    to_arrays,
    to_arrays_2d,
)
from project1.experiments.benchmark_reporting import (
    append_ranked_rows,
    build_aggregate_rows,
    build_benchmark_report,
    real_partition_sample_count,
    write_partition_leaderboards,
    write_utf8_bom_text,
)
from project1.experiments.benchmark_runners import run_benchmark, run_benchmark_2d


def _partition_samples(samples: list[AnySample], partition_mode: str) -> dict[str, list[AnySample]]:
    return partition_samples(samples, partition_mode)


def _real_partition_sample_count(items: list[dict[str, object]], partition: str) -> int:
    return real_partition_sample_count(items, partition)


def _append_ranked_rows(
    summary_lines: list[str],
    rows: list[dict[str, object]],
    target: str,
    split_mode: str,
    ranking_metric: str,
) -> None:
    append_ranked_rows(summary_lines, rows, target, split_mode, ranking_metric)


def _cv_splits(samples: list[AnySample], mode: str, requested_splits: int):
    return cv_splits(samples, mode, requested_splits)


def _resolve_kfold_splits(sample_count: int, requested_splits: int) -> int:
    return resolve_kfold_splits(sample_count, requested_splits)


def _write_partition_leaderboards(rows: list[dict[str, object]], output, radial_mode: str) -> None:
    write_partition_leaderboards(rows, output, radial_mode)


def _write_utf8_bom_text(path, content: str) -> None:
    write_utf8_bom_text(path, content)


def main() -> int:
    parser = argparse.ArgumentParser(description="运行离线模型评估")
    parser.add_argument("--input-dir", default="./data/input/sample", help="输入数据目录")
    parser.add_argument("--output-dir", default="./data/output", help="输出目录")
    parser.add_argument(
        "--radial-mode",
        choices=["full", "edge_only", "both"],
        default="both",
        help="选择全部径向点，或仅使用轮毂/叶尖径向点",
    )
    parser.add_argument("--include-gpr", action="store_true", help="在评估中包含较慢的 GPR 模型")
    parser.add_argument("--max-samples", type=int, default=900, help="每种径向模式允许参与训练的最大样本数")
    parser.add_argument(
        "--partition-mode",
        choices=["single", "family", "component", "multi", "none", "component_stage"],
        default="multi",
        help="分区评估模式：推荐使用 multi；同时兼容旧别名 none/component_stage",
    )
    args = parser.parse_args()

    modes = ["full", "edge_only"] if args.radial_mode == "both" else [args.radial_mode]
    results = {}
    for mode in modes:
        results[mode] = run_benchmark(
            args.input_dir,
            args.output_dir,
            mode,
            include_gpr=args.include_gpr,
            max_samples=args.max_samples,
            partition_mode=args.partition_mode,
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


__all__ = [
    "AnySample",
    "EndpointRegression",
    "FlexSample",
    "HierarchicalWorklineRegressor",
    "LOWESSLikeRegressor",
    "MODEL_CATEGORY",
    "PhysicsClampedRegressor",
    "PiecewiseRidgeRegressor",
    "RBFRegressor",
    "SCHEMA_CONFIG",
    "Sample",
    "_append_ranked_rows",
    "_cv_splits",
    "_partition_samples",
    "_real_partition_sample_count",
    "_resolve_kfold_splits",
    "_write_partition_leaderboards",
    "_write_utf8_bom_text",
    "all_targets",
    "build_aggregate_rows",
    "build_benchmark_report",
    "cv_splits",
    "load_predict_samples",
    "load_samples",
    "main",
    "metrics",
    "partition_samples",
    "prepare_model_for_fold",
    "real_partition_sample_count",
    "resolve_1d_input_files",
    "resolve_kfold_splits",
    "run_benchmark",
    "run_benchmark_1d",
    "run_benchmark_2d",
    "schema_input_type",
    "to_arrays",
    "to_arrays_2d",
    "write_partition_leaderboards",
    "write_utf8_bom_text",
]


if __name__ == "__main__":
    raise SystemExit(main())
