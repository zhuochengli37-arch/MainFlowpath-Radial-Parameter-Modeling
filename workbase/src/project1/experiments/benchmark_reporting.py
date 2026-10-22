from __future__ import annotations

import json
import csv
from collections import defaultdict
from pathlib import Path

import numpy as np


RADIAL_SPLIT_MODES = ("condition", "random", "rpm", "wcor")

FORMAL_METRICS = ("RMSE", "MAE", "MAPE", "R2", "NRMSE_range")
FORMAL_ID_FIELDS = ("experiment_id", "run_id", "case_id", "model_key", "family", "paper_family_code",
                    "component", "stage", "section", "target", "dataset", "split_hash", "split_seed",
                    "train_condition_hash", "validation_condition_hash", "test_condition_hash",
                    "stochasticity_type", "model_seed", "repeat_index")
FORMAL_RESULT_FIELDS = {
    "model_accuracy": (*FORMAL_ID_FIELDS, "evaluation_subset", *FORMAL_METRICS, "NRMSE_range_reason",
                       "target_range", "sample_count", "train_sample_count", "status"),
    "model_runtime": (*FORMAL_ID_FIELDS, "fit_time", "validation_inference_time", "test_inference_time",
                      "memory", "memory_note", "hardware", "status"),
    "model_metadata": (*FORMAL_ID_FIELDS, "implementation_status", "version", "parameters", "feature_order",
                       "target_definition", "model_path", "metadata_path", "status"),
    "model_summary": (*FORMAL_ID_FIELDS, "accuracy_record", "runtime_record", "record_path", "status",
                      "failure_stage", "exception_type", "reason"),
}


def formal_metrics(y_true: np.ndarray, y_pred: np.ndarray, metric_protocol: dict) -> dict[str, object]:
    from project1.experiments.benchmark_partitioning import metrics

    truth = np.asarray(y_true, dtype=float).reshape(-1)
    prediction = np.asarray(y_pred, dtype=float).reshape(-1)
    if not len(truth) or truth.shape != prediction.shape or not np.isfinite(truth).all() or not np.isfinite(prediction).all():
        raise ValueError("formal metrics require equal-length, nonempty, finite truth/prediction arrays")
    base = metrics(truth, prediction)
    result = {name: base[name.lower()] if np.isfinite(base[name.lower()]) else None for name in FORMAL_METRICS[:4]}
    rule = metric_protocol["definitions"]["NRMSE_range"]["undefined_behavior"]
    target_range = float(np.ptp(truth))
    tolerance = rule["absolute_tolerance"] + rule["relative_tolerance"] * max(1.0, float(np.max(np.abs(truth))))
    undefined = target_range <= tolerance
    result.update(NRMSE_range=None if undefined else result["RMSE"] / target_range,
                  NRMSE_range_reason=rule["reason"] if undefined else None, target_range=target_range,
                  sample_count=len(truth))
    return result


def _write_formal_csv(path: Path, fields: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, ensure_ascii=False, sort_keys=True) if isinstance(value, (dict, list)) else value
                             for key, value in row.items()})


def rebuild_formal_reports(output_dir: str | Path) -> dict[str, object]:
    """Rebuild tables from durable per-run records, without fitting or choosing models."""
    from project1.experiments.benchmark_instance import write_benchmark_json, load_formal_records

    output = Path(output_dir)
    experiment = json.loads((output / "experiment.json").read_text(encoding="utf-8"))
    records = list(load_formal_records(output, experiment).values())
    tables = {name: [] for name in FORMAL_RESULT_FIELDS}
    groups = defaultdict(list)
    manifest = []
    for record in records:
        identity = {key: record.get(key) for key in FORMAL_ID_FIELDS}
        groups[(record["case_id"], record["model_key"])].append(record)
        for subset, values in record.get("evaluations", {}).items():
            tables["model_accuracy"].append({**identity, **values, "evaluation_subset": subset,
                                             "train_sample_count": record["train_sample_count"], "status": record["status"]})
        tables["model_runtime"].append({**identity, **record.get("runtime", {}), "status": record["status"]})
        tables["model_metadata"].append({**identity, **record.get("metadata", {}), "status": record["status"]})
        tables["model_summary"].append({**identity, "accuracy_record": "model_accuracy.csv",
                                        "runtime_record": "model_runtime.csv", "record_path": f"records/{record['run_id']}.json",
                                        **{key: record.get(key) for key in ("status", "failure_stage", "exception_type", "reason")}})
        if record["status"] == "passed":
            manifest.append(record["artifact"])
    for name, fields in FORMAL_RESULT_FIELDS.items():
        _write_formal_csv(output / "results" / f"{name}.csv", fields, tables[name])
    aggregate = []
    for _, runs in sorted(groups.items()):
        first = runs[0]
        expected_seeds = set(experiment["model_seeds"][first["model_key"]])
        complete = (len(runs) == first["scheduled_runs"] == len(expected_seeds)
                    and {r["model_seed"] for r in runs} == expected_seeds
                    and all(r["status"] == "passed" for r in runs))
        for subset in ("validation", "test"):
            row = {key: first.get(key) for key in FORMAL_ID_FIELDS if key not in {"run_id", "model_seed", "repeat_index"}}
            row.update(evaluation_subset=subset, scheduled_runs=first["scheduled_runs"], recorded_runs=len(runs),
                       successful_runs=sum(r["status"] == "passed" for r in runs),
                       status="complete" if complete else "incomplete")
            for metric in FORMAL_METRICS:
                values = [r.get("evaluations", {}).get(subset, {}).get(metric) for r in runs]
                valid = complete and all(v is not None for v in values)
                row[f"{metric}_mean"] = float(np.mean(values)) if valid else None
                row[f"{metric}_sample_std"] = float(np.std(values, ddof=1)) if valid and len(values) > 1 else None
            aggregate.append(row)
    aggregate_fields = tuple(aggregate[0]) if aggregate else ("case_id", "model_key", "evaluation_subset", "status")
    _write_formal_csv(output / "results" / "model_accuracy_aggregate.csv", aggregate_fields, aggregate)
    write_benchmark_json(output / "models" / "models_manifest.json", manifest)
    summary = {"experiment_id": experiment["experiment_id"], "purpose": experiment["purpose"],
               "planned_fits": experiment["planned_fits"], "recorded_runs": len(records),
               "passed_runs": sum(r["status"] == "passed" for r in records),
               "failed_runs": sum(r["status"] == "failed" for r in records),
               "incomplete_runs": sum(r["status"] not in {"passed", "failed"} for r in records),
               "missing_runs": experiment["planned_fits"] - len(records),
               "complete": len(records) == experiment["planned_fits"] and all(r["status"] == "passed" for r in records),
               "dataset_sha256": experiment["inventory_sha256"], "split_hash": experiment["split_hash"],
               "selection_run": False, "sample_efficiency_run": False}
    write_benchmark_json(output / "reports" / "benchmark_summary.json", summary)
    write_benchmark_json(output / "reports" / "failures.json", [r for r in records if r["status"] != "passed"])
    return summary


def real_partition_sample_count(items: list[dict[str, object]], partition: str) -> int:
    for item in items:
        if str(item["partition"]) == partition:
            return int(item["partition_size"])
    return 0


def write_utf8_bom_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8-sig")


def _safe_int(value: object, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _cv_stats(row: dict[str, object]) -> tuple[int, int, int, int]:
    samples = _safe_int(row.get("partition_size", 0))
    folds = _safe_int(row.get("folds", 0))
    cv_samples = samples * folds if folds > 0 else 0
    train_exposure = samples * max(folds - 1, 0) if folds > 0 else 0
    return folds, samples, cv_samples, train_exposure


def _cv_stats_text(row: dict[str, object]) -> str:
    folds, samples, cv_samples, train_exposure = _cv_stats(row)
    return (
        f"folds={folds}, samples={samples}, "
        f"cv_samples={cv_samples}, train_exposure={train_exposure}"
    )


def append_ranked_rows(
    summary_lines: list[str],
    rows: list[dict[str, object]],
    target: str,
    split_mode: str,
    ranking_metric: str,
) -> None:
    subset = [row for row in rows if row["target"] == target and row["split_mode"] == split_mode]
    if not subset:
        return

    ranked_rows = sorted(
        subset,
        key=lambda row: (
            float(row[ranking_metric]),
            float(row["rmse"]),
            float(row["mape"]),
            str(row["model"]),
        ),
    )
    summary_lines.append(f"##### {target}/{split_mode} 按 {ranking_metric.upper()} 排序")
    for index, row in enumerate(ranked_rows, start=1):
        summary_lines.append(
            f"- 第 {index} 名: {row['model']}（{row['model_category']}）"
            f"（RMSE={float(row['rmse']):.6f}, MAPE={float(row['mape']):.6f}, "
            f"MAE={float(row['mae']):.6f}, R²={float(row['r2']):.6f}, {_cv_stats_text(row)}）"
        )
    summary_lines.append("")


def write_partition_leaderboards(rows: list[dict[str, object]], output: Path, radial_mode: str) -> None:
    partitions: dict[str, list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        partitions[str(row["partition"])].append(row)

    for partition, items in partitions.items():
        safe_partition = str(partition).replace(" ", "_").replace(":", "_")
        partition_dir = output / safe_partition
        partition_dir.mkdir(parents=True, exist_ok=True)

        partition_leaderboard_path = partition_dir / f"leaderboard_{radial_mode}.json"
        partition_leaderboard_path.write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")

        summary_lines = [
            f"# 分区模型评估报告（{radial_mode}）- {partition}",
            "",
            f"- 分区: {partition}",
            f"- 样本数量: {real_partition_sample_count(items, partition)}",
            "- CV 统计: samples=真实样本数; cv_samples=samples*folds; train_exposure=samples*(folds-1)",
            "",
            "## 各目标与交叉验证方式的最优模型",
        ]
        partition_targets = sorted({str(row["target"]) for row in items})
        for target in partition_targets:
            for split_mode in RADIAL_SPLIT_MODES:
                subset = [row for row in items if row["target"] == target and row["split_mode"] == split_mode]
                if not subset:
                    continue
                best_rmse = min(subset, key=lambda row: row["rmse"])
                best_mape = min(subset, key=lambda row: row["mape"])
                summary_lines.append(
                    f"- {target}/{split_mode} 按 RMSE 最优: {best_rmse['model']}（{best_rmse['model_category']}）"
                    f"（RMSE={best_rmse['rmse']:.6f}, MAPE={best_rmse['mape']:.6f}, "
                    f"MAE={best_rmse['mae']:.6f}, R²={best_rmse['r2']:.6f}, {_cv_stats_text(best_rmse)}）"
                )
                summary_lines.append(
                    f"- {target}/{split_mode} 按 MAPE 最优: {best_mape['model']}（{best_mape['model_category']}）"
                    f"（MAPE={best_mape['mape']:.6f}, RMSE={best_mape['rmse']:.6f}, "
                    f"MAE={best_mape['mae']:.6f}, R²={best_mape['r2']:.6f}, {_cv_stats_text(best_mape)}）"
                )
        summary_lines.append("")
        summary_lines.append("## 全部模型误差明细（按 RMSE 排序）")
        for target in partition_targets:
            for split_mode in RADIAL_SPLIT_MODES:
                append_ranked_rows(summary_lines, items, target, split_mode, "rmse")
        summary_lines.append("## 全部模型误差明细（按 MAPE 排序）")
        for target in partition_targets:
            for split_mode in RADIAL_SPLIT_MODES:
                append_ranked_rows(summary_lines, items, target, split_mode, "mape")
        partition_report_path = partition_dir / f"benchmark_report_{radial_mode}.md"
        write_utf8_bom_text(partition_report_path, "\n".join(summary_lines) + "\n")


def build_aggregate_rows(rows: list[dict[str, object]], model_category: dict[str, str]) -> list[dict[str, object]]:
    aggregate_map: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
    for row in rows:
        key = (str(row["target"]), str(row["split_mode"]), str(row["model"]))
        aggregate_map[key].append(row)

    aggregate_rows: list[dict[str, object]] = []
    for (target, split_mode, model), items in aggregate_map.items():
        weights = np.array([float(item["partition_size"]) for item in items], dtype=float)
        if np.sum(weights) <= 0:
            continue
        normalized = weights / np.sum(weights)
        aggregate_rows.append(
            {
                "target": target,
                "split_mode": split_mode,
                "model": model,
                "model_category": model_category.get(model, "其他"),
                "partitions": [str(item["partition"]) for item in items],
                "weighted_mae": float(np.sum(normalized * np.array([float(item["mae"]) for item in items]))),
                "weighted_rmse": float(np.sum(normalized * np.array([float(item["rmse"]) for item in items]))),
                "weighted_mape": float(np.sum(normalized * np.array([float(item["mape"]) for item in items]))),
                "weighted_r2": float(np.sum(normalized * np.array([float(item["r2"]) for item in items]))),
            }
        )
    return sorted(aggregate_rows, key=lambda row: (row["target"], row["split_mode"], row["weighted_rmse"]))


def build_benchmark_report(
    samples: list[object],
    input_dir: str,
    partition_mode: str,
    targets: tuple[str, ...],
    component_source_rows: list[dict[str, object]],
    model_category: dict[str, str],
    radial_mode: str,
) -> str:
    summary_lines = [
        f"# 模型评估报告（{radial_mode}）",
        "",
        f"- 样本数量: {len(samples)}",
        f"- 输入目录: {input_dir}",
        f"- 分区模式: {partition_mode}",
        "- CV 统计: samples=真实样本数; cv_samples=samples*folds; train_exposure=samples*(folds-1)",
        "",
    ]

    component_groups: dict[str, list[dict[str, object]]] = defaultdict(list)
    for item in component_source_rows:
        partition_key = str(item["partition"])
        component = partition_key.split(":")[0] if ":" in partition_key else partition_key.split("_")[0]
        component_groups[component].append(item)

    for component in sorted(component_groups):
        items = component_groups[component]
        partitions = sorted(set(str(item["partition"]) for item in items))
        total_samples = sum(real_partition_sample_count(items, partition) for partition in partitions)

        summary_lines.append(f"## {component}（共 {len(partitions)} 个级，{total_samples} 个样本）")
        summary_lines.append("")

        component_rows: list[dict[str, object]] = []
        component_aggregate: dict[tuple[str, str, str], list[dict[str, object]]] = defaultdict(list)
        for item in items:
            component_aggregate[(item["target"], item["split_mode"], item["model"])].append(item)

        for (target, split_mode, model), partition_items in component_aggregate.items():
            weights = np.array([float(item["partition_size"]) for item in partition_items], dtype=float)
            if np.sum(weights) <= 0:
                continue
            normalized = weights / np.sum(weights)
            folds_array = np.array([_safe_int(item.get("folds", 0)) for item in partition_items], dtype=float)
            size_array = np.array([_safe_int(item.get("partition_size", 0)) for item in partition_items], dtype=float)
            component_rows.append(
                {
                    "target": target,
                    "split_mode": split_mode,
                    "model": model,
                    "model_category": model_category.get(model, "其他"),
                    "rmse": float(np.sum(normalized * np.array([float(item["rmse"]) for item in partition_items]))),
                    "mape": float(np.sum(normalized * np.array([float(item["mape"]) for item in partition_items]))),
                    "mae": float(np.sum(normalized * np.array([float(item["mae"]) for item in partition_items]))),
                    "r2": float(np.sum(normalized * np.array([float(item["r2"]) for item in partition_items]))),
                    "partition_size": int(np.sum(size_array)),
                    "folds": int(round(float(np.sum(normalized * folds_array)))),
                    "cv_samples": int(np.sum(size_array * folds_array)),
                    "train_exposure": int(np.sum(size_array * np.maximum(folds_array - 1, 0))),
                }
            )

        summary_lines.append(f"### {component} 整体最佳模型（按 RMSE）")
        for target in targets:
            for split_mode in RADIAL_SPLIT_MODES:
                subset = [row for row in component_rows if row["target"] == target and row["split_mode"] == split_mode]
                if not subset:
                    continue
                best_rmse = min(subset, key=lambda row: (float(row["rmse"]), float(row["mape"]), str(row["model"])))
                summary_lines.append(
                    f"- {target}/{split_mode}: {best_rmse['model']}（{best_rmse['model_category']}）"
                    f"（RMSE={best_rmse['rmse']:.6f}, MAPE={best_rmse['mape']:.6f}, "
                    f"MAE={best_rmse['mae']:.6f}, R²={best_rmse['r2']:.6f}, "
                    f"folds={best_rmse['folds']}, samples={best_rmse['partition_size']}, "
                    f"cv_samples={best_rmse['cv_samples']}, train_exposure={best_rmse['train_exposure']}）"
                )
        summary_lines.append("")

        summary_lines.append(f"### {component} 整体最佳模型（按 MAPE）")
        for target in targets:
            for split_mode in RADIAL_SPLIT_MODES:
                subset = [row for row in component_rows if row["target"] == target and row["split_mode"] == split_mode]
                if not subset:
                    continue
                best_mape = min(subset, key=lambda row: (float(row["mape"]), float(row["rmse"]), str(row["model"])))
                summary_lines.append(
                    f"- {target}/{split_mode}: {best_mape['model']}（{best_mape['model_category']}）"
                    f"（MAPE={best_mape['mape']:.6f}, RMSE={best_mape['rmse']:.6f}, "
                    f"MAE={best_mape['mae']:.6f}, R²={best_mape['r2']:.6f}, "
                    f"folds={best_mape['folds']}, samples={best_mape['partition_size']}, "
                    f"cv_samples={best_mape['cv_samples']}, train_exposure={best_mape['train_exposure']}）"
                )
        summary_lines.append("")

        summary_lines.append(f"### {component} 各级详细误差")
        for partition in partitions:
            partition_items = [item for item in items if item["partition"] == partition]
            partition_size = real_partition_sample_count(items, partition)
            summary_lines.append(f"#### {partition}（{partition_size} 个样本）")
            summary_lines.append("##### 按 RMSE 排序")
            for target in targets:
                for split_mode in RADIAL_SPLIT_MODES:
                    append_ranked_rows(summary_lines, partition_items, target, split_mode, "rmse")
            summary_lines.append("##### 按 MAPE 排序")
            for target in targets:
                for split_mode in RADIAL_SPLIT_MODES:
                    append_ranked_rows(summary_lines, partition_items, target, split_mode, "mape")
        summary_lines.append("")

    summary_lines.append("## 模型分类说明")
    for model_key, category in sorted(model_category.items()):
        summary_lines.append(f"- {model_key}: {category}")

    return "\n".join(summary_lines) + "\n"
