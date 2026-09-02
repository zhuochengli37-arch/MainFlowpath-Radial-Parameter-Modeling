from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np


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
            for split_mode in ("random", "rpm", "wcor"):
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
            for split_mode in ("random", "rpm", "wcor"):
                append_ranked_rows(summary_lines, items, target, split_mode, "rmse")
        summary_lines.append("## 全部模型误差明细（按 MAPE 排序）")
        for target in partition_targets:
            for split_mode in ("random", "rpm", "wcor"):
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
            for split_mode in ("random", "rpm", "wcor"):
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
            for split_mode in ("random", "rpm", "wcor"):
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
                for split_mode in ("random", "rpm", "wcor"):
                    append_ranked_rows(summary_lines, partition_items, target, split_mode, "rmse")
            summary_lines.append("##### 按 MAPE 排序")
            for target in targets:
                for split_mode in ("random", "rpm", "wcor"):
                    append_ranked_rows(summary_lines, partition_items, target, split_mode, "mape")
        summary_lines.append("")

    summary_lines.append("## 模型分类说明")
    for model_key, category in sorted(model_category.items()):
        summary_lines.append(f"- {model_key}: {category}")

    return "\n".join(summary_lines) + "\n"
