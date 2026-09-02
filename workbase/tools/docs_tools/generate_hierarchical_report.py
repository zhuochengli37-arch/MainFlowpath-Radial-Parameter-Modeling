"""
按部件与分区生成层次化 benchmark 报告。
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np


def _extract_component(partition: str) -> str:
    if ":" in partition:
        return partition.split(":", 1)[0]
    if "_" in partition:
        return partition.split("_", 1)[0]
    return partition


def _weighted_model_scores(items: list[dict]) -> dict[str, dict[str, float]]:
    model_scores: dict[str, dict[str, list[float]]] = defaultdict(
        lambda: {"rmse": [], "mape": [], "mae": [], "r2": [], "weights": []}
    )
    for item in items:
        model = str(item["model"])
        model_scores[model]["rmse"].append(float(item["rmse"]))
        model_scores[model]["mape"].append(float(item["mape"]))
        model_scores[model]["mae"].append(float(item["mae"]))
        model_scores[model]["r2"].append(float(item["r2"]))
        model_scores[model]["weights"].append(float(item["partition_size"]))

    aggregated: dict[str, dict[str, float]] = {}
    for model, scores in model_scores.items():
        weights = np.array(scores["weights"], dtype=float)
        normalized = weights / np.sum(weights)
        aggregated[model] = {
            "rmse": float(np.sum(normalized * np.array(scores["rmse"], dtype=float))),
            "mape": float(np.sum(normalized * np.array(scores["mape"], dtype=float))),
            "mae": float(np.sum(normalized * np.array(scores["mae"], dtype=float))),
            "r2": float(np.sum(normalized * np.array(scores["r2"], dtype=float))),
        }
    return aggregated


def _real_partition_sample_count(items: list[dict], partition: str) -> int:
    for item in items:
        if str(item["partition"]) == partition:
            return int(item["partition_size"])
    return 0


def _write_utf8_bom_text(path: Path, content: str) -> None:
    path.write_text(content, encoding="utf-8-sig")


def generate_hierarchical_report(output_dir: str, radial_mode: str = "mean") -> None:
    output_path = Path(output_dir)
    leaderboard_path = output_path / f"leaderboard_{radial_mode}.json"
    if not leaderboard_path.exists():
        print(f"错误：未找到排行榜文件：{leaderboard_path}")
        return

    leaderboard = json.loads(leaderboard_path.read_text(encoding="utf-8"))
    component_groups: dict[str, dict[tuple[str, str], list[dict]]] = defaultdict(lambda: defaultdict(list))

    for item in leaderboard:
        component = _extract_component(str(item["partition"]))
        key = (str(item["target"]), str(item["split_mode"]))
        component_groups[component][key].append(item)

    targets = sorted({str(item["target"]) for item in leaderboard})
    split_modes = ["random", "rpm", "wcor"]

    sample_count = 0
    if targets:
        random_partitions = {str(item["partition"]) for item in leaderboard if item["target"] == targets[0] and item["split_mode"] == "random"}
        sample_count = sum(_real_partition_sample_count(leaderboard, partition) for partition in sorted(random_partitions))

    lines = [
        f"# 层次化模型评估报告（{radial_mode}）",
        "",
        f"- 样本数量: {sample_count}",
        f"- 部件数量: {len(component_groups)}",
        "",
    ]

    for component in sorted(component_groups):
        component_data = component_groups[component]
        partitions = sorted({str(item["partition"]) for items in component_data.values() for item in items})
        total_samples = sum(_real_partition_sample_count(leaderboard, partition) for partition in partitions)

        lines.append(f"## {component}（{len(partitions)} 个分区，{total_samples} 个样本）")
        lines.append("")
        lines.append("### 整体最佳模型（按 RMSE）")
        lines.append("")

        for target in targets:
            for split_mode in split_modes:
                key = (target, split_mode)
                if key not in component_data:
                    continue
                weighted_scores = _weighted_model_scores(component_data[key])
                if not weighted_scores:
                    continue
                best_model_name, best_metrics = min(weighted_scores.items(), key=lambda entry: entry[1]["rmse"])
                lines.append(
                    f"- {target}/{split_mode}: {best_model_name} "
                    f"(RMSE={best_metrics['rmse']:.6f}, MAPE={best_metrics['mape']:.6f}, "
                    f"MAE={best_metrics['mae']:.6f}, R²={best_metrics['r2']:.6f})"
                )

        lines.append("")
        lines.append("### 整体最佳模型（按 MAPE）")
        lines.append("")

        for target in targets:
            for split_mode in split_modes:
                key = (target, split_mode)
                if key not in component_data:
                    continue
                weighted_scores = _weighted_model_scores(component_data[key])
                if not weighted_scores:
                    continue
                best_model_name, best_metrics = min(weighted_scores.items(), key=lambda entry: entry[1]["mape"])
                lines.append(
                    f"- {target}/{split_mode}: {best_model_name} "
                    f"(MAPE={best_metrics['mape']:.6f}, RMSE={best_metrics['rmse']:.6f}, "
                    f"MAE={best_metrics['mae']:.6f}, R²={best_metrics['r2']:.6f})"
                )

        lines.append("")
        lines.append("### 分区详情")
        lines.append("")

        for partition in partitions:
            partition_size = _real_partition_sample_count(leaderboard, partition)
            lines.append(f"#### {partition}（{partition_size} 个样本）")
            lines.append("")

            for target in targets:
                for split_mode in split_modes:
                    key = (target, split_mode)
                    if key not in component_data:
                        continue
                    partition_items = [item for item in component_data[key] if str(item["partition"]) == partition]
                    if not partition_items:
                        continue
                    best_rmse = min(partition_items, key=lambda item: item["rmse"])
                    best_mape = min(partition_items, key=lambda item: item["mape"])
                    lines.append(
                        f"- {target}/{split_mode} 按 RMSE 最优：{best_rmse['model']} "
                        f"(RMSE={best_rmse['rmse']:.6f}, MAPE={best_rmse['mape']:.6f}, "
                        f"MAE={best_rmse['mae']:.6f}, R²={best_rmse['r2']:.6f})"
                    )
                    lines.append(
                        f"- {target}/{split_mode} 按 MAPE 最优：{best_mape['model']} "
                        f"(MAPE={best_mape['mape']:.6f}, RMSE={best_mape['rmse']:.6f}, "
                        f"MAE={best_mape['mae']:.6f}, R²={best_mape['r2']:.6f})"
                    )

            lines.append("")

        lines.append("")

    report_path = output_path / f"benchmark_report_{radial_mode}_hierarchical.md"
    _write_utf8_bom_text(report_path, "\n".join(lines) + "\n")
    print(f"层次化报告已写入：{report_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法：python workbase/tools/docs_tools/generate_hierarchical_report.py <output_dir> [radial_mode]")
        print("示例：python workbase/tools/docs_tools/generate_hierarchical_report.py ./data/output/offline_data/DATACASE1 mean")
        raise SystemExit(1)

    target_output_dir = sys.argv[1]
    target_radial_mode = sys.argv[2] if len(sys.argv) > 2 else "mean"
    generate_hierarchical_report(target_output_dir, target_radial_mode)
