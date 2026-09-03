from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from tqdm import tqdm

from project1.experiments.benchmark_data import (
    AnySample,
    OperatingCondition,
    load_operating_conditions,
    resolve_1d_input_files,
    schema_input_type,
)
from project1.experiments.model_partitioning import safe_partition_name, workline_partition
from project1.experiments.benchmark_partitioning import prepare_model_for_fold, resolve_kfold_splits
from project1.experiments.benchmark_reporting import write_utf8_bom_text
from project1.experiments.training_metadata import artifact_identity, build_training_context
from workbase.common.model_versioning import create_model_metadata, save_model_with_metadata
from project1.modeling.factory import build_1d_models


def _sample_partition_key(sample: AnySample) -> str:
    return str(workline_partition(sample))


def _safe_partition_name(partition: str) -> str:
    return safe_partition_name(partition)


def _load_1d_samples(input_files: list[Path]) -> tuple[list[OperatingCondition], Path]:
    first_input = input_files[0]
    if first_input.is_dir():
        return load_operating_conditions(first_input), first_input

    if len(input_files) == 1 and first_input.parent.is_dir() and any(first_input.parent.rglob("*.dat")):
        return load_operating_conditions(first_input.parent), first_input.parent

    selected_paths = {path.resolve() for path in input_files}
    samples: list[OperatingCondition] = []
    for path in input_files:
        samples.extend(load_operating_conditions(path))
    samples = [sample for sample in samples if Path(sample.source_path).resolve() in selected_paths]
    input_source = first_input.parent if len(input_files) > 1 else first_input
    return samples, input_source


def _extract_partitioned_rpm_wcor_pairs(
    samples: list[AnySample],
) -> dict[str, tuple[np.ndarray, np.ndarray, dict[str, object]]]:
    if not samples:
        raise ValueError("no usable samples found in 3D-like directory input")

    pairs_by_partition: dict[str, dict[float, float]] = defaultdict(dict)
    details_by_partition: dict[str, dict[str, object]] = {}
    for sample in samples:
        # Current proxy models are defined only for the aerodynamic MAIN
        # station. This is intentionally station-based, not stage 0/999 based.
        if str(getattr(sample, "station", "MAIN")).upper() != "MAIN":
            continue
        partition = _sample_partition_key(sample)
        rpm = float(sample.rpm)
        wcor = float(sample.wcor)
        existing = pairs_by_partition[partition].get(rpm)
        if existing is not None and not np.isclose(existing, wcor, rtol=1e-9, atol=1e-12):
            raise ValueError(
                f"partition {partition!r} has multiple wcor values for rpm={rpm}: "
                f"{existing} and {wcor}; rpm -> wcor is not single-valued"
            )
        pairs_by_partition[partition][rpm] = wcor
        details_by_partition.setdefault(
            partition,
            {"component": sample.component, "stage": int(sample.stage)},
        )

    if not pairs_by_partition:
        raise ValueError("no station=MAIN samples found for 1D training")

    result: dict[str, tuple[np.ndarray, np.ndarray, dict[str, object]]] = {}
    for partition, rpm_to_wcor in sorted(pairs_by_partition.items()):
        ordered = sorted(rpm_to_wcor.items(), key=lambda item: item[0])
        result[partition] = (
            np.array([[rpm] for rpm, _ in ordered], dtype=float),
            np.array([wcor for _, wcor in ordered], dtype=float),
            details_by_partition[partition],
        )
    return result


def _subsample_partition(
    x_array: np.ndarray,
    y_array: np.ndarray,
    max_samples: int,
) -> tuple[np.ndarray, np.ndarray]:
    if len(x_array) <= max_samples:
        return x_array, y_array
    rng = np.random.default_rng(42)
    indices = np.sort(rng.choice(len(x_array), max_samples, replace=False))
    return x_array[indices], y_array[indices]


def _evaluate_partition(
    partition: str,
    x_array: np.ndarray,
    y_array: np.ndarray,
    models: dict[str, object],
    effective_splits: int,
    details: dict[str, object],
    logger: Any | None,
    pbar: tqdm,
) -> list[dict[str, float | str | int]]:
    splits = list(KFold(n_splits=effective_splits, shuffle=True, random_state=42).split(x_array))
    results: list[dict[str, float | str | int]] = []
    for model_name, model in models.items():
        scores = {"mae": [], "rmse": [], "mape": [], "r2": []}
        for train_idx, test_idx in splits:
            x_train, x_test = x_array[train_idx], x_array[test_idx]
            y_train, y_test = y_array[train_idx], y_array[test_idx]
            try:
                candidate = prepare_model_for_fold(model, len(train_idx))
                candidate.fit(x_train, y_train)
                y_pred = candidate.predict(x_test)
                scores["mae"].append(mean_absolute_error(y_test, y_pred))
                scores["rmse"].append(np.sqrt(mean_squared_error(y_test, y_pred)))
                denom = np.maximum(np.abs(y_test), 1e-9)
                scores["mape"].append(float(np.mean(np.abs((y_test - y_pred) / denom))))
                scores["r2"].append(float(r2_score(y_test, y_pred)) if len(y_test) >= 2 else float("nan"))
            except Exception as exc:
                if logger is not None:
                    logger.warning(f"    {partition}/{model_name} failed: {exc}")

        completed_folds = len(scores["mae"])
        if completed_folds == effective_splits:
            r2_values = np.asarray(scores["r2"], dtype=float)
            results.append(
                {
                    "partition": partition,
                    "model_dimension": "1D",
                    "component": str(details["component"]),
                    "stage": int(details["stage"]),
                    "station": "MAIN",
                    "section": None,
                    "target": "wcor",
                    "model": model_name,
                    "path": f"models/{_safe_partition_name(partition)}/{model_name}_wcor.pkl",
                    "folds": completed_folds,
                    "partition_size": int(len(x_array)),
                    "cv_samples": int(len(x_array) * effective_splits),
                    "train_exposure": int(len(x_array) * (effective_splits - 1)),
                    "mae": float(np.mean(scores["mae"])),
                    "rmse": float(np.mean(scores["rmse"])),
                    "mape": float(np.mean(scores["mape"])),
                    "r2": float(np.nanmean(r2_values)) if not np.all(np.isnan(r2_values)) else float("nan"),
                }
            )
        elif completed_folds and logger is not None:
            logger.warning(
                f"    Skip {partition}/{model_name}: completed {completed_folds}/{effective_splits} folds"
            )
        pbar.update(1)
        pbar.set_postfix({"partition": partition, "model": model_name})
    return results


def run_benchmark_1d(
    input_files: list[Path],
    output_dir: str,
    n_splits: int,
    max_samples: int,
    include_gpr: bool,
    logger: Any | None = None,
) -> list[dict[str, float | str | int]]:
    if not input_files:
        raise ValueError("no input files provided")

    if logger is not None:
        logger.info("[1/5] Loading component-partitioned rpm->wcor data...")

    samples, input_source = _load_1d_samples(input_files)
    extracted_partitions = _extract_partitioned_rpm_wcor_pairs(samples)
    source_samples_by_partition: dict[str, list[OperatingCondition]] = defaultdict(list)
    for sample in samples:
        if sample.station.upper() == "MAIN":
            source_samples_by_partition[_sample_partition_key(sample)].append(sample)
    partition_data: dict[str, tuple[np.ndarray, np.ndarray, dict[str, object], int]] = {}
    for partition, (x_array, y_array, details) in extracted_partitions.items():
        x_array, y_array = _subsample_partition(x_array, y_array, max_samples)
        effective_splits = resolve_kfold_splits(len(x_array), n_splits)
        if effective_splits < 2:
            if logger is not None:
                logger.warning(f"  Skip {partition}: fewer than 2 unique rpm->wcor samples")
            continue
        partition_data[partition] = (x_array, y_array, details, effective_splits)
        if logger is not None:
            logger.info(
                f"  Partition {partition}: samples={len(x_array)}, "
                f"requested_folds={n_splits}, effective_folds={effective_splits}"
            )

    if not partition_data:
        raise ValueError("no component partition has enough unique rpm->wcor samples for 1D training")

    models = build_1d_models(include_gpr=include_gpr)
    if logger is not None:
        logger.info("[2/5] Cross-validation by component partition (rpm -> wcor)...")

    results: list[dict[str, float | str | int]] = []
    total_work = len(models) * len(partition_data)
    with tqdm(total=total_work, desc="模型评估", unit="模型") as pbar:
        for partition, (x_array, y_array, details, effective_splits) in partition_data.items():
            results.extend(
                _evaluate_partition(
                    partition=partition,
                    x_array=x_array,
                    y_array=y_array,
                    models=models,
                    effective_splits=effective_splits,
                    details=details,
                    logger=logger,
                    pbar=pbar,
                )
            )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    condition_manifest = []
    for partition, partition_samples in sorted(source_samples_by_partition.items()):
        unique_pairs = {
            (sample.speed_parameter, sample.flow_parameter)
            for sample in partition_samples
        }
        condition_manifest.append(
            {
                "model_dimension": "1D",
                "component": partition_samples[0].component,
                "stage": partition_samples[0].stage,
                "station": "MAIN",
                "section": None,
                "partition": partition,
                "source_file_count": len({sample.source_file for sample in partition_samples}),
                "operating_condition_record_count": len(partition_samples),
                "unique_speed_flow_points": len(unique_pairs),
                "conflicting_speed_count": 0,
            }
        )
    condition_manifest_path = output_path / "operating_conditions_manifest.json"
    condition_manifest_path.write_text(
        json.dumps(condition_manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    if logger is not None:
        logger.info("[3/5] Writing partitioned leaderboard...")
    leaderboard_path = output_path / "leaderboard.json"
    sorted_results = sorted(results, key=lambda item: (str(item["partition"]), float(item["rmse"])))
    leaderboard_path.write_text(json.dumps(sorted_results, ensure_ascii=False, indent=2), encoding="utf-8")
    report_path = output_path / "benchmark_report_1d.md"
    write_utf8_bom_text(
        report_path,
        _build_1d_report(
            sorted_results=sorted_results,
            input_source=Path(input_source),
            requested_folds=n_splits,
            partition_data=partition_data,
        ),
    )

    if logger is not None:
        logger.info("[4/5] Training final models by component partition...")
    models_dir = output_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    saved_models_info: list[dict[str, object]] = []
    with tqdm(total=total_work, desc="保存模型", unit="模型") as pbar:
        for partition, (x_array, y_array, details, effective_splits) in partition_data.items():
            partition_models_dir = models_dir / _safe_partition_name(partition)
            partition_models_dir.mkdir(parents=True, exist_ok=True)
            for model_name, model in models.items():
                try:
                    final_model = prepare_model_for_fold(model, len(x_array))
                    final_model.fit(x_array, y_array)
                    model_path = partition_models_dir / f"{model_name}_wcor.pkl"
                    best_result = next(
                        (
                            row
                            for row in results
                            if row["partition"] == partition and row["model"] == model_name
                        ),
                        None,
                    )
                    metrics = (
                        {
                            "mae": float(best_result["mae"]),
                            "rmse": float(best_result["rmse"]),
                            "r2": float(best_result["r2"]),
                        }
                        if best_result
                        else None
                    )
                    model_context = build_training_context(
                        source_samples_by_partition[partition],
                        model_dimension="1D",
                        partition=partition,
                        inputs=["speed_parameter"],
                        target="wcor",
                        training_source_root=input_source,
                        sample_count=len(x_array),
                    )
                    metadata = create_model_metadata(
                        model_name=model_name,
                        model_type="1D",
                        input_type=schema_input_type(((0, "rpm"),)),
                        data_path=Path(input_source),
                        config={
                            "partition_mode": "component_stage",
                            "partition_key": partition,
                            "requested_n_splits": n_splits,
                            "effective_n_splits": effective_splits,
                            "input_column": "rpm",
                            "input_name": "rpm",
                            "output_column": "wcor",
                            "include_gpr": include_gpr,
                            "train_samples": len(x_array),
                            "data_layout": "3d_like_directory",
                            "cv_samples": len(x_array) * effective_splits,
                            "train_exposure": len(x_array) * (effective_splits - 1),
                            "input_files": [str(path) for path in input_files],
                        },
                        metrics=metrics,
                        additional_info={
                            "component": details["component"],
                            "stage": details["stage"],
                        },
                        model_context=model_context,
                    )
                    save_model_with_metadata(final_model, model_path, metadata)
                    saved_models_info.append(
                        {
                            **artifact_identity(model_context),
                            "target": "wcor",
                            "model": model_name,
                            "path": str(model_path.relative_to(output_path)),
                            "train_samples": int(len(x_array)),
                            "sample_count": int(len(x_array)),
                            "metrics": metrics,
                            "cv_samples": int(len(x_array) * effective_splits),
                            "train_exposure": int(len(x_array) * max(effective_splits - 1, 0)),
                        }
                    )
                except Exception as exc:
                    if logger is not None:
                        logger.warning(f"  failed to save {partition}/{model_name}_wcor: {exc}")
                pbar.update(1)
                pbar.set_postfix({"partition": partition, "model": model_name})

    models_manifest_path = models_dir / "models_manifest.json"
    models_manifest_path.write_text(json.dumps(saved_models_info, ensure_ascii=False, indent=2), encoding="utf-8")

    if logger is not None:
        logger.info(f"  Models saved to {models_dir}")
        logger.info(f"  Models manifest saved to {models_manifest_path}")
        logger.info(f"  Operating conditions manifest saved to {condition_manifest_path}")
        logger.info(f"  Report saved to {report_path}")
        logger.info("[5/5] Done")
    return sorted_results


def resolve_1d_dataset_inputs(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        return [input_path]
    return resolve_1d_input_files(input_path)


def _build_1d_report(
    sorted_results: list[dict[str, float | str | int]],
    input_source: Path,
    requested_folds: int,
    partition_data: dict[str, tuple[np.ndarray, np.ndarray, dict[str, object], int]],
) -> str:
    lines: list[str] = [
        "# 1D Benchmark 评估报告",
        "",
        f"- 输入源: `{input_source}`",
        "- 输入: `rpm`",
        "- 输出: `wcor`",
        "- 分区方式: `component + stage`",
        f"- 请求折数: {requested_folds}",
        f"- 有效分区数: {len(partition_data)}",
        "",
    ]

    if not sorted_results:
        lines.extend(["## 结果", "", "- 无可用模型结果。", ""])
        return "\n".join(lines)

    for partition, (_, _, _, effective_folds) in partition_data.items():
        partition_results = [row for row in sorted_results if row["partition"] == partition]
        if not partition_results:
            continue
        sample_count = int(partition_results[0]["partition_size"])
        best_rmse = min(partition_results, key=lambda item: float(item["rmse"]))
        best_mape = min(partition_results, key=lambda item: float(item.get("mape", float("inf"))))
        lines.extend(
            [
                f"## 分区 {partition}",
                "",
                f"- 实际折数: {effective_folds}",
                f"- 样本数: {sample_count}",
                f"- 按 RMSE 最优: `{best_rmse['model']}` "
                f"(RMSE={float(best_rmse['rmse']):.6f}, MAPE={float(best_rmse['mape']):.6f})",
                f"- 按 MAPE 最优: `{best_mape['model']}` "
                f"(MAPE={float(best_mape['mape']):.6f}, RMSE={float(best_mape['rmse']):.6f})",
                "",
                "| 排名 | 模型 | RMSE | MAPE | MAE | R² | folds | samples |",
                "|---:|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for index, item in enumerate(partition_results, start=1):
            lines.append(
                f"| {index} | `{item['model']}` | {float(item['rmse']):.6f} | "
                f"{float(item['mape']):.6f} | {float(item['mae']):.6f} | "
                f"{float(item['r2']):.6f} | {int(item['folds'])} | {int(item['partition_size'])} |"
            )
        lines.append("")
    return "\n".join(lines)
