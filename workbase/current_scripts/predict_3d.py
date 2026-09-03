"""Legacy file wrapper around the shared 3D radial prediction core."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Iterable

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKBASE = PROJECT_ROOT / "workbase"
SRC = WORKBASE / "src"
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from workbase.common.runtime_env import ensure_project_venv

if __name__ == "__main__":
    ensure_project_venv(PROJECT_ROOT)

from project1.experiments.model_partitioning import (
    parse_partition_key,
    radial_partition,
    safe_partition_name,
)
from project1.experiments.model_registry import ModelRegistry
from project1.experiments.prediction_results import PredictionResult
from project1.experiments.prediction_samples import PredictionSample, load_prediction_batch
from project1.experiments.radial_prediction import predict_radial
from workbase.common.config_loader import load_config
from workbase.common.logger_config import setup_logger
from workbase.common.prediction_output import (
    find_matching_template_file,
    load_tabular_template,
    write_table_like_template,
)
from workbase.common.validators import ValidationError, validate_directory_exists


CONFIG_PATH = "config/benchmark_config.yaml"
config = load_config(CONFIG_PATH)
logger = setup_logger("predict_3d", log_dir=config.log_dir)

PREDICT_PATH = config.predict_3d_input_path
OUTPUT_DIR = config.predict_3d_model_output_dir
PREDICTION_OUTPUT = config.predict_3d_output_path
TRAIN_TEMPLATE_ROOT = Path(config.benchmark_3d_input_path)
RADIAL_MODE = config.radial_mode
PARTITION = config.predict_3d_partition
MODEL_NAME = config.predict_3d_model_name
SELECTION_SPLIT_MODE = config.predict_3d_selection_split_mode
SELECTION_METRIC = config.predict_3d_selection_metric


def _schema_input_field(index: int, default: str) -> str:
    inputs = config.schema_inputs
    return str(inputs[index]) if index < len(inputs) else default


def _value_for_template_column(
    sample: PredictionSample,
    column: str,
    predictions: dict[str, np.ndarray],
    row_index: int,
) -> object:
    normalized = str(column).strip().casefold()
    if normalized == _schema_input_field(2, "xi").casefold():
        return sample.xi
    if normalized == _schema_input_field(0, "rpm").casefold():
        return sample.speed_parameter
    if normalized == _schema_input_field(1, "wcor").casefold():
        return sample.flow_parameter
    values = predictions.get(column)
    return values[row_index] if values is not None else ""


def _partition_from_sample(sample: PredictionSample) -> str:
    return str(radial_partition(sample))


def sample_partition_key(sample: PredictionSample) -> str:
    return _partition_from_sample(sample)


def normalize_partition_name(name: str) -> str:
    return safe_partition_name(name)


def get_targets_from_models(output_dir: Path, partition: str) -> list[str]:
    """Compatibility helper backed by the shared registry when possible."""

    logical = partition
    if ":" not in logical:
        tokens = logical.split("_", 2)
        if len(tokens) >= 2 and tokens[1].upper().startswith("S"):
            logical = ":".join(tokens)
    registry = ModelRegistry(output_dir, "3D", radial_mode=RADIAL_MODE)
    targets = list(registry.available_targets(parse_partition_key(logical)))
    if not targets:
        raise ValueError(f"分区 {logical} 没有可用3D模型")
    return targets


def load_partition_leaderboard_rows(output_dir: Path, radial_mode: str) -> list[dict[str, object]]:
    leaderboard_path = output_dir / f"leaderboard_{radial_mode}.json"
    if not leaderboard_path.exists():
        raise FileNotFoundError(f"排行榜文件不存在: {leaderboard_path}")
    payload = json.loads(leaderboard_path.read_text(encoding="utf-8"))
    return [dict(row) for row in payload]


def _validate_selection_config(selection_split_mode: str, selection_metric: str) -> None:
    valid_split_modes = {"random", "rpm", "wcor", "any"}
    valid_metrics = {"rmse", "mape", "mae"}
    if selection_split_mode not in valid_split_modes:
        raise ValueError(
            f"predict_3d.selection_split_mode 无效: {selection_split_mode}. "
            f"可选值: {sorted(valid_split_modes)}"
        )
    if selection_metric not in valid_metrics:
        raise ValueError(
            f"predict_3d.selection_metric 无效: {selection_metric}. "
            f"可选值: {sorted(valid_metrics)}"
        )


def find_best_model_3d_for_partition(
    leaderboard_rows: list[dict],
    target: str,
    partition: str,
    selection_split_mode: str,
    selection_metric: str,
):
    """Backward-compatible pure leaderboard helper."""

    candidates = [
        row
        for row in leaderboard_rows
        if row.get("target") == target
        and row.get("partition") == partition
        and (
            selection_split_mode == "any"
            or row.get("split_mode") == selection_split_mode
        )
    ]
    if not candidates:
        raise ValueError(
            f"在排行榜中未找到分区 '{partition}'、目标 '{target}'、"
            f"split_mode='{selection_split_mode}' 的模型"
        )
    if any(selection_metric not in row for row in candidates):
        raise ValueError(f"排行榜中缺少选择指标 '{selection_metric}'")
    best = min(candidates, key=lambda row: float(row[selection_metric]))
    return best["model"], best


def warn_if_legacy_all_partition_present(
    output_dir: Path,
    effective_partitions: dict[str, list[int]],
) -> None:
    legacy = output_dir / "models" / "all"
    if legacy.exists() and any(key != "all" for key in effective_partitions):
        logger.warning("检测到旧models/all；共享core仍按样本component+stage选择，不自动fallback")


def _legacy_partition_override(
    samples: list[PredictionSample],
    partition: str | None,
) -> list[PredictionSample]:
    if not partition:
        return samples
    if any(sample.section is not None for sample in samples):
        raise ValueError("explicit legacy partition override is not allowed for Multi-Section input")
    if partition == "all":
        return samples
    parsed = parse_partition_key(partition)
    if parsed.section is not None:
        raise ValueError("legacy Single partition override cannot contain section")
    return [replace(sample, component=parsed.component, stage=parsed.stage) for sample in samples]


def _write_canonical_debug(results: list[PredictionResult], output_root: Path) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "canonical_prediction_results_3d.json"
    path.write_text(
        json.dumps([result.as_dict() for result in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _write_legacy_single_results(
    samples: list[PredictionSample],
    results: list[PredictionResult],
    predict_root: Path,
    output_root: Path,
) -> None:
    target_names = list(dict.fromkeys(result.target for result in results))
    predictions = {
        target: np.full(len(samples), np.nan, dtype=float) for target in target_names
    }
    for result in results:
        predictions[result.target][result.sample_index] = result.predicted_value
    grouped: dict[tuple[str, int, float, float], list[tuple[int, PredictionSample]]] = defaultdict(list)
    for index, sample in enumerate(samples):
        grouped[
            (
                sample.database,
                sample.stage,
                sample.speed_parameter,
                float(sample.flow_parameter),
            )
        ].append((index, sample))

    for (database, stage, speed, flow), sample_group in grouped.items():
        template_file = find_matching_template_file(
            TRAIN_TEMPLATE_ROOT,
            predict_file=sample_group[0][1].source_file,
            predict_root=predict_root,
        )
        template = load_tabular_template(template_file)
        output_dir = output_root / database / f"STAGE_{stage}" / f"RPM_{speed}"
        output_file = output_dir / f"{flow:.5f}{template.suffix}"
        rows = [
            {
                column: _value_for_template_column(sample, column, predictions, index)
                for column in template.columns
            }
            for index, sample in sample_group
        ]
        write_table_like_template(output_file, template, rows)


def predict_3d(
    predict_dir: str,
    output_dir: Path,
    radial_mode: str,
    partition: str | None,
    model_name: str | None,
    selection_split_mode: str,
    selection_metric: str,
    targets: str | Iterable[str] = "auto",
    missing_target_policy: str = "error",
) -> list[PredictionResult]:
    logger.info("=" * 70)
    logger.info("3D输入预测（speed_parameter + flow_parameter + xi）")
    logger.info("=" * 70)
    _validate_selection_config(selection_split_mode, selection_metric)
    validate_directory_exists(Path(predict_dir), "预测数据目录")
    validate_directory_exists(output_dir, "模型输出目录")

    batch = load_prediction_batch(predict_dir, radial_mode=radial_mode, require_flow=True)
    for exclusion in batch.exclusions:
        logger.warning(f"排除预测文件: {exclusion.source_file} ({exclusion.reason})")
    samples = _legacy_partition_override(list(batch.samples), partition)
    if not samples:
        raise ValueError("预测目录中没有有效MAIN 3D预测样本")
    registry = ModelRegistry(
        output_dir,
        "3D",
        radial_mode=radial_mode,
        legacy_partition_override="all" if partition == "all" else None,
    )
    results = predict_radial(
        samples,
        registry,
        dimension="3D",
        targets=targets,
        missing_target_policy=missing_target_policy,
        selection_metric=selection_metric,
        selection_split_mode=selection_split_mode,
        model_name=model_name,
    )

    output_root = Path(PREDICTION_OUTPUT)
    if any(sample.section is not None for sample in samples):
        path = _write_canonical_debug(results, output_root)
        logger.info(f"Multi-Section canonical调试结果: {path}")
        logger.info("最终研究所Multi-Section宽表输出尚未实现")
    else:
        _write_legacy_single_results(
            samples,
            results,
            Path(predict_dir),
            output_root,
        )
    logger.info(f"3D预测完成，共{len(results)}条canonical结果")
    return results


def main() -> None:
    try:
        predict_3d(
            predict_dir=PREDICT_PATH,
            output_dir=Path(OUTPUT_DIR),
            radial_mode=RADIAL_MODE,
            partition=PARTITION,
            model_name=MODEL_NAME,
            selection_split_mode=SELECTION_SPLIT_MODE,
            selection_metric=SELECTION_METRIC,
        )
    except ValidationError as exc:
        logger.error(f"校验失败: {exc}")
        raise


if __name__ == "__main__":
    main()
