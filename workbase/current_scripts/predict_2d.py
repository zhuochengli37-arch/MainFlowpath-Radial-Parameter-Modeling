"""Legacy file wrapper around the shared 2D radial prediction core."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import replace
import json
from pathlib import Path
import sys
from typing import Iterable

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
    TabularTemplate,
    detect_tabular_delimiter,
    write_table_like_template,
)
from workbase.common.validators import ValidationError, validate_directory_exists


CONFIG_PATH = "config/benchmark_config.yaml"
config = load_config(CONFIG_PATH)
logger = setup_logger("predict_2d", log_dir=config.log_dir)

PREDICT_PATH = Path(config.predict_2d_input_path)
OUTPUT_DIR = Path(config.predict_2d_model_output_dir)
PREDICTION_OUTPUT = Path(config.predict_2d_output_path)
RADIAL_MODE = config.radial_mode_2d
PARTITION = config.predict_2d_partition

# Import compatibility for callers that used the old script-local class.
Predict2DSample = PredictionSample


def _normalize_column_name(name: str) -> str:
    return str(name).strip().replace("\ufeff", "")


def _partition_key(sample: PredictionSample) -> str:
    return str(radial_partition(sample))


def _normalize_partition(name: str) -> str:
    return safe_partition_name(name)


def _build_samples() -> tuple[list[PredictionSample], dict[Path, list[str]]]:
    """Compatibility wrapper; path parsing stops at the normalized batch boundary."""

    batch = load_prediction_batch(PREDICT_PATH, radial_mode=RADIAL_MODE, require_flow=False)
    return list(batch.samples), {
        Path(path): list(columns) for path, columns in batch.source_columns.items()
    }


def _targets_in_partition(partition: str) -> list[str]:
    registry = ModelRegistry(OUTPUT_DIR, "2D", radial_mode=RADIAL_MODE)
    return list(registry.available_targets(parse_partition_key(partition)))


def _find_best_model(
    leaderboard_rows: list[dict[str, object]],
    partition: str,
    target: str,
) -> tuple[str, dict[str, object]]:
    """Backward-compatible pure helper retained for external callers/tests."""

    rows = [
        row
        for row in leaderboard_rows
        if row.get("partition") == partition and row.get("target") == target
    ]
    if not rows:
        raise ValueError(f"未找到分区 {partition}、目标 {target} 的排行榜记录")
    random_rows = [row for row in rows if row.get("split_mode") == "random"]
    candidates = random_rows if random_rows else rows
    best = min(candidates, key=lambda row: float(row["rmse"]))
    return str(best["model"]), best


def _build_template(
    source_file: Path,
    base_columns: list[str],
    predicted_targets: list[str],
) -> TabularTemplate:
    columns = [_normalize_column_name(column) for column in base_columns]
    existing = {column.casefold() for column in columns}
    for target in predicted_targets:
        if target.casefold() not in existing:
            columns.append(target)
    return TabularTemplate(
        source_path=source_file,
        columns=columns,
        delimiter=detect_tabular_delimiter(source_file),
        suffix=source_file.suffix or ".dat",
    )


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


def _write_canonical_debug(
    results: list[PredictionResult],
    output_root: Path,
) -> Path:
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / "canonical_prediction_results_2d.json"
    path.write_text(
        json.dumps([result.as_dict() for result in results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return path


def _write_legacy_single_results(
    samples: list[PredictionSample],
    source_columns: dict[str, tuple[str, ...]],
    results: list[PredictionResult],
    predict_root: Path,
    output_root: Path,
) -> None:
    values = {(result.sample_index, result.target): result.predicted_value for result in results}
    grouped: dict[str, list[tuple[int, PredictionSample]]] = defaultdict(list)
    for index, sample in enumerate(samples):
        grouped[sample.source_file].append((index, sample))
    for source_name, indexed_samples in grouped.items():
        source_file = Path(source_name)
        targets = list(
            dict.fromkeys(
                result.target for result in results if result.source_file == source_name
            )
        )
        template = _build_template(source_file, list(source_columns[source_name]), targets)
        relative = source_file.relative_to(predict_root) if predict_root.is_dir() else Path(source_file.name)
        output_file = (output_root / relative).with_suffix(template.suffix)
        rows: list[dict[str, object]] = []
        for sample_index, sample in indexed_samples:
            row: dict[str, object] = {}
            for column in template.columns:
                normalized = column.casefold()
                if normalized == "rpm":
                    row[column] = sample.speed_parameter
                elif normalized == "xi":
                    row[column] = sample.xi
                elif (sample_index, column) in values:
                    row[column] = values[(sample_index, column)]
                else:
                    row[column] = sample.original_row.get(column, "")
            rows.append(row)
        write_table_like_template(output_file, template, rows)
        logger.info(f"输出文件: {output_file}")


def predict_2d(
    *,
    targets: str | Iterable[str] = "auto",
    missing_target_policy: str = "error",
    model_name: str | None = None,
) -> list[PredictionResult]:
    logger.info("=" * 70)
    logger.info("2D 预测开始（输入: speed_parameter + xi）")
    logger.info("=" * 70)
    validate_directory_exists(OUTPUT_DIR, "2D 模型输出目录")

    batch = load_prediction_batch(PREDICT_PATH, radial_mode=RADIAL_MODE, require_flow=False)
    for exclusion in batch.exclusions:
        logger.warning(f"排除预测文件: {exclusion.source_file} ({exclusion.reason})")
    samples = _legacy_partition_override(list(batch.samples), PARTITION)
    if not samples:
        raise ValueError("未解析到有效MAIN 2D预测样本（需要speed_parameter + xi）")
    registry = ModelRegistry(
        OUTPUT_DIR,
        "2D",
        radial_mode=RADIAL_MODE,
        legacy_partition_override="all" if PARTITION == "all" else None,
    )
    results = predict_radial(
        samples,
        registry,
        dimension="2D",
        targets=targets,
        missing_target_policy=missing_target_policy,
        selection_metric="rmse",
        selection_split_mode="random",
        model_name=model_name,
    )

    if any(sample.section is not None for sample in samples):
        path = _write_canonical_debug(results, PREDICTION_OUTPUT)
        logger.info(f"Multi-Section canonical调试结果: {path}")
        logger.info("最终研究所Multi-Section宽表输出尚未实现")
    else:
        _write_legacy_single_results(
            samples,
            batch.source_columns,
            results,
            PREDICT_PATH,
            PREDICTION_OUTPUT,
        )
    logger.info(f"[OK] 2D预测完成，共{len(results)}条canonical结果")
    return results


def main() -> None:
    try:
        predict_2d()
    except ValidationError as exc:
        logger.error(f"校验失败: {exc}")
        raise


if __name__ == "__main__":
    main()
