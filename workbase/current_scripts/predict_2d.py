"""
2D 预测入口脚本。
当前场景只读取 rpm + xi，并预测目标性能列。
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import json
import sys

import joblib
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

from workbase.common.config_loader import load_config
from workbase.common.logger_config import setup_logger
from workbase.common.model_versioning import check_version_compatibility, load_model_metadata
from workbase.common.prediction_output import (
    TabularTemplate,
    detect_tabular_delimiter,
    write_table_like_template,
)
from workbase.common.validators import ValidationError, validate_directory_exists
from project1.services.data_reader import read_tabular_file
from project1.services.meta_parser import parse_metadata_from_path

CONFIG_PATH = "config/benchmark_config.yaml"
config = load_config(CONFIG_PATH)

logger = setup_logger("predict_2d", log_dir=config.log_dir)

PREDICT_PATH = Path(config.predict_2d_input_path)
OUTPUT_DIR = Path(config.predict_2d_model_output_dir)
PREDICTION_OUTPUT = Path(config.predict_2d_output_path)
RADIAL_MODE = config.radial_mode_2d
PARTITION = config.predict_2d_partition


@dataclass
class Predict2DSample:
    source_file: Path
    row_index: int
    component: str
    stage: int
    database: str
    rpm: float
    xi: float
    original_row: dict[str, object]


def _schema_alias_candidates(field: str) -> list[str]:
    aliases = config.schema_aliases.get(field, [])
    names = [field, *aliases]
    seen: set[str] = set()
    result: list[str] = []
    for name in names:
        normalized = str(name).strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(normalized)
    return result


def _normalize_column_name(name: str) -> str:
    return str(name).strip().replace("\ufeff", "")


def _match_column(columns: list[str], logical_field: str) -> str | None:
    lowered = {_normalize_column_name(col).lower(): col for col in columns}
    for candidate in _schema_alias_candidates(logical_field):
        matched = lowered.get(_normalize_column_name(candidate).lower())
        if matched is not None:
            return matched
    return None


def _list_predict_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    validate_directory_exists(path, "2D 预测目录")
    files = sorted(
        file_path
        for file_path in path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in {".txt", ".dat", ".csv"}
    )
    if not files:
        raise FileNotFoundError(f"在 {path} 下没有可预测文件")
    return files


def _infer_component(meta: dict[str, object | None], row: dict[str, object], columns: list[str]) -> str | None:
    col = _match_column(columns, "component")
    if col and row.get(col) is not None:
        return str(row[col]).upper()
    if meta.get("component") is not None:
        return str(meta["component"]).upper()
    return None


def _infer_stage(meta: dict[str, object | None], row: dict[str, object], columns: list[str]) -> int | None:
    col = _match_column(columns, "stage")
    if col and row.get(col) is not None:
        return int(float(row[col]))
    if meta.get("stage") is not None:
        return int(meta["stage"])
    return None


def _infer_database(meta: dict[str, object | None], component: str) -> str:
    if meta.get("database") is not None:
        return str(meta["database"])
    return f"DATABASE_{component}"


def _infer_rpm(meta: dict[str, object | None], row: dict[str, object], columns: list[str]) -> float | None:
    col = _match_column(columns, "rpm")
    if col and row.get(col) is not None:
        return float(row[col])
    if meta.get("rpm") is not None:
        return float(meta["rpm"])
    return None


def _infer_xi(row: dict[str, object], columns: list[str]) -> float | None:
    col = _match_column(columns, "xi")
    if col and row.get(col) is not None:
        return float(row[col])
    return None


def _partition_key(sample: Predict2DSample) -> str:
    return f"{sample.component}:S{sample.stage}"


def _normalize_partition(name: str) -> str:
    return name.replace(" ", "_").replace(":", "_")


def _build_samples() -> tuple[list[Predict2DSample], dict[Path, list[str]]]:
    samples: list[Predict2DSample] = []
    source_columns: dict[Path, list[str]] = {}
    for predict_file in _list_predict_files(PREDICT_PATH):
        parsed = read_tabular_file(str(predict_file))
        columns = [str(col) for col in parsed["columns"]]
        source_columns[predict_file] = columns
        meta = parse_metadata_from_path(str(predict_file))
        for row_index, row in enumerate(parsed["rows"]):
            component = _infer_component(meta, row, columns)
            stage = _infer_stage(meta, row, columns)
            if component is None or stage is None:
                continue
            rpm = _infer_rpm(meta, row, columns)
            xi = _infer_xi(row, columns)
            if rpm is None or xi is None:
                continue
            samples.append(
                Predict2DSample(
                    source_file=predict_file,
                    row_index=row_index,
                    component=component,
                    stage=stage,
                    database=_infer_database(meta, component),
                    rpm=float(rpm),
                    xi=float(xi),
                    original_row=dict(row),
                )
            )
    if not samples:
        raise ValueError("未解析到有效 2D 预测样本（需要 rpm + xi）")
    return samples, source_columns


def _load_leaderboard() -> list[dict[str, object]]:
    leaderboard_path = OUTPUT_DIR / f"leaderboard_2d_{RADIAL_MODE}.json"
    if not leaderboard_path.exists():
        raise FileNotFoundError(f"未找到 2D 排行榜: {leaderboard_path}")
    return json.loads(leaderboard_path.read_text(encoding="utf-8"))


def _targets_in_partition(partition: str) -> list[str]:
    partition_dir = OUTPUT_DIR / "models_2d" / _normalize_partition(partition)
    if not partition_dir.exists():
        return []
    targets = set()
    for model_file in partition_dir.glob("*.pkl"):
        stem = model_file.stem
        parts = stem.rsplit("_", 1)
        if len(parts) == 2:
            targets.add(parts[1])
    return sorted(targets)


def _find_best_model(
    leaderboard_rows: list[dict[str, object]],
    partition: str,
    target: str,
) -> tuple[str, dict[str, object]]:
    rows = [row for row in leaderboard_rows if row.get("partition") == partition and row.get("target") == target]
    if not rows:
        raise ValueError(f"未找到分区 {partition}、目标 {target} 的排行榜记录")
    random_rows = [row for row in rows if row.get("split_mode") == "random"]
    candidates = random_rows if random_rows else rows
    best = min(candidates, key=lambda row: float(row["rmse"]))
    return str(best["model"]), best


def _build_template(source_file: Path, base_columns: list[str], predicted_targets: list[str]) -> TabularTemplate:
    columns = [_normalize_column_name(col) for col in base_columns]
    existing_lower = {col.lower() for col in columns}
    for target in predicted_targets:
        if target.lower() not in existing_lower:
            columns.append(target)
    return TabularTemplate(
        source_path=source_file,
        columns=columns,
        delimiter=detect_tabular_delimiter(source_file),
        suffix=source_file.suffix or ".dat",
    )


def predict_2d():
    logger.info("=" * 70)
    logger.info("2D 预测开始（输入: rpm + xi）")
    logger.info("=" * 70)

    validate_directory_exists(OUTPUT_DIR, "2D 模型输出目录")

    samples, source_columns = _build_samples()
    logger.info(f"有效样本数: {len(samples)}")

    partition_to_indices: dict[str, list[int]] = defaultdict(list)
    if PARTITION:
        partition_to_indices[PARTITION] = list(range(len(samples)))
        logger.info(f"使用指定分区: {PARTITION}")
    else:
        for idx, sample in enumerate(samples):
            partition_to_indices[_partition_key(sample)].append(idx)
        logger.info(f"自动识别分区: {sorted(partition_to_indices)}")

    leaderboard_rows = _load_leaderboard()
    predictions: dict[str, np.ndarray] = {}

    for partition, indices in partition_to_indices.items():
        targets = _targets_in_partition(partition)
        if not targets:
            logger.warning(f"分区无可用模型，跳过: {partition}")
            continue
        x_partition = np.array([[samples[i].rpm, samples[i].xi] for i in indices], dtype=float)
        for target in targets:
            model_name, best_info = _find_best_model(leaderboard_rows, partition, target)
            model_path = OUTPUT_DIR / "models_2d" / _normalize_partition(partition) / f"{model_name}_{target}.pkl"
            if not model_path.exists():
                logger.warning(f"模型文件不存在，跳过: {model_path}")
                continue

            metadata = load_model_metadata(model_path)
            if metadata:
                is_compatible, warnings = check_version_compatibility(metadata)
                if not is_compatible:
                    logger.warning(f"模型版本不兼容，跳过: {model_path}")
                    continue
                for warning in warnings:
                    logger.warning(f"  {warning}")

            logger.info(
                f"分区 {partition} / 目标 {target}: 使用模型 {model_name} "
                f"(split={best_info.get('split_mode')}, rmse={float(best_info['rmse']):.6f})"
            )
            model = joblib.load(model_path)
            pred = model.predict(x_partition)
            if target not in predictions:
                predictions[target] = np.full(len(samples), np.nan, dtype=float)
            predictions[target][indices] = pred

    grouped_by_file: dict[Path, list[tuple[int, Predict2DSample]]] = defaultdict(list)
    for idx, sample in enumerate(samples):
        grouped_by_file[sample.source_file].append((idx, sample))

    for source_file, grouped_rows in grouped_by_file.items():
        predicted_targets = sorted(predictions.keys())
        template = _build_template(source_file, source_columns[source_file], predicted_targets)
        relative_path = source_file.relative_to(PREDICT_PATH) if PREDICT_PATH.is_dir() else Path(source_file.name)
        output_file = (PREDICTION_OUTPUT / relative_path).with_suffix(template.suffix)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        rows_to_write: list[dict[str, object]] = []
        for global_index, sample in grouped_rows:
            row_data: dict[str, object] = {}
            for column in template.columns:
                lower = column.lower()
                if lower == "rpm":
                    row_data[column] = sample.rpm
                elif lower == "xi":
                    row_data[column] = sample.xi
                elif column in predictions:
                    row_data[column] = predictions[column][global_index]
                else:
                    row_data[column] = sample.original_row.get(column, "")
            rows_to_write.append(row_data)

        write_table_like_template(output_file, template, rows_to_write)
        logger.info(f"输出文件: {output_file}")

    logger.info("=" * 70)
    logger.info("[OK] 2D 预测完成")
    logger.info("=" * 70)


def main():
    try:
        predict_2d()
    except ValidationError as exc:
        logger.error(f"校验失败: {exc}")
        raise


if __name__ == "__main__":
    main()
