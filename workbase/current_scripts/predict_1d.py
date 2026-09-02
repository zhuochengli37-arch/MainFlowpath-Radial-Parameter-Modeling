"""
1D 预测入口脚本。
当前场景只读取 rpm，并预测 wcor。
"""

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
from workbase.common.validators import ValidationError, validate_directory_exists, validate_file_exists
from project1.services.data_reader import read_tabular_file
from project1.services.meta_parser import parse_metadata_from_path

CONFIG_PATH = "config/benchmark_config.yaml"
config = load_config(CONFIG_PATH)

logger = setup_logger("predict_1d", log_dir=config.log_dir)

PREDICT_PATH = config.predict_1d_input_path
OUTPUT_DIR = config.predict_1d_model_output_dir
PREDICTION_OUTPUT = config.predict_1d_output_path
TARGET_NAME = "wcor"


def _normalize_column_name(name: str) -> str:
    return str(name).strip().replace("\ufeff", "")


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


def _match_column(columns: list[str], logical_field: str) -> str | None:
    lowered = {_normalize_column_name(col).lower(): col for col in columns}
    for candidate in _schema_alias_candidates(logical_field):
        matched = lowered.get(_normalize_column_name(candidate).lower())
        if matched is not None:
            return matched
    return None


def _resolve_rpm_column(columns: list[str]) -> str | None:
    return _match_column(columns, "rpm")


def _list_predict_files(path: Path) -> list[Path]:
    if path.is_file():
        validate_file_exists(path, "预测数据文件")
        return [path]
    validate_directory_exists(path, "预测数据目录")
    files = sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in {".txt", ".dat", ".csv"}
    )
    if not files:
        raise FileNotFoundError(f"在 {path} 中未找到可预测文件")
    return files


def _load_rpm_array(parsed: dict[str, object], source_file: Path) -> tuple[np.ndarray, list[float], str]:
    columns = [str(col) for col in parsed["columns"]]
    rpm_col = _resolve_rpm_column(columns)
    rpm_values: list[float] = []
    used_column = rpm_col or "path:rpm"
    if rpm_col is not None:
        for row in parsed["rows"]:
            value = row.get(rpm_col)
            if value is None:
                continue
            rpm_values.append(float(value))
    else:
        meta = parse_metadata_from_path(str(source_file))
        rpm_from_path = meta.get("rpm")
        if rpm_from_path is None:
            raise ValueError(f"文件中没有 rpm 列，且路径无法解析 rpm: {source_file}")
        for _ in parsed["rows"]:
            rpm_values.append(float(rpm_from_path))
    if not rpm_values:
        raise ValueError(f"未从文件中解析出 rpm 数据: {source_file}")
    x_array = np.array([[value] for value in rpm_values], dtype=float)
    return x_array, rpm_values, used_column


def _normalize_partition_name(partition: str) -> str:
    return partition.replace(" ", "_").replace(":", "_")


def _partition_from_source_file(source_file: Path) -> str:
    meta = parse_metadata_from_path(str(source_file))
    component = str(meta.get("component") or "").upper()
    stage = meta.get("stage")
    if not component:
        raise ValueError(f"路径中无法解析部件类型: {source_file}")

    if config.schema_partition_mode == "keys":
        parts: list[str] = []
        for key in config.schema_partition_keys:
            value = meta.get(str(key))
            if value is None or value == "":
                continue
            if str(key).lower() == "component":
                parts.append(str(value).upper())
            elif str(key).lower() == "stage":
                parts.append(f"S{int(float(value))}")
            else:
                parts.append(str(value))
        if parts:
            return ":".join(parts)

    if stage is None:
        raise ValueError(f"路径中无法解析级号: {source_file}")
    return f"{component}:S{int(float(stage))}"


def _find_best_model_1d(output_dir: Path, target: str, partition: str) -> tuple[str, dict[str, object]]:
    leaderboard_path = output_dir / "leaderboard.json"
    if not leaderboard_path.exists():
        raise FileNotFoundError(f"排行榜文件不存在: {leaderboard_path}")
    leaderboard = json.loads(leaderboard_path.read_text(encoding="utf-8"))
    partitioned = any(str(item.get("partition") or "") for item in leaderboard)
    candidates = [item for item in leaderboard if str(item.get("target")) == target]
    if partitioned:
        candidates = [item for item in candidates if str(item.get("partition")) == partition]
    if not candidates:
        available = sorted(
            {str(item.get("partition")) for item in leaderboard if str(item.get("partition") or "")}
        )
        raise ValueError(f"排行榜中未找到分区 {partition!r}、目标 {target!r}；可用分区: {available}")
    best = min(candidates, key=lambda item: float(item["rmse"]))
    return str(best["model"]), best


def _build_template(source_file: Path, parsed_columns: list[str]) -> TabularTemplate:
    columns = [_normalize_column_name(col) for col in parsed_columns]
    if TARGET_NAME not in {col.lower() for col in columns}:
        columns.append(TARGET_NAME)
    return TabularTemplate(
        source_path=source_file,
        columns=columns,
        delimiter=detect_tabular_delimiter(source_file),
        suffix=source_file.suffix or ".dat",
    )


def predict_1d_file(predict_file: Path, output_dir: Path):
    logger.info("=" * 70)
    logger.info(f"1D 预测文件: {predict_file}")
    logger.info("=" * 70)

    parsed = read_tabular_file(str(predict_file))
    if not parsed["columns"]:
        raise ValueError(f"文件没有列名: {predict_file}")

    x_array, rpm_values, rpm_source = _load_rpm_array(parsed, predict_file)
    logger.info(f"  输入来源: {rpm_source}")
    logger.info(f"  样本数: {len(x_array)}")

    partition = _partition_from_source_file(predict_file)
    model_name, best_info = _find_best_model_1d(output_dir, TARGET_NAME, partition)
    selected_partition = str(best_info.get("partition") or "")
    logger.info(f"  自动识别分区: {partition}")
    logger.info(f"  自动选择模型: {model_name} (RMSE={best_info['rmse']:.6f})")

    if selected_partition:
        model_path = (
            output_dir
            / "models"
            / _normalize_partition_name(selected_partition)
            / f"{model_name}_{TARGET_NAME}.pkl"
        )
    else:
        logger.warning("检测到旧版未分区流量模型；重新训练后可按部件和级号选择模型。")
        model_path = output_dir / "models" / f"{model_name}_{TARGET_NAME}.pkl"
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    metadata = load_model_metadata(model_path)
    if metadata:
        is_compatible, warnings = check_version_compatibility(metadata)
        if not is_compatible:
            raise RuntimeError(f"模型版本不兼容: {model_path}")
        for warning in warnings:
            logger.warning(f"  {warning}")

    model = joblib.load(model_path)
    prediction_values = model.predict(x_array)

    template = _build_template(predict_file, [str(col) for col in parsed["columns"]])
    output_root = Path(PREDICTION_OUTPUT)
    relative_path = predict_file.relative_to(Path(PREDICT_PATH)) if Path(PREDICT_PATH).is_dir() else Path(predict_file.name)
    output_file = (output_root / relative_path).with_suffix(template.suffix)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    rows_to_write: list[dict[str, object]] = []
    rpm_col = _resolve_rpm_column([str(col) for col in parsed["columns"]])
    for index, row in enumerate(parsed["rows"]):
        out_row: dict[str, object] = {}
        for col in template.columns:
            normalized = str(col).lower()
            if normalized == "rpm":
                out_row[col] = row.get(rpm_col) if rpm_col else rpm_values[index]
            elif normalized == TARGET_NAME:
                out_row[col] = float(prediction_values[index])
            else:
                out_row[col] = row.get(col, "")
        rows_to_write.append(out_row)

    write_table_like_template(output_file, template, rows_to_write)
    logger.info(f"  预测结果已保存: {output_file}")


def main():
    try:
        output_path = Path(OUTPUT_DIR)
        validate_directory_exists(output_path, "模型输出目录")
        predict_files = _list_predict_files(Path(PREDICT_PATH))
        logger.info(f"共找到 {len(predict_files)} 个预测文件")
        for file_path in predict_files:
            predict_1d_file(file_path, output_path)
    except ValidationError as exc:
        logger.error(f"校验失败: {exc}")
        raise


if __name__ == "__main__":
    main()
