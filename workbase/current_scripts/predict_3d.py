"""
3D 预测入口脚本。

使用已保存的 3D 模型预测目录结构化的离线/在线数据，
并按结构化输出格式写回结果。
"""

from collections import defaultdict
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
    find_matching_template_file,
    load_tabular_template,
    write_table_like_template,
)
from workbase.common.validators import (
    ValidationError,
    validate_array_shape,
    validate_directory_exists,
    validate_numeric_array,
)
from project1.experiments.benchmark_data import load_predict_samples

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
    if index < len(inputs):
        return str(inputs[index])
    return default


def _value_for_template_column(sample, column: str, predictions: dict[str, np.ndarray], row_index: int) -> object:
    normalized = str(column).strip().lower()
    if normalized == _schema_input_field(2, "xi").lower():
        return sample.xi
    if normalized == _schema_input_field(0, "rpm").lower():
        return sample.rpm
    if normalized == _schema_input_field(1, "wcor").lower():
        return sample.wcor
    values = predictions.get(column)
    if values is not None:
        return values[row_index]
    return ""


def _partition_from_sample(sample) -> str:
    if config.schema_partition_mode == "keys":
        parts: list[str] = []
        for key in config.schema_partition_keys:
            value = getattr(sample, key, None)
            if value is None:
                continue
            if str(key).lower() == "stage":
                parts.append(f"S{int(value)}")
            else:
                parts.append(str(value))
        return ":".join(parts) if parts else "all"
    return f"{sample.component}:S{sample.stage}"


def get_targets_from_models(output_dir: Path, partition: str):
    models_dir = output_dir / "models" / partition
    if not models_dir.exists():
        raise FileNotFoundError(f"模型目录不存在: {models_dir}")

    targets = set()
    for model_file in models_dir.glob("*.pkl"):
        parts = model_file.stem.rsplit("_", 1)
        if len(parts) == 2:
            targets.add(parts[1])

    if not targets:
        raise ValueError(f"在 {models_dir} 中未找到任何模型文件")

    targets_list = sorted(targets)
    logger.info(f"  自动检测到目标列: {targets_list}")
    return targets_list


def load_partition_leaderboard_rows(output_dir: Path, radial_mode: str):
    leaderboard_path = output_dir / f"leaderboard_{radial_mode}.json"
    if not leaderboard_path.exists():
        raise FileNotFoundError(f"排行榜文件不存在: {leaderboard_path}")

    with open(leaderboard_path, "r", encoding="utf-8") as f:
        return json.load(f)


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
    candidates = [
        item
        for item in leaderboard_rows
        if item["target"] == target and item["partition"] == partition
    ]
    if selection_split_mode != "any":
        candidates = [item for item in candidates if item.get("split_mode") == selection_split_mode]

    if not candidates:
        raise ValueError(
            f"在排行榜中未找到分区 '{partition}'、目标 '{target}'、"
            f"split_mode='{selection_split_mode}' 的模型"
        )

    missing_metric = [item for item in candidates if selection_metric not in item]
    if missing_metric:
        raise ValueError(f"排行榜中缺少选择指标 '{selection_metric}'")

    best = min(candidates, key=lambda item: item[selection_metric])
    return best["model"], best


def sample_partition_key(sample) -> str:
    return _partition_from_sample(sample)


def normalize_partition_name(name: str) -> str:
    return name.replace(" ", "_").replace(":", "_")


def warn_if_legacy_all_partition_present(output_dir: Path, effective_partitions: dict[str, list[int]]):
    models_dir = output_dir / "models"
    legacy_all_dir = models_dir / "all"
    if not legacy_all_dir.exists():
        return

    multi_partition_keys = [key for key in effective_partitions if key != "all"]
    if not multi_partition_keys:
        return

    logger.warning("检测到旧的单分区模型目录 models/all，与当前多分区预测样本同时存在。")
    logger.warning("当前将优先按样本所属分区进行预测，而不是使用 legacy all 分区模型。")
    logger.warning(f"自动识别到的分区有: {sorted(multi_partition_keys)}，如需强制指定，请设置 predict_3d.partition。")


def predict_3d(
    predict_dir: str,
    output_dir: Path,
    radial_mode: str,
    partition: str | None,
    model_name: str | None,
    selection_split_mode: str,
    selection_metric: str,
):
    logger.info("=" * 70)
    logger.info("3D 输入预测")
    logger.info("=" * 70)

    _validate_selection_config(selection_split_mode, selection_metric)

    validate_directory_exists(Path(predict_dir), "预测数据目录")
    validate_directory_exists(output_dir, "模型输出目录")

    logger.info("[1/4] 加载预测数据...")
    predict_samples = load_predict_samples(predict_dir, radial_mode)
    logger.info(f"  加载了 {len(predict_samples)} 个样本")
    if not predict_samples:
        raise ValueError(f"预测目录中没有找到数据: {predict_dir}")

    x_predict = np.array([[s.rpm, s.wcor, s.xi] for s in predict_samples], dtype=float)
    validate_array_shape(x_predict, expected_dims=2, min_samples=1, array_name="预测输入数据")
    validate_numeric_array(x_predict, array_name="预测输入数据")

    if partition is not None:
        effective_partitions = {partition: list(range(len(predict_samples)))}
        logger.info(f"使用指定分区: {partition}")
    else:
        effective_partitions = defaultdict(list)
        for index, sample in enumerate(predict_samples):
            effective_partitions[sample_partition_key(sample)].append(index)
        logger.info(f"自动识别到预测分区: {sorted(effective_partitions)}")
        warn_if_legacy_all_partition_present(output_dir, effective_partitions)

    leaderboard_rows = load_partition_leaderboard_rows(output_dir, radial_mode)
    logger.info(f"模型选择规则: split_mode={selection_split_mode}, metric={selection_metric}")

    all_targets = set()
    for partition_key in effective_partitions:
        all_targets.update(get_targets_from_models(output_dir, partition=normalize_partition_name(partition_key)))
    targets = sorted(all_targets)

    all_predictions = {target: np.full(len(predict_samples), np.nan, dtype=float) for target in targets}

    logger.info("[2/4] 按分区加载模型并预测...")
    for partition_key, indices in effective_partitions.items():
        normalized_partition = normalize_partition_name(partition_key)
        logger.info(f"  分区: {partition_key} ({len(indices)} 个样本)")
        partition_targets = get_targets_from_models(output_dir, partition=normalized_partition)
        partition_inputs = x_predict[indices]

        for target in partition_targets:
            if model_name is None:
                best_model_name, best_info = find_best_model_3d_for_partition(
                    leaderboard_rows,
                    target=target,
                    partition=partition_key,
                    selection_split_mode=selection_split_mode,
                    selection_metric=selection_metric,
                )
                logger.info(
                    f"    目标 {target}: 自动选择最优模型 {best_model_name} "
                    f"(split={best_info.get('split_mode')}, {selection_metric.upper()}={best_info[selection_metric]:.6f})"
                )
            else:
                best_model_name = model_name
                logger.info(f"    目标 {target}: 使用指定模型 {best_model_name}")

            model_path = output_dir / "models" / normalized_partition / f"{best_model_name}_{target}.pkl"
            if not model_path.exists():
                logger.warning(f"    [WARNING] 模型文件不存在: {model_path}")
                continue

            metadata = load_model_metadata(model_path)
            if metadata:
                is_compatible, warnings = check_version_compatibility(metadata)
                if not is_compatible:
                    logger.error("    [ERROR] 模型版本不兼容")
                    for warning in warnings:
                        logger.error(f"      - {warning}")
                    continue
                for warning in warnings:
                    logger.warning(f"      - {warning}")
            else:
                logger.warning("    [WARNING] 未找到模型元数据，无法验证版本兼容性")

            model = joblib.load(model_path)
            predictions = model.predict(partition_inputs)
            all_predictions[target][indices] = predictions
            logger.info(f"    目标 {target}: 预测完成 {len(predictions)} 个样本")

    logger.info("[3/4] 保存预测结果...")
    output_base = Path(PREDICTION_OUTPUT)
    grouped = defaultdict(list)
    for i, sample in enumerate(predict_samples):
        key = (sample.database, sample.stage, sample.rpm, sample.wcor)
        grouped[key].append((i, sample))

    for (database, stage, rpm, wcor), samples_group in grouped.items():
        template_file = find_matching_template_file(
            TRAIN_TEMPLATE_ROOT,
            predict_file=samples_group[0][1].source_path,
            predict_root=PREDICT_PATH,
        )
        template = load_tabular_template(template_file)
        sample_output_dir = output_base / database / f"STAGE_{stage}" / f"RPM_{rpm}"
        sample_output_dir.mkdir(parents=True, exist_ok=True)
        output_file = sample_output_dir / f"{wcor:.5f}{template.suffix}"

        rows_to_write: list[dict[str, object]] = []
        for i, sample in samples_group:
            rows_to_write.append(
                {
                    column: _value_for_template_column(sample, column, all_predictions, i)
                    for column in template.columns
                }
            )

        write_table_like_template(output_file, template, rows_to_write)

    logger.info(f"  预测结果已保存到: {output_base}")
    logger.info("[4/4] 完成")


def main():
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
