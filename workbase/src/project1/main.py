import argparse
import logging
from typing import Iterable
from project1.config import load_settings
from project1.logging_utils import setup_logging
from project1.pipeline.offline_pipeline import run_pipeline

logger = logging.getLogger(__name__)


def run_offline_job(input_dir: str, output_dir: str) -> int:
    code, result = run_pipeline(input_dir, output_dir)
    if code != 0:
        logger.warning("输入目录不存在: %s", input_dir)
        return code

    if result is None:
        logger.error("流程执行结束，但未返回结果")
        return 1

    logger.info("发现数据文件 %d 个", result.processed_files)
    logger.info("建立索引 %d 个", result.indexed_files)
    logger.info("失败文件 %d 个", result.failed_files)
    logger.info("已写入清单: %s", result.manifest_path)
    logger.info("已写入报告: %s", result.report_path)
    return 0


def run_prediction_job(
    train_dir: str,
    model_name: str,
    x_columns: str | Iterable[str],
    y_columns: str | Iterable[str],
    predict_dir: str | None = None,
    output_dir: str | None = None,
    x_pred_columns: str | Iterable[str] | None = None,
    delimiter: str | None = None,
    include_gpr: bool = True,
    multi_output: bool | None = None,
) -> int:
    from project1.experiments.prediction import predict_tabular_folder

    result = predict_tabular_folder(
        train_dir=train_dir,
        model_name=model_name,
        x_columns=x_columns,
        y_columns=y_columns,
        predict_dir=predict_dir,
        x_pred_columns=x_pred_columns,
        delimiter=delimiter,
        output_dir=output_dir,
        include_gpr=include_gpr,
        multi_output=multi_output,
    )
    logger.info("模型 %s 使用 %d 行训练，并完成 %d 行预测", model_name, len(result["train_records"]), len(result["predict_records"]))
    if result["output_path"]:
        logger.info("预测结果已写入: %s", result["output_path"])
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Project1 离线入口")
    parser.add_argument("--input-dir", default="./data/input", help="输入数据目录")
    parser.add_argument("--output-dir", default="./data/output", help="输出目录")
    return parser


def main() -> int:
    settings = load_settings()
    setup_logging(settings.log_level, settings.log_file)
    args = build_parser().parse_args()

    logger.info("以离线模式启动 %s", settings.app_name)
    return run_offline_job(args.input_dir, args.output_dir)


if __name__ == "__main__":
    raise SystemExit(main())
