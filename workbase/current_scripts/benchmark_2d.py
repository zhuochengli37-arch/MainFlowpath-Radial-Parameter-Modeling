"""
2D 训练评估入口脚本。
保持 3D 目录结构读取，但训练输入固定为 (rpm, xi)。
"""

from pathlib import Path
import sys

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

from workbase.common.config_loader import load_config, save_config_snapshot
from workbase.common.logger_config import setup_logger
from project1.experiments.benchmark_runners import run_benchmark_2d

CONFIG_PATH = "config/benchmark_config.yaml"
config = load_config(CONFIG_PATH)

logger = setup_logger("benchmark_2d", log_dir=config.log_dir)

INPUT_PATH = config.benchmark_2d_input_path
OUTPUT_DIR = config.benchmark_2d_output_dir
RADIAL_MODE = config.radial_mode_2d
MAX_SAMPLES = config.max_samples_2d
INCLUDE_GPR = config.include_gpr_2d
PARTITION_MODE = config.partition_mode_2d
N_SPLITS = config.n_splits_2d


def main():
    logger.info("=" * 70)
    logger.info("2D 训练评估开始")
    logger.info("=" * 70)
    logger.info(f"输入路径: {INPUT_PATH}")
    logger.info(f"输出路径: {OUTPUT_DIR}")
    logger.info(f"径向模式: {RADIAL_MODE}")
    logger.info(f"分区模式: {PARTITION_MODE}")
    logger.info(f"最大样本数: {MAX_SAMPLES}")
    logger.info(f"包含 GPR: {INCLUDE_GPR}")
    logger.info(f"CV 折数上限: {N_SPLITS}")

    try:
        run_benchmark_2d(
            input_dir=INPUT_PATH,
            output_dir=OUTPUT_DIR,
            radial_mode=RADIAL_MODE,
            include_gpr=INCLUDE_GPR,
            max_samples=MAX_SAMPLES,
            partition_mode=PARTITION_MODE,
            n_splits=N_SPLITS,
        )
        config_snapshot = save_config_snapshot(CONFIG_PATH, OUTPUT_DIR)

        logger.info("=" * 70)
        logger.info("[OK] 2D 训练评估完成")
        logger.info("=" * 70)
        logger.info("查看结果:")
        logger.info(f"  排行榜: {OUTPUT_DIR}/leaderboard_2d_{RADIAL_MODE}.json")
        logger.info(f"  汇总榜: {OUTPUT_DIR}/leaderboard_2d_{RADIAL_MODE}_aggregate.json")
        logger.info(f"  报告: {OUTPUT_DIR}/benchmark_report_2d_{RADIAL_MODE}.md")
        logger.info(f"  模型目录: {OUTPUT_DIR}/models_2d/")
        logger.info(f"  模型清单: {OUTPUT_DIR}/models_2d/models_manifest.json")
        logger.info(f"  配置快照: {config_snapshot['snapshot']}")
        logger.info(f"  最新配置: {config_snapshot['latest']}")

    except Exception as exc:
        logger.error("=" * 70)
        logger.error("[ERROR] 2D 训练评估失败")
        logger.error("=" * 70)
        logger.error(f"错误信息: {exc}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
