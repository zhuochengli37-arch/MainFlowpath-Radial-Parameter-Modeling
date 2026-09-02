"""
3D 训练评估入口脚本。
使用目录结构化数据训练模型，默认输入为 `(rpm, wcor, xi)`。
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
from workbase.common.validators import ValidationError, validate_directory_exists
from project1.experiments.benchmark_runners import run_benchmark

CONFIG_PATH = "config/benchmark_config.yaml"
config = load_config(CONFIG_PATH)

logger = setup_logger("benchmark_3d", log_dir=config.log_dir)

INPUT_PATH = config.benchmark_3d_input_path
OUTPUT_DIR = config.benchmark_3d_output_dir
RADIAL_MODE = config.radial_mode
MAX_SAMPLES = config.max_samples_3d
INCLUDE_GPR = config.include_gpr_3d
PARTITION_MODE = config.partition_mode
N_SPLITS = config.n_splits_3d


def main():
    logger.info("=" * 70)
    logger.info("3D Benchmark 开始")
    logger.info("=" * 70)

    logger.info(f"交叉验证折数上限: {N_SPLITS}")

    try:
        output_path = Path(OUTPUT_DIR)
        if not output_path.exists():
            logger.warning(f"输出目录不存在，将自动创建: {output_path}")
            output_path.mkdir(parents=True, exist_ok=True)
        elif not output_path.is_dir():
            raise ValidationError(f"输出路径已存在，但不是目录: {output_path}")

        validate_directory_exists(Path(INPUT_PATH), "输入目录")
        logger.info("配置参数:")
        logger.info(f"  输入目录: {INPUT_PATH}")
        logger.info(f"  输出目录: {OUTPUT_DIR}")
        logger.info(f"  径向模式: {RADIAL_MODE}")
        logger.info(f"  分区模式: {PARTITION_MODE}")
        logger.info(f"  最大样本数: {MAX_SAMPLES}")
        logger.info(f"  包含 GPR: {INCLUDE_GPR}")

        run_benchmark(
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
        logger.info("[OK] 3D Benchmark 完成")
        logger.info("=" * 70)
        logger.info("查看结果:")
        logger.info(f"  排行榜: {OUTPUT_DIR}/leaderboard_{RADIAL_MODE}_aggregate.json")
        logger.info(f"  报告: {OUTPUT_DIR}/benchmark_report_{RADIAL_MODE}.md")
        logger.info(f"  模型目录: {OUTPUT_DIR}/models/")
        logger.info(f"  配置快照: {config_snapshot['snapshot']}")
        logger.info(f"  最新配置: {config_snapshot['latest']}")

    except ValidationError as exc:
        logger.error(f"输入校验失败: {exc}")
        raise
    except Exception as exc:
        logger.error("=" * 70)
        logger.error("[ERROR] 3D Benchmark 失败")
        logger.error("=" * 70)
        logger.error(f"错误信息: {exc}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
