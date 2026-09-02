"""1D 训练评估入口脚本。"""

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
from project1.experiments.benchmark_1d_runner import run_benchmark_1d, resolve_1d_dataset_inputs

CONFIG_PATH = "config/benchmark_config.yaml"
config = load_config(CONFIG_PATH)

logger = setup_logger("benchmark_1d", log_dir=config.log_dir)

INPUT_PATH = config.benchmark_1d_input_path
OUTPUT_DIR = config.benchmark_1d_output_dir
N_SPLITS = config.n_splits_1d
MAX_SAMPLES = config.max_samples_1d
INCLUDE_GPR = config.include_gpr_1d


def main():
    logger.info("=" * 70)
    logger.info("1D Benchmark 开始")
    logger.info("=" * 70)

    try:
        input_files = resolve_1d_dataset_inputs(Path(INPUT_PATH))
        logger.info("配置参数:")
        if len(input_files) == 1 and input_files[0].is_dir():
            logger.info(f"  输入目录(3D结构): {input_files[0]}")
        elif len(input_files) == 1:
            logger.info(f"  输入文件: {input_files[0]}")
        else:
            logger.info(f"  输入目录: {Path(INPUT_PATH)}")
            logger.info(f"  输入文件数: {len(input_files)}")
        logger.info(f"  输出目录: {OUTPUT_DIR}")
        logger.info(f"  交叉验证折数: {N_SPLITS}")
        logger.info(f"  最大样本数: {MAX_SAMPLES}")
        logger.info(f"  包含GPR: {INCLUDE_GPR}")

        run_benchmark_1d(
            input_files=input_files,
            output_dir=OUTPUT_DIR,
            n_splits=N_SPLITS,
            max_samples=MAX_SAMPLES,
            include_gpr=INCLUDE_GPR,
            logger=logger,
        )
        config_snapshot = save_config_snapshot(CONFIG_PATH, OUTPUT_DIR)

        logger.info("=" * 70)
        logger.info("[OK] 1D Benchmark 完成")
        logger.info("=" * 70)
        logger.info("查看结果:")
        logger.info(f"  排行榜: {OUTPUT_DIR}/leaderboard.json")
        logger.info(f"  报告: {OUTPUT_DIR}/benchmark_report_1d.md")
        logger.info(f"  模型目录: {OUTPUT_DIR}/models/")
        logger.info(f"  模型清单: {OUTPUT_DIR}/models/models_manifest.json")
        logger.info(f"  配置快照: {config_snapshot['snapshot']}")
        logger.info(f"  最新配置: {config_snapshot['latest']}")

    except Exception as exc:
        logger.error("=" * 70)
        logger.error("[ERROR] 1D Benchmark 失败")
        logger.error("=" * 70)
        logger.error(f"错误信息: {exc}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
