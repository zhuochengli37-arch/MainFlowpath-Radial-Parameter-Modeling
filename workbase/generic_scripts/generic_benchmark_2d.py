from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKBASE = PROJECT_ROOT / "workbase"
SRC = WORKBASE / "src"
for candidate in (PROJECT_ROOT, SRC):
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from workbase.common.runtime_env import ensure_project_venv

if __name__ == "__main__":
    ensure_project_venv(PROJECT_ROOT)

from workbase.common.config_loader import load_generic_config, save_config_snapshot
from workbase.common.generic_tabular import run_generic_benchmark
from workbase.common.logger_config import setup_logger


CONFIG_PATH = PROJECT_ROOT / "config" / "generic_config.yaml"


def main() -> None:
    config = load_generic_config(str(CONFIG_PATH))
    logger = setup_logger("generic_benchmark_2d", log_dir=config.log_dir)

    parser = argparse.ArgumentParser(description="通用 2D 表格 benchmark")
    parser.add_argument("--input", default=None, help="训练文件或目录")
    parser.add_argument("--output", default=None, help="模型输出目录")
    parser.add_argument("--max-samples", type=int, default=None, help="最大采样行数")
    parser.add_argument("--n-splits", type=int, default=None, help="交叉验证折数上限")
    parser.add_argument("--include-gpr", action="store_true", help="启用 GPR 候选模型")
    args = parser.parse_args()

    input_path = args.input or config.benchmark_input_path(2)
    output_dir = args.output or config.benchmark_output_dir(2)
    max_samples = args.max_samples if args.max_samples is not None else config.benchmark_max_samples(2)
    n_splits = args.n_splits if args.n_splits is not None else config.benchmark_n_splits(2)
    include_gpr = args.include_gpr or config.benchmark_include_gpr(2)

    logger.info("=" * 70)
    logger.info("通用 2D benchmark 开始")
    logger.info("=" * 70)
    logger.info("配置参数:")
    logger.info(f"  输入: {input_path}")
    logger.info(f"  输出: {output_dir}")
    logger.info(f"  最大样本数: {max_samples}")
    logger.info(f"  请求折数: {n_splits}")
    logger.info(f"  启用 GPR: {include_gpr}")

    try:
        result = run_generic_benchmark(
            input_dim=2,
            input_path=input_path,
            output_dir=output_dir,
            include_gpr=include_gpr,
            max_samples=max_samples,
            n_splits=n_splits,
        )
        config_snapshot = save_config_snapshot(CONFIG_PATH, output_dir, snapshot_prefix="generic_config")

        logger.info("=" * 70)
        logger.info("[OK] 通用 2D benchmark 完成")
        logger.info("=" * 70)
        logger.info(f"  记录数: {result['row_count']}")
        logger.info(f"  实际折数: {result['effective_n_splits']}")
        logger.info(f"  排行榜: {result['leaderboard_path']}")
        logger.info(f"  清单: {result['manifest_path']}")
        logger.info(f"  配置快照: {config_snapshot['snapshot']}")
        logger.info(f"  最新配置: {config_snapshot['latest']}")
    except Exception as exc:
        logger.error("=" * 70)
        logger.error("[ERROR] 通用 2D benchmark 失败")
        logger.error("=" * 70)
        logger.error(f"错误信息: {exc}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
