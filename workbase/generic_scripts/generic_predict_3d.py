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

from workbase.common.config_loader import load_generic_config
from workbase.common.generic_tabular import run_generic_predict
from workbase.common.logger_config import setup_logger


CONFIG_PATH = PROJECT_ROOT / "config" / "generic_config.yaml"


def main() -> None:
    config = load_generic_config(str(CONFIG_PATH))
    logger = setup_logger("generic_predict_3d", log_dir=config.log_dir)

    parser = argparse.ArgumentParser(description="通用 3D 表格预测")
    parser.add_argument("--input", default=None, help="预测文件或目录")
    parser.add_argument("--model-output", default=None, help="训练输出目录")
    parser.add_argument("--output", default=None, help="预测输出目录")
    args = parser.parse_args()

    input_path = args.input or config.predict_input_path(3)
    model_output = args.model_output or config.predict_model_output_dir(3)
    output_dir = args.output or config.predict_output_dir(3)

    logger.info("=" * 70)
    logger.info("通用 3D 预测开始")
    logger.info("=" * 70)
    logger.info("配置参数:")
    logger.info(f"  输入: {input_path}")
    logger.info(f"  模型输出: {model_output}")
    logger.info(f"  结果输出: {output_dir}")

    try:
        result = run_generic_predict(
            input_dim=3,
            input_path=input_path,
            model_output_dir=model_output,
            output_dir=output_dir,
        )
        logger.info("=" * 70)
        logger.info("[OK] 通用 3D 预测完成")
        logger.info("=" * 70)
        logger.info(f"  输出文件数: {len(result['outputs'])}")
        for output_file in result["outputs"]:
            logger.info(f"  - {output_file}")
        logger.info(f"  清单: {result['manifest_path']}")
    except Exception as exc:
        logger.error("=" * 70)
        logger.error("[ERROR] 通用 3D 预测失败")
        logger.error("=" * 70)
        logger.error(f"错误信息: {exc}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
