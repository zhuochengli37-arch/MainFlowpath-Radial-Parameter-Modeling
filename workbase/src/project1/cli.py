from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Sequence

from project1.config import load_settings
from project1.logging_utils import setup_logging
from project1.unified_entry import run_workflow

logger = logging.getLogger(__name__)


CURRENT_WORKFLOWS: tuple[tuple[str, str, str, tuple[str, ...]], ...] = (
    ("benchmark-1d", "current_benchmark_1d", "运行领域专用 1D benchmark", ()),
    ("predict-1d", "current_predict_1d", "运行领域专用 1D 预测", ()),
    ("benchmark-2d", "current_benchmark_2d", "运行领域专用 2D benchmark", ()),
    ("train-2d", "current_train_2d", "训练领域专用 2D 模型（输入 rpm, xi）", ()),
    ("train-workline", "current_train_workline", "训练 2D workline 模型", ()),
    ("predict-2d", "current_predict_2d", "运行领域专用 2D 预测（输入 rpm, xi）", ()),
    ("predict-2d-with-workline", "current_predict_2d_workline", "运行 2D + workline 预测（兼容旧流程）", ()),
    ("benchmark-3d", "current_benchmark_3d", "运行领域专用 3D benchmark", ()),
    ("predict-3d", "current_predict_3d", "运行领域专用 3D 预测", ()),
)

CURRENT_WORKFLOWS = tuple(
    item
    for item in CURRENT_WORKFLOWS
    if item[1] not in {"current_train_2d", "current_train_workline", "current_predict_2d_workline"}
)

GENERIC_WORKFLOWS: tuple[tuple[str, str, str, str], ...] = (
    ("benchmark-1d", "generic_benchmark_1d", "运行通用 1D benchmark", "benchmark"),
    ("predict-1d", "generic_predict_1d", "运行通用 1D 预测", "predict"),
    ("benchmark-2d", "generic_benchmark_2d", "运行通用 2D benchmark", "benchmark"),
    ("predict-2d", "generic_predict_2d", "运行通用 2D 预测", "predict"),
    ("benchmark-3d", "generic_benchmark_3d", "运行通用 3D benchmark", "benchmark"),
    ("predict-3d", "generic_predict_3d", "运行通用 3D 预测", "predict"),
)

TOP_LEVEL_EXAMPLES = """Examples:
  python -m project1 current benchmark-3d
  python -m project1 current predict-3d
  python -m project1 generic benchmark-2d --input data\\generic\\2d\\train
  python -m project1 generic predict-2d --input data\\generic\\2d\\predict --model-output data\\generic\\2d\\output
"""

CURRENT_EXAMPLES = """Examples:
  python -m project1 current benchmark-1d
  python -m project1 current train-workline
  python -m project1 current predict-3d

Notes:
  Current workflows read settings from `config/benchmark_config.yaml`.
  You can run `python -m project1 current <command> --help` for details.
"""

CURRENT_EXAMPLES = """Examples:
  python -m project1 current benchmark-1d
  python -m project1 current benchmark-2d
  python -m project1 current predict-3d

Notes:
  Current workflows read settings from `config/benchmark_config.yaml`.
  You can run `python -m project1 current <command> --help` for details.
"""

GENERIC_EXAMPLES = """Examples:
  python -m project1 generic benchmark-1d
  python -m project1 generic benchmark-2d --input data\\generic\\2d\\train --output data\\generic\\2d\\output
  python -m project1 generic predict-3d --input data\\generic\\3d\\predict --model-output data\\generic\\3d\\output

Notes:
  Generic workflows first parse common args in this unified entry, then forward.
"""


def _parse_columns(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _json_ready(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _prediction_summary(result: dict[str, Any]) -> dict[str, Any]:
    predictions = result.get("predictions")
    prediction_shape = tuple(int(dim) for dim in getattr(predictions, "shape", ()))
    return {
        "model": result.get("model"),
        "train_dir": result.get("train_dir"),
        "predict_dir": result.get("predict_dir"),
        "output_path": result.get("output_path"),
        "x_columns": result.get("x_columns"),
        "pred_columns": result.get("pred_columns"),
        "train_record_count": len(result.get("train_records", [])),
        "predict_record_count": len(result.get("predict_records", [])),
        "prediction_shape": prediction_shape,
    }


def _workflow_prog_name(args: argparse.Namespace) -> str:
    if args.command == "current":
        return f"python -m project1 current {args.current_command}"
    if args.command == "generic":
        return f"python -m project1 generic {args.generic_command}"
    return "python -m project1"


def _default_prog_name() -> str:
    executable_name = Path(sys.argv[0]).name.lower()
    if executable_name == "__main__.py":
        return "python -m project1"
    if executable_name.startswith("project1"):
        return "project1"
    return executable_name or "project1"


def _append_optional_arg(argv: list[str], option: str, value: Any) -> None:
    if value is None:
        return
    if isinstance(value, bool):
        if value:
            argv.append(option)
        return
    argv.extend([option, str(value)])


def _generic_workflow_argv(args: argparse.Namespace) -> list[str]:
    forwarded: list[str] = []
    _append_optional_arg(forwarded, "--input", getattr(args, "input", None))

    mode = getattr(args, "generic_mode", None)
    if mode == "benchmark":
        _append_optional_arg(forwarded, "--output", getattr(args, "output", None))
        _append_optional_arg(forwarded, "--max-samples", getattr(args, "max_samples", None))
        _append_optional_arg(forwarded, "--n-splits", getattr(args, "n_splits", None))
        _append_optional_arg(forwarded, "--include-gpr", getattr(args, "include_gpr", False))
        return forwarded

    if mode == "predict":
        _append_optional_arg(forwarded, "--model-output", getattr(args, "model_output", None))
        _append_optional_arg(forwarded, "--output", getattr(args, "output", None))
        return forwarded

    raise ValueError(f"unknown generic workflow mode: {mode}")


def _add_current_leaf_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    command: str,
    workflow: str,
    help_text: str,
    aliases: tuple[str, ...] = (),
) -> None:
    parser = subparsers.add_parser(
        command,
        help=help_text,
        description=f"{help_text}. This command runs with `config/benchmark_config.yaml`.",
        formatter_class=argparse.RawTextHelpFormatter,
        aliases=list(aliases),
    )
    parser.set_defaults(handler=f"workflow:{workflow}")


def _add_generic_leaf_parser(
    subparsers: argparse._SubParsersAction[argparse.ArgumentParser],
    command: str,
    workflow: str,
    help_text: str,
    mode: str,
) -> None:
    parser = subparsers.add_parser(
        command,
        help=help_text,
        description=f"{help_text}. Args are parsed in unified CLI before forwarding.",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.set_defaults(handler=f"workflow:{workflow}", generic_mode=mode)
    parser.add_argument("--input", default=None, help="Input file or directory")

    if mode == "benchmark":
        parser.add_argument("--output", default=None, help="Model output directory")
        parser.add_argument("--max-samples", type=int, default=None, help="Max samples")
        parser.add_argument("--n-splits", type=int, default=None, help="CV folds cap")
        parser.add_argument("--include-gpr", action="store_true", help="Enable GPR candidates")
        return

    if mode == "predict":
        parser.add_argument("--model-output", default=None, help="Training output directory")
        parser.add_argument("--output", default=None, help="Prediction output directory")
        return

    raise ValueError(f"unknown generic workflow mode: {mode}")


def _add_current_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    current_parser = subparsers.add_parser(
        "current",
        help="Run current-domain workflows",
        description="Unified entry for current-domain workflows.",
        epilog=CURRENT_EXAMPLES,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    current_subparsers = current_parser.add_subparsers(dest="current_command", required=True)
    for command, workflow, help_text, aliases in CURRENT_WORKFLOWS:
        _add_current_leaf_parser(current_subparsers, command, workflow, help_text, aliases=aliases)


def _add_generic_parser(subparsers: argparse._SubParsersAction[argparse.ArgumentParser]) -> None:
    generic_parser = subparsers.add_parser(
        "generic",
        help="Run generic tabular workflows",
        description="Unified entry for generic tabular workflows.",
        epilog=GENERIC_EXAMPLES,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    generic_subparsers = generic_parser.add_subparsers(dest="generic_command", required=True)
    for command, workflow, help_text, mode in GENERIC_WORKFLOWS:
        _add_generic_leaf_parser(generic_subparsers, command, workflow, help_text, mode)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog=_default_prog_name(),
        description="Project1 unified CLI",
        epilog=TOP_LEVEL_EXAMPLES,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="Run offline main workflow")
    run_parser.add_argument("--input-dir", default="./data/input", help="Input directory")
    run_parser.add_argument("--output-dir", default="./data/output", help="Output directory")
    run_parser.set_defaults(handler="run")

    benchmark_parser = subparsers.add_parser("benchmark", help="Run compatible benchmark entry")
    benchmark_parser.add_argument("--input-dir", default="./data/input/sample", help="Input directory")
    benchmark_parser.add_argument("--output-dir", default="./data/output", help="Output directory")
    benchmark_parser.add_argument(
        "--radial-mode",
        choices=["full", "edge_only", "both"],
        default="both",
        help="Radial mode",
    )
    benchmark_parser.add_argument("--exclude-gpr", action="store_true", help="Exclude GPR")
    benchmark_parser.add_argument("--max-samples", type=int, default=900, help="Max samples")
    benchmark_parser.add_argument(
        "--partition-mode",
        choices=["single", "family", "component", "multi", "none", "component_stage"],
        default="multi",
        help="Partition mode",
    )
    benchmark_parser.set_defaults(handler="benchmark")

    predict_parser = subparsers.add_parser("predict", help="Run compatible tabular prediction entry")
    predict_parser.add_argument("--train-dir", required=True, help="Training directory")
    predict_parser.add_argument("--predict-dir", default=None, help="Prediction input directory")
    predict_parser.add_argument("--output-dir", default="./data/output", help="Prediction output directory")
    predict_parser.add_argument("--model-name", default="ridge_deg2", help="Model name")
    predict_parser.add_argument("--x-columns", required=True, help="Input columns, comma-separated")
    predict_parser.add_argument("--y-columns", required=True, help="Target columns, comma-separated")
    predict_parser.add_argument("--x-pred-columns", default=None, help="Prediction input columns")
    predict_parser.add_argument("--delimiter", default=None, help="Input delimiter")
    predict_parser.add_argument("--exclude-gpr", action="store_true", help="Exclude GPR")
    predict_parser.add_argument("--multi-output", action="store_true", help="Enable multi-output wrapper")
    predict_parser.set_defaults(handler="predict")

    generate_parser = subparsers.add_parser("generate-sample5", help="Generate sample5 synthetic data")
    generate_parser.add_argument("--input-dir", default="./data/input/sample4", help="Source directory")
    generate_parser.add_argument("--output-dir", default="./data/input/sample5", help="Target directory")
    generate_parser.add_argument("--per-pair", type=int, default=3, help="Generated files per adjacent pair")
    generate_parser.add_argument("--noise-ratio", type=float, default=0.002, help="Noise ratio")
    generate_parser.add_argument("--clamp-ratio", type=float, default=0.02, help="Clamp ratio")
    generate_parser.add_argument("--seed", type=int, default=42, help="Random seed")
    generate_parser.add_argument("--clean-output", action="store_true", help="Clean output before generation")
    generate_parser.set_defaults(handler="generate-sample5")

    qc_parser = subparsers.add_parser("qc-sample5", help="Run sample5 QC")
    qc_parser.add_argument("--base-dir", default="./data/input/sample4", help="Base data directory")
    qc_parser.add_argument("--sample5-dir", default="./data/input/sample5", help="Sample5 directory")
    qc_parser.add_argument("--output-dir", default="./data/output", help="QC report output directory")
    qc_parser.set_defaults(handler="qc-sample5")

    _add_current_parser(subparsers)
    _add_generic_parser(subparsers)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args, extra_args = parser.parse_known_args(argv)
    handler = getattr(args, "handler", None)

    if handler is None:
        parser.error("No executable handler found")

    if handler.startswith("workflow:"):
        workflow_name = handler.split(":", 1)[1]
        if extra_args:
            parser.error(f"Unrecognized args: {' '.join(extra_args)}")
        forwarded_args = _generic_workflow_argv(args) if args.command == "generic" else ()
        return run_workflow(
            workflow_name,
            forwarded_args,
            prog_name=_workflow_prog_name(args),
        )

    if extra_args:
        parser.error(f"Unrecognized args: {' '.join(extra_args)}")

    settings = load_settings()
    setup_logging(settings.log_level, settings.log_file)

    if handler == "run":
        from project1.main import run_offline_job

        return run_offline_job(args.input_dir, args.output_dir)

    if handler == "benchmark":
        from project1.experiments.benchmark import run_benchmark

        run_benchmark(
            args.input_dir,
            args.output_dir,
            args.radial_mode,
            include_gpr=not args.exclude_gpr,
            max_samples=args.max_samples,
            partition_mode=args.partition_mode,
        )
        return 0

    if handler == "predict":
        from project1.experiments.prediction import predict_tabular_folder

        x_columns = _parse_columns(args.x_columns)
        y_columns = _parse_columns(args.y_columns)
        x_pred_columns = _parse_columns(args.x_pred_columns) if args.x_pred_columns else None
        result = predict_tabular_folder(
            train_dir=args.train_dir,
            model_name=args.model_name,
            x_columns=x_columns,
            y_columns=y_columns,
            predict_dir=args.predict_dir,
            x_pred_columns=x_pred_columns,
            delimiter=args.delimiter,
            output_dir=args.output_dir,
            include_gpr=not args.exclude_gpr,
            multi_output=args.multi_output,
        )
        print(json.dumps(_json_ready(_prediction_summary(result)), ensure_ascii=False, indent=2))
        return 0

    if handler == "generate-sample5":
        from project1.experiments.generate_sample5 import generate_sample5

        result = generate_sample5(
            input_dir=args.input_dir,
            output_dir=args.output_dir,
            per_pair=args.per_pair,
            noise_ratio=args.noise_ratio,
            clamp_ratio=args.clamp_ratio,
            seed=args.seed,
            clean_output=args.clean_output,
        )
        print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))
        return 0

    if handler == "qc-sample5":
        from project1.experiments.qc_sample5 import run_qc

        result = run_qc(
            base_dir=args.base_dir,
            sample5_dir=args.sample5_dir,
            output_dir=args.output_dir,
        )
        print(json.dumps(_json_ready(result), ensure_ascii=False, indent=2))
        return 0

    logger.error("Unknown command handler: %s", handler)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
