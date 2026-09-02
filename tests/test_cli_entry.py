from __future__ import annotations

import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
WORKBASE_SRC = ROOT / "workbase" / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(WORKBASE_SRC) not in sys.path:
    sys.path.insert(0, str(WORKBASE_SRC))


def test_project1_cli_importable():
    import project1.cli as cli

    assert callable(cli.main)


def test_cli_parser_resolves_current_workflow():
    from project1.cli import build_parser

    parser = build_parser()
    args, extra = parser.parse_known_args(["current", "benchmark-1d"])

    assert args.handler == "workflow:current_benchmark_1d"
    assert extra == []


def test_cli_parser_supports_current_2d_prediction():
    from project1.cli import build_parser

    parser = build_parser()
    args, extra = parser.parse_known_args(["current", "predict-2d"])

    assert args.handler == "workflow:current_predict_2d"
    assert extra == []


def test_cli_parser_parses_generic_arguments():
    from project1.cli import build_parser

    parser = build_parser()
    args, extra = parser.parse_known_args(
        ["generic", "benchmark-2d", "--input", "data/train", "--output", "data/out", "--include-gpr"]
    )

    assert args.handler == "workflow:generic_benchmark_2d"
    assert args.input == "data/train"
    assert args.output == "data/out"
    assert args.include_gpr is True
    assert extra == []


def test_cli_parser_handles_generic_help_locally():
    from project1.cli import build_parser

    parser = build_parser()
    with pytest.raises(SystemExit) as exc_info:
        parser.parse_args(["generic", "benchmark-2d", "--help"])

    assert exc_info.value.code == 0


def test_prediction_module_importable_after_phase4_changes():
    from project1.experiments.prediction import build_model

    model = build_model("ridge_deg2", feature_count=2, include_gpr=False)

    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
