"""Runner executed by the isolated PySR Python environment."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
from pathlib import Path

import numpy as np
from pysr import PySRRegressor
from pysr.julia_import import jl


def _version(package: str) -> str:
    return importlib.metadata.version(package)


def _finite_float(value):
    number = float(value)
    return number if np.isfinite(number) else None


def train(input_path: Path, config_path: Path, artifact_dir: Path, response_path: Path) -> None:
    arrays = np.load(input_path, allow_pickle=False)
    x = np.asarray(arrays["X"], dtype=float)
    y = np.asarray(arrays["y"], dtype=float)
    config = json.loads(config_path.read_text(encoding="utf-8"))
    feature_names = config.pop("feature_names")
    context = config.pop("context")

    started = time.perf_counter()
    model = PySRRegressor(
        **config,
        output_directory=str(artifact_dir.parent),
        run_id=artifact_dir.name,
    )
    model.fit(x, y, variable_names=feature_names)
    training_time = time.perf_counter() - started

    selected = model.get_best()
    columns = [name for name in ("complexity", "loss", "score", "equation") if name in model.equations_]
    equation_table = [
        {
            name: (str(value) if name == "equation" else _finite_float(value))
            for name, value in row.items()
        }
        for row in model.equations_[columns].to_dict(orient="records")
    ]
    metadata = {
        "selected_equation": str(selected["equation"]),
        "selected_complexity": int(selected["complexity"]),
        "selected_loss": _finite_float(selected["loss"]),
        "selected_score": _finite_float(selected["score"]),
        "equation_table": equation_table,
        "feature_names": feature_names,
        "input_scaling": "none; raw Direct-3D variables",
        "search_config": config,
        "model_selection_rule": (
            "official 'best': among equations with loss <= 1.5 * minimum loss, "
            "select the maximum score"
        ),
        "context": context,
        "training_time_seconds": training_time,
        "parallelism": config["parallelism"],
        "process_count": 1,
        "environment_identifier": ".venv-pysr",
        "versions": {
            "python": platform.python_version(),
            "pysr": _version("pysr"),
            "juliacall": _version("juliacall"),
            "numpy": _version("numpy"),
            "pandas": _version("pandas"),
            "scikit-learn": _version("scikit-learn"),
            "julia": str(jl.seval("VERSION")),
            "symbolic_regression_jl": str(jl.seval("Base.pkgversion(SymbolicRegression)")),
        },
    }
    (artifact_dir / "project1_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    response_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")


def predict(input_path: Path, artifact_dir: Path, output_path: Path) -> None:
    arrays = np.load(input_path, allow_pickle=False)
    x = np.asarray(arrays["X"], dtype=float)
    started = time.perf_counter()
    model = PySRRegressor.from_file(run_directory=artifact_dir)
    prediction = np.asarray(model.predict(x), dtype=float).reshape(-1)
    elapsed = time.perf_counter() - started
    if len(prediction) != len(x) or not np.isfinite(prediction).all():
        raise ValueError("PySR returned invalid predictions")
    np.savez_compressed(output_path, prediction=prediction, inference_time_seconds=elapsed)


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--input", type=Path, required=True)
    train_parser.add_argument("--config", type=Path, required=True)
    train_parser.add_argument("--artifact-dir", type=Path, required=True)
    train_parser.add_argument("--response", type=Path, required=True)
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--input", type=Path, required=True)
    predict_parser.add_argument("--artifact-dir", type=Path, required=True)
    predict_parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args.input, args.config, args.artifact_dir, args.response)
    else:
        predict(args.input, args.artifact_dir, args.output)


if __name__ == "__main__":
    main()
