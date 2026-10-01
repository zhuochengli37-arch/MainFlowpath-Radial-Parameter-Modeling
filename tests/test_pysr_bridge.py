from __future__ import annotations

import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from project1.experiments.benchmark_data import Sample
from project1.experiments.benchmark_runners import run_benchmark
from project1.modeling.pysr_estimator import ExternalPySRRegressor


def _argument(command: list[str], name: str) -> Path:
    return Path(command[command.index(name) + 1])


def _fake_pysr_process(seen: dict[str, object]):
    def run(command, **kwargs):
        seen["environment"] = kwargs["env"]
        action = command[2]
        arrays = np.load(_argument(command, "--input"), allow_pickle=False)
        if action == "train":
            seen["train_keys"] = set(arrays.files)
            seen["train_conditions"] = arrays["condition_ids"].tolist()
            config = json.loads(_argument(command, "--config").read_text(encoding="utf-8"))
            seen["config"] = config
            artifact = _argument(command, "--artifact-dir")
            artifact.mkdir()
            (artifact / "checkpoint.pkl").write_bytes(b"official-pysr-placeholder")
            report = {
                "selected_equation": "n + xi",
                "selected_complexity": 3,
                "selected_loss": 0.1,
                "selected_score": 1.0,
                "equation_table": [
                    {"complexity": 3.0, "loss": 0.1, "score": 1.0, "equation": "n + xi"}
                ],
                "training_time_seconds": 0.01,
                "versions": {"pysr": "2.5.2", "julia": "1.11.9"},
            }
            _argument(command, "--response").write_text(json.dumps(report), encoding="utf-8")
        else:
            seen["predict_keys"] = set(arrays.files)
            seen["predict_conditions"] = arrays["condition_ids"].tolist()
            prediction = arrays["X"][:, 0] + arrays["X"][:, 2]
            np.savez_compressed(
                _argument(command, "--output"),
                prediction=prediction,
                inference_time_seconds=0.02,
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    return run


def test_pysr_bridge_preserves_feature_order_symbolic_data_and_no_test_targets(monkeypatch):
    seen: dict[str, object] = {}
    monkeypatch.setenv("JULIA_DEPOT_PATH", "short-julia-depot")
    monkeypatch.setattr("project1.modeling.pysr_estimator.subprocess.run", _fake_pysr_process(seen))
    x = np.array([[0.8, 10.0, 0.0], [0.8, 10.0, 1.0], [1.0, 11.0, 0.5]])
    y = np.array([0.2, 1.2, 0.7])
    model = ExternalPySRRegressor(python_executable=sys.executable)
    model.set_run_context(
        train_condition_ids=["condition-a", "condition-a", "condition-b"],
        split_mode="condition",
        feature_order=["n", "w", "xi"],
    )
    model.fit(x, y)
    model.set_prediction_context(condition_ids=["condition-c"])
    prediction = model.predict(x[:1])
    restored = pickle.loads(pickle.dumps(model))

    assert seen["train_keys"] == {"X", "y", "condition_ids"}
    assert seen["predict_keys"] == {"X", "condition_ids"}
    assert seen["train_conditions"] == ["condition-a", "condition-a", "condition-b"]
    assert seen["predict_conditions"] == ["condition-c"]
    assert seen["config"]["feature_names"] == ["n", "w", "xi"]
    assert seen["config"]["model_selection"] == "best"
    assert Path(seen["environment"]["PYTHON_JULIACALL_PROJECT"]) == (
        model._project_root() / "envs" / "pysr-julia"
    )
    assert Path(seen["environment"]["PYTHON_JULIACALL_EXE"]).name in {"julia", "julia.exe"}
    assert seen["environment"]["JULIA_DEPOT_PATH"] == "short-julia-depot"
    assert model.feature_names_in_.tolist() == ["n", "w", "xi"]
    assert model.selected_equation_ == "n + xi"
    assert model.complexity_ == 3
    assert model.equation_table_
    assert np.allclose(prediction, [0.8])
    assert np.allclose(restored.predict(x[:1]), prediction)


def test_pysr_bridge_reports_external_process_failure(monkeypatch):
    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, stderr="synthetic Julia failure")

    monkeypatch.setattr("project1.modeling.pysr_estimator.subprocess.run", fail)
    model = ExternalPySRRegressor(python_executable=sys.executable)
    with pytest.raises(RuntimeError, match="PySR external training failed.*synthetic Julia failure"):
        model.fit(np.ones((2, 3)), np.ones(2))


def test_pysr_failure_is_isolated_in_benchmark_log(tmp_path: Path, monkeypatch):
    samples = [
        Sample(
            component="CMP",
            family="CMP",
            station="MAIN",
            stage=1,
            rpm=0.7 + condition * 0.1,
            wcor=10.0 + condition,
            xi=xi,
            psi=1.0 + condition * 0.1 + xi,
            tsi=2.0,
            mai=3.0,
            source_file=f"condition_{condition}.dat",
        )
        for condition in range(4)
        for xi in (0.0, 1.0)
    ]
    monkeypatch.setattr(
        "project1.experiments.benchmark_runners.load_training_samples",
        lambda input_dir, radial_mode: samples,
    )
    monkeypatch.setattr(
        "project1.experiments.benchmark_runners._build_models",
        lambda include_gpr: {"pysr": ExternalPySRRegressor(python_executable="missing-python")},
    )
    result = run_benchmark(
        input_dir=str(tmp_path / "unused"),
        output_dir=str(tmp_path / "output"),
        radial_mode="full",
        include_gpr=False,
        max_samples=100,
        partition_mode="single",
        n_splits=2,
    )
    failures = json.loads(Path(result["failures_path"]).read_text(encoding="utf-8"))

    assert failures
    assert all(row["model_key"] == "pysr" for row in failures)
    assert all({"status", "exception_type", "reason", "stage"} <= row.keys() for row in failures)
