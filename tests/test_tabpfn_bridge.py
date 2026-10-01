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
from project1.modeling.tabpfn_estimator import ExternalTabPFNRegressor


def _argument(command: list[str], name: str) -> Path:
    return Path(command[command.index(name) + 1])


def _fake_tabpfn_process(seen: dict[str, object]):
    def run(command, **kwargs):
        seen["environment"] = kwargs["env"]
        seen["timeout"] = kwargs["timeout"]
        action = command[2]
        arrays = np.load(_argument(command, "--input"), allow_pickle=False)
        if action == "train":
            seen["train_keys"] = set(arrays.files)
            seen["train_conditions"] = arrays["condition_ids"].tolist()
            config = json.loads(_argument(command, "--config").read_text(encoding="utf-8"))
            seen["config"] = config
            artifact = _argument(command, "--artifact-dir")
            artifact.mkdir()
            (artifact / "model.tabpfn_fit").write_bytes(b"official-fit-state-placeholder")
            report = {
                "fit_context_time_seconds": 0.1,
                "checkpoint_filename": "tabpfn-v3.5-fast-20260909.safetensors",
                "versions": {"tabpfn": "9.0.0", "torch": "2.14.0+cpu"},
            }
            _argument(command, "--response").write_text(json.dumps(report), encoding="utf-8")
        else:
            seen["predict_keys"] = set(arrays.files)
            seen["predict_conditions"] = arrays["condition_ids"].tolist()
            prediction = arrays["X"][:, 0] + arrays["X"][:, 2]
            np.savez_compressed(
                _argument(command, "--output"),
                prediction=prediction,
                load_time_seconds=0.2,
                inference_time_seconds=0.03,
            )
        return subprocess.CompletedProcess(command, 0, "", "")

    return run


def test_tabpfn_bridge_preserves_feature_order_context_and_no_test_targets(
    tmp_path: Path, monkeypatch
):
    seen: dict[str, object] = {}
    checkpoint = tmp_path / "tabpfn-v3.5-fast-20260909.safetensors"
    checkpoint.write_bytes(b"checkpoint-placeholder")
    monkeypatch.setattr(
        "project1.modeling.tabpfn_estimator.subprocess.run", _fake_tabpfn_process(seen)
    )
    x = np.array([[0.8, 10.0, 0.0], [0.8, 10.0, 1.0], [1.0, 11.0, 0.5]])
    y = np.array([0.2, 1.2, 0.7])
    model = ExternalTabPFNRegressor(
        python_executable=sys.executable,
        checkpoint_path=str(checkpoint),
    )
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
    assert seen["config"]["n_estimators"] == 1
    assert seen["environment"]["HF_HUB_OFFLINE"] == "1"
    assert seen["environment"]["TABPFN_NO_BROWSER"] == "1"
    assert Path(seen["environment"]["TABPFN_MODEL_CACHE_DIR"]).name == "model_cache"
    assert model.feature_names_in_.tolist() == ["n", "w", "xi"]
    assert np.allclose(prediction, [0.8])
    assert np.allclose(restored.predict(x[:1]), prediction)


def test_tabpfn_bridge_reports_external_process_failure(tmp_path: Path, monkeypatch):
    checkpoint = tmp_path / "tabpfn-v3.5-fast-20260909.safetensors"
    checkpoint.write_bytes(b"checkpoint-placeholder")

    def fail(command, **kwargs):
        raise subprocess.CalledProcessError(1, command, stderr="synthetic TabPFN failure")

    monkeypatch.setattr("project1.modeling.tabpfn_estimator.subprocess.run", fail)
    model = ExternalTabPFNRegressor(
        python_executable=sys.executable,
        checkpoint_path=str(checkpoint),
    )
    with pytest.raises(RuntimeError, match="TabPFN external training failed.*synthetic"):
        model.fit(np.ones((2, 3)), np.ones(2))


def test_tabpfn_failure_is_isolated_in_benchmark_log(tmp_path: Path, monkeypatch):
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
        lambda include_gpr: {"tabpfn": ExternalTabPFNRegressor(python_executable="missing-python")},
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
    assert all(row["model_key"] == "tabpfn" for row in failures)
    assert all({"status", "exception_type", "reason", "stage"} <= row.keys() for row in failures)
