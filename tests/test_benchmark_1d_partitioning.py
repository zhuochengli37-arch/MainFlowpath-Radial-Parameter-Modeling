from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from sklearn.linear_model import LinearRegression

from project1.experiments.benchmark_1d_runner import (
    _extract_partitioned_rpm_wcor_pairs,
    run_benchmark_1d,
)
from project1.experiments.benchmark_data import Sample
from workbase.current_scripts.predict_1d import _find_best_model_1d, _partition_from_source_file


def _sample(component: str, stage: int, rpm: float, wcor: float, xi: float) -> Sample:
    return Sample(
        component=component,
        family=component,
        station="MAIN",
        stage=stage,
        rpm=rpm,
        wcor=wcor,
        xi=xi,
        psi=1.0,
        tsi=1.0,
        mai=1.0,
    )


def _write_curve(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("xi Cpt\n0.0 1.0\n1.0 1.1\n", encoding="utf-8")


def test_rpm_wcor_pairs_are_deduplicated_within_component_partition():
    samples = [
        _sample("CMP", 1, 0.6, 0.004, 0.0),
        _sample("CMP", 1, 0.6, 0.004, 1.0),
        _sample("CMP", 1, 0.8, 0.006, 0.0),
        _sample("HTB", 1, 0.6, 0.015, 0.0),
        _sample("HTB", 1, 0.6, 0.015, 1.0),
        _sample("HTB", 1, 0.8, 0.010, 0.0),
    ]

    partitions = _extract_partitioned_rpm_wcor_pairs(samples)

    assert set(partitions) == {"CMP:S1", "HTB:S1"}
    cmp_x, cmp_y, _ = partitions["CMP:S1"]
    htb_x, htb_y, _ = partitions["HTB:S1"]
    np.testing.assert_allclose(cmp_x[:, 0], [0.6, 0.8])
    np.testing.assert_allclose(cmp_y, [0.004, 0.006])
    np.testing.assert_allclose(htb_x[:, 0], [0.6, 0.8])
    np.testing.assert_allclose(htb_y, [0.015, 0.010])


def test_rpm_wcor_pairs_reject_multiple_flows_for_same_component_rpm():
    samples = [
        _sample("HTB", 1, 0.6, 0.010, 0.0),
        _sample("HTB", 1, 0.6, 0.012, 1.0),
    ]

    with pytest.raises(ValueError, match="not single-valued"):
        _extract_partitioned_rpm_wcor_pairs(samples)


def test_benchmark_saves_models_by_component_partition(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    train = tmp_path / "train"
    output = tmp_path / "output"
    _write_curve(train / "DATABASE_CMP" / "STAGE_1" / "rpm_0.6" / "0.004.txt")
    _write_curve(train / "DATABASE_CMP" / "STAGE_1" / "rpm_0.8" / "0.006.txt")
    _write_curve(train / "DATABASE_HTB" / "STAGE_1" / "rpm_0.6" / "0.015.txt")
    _write_curve(train / "DATABASE_HTB" / "STAGE_1" / "rpm_0.8" / "0.010.txt")
    monkeypatch.setattr(
        "project1.experiments.benchmark_1d_runner.build_1d_models",
        lambda include_gpr: {"linear": LinearRegression()},
    )

    results = run_benchmark_1d(
        input_files=[train],
        output_dir=str(output),
        n_splits=2,
        max_samples=100,
        include_gpr=False,
    )

    assert {str(row["partition"]) for row in results} == {"CMP:S1", "HTB:S1"}
    assert (output / "models" / "CMP_S1" / "linear_wcor.pkl").exists()
    assert (output / "models" / "HTB_S1" / "linear_wcor.pkl").exists()
    manifest = json.loads((output / "models" / "models_manifest.json").read_text(encoding="utf-8"))
    assert {str(row["partition"]) for row in manifest} == {"CMP:S1", "HTB:S1"}
    assert {row["section"] for row in manifest} == {None}
    assert {row["model_dimension"] for row in manifest} == {"1D"}
    condition_manifest = json.loads(
        (output / "operating_conditions_manifest.json").read_text(encoding="utf-8")
    )
    assert {row["partition"] for row in condition_manifest} == {"CMP:S1", "HTB:S1"}
    assert {row["operating_condition_record_count"] for row in condition_manifest} == {2}
    assert {row["unique_speed_flow_points"] for row in condition_manifest} == {2}
    metadata = json.loads(
        (output / "models" / "CMP_S1" / "linear_wcor.json").read_text(encoding="utf-8")
    )
    assert metadata["model_context"]["section"] is None
    assert metadata["model_context"]["partition"] == "CMP:S1"


def test_prediction_selects_best_model_from_requested_partition(tmp_path: Path):
    leaderboard = [
        {"partition": "CMP:S1", "target": "wcor", "model": "cmp_model", "rmse": 0.001},
        {"partition": "HTB:S1", "target": "wcor", "model": "htb_model", "rmse": 0.002},
    ]
    (tmp_path / "leaderboard.json").write_text(json.dumps(leaderboard), encoding="utf-8")

    model_name, best = _find_best_model_1d(tmp_path, "wcor", "HTB:S1")

    assert model_name == "htb_model"
    assert best["partition"] == "HTB:S1"
    source = Path("DATABASE_HTB") / "STAGE_1" / "rpm_0.6" / "predict.txt"
    assert _partition_from_source_file(source) == "HTB:S1"
