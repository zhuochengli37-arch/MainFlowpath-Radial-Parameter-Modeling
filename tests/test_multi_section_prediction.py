from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pytest

from project1.experiments.model_partitioning import radial_partition
from project1.experiments.model_registry import ModelIdentity, ModelRegistry
from project1.experiments.prediction_samples import (
    PredictionSample,
    load_prediction_batch,
)
from project1.experiments.radial_prediction import predict_radial


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "institute_four"


class ConstantModel:
    def __init__(self, value: float):
        self.value = value

    def predict(self, x_array):
        return np.full(len(x_array), self.value, dtype=float)


class FlowEchoModel:
    def predict(self, x_array):
        return np.asarray(x_array, dtype=float)[:, 1]


def _sample(
    *,
    section: str | None,
    xi: float = 0.2,
    flow: float | None = 0.512,
    source_file: str = "DATABASE_CMP/STAGE_1/CNC_0.9/0.01509.txt",
    row_index: int = 0,
    section_row_index: int = 0,
) -> PredictionSample:
    return PredictionSample(
        source_file=source_file,
        component="CMP",
        stage=1,
        station="MAIN",
        section=section,
        database="DATABASE_CMP",
        speed_parameter=0.9,
        speed_parameter_source_name="CNC",
        flow_parameter=flow,
        flow_parameter_source_name="W25COR",
        xi=xi,
        schema_name="institute_multi_section" if section else "institute_single_section",
        schema_version="v1",
        row_index=row_index,
        section_row_index=section_row_index,
        source_fields={"xi": f"xi_{section}" if section else "xi"},
        original_row={},
    )


def _write_new_model(
    root: Path,
    *,
    dimension: str,
    section: str | None,
    target: str,
    model,
    value: float = 0.1,
    context_overrides: dict[str, object] | None = None,
) -> dict[str, object]:
    partition = "CMP:S1" + (f":{section}" if section else "")
    safe_partition = partition.replace(":", "_")
    models_name = "models_2d" if dimension == "2D" else "models"
    model_name = f"constant_{int(value)}"
    relative = Path(models_name) / safe_partition / f"artifact_{len(target)}_{model_name}.pkl"
    model_path = root / relative
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path)
    context = {
        "model_dimension": dimension,
        "component": "CMP",
        "stage": 1,
        "station": "MAIN",
        "section": section,
        "partition": partition,
        "schema_name": "institute_multi_section" if section else "institute_single_section",
        "schema_version": "v1",
        "inputs": (
            ["speed_parameter", "xi"]
            if dimension == "2D"
            else ["speed_parameter", "flow_parameter", "xi"]
        ),
        "target": target,
    }
    context.update(context_overrides or {})
    model_path.with_suffix(".json").write_text(
        json.dumps({"model_context": context}), encoding="utf-8"
    )
    return {
        "model_dimension": dimension,
        "component": "CMP",
        "stage": 1,
        "station": "MAIN",
        "section": section,
        "partition": partition,
        "target": target,
        "model": model_name,
        "path": str(relative),
        "sample_count": 8,
        "metrics": {"rmse": value},
    }


def _write_registry(root: Path, dimension: str, records: list[dict[str, object]]) -> None:
    models_name = "models_2d" if dimension == "2D" else "models"
    manifest = root / models_name / "models_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(records), encoding="utf-8")
    prefix = "leaderboard_2d_full.json" if dimension == "2D" else "leaderboard_full.json"
    leaderboard = [
        {
            **record,
            "split_mode": "random",
            "rmse": record["metrics"]["rmse"],
        }
        for record in records
    ]
    (root / prefix).write_text(json.dumps(leaderboard), encoding="utf-8")


def test_prediction_loader_unifies_multi_sections_and_excludes_boundaries():
    batch = load_prediction_batch(FIXTURE_ROOT, radial_mode="full", require_flow=True)

    assert {sample.component for sample in batch.samples} == {"FAN", "CMP", "HPTB", "LPTB"}
    assert {sample.section for sample in batch.samples} == {"RI", "RO", "SI", "SO"}
    assert {sample.station for sample in batch.samples} == {"MAIN"}
    assert all(sample.flow_parameter is not None for sample in batch.samples)
    assert all(sample.section_row_index >= 0 for sample in batch.samples)
    assert {item.station for item in batch.exclusions} == {"INLET", "OUTLET"}
    assert all("station is not MAIN" in item.reason for item in batch.exclusions)


def test_prediction_loader_rejects_duplicate_section_aliases(tmp_path: Path):
    source = tmp_path / "DATABASE_CMP" / "STAGE_1" / "CNC_0.9" / "W25COR_0.512.dat"
    source.parent.mkdir(parents=True)
    source.write_text(
        "xi_RI Cpt_RI xi_rotor_inlet Cpt_rotor_inlet\n"
        "0.0 1.0 0.0 1.0\n",
        encoding="utf-8",
    )

    batch = load_prediction_batch(source, require_flow=True)

    assert not batch.samples
    assert len(batch.exclusions) == 1
    assert "duplicate section 'RI'" in batch.exclusions[0].reason


def test_multi_section_prediction_preserves_source_row_indices(tmp_path: Path):
    source = tmp_path / "DATABASE_CMP" / "STAGE_1" / "CNC_0.9" / "W25COR_0.512.csv"
    source.parent.mkdir(parents=True)
    source.write_text("xi_A,Cpt_A\n0.0,1.0\n,\n1.0,2.0\n", encoding="utf-8")

    samples = load_prediction_batch(source, require_flow=True).samples

    assert [sample.row_index for sample in samples] == [0, 2]
    assert [sample.section_row_index for sample in samples] == [0, 1]


def test_file_loader_content_flow_has_priority_over_numeric_filename(tmp_path: Path):
    source = tmp_path / "DATABASE_CMP" / "STAGE_1" / "CNC_0.9" / "0.01509.txt"
    source.parent.mkdir(parents=True)
    source.write_text("rpm wcor xi\n0.9 0.512 0.25\n", encoding="utf-8")

    sample = load_prediction_batch(source, require_flow=True).samples[0]

    assert sample.speed_parameter == pytest.approx(0.9)
    assert sample.flow_parameter == pytest.approx(0.512)
    assert sample.flow_parameter_source_name == "wcor"


def test_legacy_filename_inference_fills_missing_flow_once(tmp_path: Path):
    source = tmp_path / "DATABASE_CMP" / "STAGE_1" / "RPM_0.9" / "0.01509.txt"
    source.parent.mkdir(parents=True)
    source.write_text("xi\n0.25\n", encoding="utf-8")

    sample = load_prediction_batch(source, require_flow=True).samples[0]

    assert sample.speed_parameter == pytest.approx(0.9)
    assert sample.flow_parameter == pytest.approx(0.01509)


def test_legacy_path_metadata_precedes_filename_inference(tmp_path: Path):
    source = (
        tmp_path
        / "DATABASE_CMP"
        / "STAGE_1"
        / "RPM_0.9"
        / "WCOR_0.512"
        / "0.01509.txt"
    )
    source.parent.mkdir(parents=True)
    source.write_text("xi\n0.25\n", encoding="utf-8")

    sample = load_prediction_batch(source, require_flow=True).samples[0]

    assert sample.flow_parameter == pytest.approx(0.512)


def test_input_only_single_file_can_use_institute_single_model(tmp_path: Path):
    source = tmp_path / "DATABASE_CMP" / "STAGE_1" / "RPM_0.9" / "0.512.txt"
    source.parent.mkdir(parents=True)
    source.write_text("rpm wcor xi\n0.9 0.512 0.25\n", encoding="utf-8")
    record = _write_new_model(
        tmp_path / "model",
        dimension="3D",
        section=None,
        target="Cpt",
        model=FlowEchoModel(),
        context_overrides={"schema_name": "institute_single_section"},
    )
    _write_registry(tmp_path / "model", "3D", [record])

    sample = load_prediction_batch(source, require_flow=True).samples[0]
    result = predict_radial(
        [sample], ModelRegistry(tmp_path / "model", "3D"), dimension="3D"
    )[0]

    assert sample.schema_name == "unclassified_prediction"
    assert result.predicted_value == pytest.approx(0.512)


def test_2d_core_selects_section_models_without_order_assumptions(tmp_path: Path):
    records = [
        _write_new_model(
            tmp_path, dimension="2D", section="RI", target="Cpt", model=ConstantModel(111), value=111
        ),
        _write_new_model(
            tmp_path, dimension="2D", section="RO", target="Cpt", model=ConstantModel(222), value=222
        ),
    ]
    _write_registry(tmp_path, "2D", records)
    samples = [_sample(section="RO", xi=0.8, flow=None), _sample(section="RI", xi=0.1)]

    results = predict_radial(samples, ModelRegistry(tmp_path, "2D"), dimension="2D")

    assert [(item.section, item.predicted_value) for item in results] == [("RO", 222), ("RI", 111)]
    assert {item.partition for item in results} == {"CMP:S1:RI", "CMP:S1:RO"}
    assert all(item.model_identity.dimension == "2D" for item in results)


def test_registry_uses_leaderboard_metric_with_manifest_identity(tmp_path: Path):
    records = [
        _write_new_model(
            tmp_path, dimension="2D", section="RI", target="Cpt", model=ConstantModel(10), value=10
        ),
        _write_new_model(
            tmp_path, dimension="2D", section="RI", target="Cpt", model=ConstantModel(1), value=1
        ),
    ]
    _write_registry(tmp_path, "2D", records)

    result = predict_radial(
        [_sample(section="RI")], ModelRegistry(tmp_path, "2D"), dimension="2D"
    )[0]

    assert result.predicted_value == 1
    assert result.model_name == "constant_1"


def test_3d_core_uses_structured_flow_not_numeric_filename(tmp_path: Path):
    record = _write_new_model(
        tmp_path, dimension="3D", section="RI", target="Cpt", model=FlowEchoModel(), value=1
    )
    _write_registry(tmp_path, "3D", [record])
    sample = _sample(section="RI", flow=0.512, source_file="DATABASE_CMP/STAGE_1/CNC_0.9/0.01509.txt")

    result = predict_radial([sample], ModelRegistry(tmp_path, "3D"), dimension="3D")[0]

    assert result.predicted_value == pytest.approx(0.512)
    assert result.flow_parameter == pytest.approx(0.512)


def test_registry_keeps_dimension_and_section_strict(tmp_path: Path):
    records_2d = [
        _write_new_model(
            tmp_path, dimension="2D", section="RI", target="Cpt", model=ConstantModel(2), value=2
        )
    ]
    records_3d = [
        _write_new_model(
            tmp_path, dimension="3D", section="RI", target="Cpt", model=ConstantModel(3), value=3
        )
    ]
    _write_registry(tmp_path, "2D", records_2d)
    _write_registry(tmp_path, "3D", records_3d)

    result_2d = predict_radial([_sample(section="RI")], ModelRegistry(tmp_path, "2D"), dimension="2D")[0]
    result_3d = predict_radial([_sample(section="RI")], ModelRegistry(tmp_path, "3D"), dimension="3D")[0]

    assert result_2d.predicted_value == 2
    assert result_3d.predicted_value == 3
    with pytest.raises(ValueError, match="dimension"):
        predict_radial([_sample(section="RI")], ModelRegistry(tmp_path, "2D"), dimension="3D")


def test_multi_never_falls_back_to_single_model(tmp_path: Path):
    record = _write_new_model(
        tmp_path, dimension="2D", section=None, target="Cpt", model=ConstantModel(9), value=9
    )
    _write_registry(tmp_path, "2D", [record])

    with pytest.raises(LookupError, match="CMP:S1:RO"):
        predict_radial([_sample(section="RO")], ModelRegistry(tmp_path, "2D"), dimension="2D")


def test_manifest_preserves_underscore_target_and_arbitrary_section(tmp_path: Path):
    target = "Total_Pressure_Coefficient"
    record = _write_new_model(
        tmp_path,
        dimension="2D",
        section="ROTOR_OUTLET",
        target=target,
        model=ConstantModel(7),
        value=7,
    )
    _write_registry(tmp_path, "2D", [record])
    sample = _sample(section="ROTOR_OUTLET")

    registry = ModelRegistry(tmp_path, "2D")
    assert registry.available_targets(radial_partition(sample)) == (target,)
    result = predict_radial([sample], registry, dimension="2D")[0]

    assert result.target == target
    assert result.predicted_value == 7
    assert result.section == "ROTOR_OUTLET"


def test_explicit_targets_error_when_one_partition_lacks_target(tmp_path: Path):
    records = [
        _write_new_model(
            tmp_path, dimension="2D", section="RI", target="Cpt", model=ConstantModel(1), value=1
        ),
        _write_new_model(
            tmp_path, dimension="2D", section="RI", target="Alpha", model=ConstantModel(2), value=2
        ),
        _write_new_model(
            tmp_path, dimension="2D", section="RO", target="Cpt", model=ConstantModel(3), value=3
        ),
    ]
    _write_registry(tmp_path, "2D", records)

    auto = predict_radial(
        [_sample(section="RI"), _sample(section="RO")],
        ModelRegistry(tmp_path, "2D"),
        dimension="2D",
    )
    assert {(result.section, result.target) for result in auto} == {
        ("RI", "Cpt"),
        ("RI", "Alpha"),
        ("RO", "Cpt"),
    }

    with pytest.raises(LookupError, match="Alpha"):
        predict_radial(
            [_sample(section="RI"), _sample(section="RO")],
            ModelRegistry(tmp_path, "2D"),
            dimension="2D",
            targets=["Cpt", "Alpha"],
        )

    skipped = predict_radial(
        [_sample(section="RI"), _sample(section="RO")],
        ModelRegistry(tmp_path, "2D"),
        dimension="2D",
        targets=["Cpt", "Alpha"],
        missing_target_policy="skip",
    )
    assert {(result.section, result.target) for result in skipped} == {
        ("RI", "Cpt"),
        ("RI", "Alpha"),
        ("RO", "Cpt"),
    }


def test_metadata_identity_mismatch_is_rejected(tmp_path: Path):
    record = _write_new_model(
        tmp_path,
        dimension="3D",
        section="RO",
        target="Cpt",
        model=ConstantModel(1),
        context_overrides={"section": "RI"},
    )
    _write_registry(tmp_path, "3D", [record])

    with pytest.raises(ValueError, match="section"):
        predict_radial([_sample(section="RO")], ModelRegistry(tmp_path, "3D"), dimension="3D")


def test_metadata_dimension_mismatch_is_rejected(tmp_path: Path):
    record = _write_new_model(
        tmp_path,
        dimension="3D",
        section="RI",
        target="Cpt",
        model=ConstantModel(1),
        context_overrides={"model_dimension": "2D"},
    )
    _write_registry(tmp_path, "3D", [record])

    with pytest.raises(ValueError, match="model_dimension"):
        predict_radial([_sample(section="RI")], ModelRegistry(tmp_path, "3D"), dimension="3D")


def test_legacy_single_model_without_manifest_remains_usable(tmp_path: Path):
    model_path = tmp_path / "models_2d" / "CMP_S1" / "linear_Cpt.pkl"
    model_path.parent.mkdir(parents=True)
    joblib.dump(ConstantModel(5), model_path)
    (tmp_path / "leaderboard_2d_full.json").write_text(
        json.dumps(
            [
                {
                    "partition": "CMP:S1",
                    "target": "Cpt",
                    "model": "linear",
                    "split_mode": "random",
                    "rmse": 0.1,
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.warns(UserWarning, match="legacy Single"):
        result = predict_radial(
            [_sample(section=None)], ModelRegistry(tmp_path, "2D"), dimension="2D"
        )[0]

    assert result.predicted_value == 5
    assert result.partition == "CMP:S1"
    assert result.model_identity.section is None


def test_explicit_legacy_all_partition_remains_single_only(tmp_path: Path):
    model_path = tmp_path / "models_2d" / "all" / "linear_Cpt.pkl"
    model_path.parent.mkdir(parents=True)
    joblib.dump(ConstantModel(7), model_path)
    (tmp_path / "leaderboard_2d_full.json").write_text(
        json.dumps(
            [
                {
                    "partition": "all",
                    "target": "Cpt",
                    "model": "linear",
                    "split_mode": "random",
                    "rmse": 0.1,
                }
            ]
        ),
        encoding="utf-8",
    )
    registry = ModelRegistry(
        tmp_path, "2D", legacy_partition_override="all"
    )

    with pytest.warns(UserWarning, match="legacy Single"):
        result = predict_radial([_sample(section=None)], registry, dimension="2D")[0]

    assert result.predicted_value == pytest.approx(7)
    assert result.partition == "CMP:S1"
    assert Path(result.model_path).parts[-3:] == ("models_2d", "all", "linear_Cpt.pkl")
    assert registry.available_targets(_sample(section="RI")) == ()


def test_3d_requires_flow_in_standard_sample(tmp_path: Path):
    record = _write_new_model(
        tmp_path, dimension="3D", section="RI", target="Cpt", model=ConstantModel(1)
    )
    _write_registry(tmp_path, "3D", [record])

    with pytest.raises(ValueError, match="flow_parameter"):
        predict_radial(
            [_sample(section="RI", flow=None)],
            ModelRegistry(tmp_path, "3D"),
            dimension="3D",
        )


def test_legacy_single_2d_wrapper_keeps_relative_output_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import workbase.current_scripts.predict_2d as wrapper

    source = tmp_path / "input" / "DATABASE_CMP" / "STAGE_1" / "RPM_0.9" / "0.512.dat"
    source.parent.mkdir(parents=True)
    source.write_text("rpm xi\n0.9 0.1\n0.9 0.8\n", encoding="utf-8")
    record = _write_new_model(
        tmp_path / "model",
        dimension="2D",
        section=None,
        target="Cpt",
        model=ConstantModel(5),
        context_overrides={"schema_name": "legacy_validation"},
    )
    _write_registry(tmp_path / "model", "2D", [record])
    output = tmp_path / "output"
    monkeypatch.setattr(wrapper, "PREDICT_PATH", tmp_path / "input")
    monkeypatch.setattr(wrapper, "OUTPUT_DIR", tmp_path / "model")
    monkeypatch.setattr(wrapper, "PREDICTION_OUTPUT", output)
    monkeypatch.setattr(wrapper, "RADIAL_MODE", "full")
    monkeypatch.setattr(wrapper, "PARTITION", None)

    results = wrapper.predict_2d()

    expected = output / source.relative_to(tmp_path / "input")
    assert expected.exists()
    assert len(results) == 2
    assert all(result.partition == "CMP:S1" for result in results)
    assert "Cpt" in expected.read_text(encoding="utf-8").splitlines()[0]


def test_legacy_single_3d_wrapper_keeps_output_and_structured_flow(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import workbase.current_scripts.predict_3d as wrapper

    source = tmp_path / "input" / "DATABASE_CMP" / "STAGE_1" / "RPM_0.9" / "0.01509.dat"
    source.parent.mkdir(parents=True)
    source.write_text("rpm wcor xi Cpt\n0.9 0.512 0.1 0\n0.9 0.512 0.8 0\n", encoding="utf-8")
    record = _write_new_model(
        tmp_path / "model",
        dimension="3D",
        section=None,
        target="Cpt",
        model=FlowEchoModel(),
        context_overrides={"schema_name": "legacy_validation"},
    )
    _write_registry(tmp_path / "model", "3D", [record])
    output = tmp_path / "output"
    monkeypatch.setattr(wrapper, "PREDICTION_OUTPUT", str(output))
    monkeypatch.setattr(wrapper, "TRAIN_TEMPLATE_ROOT", source)

    results = wrapper.predict_3d(
        predict_dir=str(tmp_path / "input"),
        output_dir=tmp_path / "model",
        radial_mode="full",
        partition=None,
        model_name=None,
        selection_split_mode="random",
        selection_metric="rmse",
    )

    expected = output / "DATABASE_CMP" / "STAGE_1" / "RPM_0.9" / "0.51200.dat"
    assert expected.exists()
    assert len(results) == 2
    assert all(result.predicted_value == pytest.approx(0.512) for result in results)


def test_direct_3d_writes_multi_section_wide_table_with_unequal_xi(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    import workbase.current_scripts.predict_3d as wrapper
    from project1.services.data_reader import read_tabular_file

    source = tmp_path / "input" / "DATABASE_CMP" / "STAGE_1" / "CNC_0.9" / "W25COR_0.512.csv"
    source.parent.mkdir(parents=True)
    source.write_text(
        "xi_A,Cpt_A,xi_B,Cpt_B\n"
        "0.0,9.0,0.0,9.0\n"
        "0.5,9.0,,\n"
        "1.0,9.0,1.0,9.0\n",
        encoding="utf-8",
    )
    records = [
        _write_new_model(
            tmp_path / "model",
            dimension="3D",
            section=section,
            target="Cpt",
            model=ConstantModel(value),
            value=value,
        )
        for section, value in (("A", 1), ("B", 2))
    ]
    _write_registry(tmp_path / "model", "3D", records)
    output = tmp_path / "output"
    monkeypatch.setattr(wrapper, "PREDICTION_OUTPUT", str(output))

    results = wrapper.predict_3d(
        predict_dir=str(tmp_path / "input"),
        output_dir=tmp_path / "model",
        radial_mode="full",
        partition=None,
        model_name=None,
        selection_split_mode="random",
        selection_metric="rmse",
    )

    expected = output / source.relative_to(tmp_path / "input")
    parsed = read_tabular_file(str(expected))
    assert len(results) == 5
    assert [row["Cpt_A"] for row in parsed["rows"]] == [1.0, 1.0, 1.0]
    assert [parsed["rows"][index]["Cpt_B"] for index in (0, 2)] == [2.0, 2.0]
    assert np.isnan(parsed["rows"][1]["xi_B"])
    assert np.isnan(parsed["rows"][1]["Cpt_B"])
    assert (output / "canonical_prediction_results_3d.json").exists()
