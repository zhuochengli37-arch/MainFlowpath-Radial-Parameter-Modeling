from __future__ import annotations

from pathlib import Path

import pytest

from project1.experiments.benchmark_1d_runner import _extract_partitioned_rpm_wcor_pairs
from project1.experiments.benchmark_data import inspect_dataset_file, load_predict_samples, load_samples
from project1.services.dataset_schema import (
    INSTITUTE_FOUR_SECTION_SCHEMA,
    INSTITUTE_SINGLE_SECTION_SCHEMA,
    LEGACY_VALIDATION_SCHEMA,
    detect_dataset_schema,
)
from project1.services.meta_parser import parse_metadata_from_path


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "institute_single"
FORMAL_TARGETS = {"Cpt", "Ctt", "Cps", "Cts", "MA"}


@pytest.mark.parametrize(
    ("component", "relative_path", "speed_parameter", "flow_parameter", "first_cpt"),
    [
        ("FAN", "DATABASE_FAN/STAGE_1/CNC_0.78/W2COR_12.968.dat", 0.78, 12.968, 1.051834),
        ("CMP", "DATABASE_CMP/STAGE_1/CNC_0.78/W25COR_27.998.dat", 0.78, 27.998, 1.131541),
        ("HPTB", "DATABASE_HPTB/STAGE_1/CNC_0.82/W4COR_20.698.dat", 0.82, 20.698, 0.879670),
        ("LPTB", "DATABASE_LPTB/STAGE_1/CNC_0.80/W5COR_23.616.dat", 0.80, 23.616, 0.944288),
    ],
)
def test_institute_main_component_is_read_from_formal_fixture(
    component: str,
    relative_path: str,
    speed_parameter: float,
    flow_parameter: float,
    first_cpt: float,
):
    source = FIXTURE_ROOT / relative_path
    samples = load_samples(str(source.parent.parent.parent), radial_mode="full")
    matching = [sample for sample in samples if Path(sample.source_file) == source]

    assert len(matching) == 5
    sample = matching[0]
    assert sample.component == component
    assert sample.stage == 1
    assert sample.station == "MAIN"
    assert sample.section is None
    assert sample.schema == INSTITUTE_SINGLE_SECTION_SCHEMA
    assert sample.speed_parameter == pytest.approx(speed_parameter)
    assert sample.flow_parameter == pytest.approx(flow_parameter)
    assert sample.rpm == pytest.approx(speed_parameter)
    assert sample.wcor == pytest.approx(flow_parameter)
    assert sample.xi == pytest.approx(0.0)
    assert set(sample.output_columns) == FORMAL_TARGETS
    assert sample.get_output("Cpt") == pytest.approx(first_cpt)


@pytest.mark.parametrize(
    ("relative_path", "station", "stage"),
    [
        ("DATABASE_CMP_INLET/STAGE_0/CNC_0.78/W25COR_27.998.dat", "INLET", 0),
        ("DATABASE_CMP_OUTLET/STAGE_999/CNC_0.78/W25COR_27.998.dat", "OUTLET", 999),
    ],
)
def test_inlet_and_outlet_are_recognized_but_not_trainable(relative_path: str, station: str, stage: int):
    source = FIXTURE_ROOT / relative_path
    metadata = parse_metadata_from_path(str(source))
    inspection = inspect_dataset_file(source)

    assert metadata["component"] == "CMP"
    assert metadata["station"] == station
    assert metadata["stage"] == stage
    assert metadata["section"] is None
    assert metadata["speed_parameter"] == pytest.approx(0.78)
    assert metadata["flow_parameter"] == pytest.approx(27.998)
    assert inspection["schema"] == INSTITUTE_SINGLE_SECTION_SCHEMA
    assert inspection["trainable"] is False


def test_training_loader_uses_only_main_station_from_mixed_fixture():
    samples = load_samples(str(FIXTURE_ROOT), radial_mode="full")

    assert len(samples) == 20
    assert {sample.component for sample in samples} == {"FAN", "CMP", "HPTB", "LPTB"}
    assert {sample.station for sample in samples} == {"MAIN"}
    assert {sample.schema for sample in samples} == {INSTITUTE_SINGLE_SECTION_SCHEMA}
    assert not any("_INLET" in sample.source_file or "_OUTLET" in sample.source_file for sample in samples)


def test_station_filter_does_not_use_stage_sentinels(tmp_path: Path):
    curve = "xi Cpt Ctt Cps Cts MA\n0.0 1.0 1.0 1.0 1.0 0.2\n"
    main = tmp_path / "DATABASE_CMP" / "STAGE_0" / "CNC_0.78" / "W25COR_27.998.dat"
    inlet = tmp_path / "DATABASE_CMP_INLET" / "STAGE_1" / "CNC_0.78" / "W25COR_27.998.dat"
    main.parent.mkdir(parents=True)
    inlet.parent.mkdir(parents=True)
    main.write_text(curve, encoding="utf-8")
    inlet.write_text(curve, encoding="utf-8")

    samples = load_samples(str(tmp_path), radial_mode="full")

    assert len(samples) == 1
    assert samples[0].station == "MAIN"
    assert samples[0].stage == 0
    assert Path(samples[0].source_file) == main


def test_single_section_1d_partition_remains_component_and_stage_only():
    samples = load_samples(str(FIXTURE_ROOT), radial_mode="full")
    partitions = _extract_partitioned_rpm_wcor_pairs(samples)

    assert set(partitions) == {"FAN:S1", "CMP:S1", "HPTB:S1", "LPTB:S1"}
    assert all(sample.section is None for sample in samples)


def test_four_section_header_is_distinct_and_not_loaded_for_current_models(tmp_path: Path):
    columns = [
        "xi_RI", "Cpt_RI", "xi_RO", "Cpt_RO",
        "xi_SI", "Cpt_SI", "xi_SO", "Cpt_SO",
    ]
    assert detect_dataset_schema(columns) == INSTITUTE_FOUR_SECTION_SCHEMA

    source = tmp_path / "DATABASE_CMP" / "STAGE_1" / "CNC_0.76" / "W25COR_20.1.dat"
    source.parent.mkdir(parents=True)
    source.write_text(" ".join(columns) + "\n" + " ".join(["0.1", "1.0"] * 4) + "\n", encoding="utf-8")

    inspection = inspect_dataset_file(source)
    assert inspection["schema"] == INSTITUTE_FOUR_SECTION_SCHEMA
    assert inspection["trainable"] is False
    assert load_samples(str(tmp_path), radial_mode="full") == []
    assert load_predict_samples(str(tmp_path), radial_mode="full") == []


@pytest.mark.parametrize(
    "columns",
    [
        ["xi", "Cpt", "Cps", "Ctt", "Cts", "Vz", "Rho"],
        ["xi", "Cpt", "Cps", "Ctt", "Cts", "Ma"],
    ],
)
def test_four_section_boundary_headers_remain_four_section_schema(columns: list[str]):
    assert detect_dataset_schema(columns) == INSTITUTE_FOUR_SECTION_SCHEMA


def test_legacy_validation_schema_remains_compatible(tmp_path: Path):
    source = tmp_path / "DATABASE_HTB" / "STAGE_1" / "RPM_0.6" / "0.015.dat"
    source.parent.mkdir(parents=True)
    source.write_text(
        "xi psi tsi mai\n"
        "0.0 1.2 0.9 0.3\n"
        "1.0 1.3 1.0 0.4\n",
        encoding="utf-8",
    )

    samples = load_samples(str(tmp_path), radial_mode="full")

    assert len(samples) == 2
    assert {sample.schema for sample in samples} == {LEGACY_VALIDATION_SCHEMA}
    assert {sample.output_columns for sample in samples} == {("psi", "tsi", "mai")}
    assert samples[0].speed_parameter == pytest.approx(0.6)
    assert samples[0].flow_parameter == pytest.approx(0.015)
    assert samples[0].section is None
