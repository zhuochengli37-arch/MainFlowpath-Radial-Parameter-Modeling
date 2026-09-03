from __future__ import annotations

from pathlib import Path

import pytest

from project1.experiments.benchmark_1d_runner import _extract_partitioned_rpm_wcor_pairs
from project1.experiments.benchmark_data import inspect_dataset_file, load_predict_samples, load_samples
from project1.experiments.four_section_adapter import (
    CANONICAL_SECTION_ORDER,
    FourSectionAdapterError,
    adapt_four_section_file,
    four_section_layout,
)
from project1.services.dataset_schema import (
    INSTITUTE_FOUR_SECTION_SCHEMA,
    INSTITUTE_SINGLE_SECTION_SCHEMA,
    LEGACY_VALIDATION_SCHEMA,
    detect_dataset_schema,
)


FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "institute_four"
SINGLE_FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "institute_single"

MAIN_CASES = [
    (
        "FAN",
        "DATABASE_FAN/STAGE_1/CNC_0.76/W2COR_20.1.dat",
        ("RI", "RO", "SI", "SO"),
        {"Cpt", "Cps", "Ctt", "Cts", "Vz", "Rho"},
    ),
    (
        "CMP",
        "DATABASE_CMP/STAGE_1/CNC_0.76/W25COR_20.1.dat",
        ("RI", "RO", "SI", "SO"),
        {"Cpt", "Cps", "Ctt", "Cts", "Vz", "Rho"},
    ),
    (
        "HPTB",
        "DATABASE_HPTB/STAGE_1/CNC_0.71/W4COR_18.2.dat",
        ("SI", "SO", "RI", "RO"),
        {"Cpt", "Cps", "Ctt", "Cts", "MA"},
    ),
    (
        "LPTB",
        "DATABASE_LPTB/STAGE_1/CNC_0.76/W5COR_28.2.dat",
        ("SI", "SO", "RI", "RO"),
        {"Cpt", "Cps", "Ctt", "Cts", "MA"},
    ),
]


@pytest.mark.parametrize(("component", "relative_path", "source_order", "outputs"), MAIN_CASES)
def test_main_file_is_identified_and_split_by_header_suffix(
    component: str,
    relative_path: str,
    source_order: tuple[str, ...],
    outputs: set[str],
):
    source = FIXTURE_ROOT / relative_path
    inspection = inspect_dataset_file(source)
    columns = list(inspection["columns"])
    layout = four_section_layout(columns)
    samples = adapt_four_section_file(source)

    assert inspection["schema"] == INSTITUTE_FOUR_SECTION_SCHEMA
    assert inspection["station"] == "MAIN"
    assert inspection["trainable"] is False
    assert layout["source_section_order"] == source_order
    assert len(samples) == 8
    assert tuple(dict.fromkeys(sample.section for sample in samples)) == CANONICAL_SECTION_ORDER
    assert {sample.component for sample in samples} == {component}
    assert {sample.station for sample in samples} == {"MAIN"}
    assert {sample.stage for sample in samples} == {1}
    assert {sample.schema for sample in samples} == {INSTITUTE_FOUR_SECTION_SCHEMA}
    assert {frozenset(sample.output_columns) for sample in samples} == {frozenset(outputs)}


@pytest.mark.parametrize(
    ("relative_path", "expected_cpt", "expected_extra"),
    [
        (
            "DATABASE_CMP/STAGE_1/CNC_0.76/W25COR_20.1.dat",
            {"RI": 1.000000, "RO": 1.010000, "SI": 1.020000, "SO": 1.030000},
            {"RI": 112.010000, "RO": 113.010000, "SI": 113.010000, "SO": 114.010000},
        ),
        (
            "DATABASE_HPTB/STAGE_1/CNC_0.71/W4COR_18.2.dat",
            {"RI": 0.923299, "RO": 0.901863, "SI": 0.966581, "SO": 0.944772},
            {"RI": 0.258572, "RO": 0.264256, "SI": 0.236676, "SO": 0.245752},
        ),
    ],
)
def test_each_section_uses_its_own_xi_and_output_columns(
    relative_path: str,
    expected_cpt: dict[str, float],
    expected_extra: dict[str, float],
):
    samples = adapt_four_section_file(FIXTURE_ROOT / relative_path)
    first_by_section = {section: next(sample for sample in samples if sample.section == section) for section in CANONICAL_SECTION_ORDER}
    extra = "Vz" if "CMP" in relative_path else "MA"

    for section, sample in first_by_section.items():
        assert sample.xi == pytest.approx(0.01 if extra == "Vz" else 0.0)
        assert sample.get_output("Cpt") == pytest.approx(expected_cpt[section])
        assert sample.get_output(extra) == pytest.approx(expected_extra[section])
        assert sample.source_fields["xi"] == f"xi_{section}"
        expected_source_extra = f"Vz_{section}" if extra == "Vz" else f"Ma_{section}"
        assert sample.source_fields[extra] == expected_source_extra


@pytest.mark.parametrize(
    ("relative_path", "station", "component", "stage"),
    [
        ("DATABASE_FAN_INLET/STAGE_0/CNC_0.76/W2COR_20.1.dat", "INLET", "FAN", 0),
        ("DATABASE_FAN_OUTLET/STAGE_999/CNC_0.76/W2COR_20.1.dat", "OUTLET", "FAN", 999),
        ("DATABASE_HPTB_INLET/STAGE_0/CNC_0.71/W4COR_18.2.dat", "INLET", "HPTB", 0),
        ("DATABASE_HPTB_OUTLET/STAGE_999/CNC_0.71/W4COR_18.2.dat", "OUTLET", "HPTB", 999),
    ],
)
def test_boundaries_keep_actual_single_schema_and_have_no_section(
    relative_path: str,
    station: str,
    component: str,
    stage: int,
):
    source = FIXTURE_ROOT / relative_path
    inspection = inspect_dataset_file(source)

    assert inspection["schema"] == INSTITUTE_SINGLE_SECTION_SCHEMA
    assert inspection["component"] == component
    assert inspection["station"] == station
    assert inspection["stage"] == stage
    assert inspection["section"] is None
    assert inspection["trainable"] is False
    with pytest.raises(FourSectionAdapterError, match="not a four-section MAIN file"):
        adapt_four_section_file(source)


def test_single_section_adapter_behavior_is_unchanged():
    samples = load_samples(str(SINGLE_FIXTURE_ROOT), radial_mode="full")

    assert len(samples) == 20
    assert {sample.schema for sample in samples} == {INSTITUTE_SINGLE_SECTION_SCHEMA}
    assert {sample.section for sample in samples} == {None}


def test_legacy_schema_remains_compatible(tmp_path: Path):
    source = tmp_path / "DATABASE_HTB" / "STAGE_1" / "RPM_0.6" / "0.015.dat"
    source.parent.mkdir(parents=True)
    source.write_text("xi psi tsi mai\n0.0 1.2 0.9 0.3\n", encoding="utf-8")

    assert detect_dataset_schema(["xi", "psi", "tsi", "mai"]) == LEGACY_VALIDATION_SCHEMA
    samples = load_samples(str(tmp_path), radial_mode="full")
    assert len(samples) == 1
    assert samples[0].schema == LEGACY_VALIDATION_SCHEMA
    assert samples[0].section is None


def test_four_section_data_stops_before_current_model_loaders():
    assert load_samples(str(FIXTURE_ROOT), radial_mode="full") == []
    assert load_predict_samples(str(FIXTURE_ROOT), radial_mode="full") == []


def test_existing_1d_partition_remains_component_and_stage_only():
    fan = adapt_four_section_file(FIXTURE_ROOT / MAIN_CASES[0][1])
    cmp_samples = adapt_four_section_file(FIXTURE_ROOT / MAIN_CASES[1][1])
    partitions = _extract_partitioned_rpm_wcor_pairs(fan + cmp_samples)

    assert set(partitions) == {"FAN:S1", "CMP:S1"}
    assert all(details == {"component": key.split(":")[0], "stage": 1} for key, (_, _, details) in partitions.items())


def test_layout_rejects_missing_section_and_case_alias_collision():
    with pytest.raises(FourSectionAdapterError, match="missing section"):
        four_section_layout(["xi_RI", "Cpt_RI", "xi_RO", "Cpt_RO", "xi_SI", "Cpt_SI"])

    columns = ["xi_RI", "Cpt_RI", "Ma_RI", "MA_RI"]
    for section in ("RO", "SI", "SO"):
        columns.extend([f"xi_{section}", f"Cpt_{section}", f"Ma_{section}"])
    with pytest.raises(FourSectionAdapterError, match="duplicate canonical field"):
        four_section_layout(columns)
