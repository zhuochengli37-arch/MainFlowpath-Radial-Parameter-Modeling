from __future__ import annotations

from pathlib import Path

import pytest

from project1.experiments.benchmark_data import load_predict_samples, load_samples
from project1.experiments.four_section_adapter import audit_four_section_database
from project1.experiments.multi_section_adapter import (
    adapt_multi_section_file,
    adapt_multi_section_rows,
    multi_section_layout,
)
from project1.services.dataset_schema import (
    INSTITUTE_MULTI_SECTION_SCHEMA,
    INSTITUTE_SCHEMA_VERSION,
    detect_dataset_schema,
)


FOUR_FIXTURE_ROOT = Path(__file__).resolve().parent / "fixtures" / "institute_four"


def metadata(**overrides: object) -> dict[str, object]:
    values: dict[str, object] = {
        "component": "CMP",
        "database": "DATABASE_CMP",
        "stage": 1,
        "station": "MAIN",
        "speed_parameter": 0.76,
        "flow_parameter": 20.1,
        "speed_parameter_name": "CNC",
        "flow_parameter_name": "W25COR",
    }
    values.update(overrides)
    return values


def sectioned_table(names: list[str]) -> tuple[list[str], list[dict[str, object]]]:
    columns: list[str] = []
    row: dict[str, object] = {}
    for index, name in enumerate(names):
        columns.extend([f"xi_{name}", f"Value_{name}"])
        row[f"xi_{name}"] = index / 10.0
        row[f"Value_{name}"] = 100.0 + index
    return columns, [row]


@pytest.mark.parametrize("count", [1, 3, 5])
def test_section_count_is_discovered_without_core_changes(count: int):
    names = [chr(ord("A") + index) for index in range(count)]
    columns, rows = sectioned_table(names)

    data = adapt_multi_section_rows(columns, rows, metadata())

    assert tuple(data.sections) == tuple(names)
    assert len(data.sections) == count
    assert data.schema == INSTITUTE_MULTI_SECTION_SCHEMA


def test_unknown_section_and_output_names_are_preserved():
    columns = ["xi_MIXING", "Cpt_MIXING", "Alpha_MIXING"]
    rows = [{"xi_MIXING": 0.25, "Cpt_MIXING": 1.2, "Alpha_MIXING": 7.5}]

    data = adapt_multi_section_rows(columns, rows, metadata(component="FUTURE"))
    section = data.sections["MIXING"]

    assert section.source_section == "MIXING"
    assert section.outputs == {"Cpt": (1.2,), "Alpha": (7.5,)}
    assert section.source_fields["Alpha"] == "Alpha_MIXING"


def test_sections_may_have_different_outputs_and_xi_lengths():
    columns = ["xi_A", "Cpt_A", "Alpha_A", "xi_B", "Cpt_B"]
    rows = [
        {"xi_A": 0.0, "Cpt_A": 1.0, "Alpha_A": 2.0, "xi_B": 0.2, "Cpt_B": 3.0},
        {"xi_A": 0.5, "Cpt_A": 1.1, "Alpha_A": 2.1, "xi_B": None, "Cpt_B": None},
    ]

    data = adapt_multi_section_rows(columns, rows, metadata())

    assert data.sections["A"].xi == (0.0, 0.5)
    assert data.sections["A"].output_names == ("Cpt", "Alpha")
    assert data.sections["B"].xi == (0.2,)
    assert data.sections["B"].output_names == ("Cpt",)


def test_source_column_order_has_no_section_assignment_meaning():
    columns, rows = sectioned_table(["C", "A", "B"])
    data = adapt_multi_section_rows(columns, rows, metadata())

    assert tuple(data.sections) == ("C", "A", "B")
    assert data.sections["A"].xi == (0.1,)
    assert data.sections["B"].outputs["Value"] == (102.0,)
    assert data.sections["C"].outputs["Value"] == (100.0,)


def test_section_and_variable_aliases_preserve_source_names():
    columns = [
        "xi_ROTOR_INLET",
        "Ma_ROTOR_INLET",
        "xi_CUSTOM",
        "Mach_CUSTOM",
    ]
    rows = [{
        "xi_ROTOR_INLET": 0.0,
        "Ma_ROTOR_INLET": 0.3,
        "xi_CUSTOM": 0.4,
        "Mach_CUSTOM": 0.5,
    }]

    data = adapt_multi_section_rows(
        columns,
        rows,
        metadata(),
        variable_aliases={"mach": "MA"},
    )

    assert tuple(data.sections) == ("RI", "CUSTOM")
    assert data.sections["RI"].source_section == "ROTOR_INLET"
    assert data.sections["RI"].source_fields["MA"] == "Ma_ROTOR_INLET"
    assert data.sections["CUSTOM"].source_fields["MA"] == "Mach_CUSTOM"


def test_canonical_metadata_preserves_parameter_names_and_schema_version():
    source = FOUR_FIXTURE_ROOT / "DATABASE_CMP/STAGE_1/CNC_0.76/W25COR_20.1.dat"
    data = adapt_multi_section_file(source)

    assert data.speed_parameter == pytest.approx(0.76)
    assert data.flow_parameter == pytest.approx(20.1)
    assert data.speed_parameter_name == "CNC"
    assert data.flow_parameter_name == "W25COR"
    assert data.schema_name == INSTITUTE_MULTI_SECTION_SCHEMA
    assert data.schema_version == INSTITUTE_SCHEMA_VERSION
    assert data.units == {}


def test_units_are_recorded_only_when_explicitly_supplied():
    columns, rows = sectioned_table(["A"])
    without_units = adapt_multi_section_rows(columns, rows, metadata())
    with_units = adapt_multi_section_rows(
        columns,
        rows,
        metadata(),
        units={"xi": "normalized", "Value": "declared-unit"},
    )

    assert without_units.units == {}
    assert without_units.sections["A"].units == {}
    assert with_units.sections["A"].units == {
        "xi": "normalized",
        "Value": "declared-unit",
    }


def test_multi_section_schema_is_dynamic_and_still_rejected_by_model_loaders(tmp_path: Path):
    columns, rows = sectioned_table(["A", "B", "C"])
    assert detect_dataset_schema(columns) == INSTITUTE_MULTI_SECTION_SCHEMA

    source = tmp_path / "DATABASE_CMP" / "STAGE_1" / "CNC_0.76" / "W25COR_20.1.dat"
    source.parent.mkdir(parents=True)
    source.write_text(
        " ".join(columns) + "\n" + " ".join(str(rows[0][name]) for name in columns) + "\n",
        encoding="utf-8",
    )

    assert load_samples(str(tmp_path), radial_mode="full") == []
    assert load_predict_samples(str(tmp_path), radial_mode="full") == []


def test_unassigned_global_fields_are_retained_as_source_metadata():
    layout = multi_section_layout(["case_id", "xi_A", "Value_A"])
    data = adapt_multi_section_rows(
        ["case_id", "xi_A", "Value_A"],
        [{"case_id": 7, "xi_A": 0.2, "Value_A": 1.0}],
        metadata(),
    )

    assert layout.unassigned_fields == ("case_id",)
    assert data.source_metadata_fields == ("case_id",)
    assert data.source_metadata == {"case_id": (7,)}


def test_four_section_audit_keeps_dataset_specific_completeness_check(tmp_path: Path):
    columns, rows = sectioned_table(["RI", "RO", "SI"])
    source = tmp_path / "DATABASE_CMP" / "STAGE_1" / "CNC_0.76" / "W25COR_20.1.dat"
    source.parent.mkdir(parents=True)
    source.write_text(
        " ".join(columns) + "\n" + " ".join(str(rows[0][name]) for name in columns) + "\n",
        encoding="utf-8",
    )

    report = audit_four_section_database(tmp_path)

    assert report["adapted_main_files"] == 0
    assert len(report["rejected_files"]) == 1
    assert report["missing_section_files"] == [str(source)]
