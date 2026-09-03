"""Adapter for Institute four-section wide DATABASE files.

This module deliberately stops at canonical data conversion.  It is not
imported by the existing 1D/2D/3D training or prediction loaders.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path
import csv

from project1.experiments.benchmark_data import FlexSample, family_of, inspect_dataset_file
from project1.services.data_reader import read_curve_file
from project1.services.dataset_schema import (
    INSTITUTE_FOUR_SECTION_NAMES,
    INSTITUTE_FOUR_SECTION_SCHEMA,
    INSTITUTE_SINGLE_SECTION_SCHEMA,
    detect_dataset_schema,
)
from project1.services.meta_parser import parse_metadata_from_path


CANONICAL_SECTION_ORDER = ("RI", "RO", "SI", "SO")
_CANONICAL_FIELD_NAMES = {
    "xi": "xi",
    "cpt": "Cpt",
    "cps": "Cps",
    "ctt": "Ctt",
    "cts": "Cts",
    "ma": "MA",
    "vz": "Vz",
    "rho": "Rho",
}


class FourSectionAdapterError(ValueError):
    """Raised when a file cannot be represented as four independent sections."""


def _split_section_column(column: object) -> tuple[str, str] | None:
    name = str(column).strip().replace("\ufeff", "")
    if "_" not in name:
        return None
    source_field, suffix = name.rsplit("_", 1)
    section = suffix.upper()
    if not source_field or section not in INSTITUTE_FOUR_SECTION_NAMES:
        return None
    return source_field, section


def _canonical_field(source_field: str) -> str:
    return _CANONICAL_FIELD_NAMES.get(source_field.lower(), source_field)


def four_section_layout(columns: list[object]) -> dict[str, object]:
    """Build a section-to-field map from suffixes, independent of column order."""

    fields_by_section: dict[str, dict[str, str]] = defaultdict(dict)
    source_section_order: list[str] = []
    unsuffixed: list[str] = []

    for raw_column in columns:
        column = str(raw_column).strip().replace("\ufeff", "")
        split = _split_section_column(column)
        if split is None:
            unsuffixed.append(column)
            continue
        source_field, section = split
        if section not in source_section_order:
            source_section_order.append(section)
        canonical = _canonical_field(source_field)
        if canonical in fields_by_section[section]:
            previous = fields_by_section[section][canonical]
            raise FourSectionAdapterError(
                "duplicate canonical field "
                f"{canonical!r} for section {section}: {previous!r}, {column!r}"
            )
        fields_by_section[section][canonical] = column

    present = set(fields_by_section)
    missing = set(CANONICAL_SECTION_ORDER) - present
    unexpected = present - set(CANONICAL_SECTION_ORDER)
    if missing:
        raise FourSectionAdapterError(
            "missing section(s): " + ", ".join(sorted(missing))
        )
    if unexpected:
        raise FourSectionAdapterError(
            "unexpected section(s): " + ", ".join(sorted(unexpected))
        )
    if unsuffixed:
        raise FourSectionAdapterError(
            "mixed suffixed and unsuffixed columns: " + ", ".join(unsuffixed)
        )

    expected_outputs: set[str] | None = None
    for section in CANONICAL_SECTION_ORDER:
        mapping = fields_by_section[section]
        if "xi" not in mapping:
            raise FourSectionAdapterError(f"missing xi for section {section}")
        outputs = set(mapping) - {"xi"}
        if not outputs:
            raise FourSectionAdapterError(f"no output fields for section {section}")
        if expected_outputs is None:
            expected_outputs = outputs
        elif outputs != expected_outputs:
            raise FourSectionAdapterError(
                f"inconsistent output fields for section {section}: "
                f"expected {sorted(expected_outputs)}, got {sorted(outputs)}"
            )

    return {
        "source_section_order": tuple(source_section_order),
        "sections": {
            section: dict(fields_by_section[section])
            for section in CANONICAL_SECTION_ORDER
        },
    }


def _tokens(line: str, comma_delimited: bool) -> list[str]:
    if comma_delimited:
        return [cell.strip() for cell in next(csv.reader([line]))]
    return line.strip().replace("\t", " ").split()


def _validate_source_row_widths(path: Path) -> None:
    lines = [line for line in path.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
    if len(lines) < 2:
        raise FourSectionAdapterError("file has no usable tabular content")
    comma_delimited = "," in lines[0] and "\t" not in lines[0]
    expected = len(_tokens(lines[0], comma_delimited))
    for line_number, line in enumerate(lines[1:], start=2):
        actual = len(_tokens(line, comma_delimited))
        if actual != expected:
            raise FourSectionAdapterError(
                f"row column count mismatch at line {line_number}: expected {expected}, got {actual}"
            )


def _required_metadata(meta: dict[str, object | None], source: Path) -> tuple[str, int, float, float]:
    component = str(meta.get("component") or "").upper()
    try:
        stage = int(meta["stage"])
        speed = float(meta["speed_parameter"])
        flow = float(meta["flow_parameter"])
    except (KeyError, TypeError, ValueError):
        raise FourSectionAdapterError(f"incomplete path metadata: {source}")
    if not component:
        raise FourSectionAdapterError(f"missing component metadata: {source}")
    return component, stage, speed, flow


def adapt_four_section_file(file_path: str | Path) -> list[FlexSample]:
    """Split one MAIN wide table into canonical RI/RO/SI/SO records."""

    source = Path(file_path)
    meta = parse_metadata_from_path(str(source))
    station = str(meta.get("station") or "MAIN").upper()
    if station != "MAIN":
        raise FourSectionAdapterError(
            f"station {station} is not a four-section MAIN file"
        )

    _validate_source_row_widths(source)
    parsed = read_curve_file(str(source))
    columns = list(parsed["columns"])
    schema = detect_dataset_schema(columns, station=station)
    if schema != INSTITUTE_FOUR_SECTION_SCHEMA:
        raise FourSectionAdapterError(f"schema {schema} is not institute four-section")

    layout = four_section_layout(columns)
    component, stage, speed, flow = _required_metadata(meta, source)
    database = str(meta.get("database") or f"DATABASE_{component}")
    result: list[FlexSample] = []

    for section in CANONICAL_SECTION_ORDER:
        source_fields = layout["sections"][section]
        xi_source = source_fields["xi"]
        output_sources = {
            canonical: source_column
            for canonical, source_column in source_fields.items()
            if canonical != "xi"
        }
        for row_number, row in enumerate(parsed["rows"], start=2):
            try:
                xi = float(row[xi_source])
                outputs = {
                    canonical: float(row[source_column])
                    for canonical, source_column in output_sources.items()
                }
            except (KeyError, TypeError, ValueError) as exc:
                raise FourSectionAdapterError(
                    f"invalid or missing value for section {section} at line {row_number}"
                ) from exc
            result.append(
                FlexSample(
                    component=component,
                    family=family_of(component),
                    station="MAIN",
                    database=database,
                    stage=stage,
                    rpm=speed,
                    wcor=flow,
                    xi=xi,
                    source_path=str(source),
                    outputs=outputs,
                    section=section,
                    schema=INSTITUTE_FOUR_SECTION_SCHEMA,
                    source_fields=dict(source_fields),
                )
            )

    expected_rows = int(parsed["row_count"])
    section_counts = Counter(sample.section for sample in result)
    if any(section_counts[section] != expected_rows for section in CANONICAL_SECTION_ORDER):
        raise FourSectionAdapterError("section field row counts are inconsistent")
    return result


def audit_four_section_database(input_dir: str | Path) -> dict[str, object]:
    """Inspect every DATABASE file and adapt only valid MAIN four-section files."""

    root = Path(input_dir)
    files = sorted(path for path in root.rglob("*") if path.is_file() and path.suffix.lower() in {".dat", ".txt"})
    component_counts: Counter[str] = Counter()
    station_counts: Counter[str] = Counter()
    schema_counts: Counter[str] = Counter()
    source_orders: dict[str, set[tuple[str, ...]]] = defaultdict(set)
    sections_by_component: dict[str, set[str]] = defaultdict(set)
    outputs_by_component: dict[str, set[str]] = defaultdict(set)
    rejected: list[dict[str, str]] = []
    missing_sections: list[str] = []
    duplicate_sections: list[str] = []
    header_anomalies: list[str] = []
    row_count_mismatches: list[str] = []
    parsed_files = 0
    adapted_main_files = 0
    boundary_single_files = 0

    for source in files:
        meta = parse_metadata_from_path(str(source))
        component = str(meta.get("component") or "UNKNOWN").upper()
        station = str(meta.get("station") or "MAIN").upper()
        component_counts[component] += 1
        station_counts[station] += 1
        try:
            inspection = inspect_dataset_file(source)
            schema = str(inspection["schema"])
            schema_counts[schema] += 1
            parsed_files += 1
            if station != "MAIN":
                if schema == INSTITUTE_SINGLE_SECTION_SCHEMA:
                    boundary_single_files += 1
                else:
                    header_anomalies.append(str(source))
                continue

            parsed = read_curve_file(str(source))
            layout = four_section_layout(list(parsed["columns"]))
            source_orders[component].add(tuple(layout["source_section_order"]))
            samples = adapt_four_section_file(source)
            adapted_main_files += 1
            sections_by_component[component].update(
                str(sample.section) for sample in samples if sample.section is not None
            )
            for sample in samples:
                outputs_by_component[component].update(sample.output_columns)
        except (OSError, ValueError) as exc:
            reason = str(exc)
            rejected.append({"source_file": str(source), "reason": reason})
            lowered = reason.lower()
            if "missing section" in lowered:
                missing_sections.append(str(source))
            if "duplicate" in lowered:
                duplicate_sections.append(str(source))
            if "row column count mismatch" in lowered or "row counts are inconsistent" in lowered:
                row_count_mismatches.append(str(source))
            if station == "MAIN" and str(source) not in row_count_mismatches:
                header_anomalies.append(str(source))

    return {
        "total_files": len(files),
        "component_counts": dict(sorted(component_counts.items())),
        "station_counts": dict(sorted(station_counts.items())),
        "schema_counts": dict(sorted(schema_counts.items())),
        "parsed_files": parsed_files,
        "adapted_main_files": adapted_main_files,
        "boundary_single_files": boundary_single_files,
        "source_section_order_by_component": {
            component: [list(order) for order in sorted(orders)]
            for component, orders in sorted(source_orders.items())
        },
        "sections_by_component": {
            component: sorted(sections)
            for component, sections in sorted(sections_by_component.items())
        },
        "outputs_by_component": {
            component: sorted(outputs)
            for component, outputs in sorted(outputs_by_component.items())
        },
        "rejected_files": rejected,
        "missing_section_files": missing_sections,
        "duplicate_section_files": duplicate_sections,
        "header_anomaly_files": header_anomalies,
        "row_count_mismatch_files": row_count_mismatches,
    }
