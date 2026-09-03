"""Generic adapter from sectioned wide tables to canonical multi-section data.

The data layer discovers sections and outputs from headers.  It does not select
model targets, align radial grids, convert units, or perform physical-variable
conversion.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
import csv
from typing import Mapping

from project1.experiments.benchmark_data import FlexSample, family_of, inspect_dataset_file
from project1.services.data_reader import read_curve_file
from project1.services.dataset_schema import (
    INSTITUTE_FOUR_SECTION_SCHEMA,
    INSTITUTE_MULTI_SECTION_SCHEMA,
    INSTITUTE_SCHEMA_VERSION,
    INSTITUTE_SINGLE_SECTION_SCHEMA,
    detect_dataset_schema,
)
from project1.services.meta_parser import parse_metadata_from_path
from project1.services.section_registry import (
    SectionDescriptor,
    canonical_variable_name,
    discover_sections,
    split_section_column,
)


class MultiSectionAdapterError(ValueError):
    """Raised when sectioned source data cannot be represented canonically."""


@dataclass(frozen=True)
class SectionLayout:
    """Header-derived fields for one section."""

    name: str
    source_name: str
    source_fields: dict[str, str]


@dataclass(frozen=True)
class MultiSectionLayout:
    """Dynamic source-order layout discovered from a wide-table header."""

    sections: tuple[SectionLayout, ...]
    unassigned_fields: tuple[str, ...] = ()

    @property
    def section_names(self) -> tuple[str, ...]:
        return tuple(section.name for section in self.sections)


@dataclass(frozen=True)
class CanonicalSectionData:
    """One independent radial coordinate and its available output arrays."""

    section: str
    source_section: str
    xi: tuple[float, ...]
    outputs: dict[str, tuple[float, ...]]
    source_fields: dict[str, str]
    units: dict[str, str] = field(default_factory=dict)

    @property
    def output_names(self) -> tuple[str, ...]:
        return tuple(self.outputs)


@dataclass(frozen=True)
class CanonicalMultiSectionData:
    """All sections discovered in one operating-condition source file."""

    component: str
    database: str
    stage: int
    station: str
    speed_parameter: float
    flow_parameter: float
    speed_parameter_name: str | None
    flow_parameter_name: str | None
    schema: str
    schema_name: str
    schema_version: str
    source_file: str
    sections: dict[str, CanonicalSectionData]
    source_metadata_fields: tuple[str, ...] = ()
    source_metadata: dict[str, tuple[object, ...]] = field(default_factory=dict)
    units: dict[str, str] = field(default_factory=dict)


def multi_section_layout(
    columns: list[object],
    radial_variable: str = "xi",
    section_aliases: Mapping[str, str] | None = None,
    variable_aliases: Mapping[str, str] | None = None,
) -> MultiSectionLayout:
    """Discover M sections and N outputs per section from `<variable>_<section>`."""

    try:
        descriptors = discover_sections(
            columns,
            radial_variable=radial_variable,
            section_aliases=section_aliases,
            variable_aliases=variable_aliases,
        )
    except ValueError as exc:
        raise MultiSectionAdapterError(str(exc)) from exc
    if not descriptors:
        raise MultiSectionAdapterError(
            f"no section found from radial field {radial_variable!r}"
        )

    fields: dict[str, dict[str, str]] = {
        descriptor.name: {} for descriptor in descriptors
    }
    unassigned: list[str] = []
    for raw_column in columns:
        source_column = str(raw_column).strip().replace("\ufeff", "")
        split = split_section_column(source_column, descriptors)
        if split is None:
            unassigned.append(source_column)
            continue
        source_variable, descriptor = split
        canonical_variable = canonical_variable_name(source_variable, variable_aliases)
        section_fields = fields[descriptor.name]
        if canonical_variable in section_fields:
            previous = section_fields[canonical_variable]
            raise MultiSectionAdapterError(
                "duplicate canonical field "
                f"{canonical_variable!r} for section {descriptor.name!r}: "
                f"{previous!r}, {source_column!r}"
            )
        section_fields[canonical_variable] = source_column

    radial_name = canonical_variable_name(radial_variable, variable_aliases)
    layouts: list[SectionLayout] = []
    for descriptor in descriptors:
        section_fields = fields[descriptor.name]
        if radial_name not in section_fields:
            raise MultiSectionAdapterError(
                f"missing radial field {radial_name!r} for section {descriptor.name!r}"
            )
        if len(section_fields) == 1:
            raise MultiSectionAdapterError(
                f"no output fields for section {descriptor.name!r}"
            )
        layouts.append(
            SectionLayout(
                name=descriptor.name,
                source_name=descriptor.source_name,
                source_fields=dict(section_fields),
            )
        )

    return MultiSectionLayout(
        sections=tuple(layouts),
        unassigned_fields=tuple(unassigned),
    )


def _numeric_or_none(value: object) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _required_metadata(
    metadata: Mapping[str, object | None],
    source_file: str,
) -> tuple[str, str, int, str, float, float]:
    component = str(metadata.get("component") or "").upper()
    database = str(metadata.get("database") or f"DATABASE_{component}")
    station = str(metadata.get("station") or "MAIN").upper()
    try:
        stage = int(metadata["stage"])
        speed = float(metadata["speed_parameter"])
        flow = float(metadata["flow_parameter"])
    except (KeyError, TypeError, ValueError):
        raise MultiSectionAdapterError(f"incomplete operating-condition metadata: {source_file}")
    if not component:
        raise MultiSectionAdapterError(f"missing component metadata: {source_file}")
    return component, database, stage, station, speed, flow


def adapt_multi_section_rows(
    columns: list[object],
    rows: list[dict[str, object]],
    metadata: Mapping[str, object | None],
    source_file: str = "<memory>",
    radial_variable: str = "xi",
    section_aliases: Mapping[str, str] | None = None,
    variable_aliases: Mapping[str, str] | None = None,
    units: Mapping[str, str] | None = None,
) -> CanonicalMultiSectionData:
    """Convert parsed rows without assuming section count, names, order, or outputs."""

    component, database, stage, station, speed, flow = _required_metadata(
        metadata, source_file
    )
    if station != "MAIN":
        raise MultiSectionAdapterError(
            f"station {station} is not a multi-section MAIN file"
        )
    layout = multi_section_layout(
        columns,
        radial_variable=radial_variable,
        section_aliases=section_aliases,
        variable_aliases=variable_aliases,
    )
    schema = detect_dataset_schema(columns, station=station)
    if schema not in {INSTITUTE_FOUR_SECTION_SCHEMA, INSTITUTE_MULTI_SECTION_SCHEMA}:
        raise MultiSectionAdapterError(f"schema {schema} is not institute multi-section")

    radial_name = canonical_variable_name(radial_variable, variable_aliases)
    canonical_sections: dict[str, CanonicalSectionData] = {}
    unit_map = dict(units or {})
    for section_layout in layout.sections:
        xi_source = section_layout.source_fields[radial_name]
        output_sources = {
            name: source
            for name, source in section_layout.source_fields.items()
            if name != radial_name
        }
        xi_values: list[float] = []
        output_values: dict[str, list[float]] = {
            name: [] for name in output_sources
        }
        for row_number, row in enumerate(rows, start=2):
            xi = _numeric_or_none(row.get(xi_source))
            current_outputs = {
                name: _numeric_or_none(row.get(source))
                for name, source in output_sources.items()
            }
            if xi is None:
                if any(value is not None for value in current_outputs.values()):
                    raise MultiSectionAdapterError(
                        f"section {section_layout.name!r} has outputs without xi "
                        f"at line {row_number}"
                    )
                continue
            missing_outputs = [
                name for name, value in current_outputs.items() if value is None
            ]
            if missing_outputs:
                raise MultiSectionAdapterError(
                    f"section {section_layout.name!r} has missing/non-numeric outputs "
                    f"at line {row_number}: {', '.join(missing_outputs)}"
                )
            xi_values.append(xi)
            for name, value in current_outputs.items():
                output_values[name].append(float(value))

        if not xi_values:
            raise MultiSectionAdapterError(
                f"section {section_layout.name!r} has no numeric radial data"
            )
        canonical_sections[section_layout.name] = CanonicalSectionData(
            section=section_layout.name,
            source_section=section_layout.source_name,
            xi=tuple(xi_values),
            outputs={name: tuple(values) for name, values in output_values.items()},
            source_fields=dict(section_layout.source_fields),
            units={name: unit_map[name] for name in section_layout.source_fields if name in unit_map},
        )

    speed_name = metadata.get("speed_parameter_name")
    flow_name = metadata.get("flow_parameter_name")
    return CanonicalMultiSectionData(
        component=component,
        database=database,
        stage=stage,
        station=station,
        speed_parameter=speed,
        flow_parameter=flow,
        speed_parameter_name=str(speed_name) if speed_name else None,
        flow_parameter_name=str(flow_name) if flow_name else None,
        schema=schema,
        schema_name=INSTITUTE_MULTI_SECTION_SCHEMA,
        schema_version=INSTITUTE_SCHEMA_VERSION,
        source_file=source_file,
        sections=canonical_sections,
        source_metadata_fields=layout.unassigned_fields,
        source_metadata={
            name: tuple(row.get(name) for row in rows)
            for name in layout.unassigned_fields
        },
        units=unit_map,
    )


def _tokens(line: str, comma_delimited: bool) -> list[str]:
    if comma_delimited:
        return [cell.strip() for cell in next(csv.reader([line]))]
    return line.strip().replace("\t", " ").split()


def _validate_source_row_widths(path: Path) -> None:
    lines = [
        line
        for line in path.read_text(encoding="utf-8", errors="ignore").splitlines()
        if line.strip()
    ]
    if len(lines) < 2:
        raise MultiSectionAdapterError("file has no usable tabular content")
    comma_delimited = "," in lines[0] and "\t" not in lines[0]
    expected = len(_tokens(lines[0], comma_delimited))
    for line_number, line in enumerate(lines[1:], start=2):
        actual = len(_tokens(line, comma_delimited))
        if actual != expected:
            raise MultiSectionAdapterError(
                f"row column count mismatch at line {line_number}: "
                f"expected {expected}, got {actual}"
            )


def adapt_multi_section_file(
    file_path: str | Path,
    radial_variable: str = "xi",
    section_aliases: Mapping[str, str] | None = None,
    variable_aliases: Mapping[str, str] | None = None,
    units: Mapping[str, str] | None = None,
) -> CanonicalMultiSectionData:
    """Parse one sectioned wide-table source file into canonical data."""

    source = Path(file_path)
    _validate_source_row_widths(source)
    parsed = read_curve_file(str(source))
    metadata = parse_metadata_from_path(str(source))
    return adapt_multi_section_rows(
        columns=list(parsed["columns"]),
        rows=list(parsed["rows"]),
        metadata=metadata,
        source_file=str(source),
        radial_variable=radial_variable,
        section_aliases=section_aliases,
        variable_aliases=variable_aliases,
        units=units,
    )


def flatten_multi_section_data(
    data: CanonicalMultiSectionData,
    section_order: tuple[str, ...] | None = None,
) -> list[FlexSample]:
    """Expose canonical arrays as legacy-compatible per-radial-point samples."""

    ordered_names = list(section_order or tuple(data.sections))
    ordered_names.extend(name for name in data.sections if name not in ordered_names)
    result: list[FlexSample] = []
    for section_name in ordered_names:
        section = data.sections.get(section_name)
        if section is None:
            continue
        for index, xi in enumerate(section.xi):
            outputs = {
                name: values[index] for name, values in section.outputs.items()
            }
            result.append(
                FlexSample(
                    component=data.component,
                    family=family_of(data.component),
                    station=data.station,
                    database=data.database,
                    stage=data.stage,
                    rpm=data.speed_parameter,
                    wcor=data.flow_parameter,
                    xi=xi,
                    source_path=data.source_file,
                    outputs=outputs,
                    section=section.section,
                    schema=data.schema,
                    source_fields=dict(section.source_fields),
                    schema_name=data.schema_name,
                    schema_version=data.schema_version,
                    speed_parameter_name=data.speed_parameter_name,
                    flow_parameter_name=data.flow_parameter_name,
                    units={**data.units, **section.units},
                )
            )
    return result


def audit_multi_section_database(
    input_dir: str | Path,
    expected_sections: set[str] | None = None,
) -> dict[str, object]:
    """Scan a DATABASE tree, with optional dataset-specific section validation."""

    root = Path(input_dir)
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in {".dat", ".txt"}
    )
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
        metadata = parse_metadata_from_path(str(source))
        component = str(metadata.get("component") or "UNKNOWN").upper()
        station = str(metadata.get("station") or "MAIN").upper()
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
            data = adapt_multi_section_file(source)
            if expected_sections is not None:
                present = set(data.sections)
                missing = expected_sections - present
                unexpected = present - expected_sections
                if missing:
                    raise MultiSectionAdapterError(
                        "missing section(s): " + ", ".join(sorted(missing))
                    )
                if unexpected:
                    raise MultiSectionAdapterError(
                        "unexpected section(s): " + ", ".join(sorted(unexpected))
                    )
            adapted_main_files += 1
            source_orders[component].add(
                tuple(section.source_section for section in data.sections.values())
            )
            sections_by_component[component].update(data.sections)
            for section in data.sections.values():
                outputs_by_component[component].update(section.outputs)
        except (OSError, ValueError) as exc:
            reason = str(exc)
            rejected.append({"source_file": str(source), "reason": reason})
            lowered = reason.lower()
            if "missing section" in lowered or "no section found" in lowered:
                missing_sections.append(str(source))
            if "duplicate section" in lowered:
                duplicate_sections.append(str(source))
            if "row column count mismatch" in lowered:
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
