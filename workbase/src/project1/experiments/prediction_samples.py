"""File-boundary adapters for normalized radial prediction samples.

Path and filename inference is confined to this module.  Once a
``PredictionSample`` exists, its structured values are authoritative.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from project1.experiments.benchmark_data import SCHEMA_CONFIG
from project1.experiments.multi_section_adapter import adapt_multi_section_file
from project1.services.data_reader import read_tabular_file
from project1.services.dataset_schema import (
    INSTITUTE_FOUR_SECTION_SCHEMA,
    INSTITUTE_MULTI_SECTION_SCHEMA,
    LEGACY_VALIDATION_SCHEMA,
    detect_dataset_schema,
    schema_identity,
)
from project1.services.meta_parser import parse_metadata_from_path


SUPPORTED_PREDICTION_SUFFIXES = {".txt", ".dat", ".csv"}
MULTI_SCHEMAS = {INSTITUTE_FOUR_SECTION_SCHEMA, INSTITUTE_MULTI_SECTION_SCHEMA}


@dataclass(frozen=True)
class PredictionSample:
    """One normalized radial prediction point with explicit provenance."""

    source_file: str
    component: str
    stage: int
    station: str
    section: str | None
    database: str
    speed_parameter: float
    speed_parameter_source_name: str | None
    flow_parameter: float | None
    flow_parameter_source_name: str | None
    xi: float
    schema_name: str
    schema_version: str
    row_index: int
    section_row_index: int
    source_fields: dict[str, str] = field(default_factory=dict)
    original_row: dict[str, object] = field(default_factory=dict)

    @property
    def rpm(self) -> float:
        return self.speed_parameter

    @property
    def wcor(self) -> float | None:
        return self.flow_parameter

    @property
    def source_path(self) -> str:
        return self.source_file


@dataclass(frozen=True)
class PredictionExclusion:
    source_file: str
    station: str
    reason: str


@dataclass(frozen=True)
class PredictionBatch:
    """Normalized samples plus file-boundary information for outer writers."""

    samples: tuple[PredictionSample, ...]
    source_columns: dict[str, tuple[str, ...]]
    exclusions: tuple[PredictionExclusion, ...] = ()


def _prediction_files(input_path: str | Path) -> list[Path]:
    root = Path(input_path)
    if root.is_file():
        return [root] if root.suffix.lower() in SUPPORTED_PREDICTION_SUFFIXES else []
    if not root.exists():
        raise FileNotFoundError(f"prediction input does not exist: {root}")
    return sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_PREDICTION_SUFFIXES
    )


def _aliases(logical_name: str) -> tuple[str, ...]:
    configured = SCHEMA_CONFIG.schema_aliases.get(logical_name, [])
    result: list[str] = []
    for value in (logical_name, *configured):
        normalized = str(value).strip()
        if normalized and normalized.casefold() not in {item.casefold() for item in result}:
            result.append(normalized)
    return tuple(result)


def _column(columns: Iterable[str], *logical_names: str) -> str | None:
    lookup = {str(column).strip().replace("\ufeff", "").casefold(): str(column) for column in columns}
    for logical_name in logical_names:
        for candidate in _aliases(logical_name):
            match = lookup.get(candidate.casefold())
            if match is not None:
                return match
    return None


def _float(value: object) -> float | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _integer(value: object) -> int | None:
    numeric = _float(value)
    return None if numeric is None else int(numeric)


def _value_then_path(
    row: dict[str, object],
    column: str | None,
    path_metadata: dict[str, object | None],
    metadata_key: str,
) -> object | None:
    """Apply content-before-path precedence without overwriting explicit values."""

    if column is not None:
        value = row.get(column)
        if value is not None and str(value).strip() != "":
            return value
    return path_metadata.get(metadata_key)


def _legacy_metadata_with_precedence(path: Path) -> dict[str, object | None]:
    """Resolve Legacy path metadata before falling back to filename inference.

    ``parse_metadata_from_path`` intentionally supports both sources for older
    callers. Prediction loading separates them here so the boundary contract is
    deterministic: parent-path metadata may fill a missing value before the
    filename gets a chance to do so.
    """

    path_only = parse_metadata_from_path(str(path.with_name("__prediction_input__.dat")))
    filename_only = parse_metadata_from_path(path.name)
    metadata: dict[str, object | None] = {}
    for key in ("component", "database", "section", "stage", "schema"):
        path_value = path_only.get(key)
        metadata[key] = path_value if path_value is not None else filename_only.get(key)
    path_has_component = path_only.get("component") is not None
    metadata["station"] = (
        path_only.get("station") if path_has_component else filename_only.get("station")
    )
    for value_key, name_key in (
        ("speed_parameter", "speed_parameter_name"),
        ("flow_parameter", "flow_parameter_name"),
    ):
        source = path_only if path_only.get(value_key) is not None else filename_only
        metadata[value_key] = source.get(value_key)
        metadata[name_key] = source.get(name_key)
    metadata["source_file"] = str(path)
    return metadata


def _selected_row_indices(
    rows: list[dict[str, object]],
    xi_column: str,
    radial_mode: str,
) -> list[int]:
    valid = [index for index, row in enumerate(rows) if _float(row.get(xi_column)) is not None]
    if radial_mode == "full" or len(valid) <= 2:
        return valid
    return [
        min(valid, key=lambda index: float(rows[index][xi_column])),
        max(valid, key=lambda index: float(rows[index][xi_column])),
    ]


def _load_single_file(
    path: Path,
    columns: list[str],
    rows: list[dict[str, object]],
    metadata: dict[str, object | None],
    schema: str,
    radial_mode: str,
    require_flow: bool,
) -> list[PredictionSample]:
    component_column = _column(columns, "component")
    stage_column = _column(columns, "stage")
    database_column = _column(columns, "database")
    speed_column = _column(columns, "speed_parameter", "rpm")
    flow_column = _column(columns, "flow_parameter", "wcor")
    xi_column = _column(columns, "xi")
    if xi_column is None:
        return []
    schema_name, schema_version = schema_identity(schema)
    normalized_columns = {str(column).strip().casefold() for column in columns}
    if schema == LEGACY_VALIDATION_SCHEMA and not {
        "psi",
        "tsi",
        "mai",
    }.intersection(normalized_columns):
        # Input-only legacy files do not contain enough information to assert a
        # training-data schema.  Keep the provenance explicit but do not create
        # a false schema incompatibility with an Institute Single model.
        schema_name = "unclassified_prediction"
    result: list[PredictionSample] = []
    for section_index, row_index in enumerate(
        _selected_row_indices(rows, xi_column, radial_mode)
    ):
        row = rows[row_index]
        component_value = _value_then_path(row, component_column, metadata, "component")
        stage_value = _value_then_path(row, stage_column, metadata, "stage")
        database_value = _value_then_path(row, database_column, metadata, "database")
        speed_value = _value_then_path(row, speed_column, metadata, "speed_parameter")
        flow_value = _value_then_path(row, flow_column, metadata, "flow_parameter")
        component = str(component_value or "").upper()
        stage = _integer(stage_value)
        speed = _float(speed_value)
        flow = _float(flow_value)
        xi = _float(row.get(xi_column))
        if not component or stage is None or speed is None or xi is None:
            continue
        if require_flow and flow is None:
            raise ValueError(f"3D prediction requires flow_parameter: {path}")
        result.append(
            PredictionSample(
                source_file=str(path),
                component=component,
                stage=stage,
                station="MAIN",
                section=None,
                database=str(database_value or f"DATABASE_{component}"),
                speed_parameter=speed,
                speed_parameter_source_name=(
                    speed_column or (
                        str(metadata["speed_parameter_name"])
                        if metadata.get("speed_parameter_name")
                        else None
                    )
                ),
                flow_parameter=flow,
                flow_parameter_source_name=(
                    flow_column or (
                        str(metadata["flow_parameter_name"])
                        if metadata.get("flow_parameter_name")
                        else None
                    )
                ),
                xi=xi,
                schema_name=schema_name,
                schema_version=schema_version,
                row_index=row_index,
                section_row_index=section_index,
                source_fields={
                    **({"speed_parameter": speed_column} if speed_column else {}),
                    **({"flow_parameter": flow_column} if flow_column else {}),
                    "xi": xi_column,
                },
                original_row=dict(row),
            )
        )
    return result


def _load_multi_file(path: Path, radial_mode: str, require_flow: bool) -> list[PredictionSample]:
    canonical = adapt_multi_section_file(path)
    if require_flow and canonical.flow_parameter is None:
        raise ValueError(f"3D prediction requires flow_parameter: {path}")
    result: list[PredictionSample] = []
    for section in canonical.sections.values():
        indices = list(range(len(section.xi)))
        if radial_mode != "full" and len(indices) > 2:
            indices = [
                min(indices, key=lambda index: section.xi[index]),
                max(indices, key=lambda index: section.xi[index]),
            ]
        for section_row_index in indices:
            source_row_index = section.source_row_indices[section_row_index]
            result.append(
                PredictionSample(
                    source_file=canonical.source_file,
                    component=canonical.component,
                    stage=canonical.stage,
                    station=canonical.station,
                    section=section.section,
                    database=canonical.database,
                    speed_parameter=canonical.speed_parameter,
                    speed_parameter_source_name=canonical.speed_parameter_name,
                    flow_parameter=canonical.flow_parameter,
                    flow_parameter_source_name=canonical.flow_parameter_name,
                    xi=section.xi[section_row_index],
                    schema_name=canonical.schema_name,
                    schema_version=canonical.schema_version,
                    row_index=source_row_index,
                    section_row_index=section_row_index,
                    source_fields=dict(section.source_fields),
                    original_row={
                        **{
                            name: values[source_row_index]
                            for name, values in canonical.source_metadata.items()
                        },
                        **{
                            source_name: (
                                section.xi[section_row_index]
                                if canonical_name == "xi"
                                else section.outputs[canonical_name][section_row_index]
                            )
                            for canonical_name, source_name in section.source_fields.items()
                        },
                    },
                )
            )
    return result


def load_prediction_batch(
    input_path: str | Path,
    *,
    radial_mode: str = "full",
    require_flow: bool = False,
) -> PredictionBatch:
    """Load Legacy, Single and Multi files into one prediction representation."""

    samples: list[PredictionSample] = []
    source_columns: dict[str, tuple[str, ...]] = {}
    exclusions: list[PredictionExclusion] = []
    for path in _prediction_files(input_path):
        metadata = _legacy_metadata_with_precedence(path)
        station = str(metadata.get("station") or "MAIN").upper()
        if station != "MAIN":
            exclusions.append(
                PredictionExclusion(
                    source_file=str(path),
                    station=station,
                    reason=f"station is not MAIN: {station}",
                )
            )
            continue
        try:
            parsed = read_tabular_file(str(path))
        except (OSError, ValueError) as exc:
            exclusions.append(
                PredictionExclusion(str(path), station, f"cannot parse prediction file: {exc}")
            )
            continue
        columns = [str(column) for column in parsed["columns"]]
        rows = list(parsed["rows"])
        source_columns[str(path)] = tuple(columns)
        try:
            schema = detect_dataset_schema(columns, station=station)
        except ValueError as exc:
            exclusions.append(
                PredictionExclusion(str(path), station, f"invalid prediction schema: {exc}")
            )
            continue
        if schema in MULTI_SCHEMAS:
            samples.extend(_load_multi_file(path, radial_mode, require_flow))
        else:
            samples.extend(
                _load_single_file(
                    path,
                    columns,
                    rows,
                    metadata,
                    schema,
                    radial_mode,
                    require_flow,
                )
            )
    return PredictionBatch(
        samples=tuple(samples),
        source_columns=source_columns,
        exclusions=tuple(exclusions),
    )
