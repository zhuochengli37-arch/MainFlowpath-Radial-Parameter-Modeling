from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from project1.services.data_reader import read_curve_file, read_tabular_file
from project1.services.dataset_schema import (
    INSTITUTE_FOUR_SECTION_SCHEMA,
    LEGACY_VALIDATION_SCHEMA,
    detect_dataset_schema,
    is_current_proxy_schema,
)
from project1.services.meta_parser import parse_metadata_from_path
from workbase.common.config_loader import load_config
from workbase.common.validators import ValidationError, validate_directory_exists, validate_file_exists


PROJECT_ROOT = Path(__file__).resolve().parents[4]
SCHEMA_CONFIG = load_config(str(PROJECT_ROOT / "config" / "benchmark_config.yaml"))
MISSING_VALUE_SENTINELS = tuple(SCHEMA_CONFIG.missing_value_sentinels)


@dataclass(frozen=True)
class Sample:
    component: str
    family: str
    station: str
    stage: int
    rpm: float
    wcor: float
    xi: float
    psi: float
    tsi: float
    mai: float
    section: str | None = None
    schema: str = LEGACY_VALIDATION_SCHEMA
    source_file: str = ""

    @property
    def speed_parameter(self) -> float:
        return self.rpm

    @property
    def flow_parameter(self) -> float:
        return self.wcor

    @property
    def source_path(self) -> str:
        return self.source_file

    def get_output(self, name: str) -> float:
        return float(getattr(self, name))

    @property
    def output_columns(self) -> tuple[str, ...]:
        return ("psi", "tsi", "mai")


class FlexSample:
    __slots__ = (
        "component",
        "family",
        "station",
        "section",
        "database",
        "stage",
        "speed_parameter",
        "flow_parameter",
        "xi",
        "schema",
        "source_file",
        "source_fields",
        "_outputs",
    )

    def __init__(
        self,
        component: str,
        family: str,
        station: str,
        database: str,
        stage: int,
        rpm: float,
        wcor: float,
        xi: float,
        source_path: str,
        outputs: dict[str, float],
        section: str | None = None,
        schema: str = LEGACY_VALIDATION_SCHEMA,
        source_fields: dict[str, str] | None = None,
    ) -> None:
        self.component = component
        self.family = family
        self.station = station
        self.section = section
        self.database = database
        self.stage = stage
        self.speed_parameter = rpm
        self.flow_parameter = wcor
        self.xi = xi
        self.schema = schema
        self.source_file = source_path
        self.source_fields = dict(source_fields or {})
        self._outputs = outputs

    @property
    def rpm(self) -> float:
        """Backward-compatible alias for the generic speed parameter."""

        return self.speed_parameter

    @property
    def wcor(self) -> float:
        """Backward-compatible alias for the generic flow parameter."""

        return self.flow_parameter

    @property
    def source_path(self) -> str:
        """Backward-compatible alias for the canonical source file."""

        return self.source_file

    def get_output(self, name: str) -> float:
        return float(self._outputs[name])

    @property
    def output_columns(self) -> tuple[str, ...]:
        return tuple(self._outputs.keys())

    @property
    def outputs(self) -> dict[str, float]:
        """Return a copy of the physical outputs present in this record."""

        return dict(self._outputs)


AnySample = Sample | FlexSample


def _schema_input_field(index: int, default: str) -> str:
    inputs = SCHEMA_CONFIG.schema_inputs
    if index < len(inputs):
        return str(inputs[index])
    return default


def schema_input_type(index_field_pairs: tuple[tuple[int, str], ...]) -> str:
    names = [_schema_input_field(index, field) for index, field in index_field_pairs]
    return "_".join(names)


def _schema_alias_candidates(field: str) -> list[str]:
    aliases = SCHEMA_CONFIG.schema_aliases.get(field, [])
    names = [field, *aliases]
    seen: set[str] = set()
    result: list[str] = []
    for name in names:
        normalized = str(name).strip()
        if not normalized:
            continue
        lowered = normalized.lower()
        if lowered in seen:
            continue
        seen.add(lowered)
        result.append(normalized)
    return result


def _match_column(columns: list[str], logical_field: str) -> str | None:
    lowered = {col.lower(): col for col in columns}
    for candidate in _schema_alias_candidates(logical_field):
        matched = lowered.get(candidate.lower())
        if matched is not None:
            return matched
    return None


def _coerce_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _coerce_int(value: object) -> int | None:
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _is_missing_sentinel(value: object) -> bool:
    numeric = _coerce_float(value)
    if numeric is None:
        return False
    return any(np.isclose(numeric, sentinel) for sentinel in MISSING_VALUE_SENTINELS)


def _extract_valid_outputs(row: dict[str, object], output_cols: list[str]) -> dict[str, float]:
    outputs: dict[str, float] = {}
    for col in output_cols:
        if col not in row:
            continue
        numeric = _coerce_float(row[col])
        if numeric is None or _is_missing_sentinel(numeric):
            continue
        outputs[col] = numeric
    return outputs


def all_targets(samples: list[AnySample]) -> tuple[str, ...]:
    seen: set[str] = set()
    ordered: list[str] = []
    for sample in samples:
        for target in sample.output_columns:
            if target in seen:
                continue
            seen.add(target)
            ordered.append(target)
    return tuple(ordered)


def _resolve_logical_value(
    logical_field: str,
    row: dict[str, object],
    meta: dict[str, object | None],
    columns: list[str],
) -> object | None:
    priorities = SCHEMA_CONFIG.schema_source_priority.get(logical_field, ["path", "column"])
    matched_column = _match_column(columns, logical_field)
    for source in priorities:
        if source == "path":
            value = meta.get(logical_field)
            if value is not None and value != "":
                return value
        elif source == "column" and matched_column is not None and matched_column in row:
            value = row.get(matched_column)
            if value is not None and value != "":
                return value
    if matched_column is not None:
        return row.get(matched_column)
    return meta.get(logical_field)


def _resolve_target_columns(columns: list[str], used_columns: set[str]) -> list[str]:
    matched_targets: list[str] = []
    for logical_target in SCHEMA_CONFIG.schema_targets:
        matched = _match_column(columns, logical_target)
        if matched is not None and matched not in used_columns and matched not in matched_targets:
            matched_targets.append(matched)
    if matched_targets:
        return matched_targets
    return [col for col in columns if col not in used_columns]


def _metadata_columns(columns: list[str]) -> set[str]:
    return {
        matched
        for logical_field in SCHEMA_CONFIG.schema_metadata
        for matched in [_match_column(columns, logical_field)]
        if matched is not None
    }


def inspect_dataset_file(file_path: str | Path) -> dict[str, object]:
    """Inspect path metadata and header schema without admitting data to training."""

    path = Path(file_path)
    meta = parse_metadata_from_path(str(path))
    parsed = read_curve_file(str(path))
    columns = [str(column) for column in parsed["columns"]]
    station = str(meta.get("station") or "MAIN").upper()
    schema = detect_dataset_schema(columns, station=station)
    return {
        "component": meta.get("component"),
        "stage": meta.get("stage"),
        "station": station,
        "section": meta.get("section"),
        "speed_parameter": meta.get("speed_parameter"),
        "flow_parameter": meta.get("flow_parameter"),
        "xi": None,
        "schema": schema,
        "source_file": str(path),
        "columns": tuple(columns),
        "trainable": station == "MAIN" and is_current_proxy_schema(schema),
    }


def resolve_1d_input(columns: list[str]) -> tuple[str, str]:
    for logical_input in SCHEMA_CONFIG.schema_inputs:
        matched = _match_column(columns, str(logical_input))
        if matched is not None:
            return str(logical_input), matched
    fallback = columns[0]
    return fallback, fallback


def resolve_1d_targets(columns: list[str], input_col: str) -> dict[str, str]:
    resolved: dict[str, str] = {}
    for logical_target in SCHEMA_CONFIG.schema_targets:
        matched = _match_column(columns, str(logical_target))
        if matched is not None and matched != input_col and str(logical_target) not in resolved:
            resolved[str(logical_target)] = matched
    if resolved:
        return resolved
    return {col: col for col in columns if col != input_col}


def resolve_1d_input_files(input_path: Path) -> list[Path]:
    if input_path.is_file():
        validate_file_exists(input_path, "输入文件")
        return [input_path]

    validate_directory_exists(input_path, "输入目录")
    candidate_files = sorted(
        path
        for path in input_path.rglob("*")
        if path.is_file() and path.suffix.lower() in {".txt", ".dat"} and not path.name.lower().endswith(".bak")
    )
    if not candidate_files:
        raise ValidationError(f"目录中没有找到 .txt 或 .dat 文件: {input_path}")
    return candidate_files


def sample_partition_from_keys(sample: AnySample, keys: list[str]) -> str:
    parts: list[str] = []
    for key in keys:
        value = getattr(sample, key, None)
        if value is None:
            continue
        if key.lower() == "stage":
            parts.append(f"S{int(value)}")
        else:
            parts.append(str(value))
    return ":".join(parts) if parts else "all"


def family_of(component: str) -> str:
    upper = component.upper()
    if "CMP" in upper or "FAN" in upper:
        return "compressor"
    if "TB" in upper or "TURB" in upper:
        return "turbine"
    if "CC" in upper or "COMB" in upper or "BURNER" in upper:
        return "combustor"
    return upper.lower()


def iter_data_files(input_dir: str) -> list[Path]:
    root = Path(input_dir)
    result: list[Path] = []
    for path in root.rglob("*"):
        if path.name.lower().endswith(".bak"):
            continue
        if path.is_file() and path.suffix.lower() in {".txt", ".dat"}:
            result.append(path)
    return result


def rows_for_mode(rows: list[dict[str, object]], xi_col: str, radial_mode: str) -> list[dict[str, object]]:
    valid = [row for row in rows if xi_col in row]
    if radial_mode == "full":
        return valid
    if len(valid) <= 2:
        return valid
    sorted_rows = sorted(valid, key=lambda item: float(item[xi_col]))
    return [sorted_rows[0], sorted_rows[-1]]


def load_samples(input_dir: str, radial_mode: str) -> list[FlexSample]:
    samples: list[FlexSample] = []
    for file_path in iter_data_files(input_dir):
        meta = parse_metadata_from_path(str(file_path))
        station = str(meta.get("station") or "MAIN").upper()
        if station != "MAIN":
            continue
        try:
            parsed = read_curve_file(str(file_path))
        except ValueError:
            continue
        columns = parsed["columns"]
        if not columns:
            continue
        schema = detect_dataset_schema(columns)
        if not is_current_proxy_schema(schema):
            continue
        rpm_field = _schema_input_field(0, "rpm")
        wcor_field = _schema_input_field(1, "wcor")
        xi_field = _schema_input_field(2, "xi")
        xi_col = _match_column(columns, xi_field) or columns[0]
        used_columns = {
            col
            for col in (
                _match_column(columns, rpm_field),
                _match_column(columns, wcor_field),
                _match_column(columns, xi_field),
            )
            if col is not None
        } | _metadata_columns(columns)
        output_cols = _resolve_target_columns(columns, used_columns)
        if not output_cols:
            continue
        for row in rows_for_mode(parsed["rows"], xi_col, radial_mode):
            component_value = _resolve_logical_value("component", row, meta, columns)
            stage_value = _resolve_logical_value("stage", row, meta, columns)
            database_value = _resolve_logical_value("database", row, meta, columns)
            rpm_value = _resolve_logical_value(rpm_field, row, meta, columns)
            wcor_value = _resolve_logical_value(wcor_field, row, meta, columns)
            xi_value = _resolve_logical_value(xi_field, row, meta, columns)
            component = str(component_value or "").upper()
            stage = _coerce_int(stage_value)
            rpm = _coerce_float(rpm_value)
            wcor = _coerce_float(wcor_value)
            xi = _coerce_float(xi_value)
            if not component or stage is None or rpm is None or wcor is None or xi is None:
                continue
            outputs = _extract_valid_outputs(row, output_cols)
            if not outputs:
                continue
            samples.append(
                FlexSample(
                    component=component,
                    family=family_of(component),
                    station=station,
                    database=str(database_value) if database_value else f"DATABASE_{component}",
                    stage=stage,
                    rpm=rpm,
                    wcor=wcor,
                    xi=xi,
                    source_path=str(file_path),
                    outputs=outputs,
                    section=None,
                    schema=schema,
                )
            )
    return samples


def load_predict_samples(input_dir: str, radial_mode: str) -> list[FlexSample]:
    samples: list[FlexSample] = []
    for file_path in iter_data_files(input_dir):
        meta = parse_metadata_from_path(str(file_path))
        station = str(meta.get("station") or "MAIN").upper()
        if station != "MAIN":
            continue
        try:
            parsed = read_tabular_file(str(file_path))
        except (ValueError, Exception):
            continue
        columns = parsed["columns"]
        if not columns:
            continue
        schema = detect_dataset_schema(columns)
        if schema == INSTITUTE_FOUR_SECTION_SCHEMA:
            continue
        rpm_field = _schema_input_field(0, "rpm")
        wcor_field = _schema_input_field(1, "wcor")
        xi_field = _schema_input_field(2, "xi")
        xi_col = _match_column(columns, xi_field) or columns[0]
        used_columns = {
            col
            for col in (
                _match_column(columns, rpm_field),
                _match_column(columns, wcor_field),
                _match_column(columns, xi_field),
            )
            if col is not None
        } | _metadata_columns(columns)
        output_cols = _resolve_target_columns(columns, used_columns)
        for row in rows_for_mode(parsed["rows"], xi_col, radial_mode):
            component_value = _resolve_logical_value("component", row, meta, columns)
            stage_value = _resolve_logical_value("stage", row, meta, columns)
            database_value = _resolve_logical_value("database", row, meta, columns)
            rpm_value = _resolve_logical_value(rpm_field, row, meta, columns)
            wcor_value = _resolve_logical_value(wcor_field, row, meta, columns)
            xi_value = _resolve_logical_value(xi_field, row, meta, columns)
            component = str(component_value or "").upper()
            stage = _coerce_int(stage_value)
            rpm = _coerce_float(rpm_value)
            wcor = _coerce_float(wcor_value)
            xi = _coerce_float(xi_value)
            if not component or stage is None or rpm is None or wcor is None or xi is None:
                continue
            outputs = _extract_valid_outputs(row, output_cols) if output_cols else {}
            samples.append(
                FlexSample(
                    component=component,
                    family=family_of(component),
                    station=station,
                    database=str(database_value) if database_value else f"DATABASE_{component}",
                    stage=stage,
                    rpm=rpm,
                    wcor=wcor,
                    xi=xi,
                    source_path=str(file_path),
                    outputs=outputs,
                    section=None,
                    schema=schema,
                )
            )
    return samples
