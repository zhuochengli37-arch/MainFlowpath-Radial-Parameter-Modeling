from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from project1.services.data_reader import read_curve_file, read_tabular_file
from project1.services.dataset_schema import (
    LEGACY_VALIDATION_SCHEMA,
    detect_dataset_schema,
    is_current_proxy_schema,
    schema_identity,
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
        "schema_name",
        "schema_version",
        "source_file",
        "source_fields",
        "speed_parameter_name",
        "flow_parameter_name",
        "units",
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
        schema_name: str | None = None,
        schema_version: str | None = None,
        speed_parameter_name: str | None = None,
        flow_parameter_name: str | None = None,
        units: dict[str, str] | None = None,
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
        default_schema_name, default_schema_version = schema_identity(schema)
        self.schema_name = schema_name or default_schema_name
        self.schema_version = schema_version or default_schema_version
        self.source_file = source_path
        self.source_fields = dict(source_fields or {})
        self.speed_parameter_name = speed_parameter_name
        self.flow_parameter_name = flow_parameter_name
        self.units = dict(units or {})
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


@dataclass(frozen=True)
class OperatingCondition:
    """One source-file-level operating condition for the 1D workline."""

    component: str
    stage: int
    station: str
    speed_parameter: float
    flow_parameter: float
    schema: str
    schema_name: str
    schema_version: str
    source_file: str
    speed_parameter_source_name: str | None
    flow_parameter_source_name: str | None
    section: None = None

    @property
    def rpm(self) -> float:
        return self.speed_parameter

    @property
    def wcor(self) -> float:
        return self.flow_parameter

    @property
    def source_path(self) -> str:
        return self.source_file

    @property
    def speed_parameter_name(self) -> str | None:
        return self.speed_parameter_source_name

    @property
    def flow_parameter_name(self) -> str | None:
        return self.flow_parameter_source_name


@dataclass(frozen=True)
class PartitionTargetPlan:
    """Available database outputs and targets selected for one partition."""

    available_outputs: tuple[str, ...]
    selected_targets: tuple[str, ...]
    skipped_targets: dict[str, str]


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


def _extract_valid_outputs(
    row: dict[str, object],
    output_fields: dict[str, str],
) -> dict[str, float]:
    outputs: dict[str, float] = {}
    for canonical_name, source_column in output_fields.items():
        if source_column not in row:
            continue
        numeric = _coerce_float(row[source_column])
        if numeric is None or _is_missing_sentinel(numeric):
            continue
        outputs[canonical_name] = numeric
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


def _canonical_output_name(source_column: str) -> str:
    lowered = str(source_column).strip().lower()
    for logical_name, aliases in SCHEMA_CONFIG.schema_aliases.items():
        candidates = [logical_name, *aliases]
        if lowered in {str(candidate).strip().lower() for candidate in candidates}:
            return str(logical_name)
    return str(source_column)


def _discover_output_fields(columns: list[str], used_columns: set[str]) -> dict[str, str]:
    """Discover physical outputs independently from the training target selection."""

    result: dict[str, str] = {}
    for source_column in columns:
        if source_column in used_columns:
            continue
        canonical_name = _canonical_output_name(source_column)
        previous = result.get(canonical_name)
        if previous is not None and previous != source_column:
            raise ValueError(
                f"duplicate output field {canonical_name!r}: {previous!r} and {source_column!r}"
            )
        result[canonical_name] = source_column
    return result


def sample_available_outputs(sample: AnySample) -> tuple[str, ...]:
    source_fields = getattr(sample, "source_fields", None)
    if source_fields:
        return tuple(name for name in source_fields if str(name).lower() != "xi")
    return tuple(sample.output_columns)


def _target_selection_value(targets: str | Iterable[str] | None) -> str | list[str]:
    if targets is None:
        return SCHEMA_CONFIG.schema_target_selection
    if isinstance(targets, str):
        return targets
    return [str(target) for target in targets]


def resolve_partition_targets(
    samples: list[AnySample],
    targets: str | Iterable[str] | None = None,
    missing_target_policy: str | None = None,
) -> PartitionTargetPlan:
    """Resolve targets for one radial partition without component-specific rules."""

    if not samples:
        raise ValueError("cannot resolve targets for an empty partition")
    policy = str(missing_target_policy or SCHEMA_CONFIG.missing_target_policy).strip().lower()
    if policy not in {"error", "skip"}:
        raise ValueError(f"unsupported missing_target_policy: {policy!r}")

    source_signatures: dict[tuple[str, object], tuple[str, ...]] = {}
    available_order: list[str] = []
    seen_available: set[str] = set()
    for sample in samples:
        available = sample_available_outputs(sample)
        source_key = (str(sample.source_path), getattr(sample, "section", None))
        previous = source_signatures.get(source_key)
        if previous is not None and set(previous) != set(available):
            raise ValueError(f"inconsistent available outputs within source {source_key[0]!r}")
        source_signatures[source_key] = available
        for target in available:
            if target not in seen_available:
                seen_available.add(target)
                available_order.append(target)

    signature_sets = [set(signature) for signature in source_signatures.values()]
    common = set.intersection(*signature_sets) if signature_sets else set()
    inconsistent = any(signature != signature_sets[0] for signature in signature_sets[1:])
    selection = _target_selection_value(targets)
    skipped: dict[str, str] = {}

    if isinstance(selection, str):
        if selection.strip().lower() != "auto":
            raise ValueError("dataset_schema.targets must be 'auto' or a list of target names")
        if inconsistent and policy == "error":
            details = sorted({tuple(signature) for signature in source_signatures.values()})
            raise ValueError(f"inconsistent available outputs in partition: {details}")
        selected = [target for target in available_order if target in common]
        for target in available_order:
            if target not in common:
                skipped[target] = "not available in every source of the partition"
    else:
        selected = []
        lookup = {name.casefold(): name for name in available_order}
        for requested in selection:
            canonical_requested = _canonical_output_name(requested)
            actual = lookup.get(canonical_requested.casefold())
            if actual is None or actual not in common:
                reason = "not available in every source of the partition"
                if policy == "error":
                    raise ValueError(f"requested target {requested!r} is {reason}")
                skipped[requested] = reason
                continue
            if actual not in selected:
                selected.append(actual)

    return PartitionTargetPlan(
        available_outputs=tuple(available_order),
        selected_targets=tuple(selected),
        skipped_targets=skipped,
    )


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
    schema_name, schema_version = schema_identity(schema)
    return {
        "component": meta.get("component"),
        "stage": meta.get("stage"),
        "station": station,
        "section": meta.get("section"),
        "speed_parameter": meta.get("speed_parameter"),
        "flow_parameter": meta.get("flow_parameter"),
        "speed_parameter_name": meta.get("speed_parameter_name"),
        "flow_parameter_name": meta.get("flow_parameter_name"),
        "xi": None,
        "schema": schema,
        "schema_name": schema_name,
        "schema_version": schema_version,
        "source_file": str(path),
        "units": {},
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
    if root.is_file():
        if root.suffix.lower() in {".txt", ".dat"} and not root.name.lower().endswith(".bak"):
            return [root]
        return []
    result: list[Path] = []
    for path in root.rglob("*"):
        if path.name.lower().endswith(".bak"):
            continue
        if path.is_file() and path.suffix.lower() in {".txt", ".dat"}:
            result.append(path)
    return sorted(result)


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
        output_fields = _discover_output_fields(columns, used_columns)
        if not output_fields:
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
            outputs = _extract_valid_outputs(row, output_fields)
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
                    source_fields={"xi": xi_col, **output_fields},
                    speed_parameter_name=str(meta["speed_parameter_name"]) if meta.get("speed_parameter_name") else None,
                    flow_parameter_name=str(meta["flow_parameter_name"]) if meta.get("flow_parameter_name") else None,
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
        output_fields = _discover_output_fields(columns, used_columns)
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
            outputs = _extract_valid_outputs(row, output_fields) if output_fields else {}
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
                    source_fields={"xi": xi_col, **output_fields},
                    speed_parameter_name=str(meta["speed_parameter_name"]) if meta.get("speed_parameter_name") else None,
                    flow_parameter_name=str(meta["flow_parameter_name"]) if meta.get("flow_parameter_name") else None,
                )
            )
    return samples


def load_operating_conditions(input_path: str | Path) -> list[OperatingCondition]:
    """Load one MAIN operating-condition record per source file.

    This loader intentionally does not flatten sections and does not inspect xi or
    radial outputs. It is shared by Legacy, Institute Single and Institute Multi
    schemas for 1D speed-to-flow training.
    """

    conditions: list[OperatingCondition] = []
    for file_path in iter_data_files(str(input_path)):
        try:
            inspection = inspect_dataset_file(file_path)
        except (OSError, ValueError):
            continue
        station = str(inspection.get("station") or "MAIN").upper()
        if station != "MAIN":
            continue
        component = str(inspection.get("component") or "").upper()
        stage = _coerce_int(inspection.get("stage"))
        speed = _coerce_float(inspection.get("speed_parameter"))
        flow = _coerce_float(inspection.get("flow_parameter"))
        if not component or stage is None or speed is None or flow is None:
            continue
        conditions.append(
            OperatingCondition(
                component=component,
                stage=stage,
                station=station,
                speed_parameter=speed,
                flow_parameter=flow,
                schema=str(inspection["schema"]),
                schema_name=str(inspection["schema_name"]),
                schema_version=str(inspection["schema_version"]),
                source_file=str(file_path),
                speed_parameter_source_name=(
                    str(inspection["speed_parameter_name"])
                    if inspection.get("speed_parameter_name")
                    else None
                ),
                flow_parameter_source_name=(
                    str(inspection["flow_parameter_name"])
                    if inspection.get("flow_parameter_name")
                    else None
                ),
            )
        )
    return conditions


def _copy_with_valid_outputs(sample: FlexSample) -> FlexSample | None:
    outputs = {
        name: value
        for name, value in sample.outputs.items()
        if not _is_missing_sentinel(value)
    }
    if not outputs:
        return None
    return FlexSample(
        component=sample.component,
        family=sample.family,
        station=sample.station,
        database=sample.database,
        stage=sample.stage,
        rpm=sample.rpm,
        wcor=sample.wcor,
        xi=sample.xi,
        source_path=sample.source_path,
        outputs=outputs,
        section=sample.section,
        schema=sample.schema,
        source_fields=sample.source_fields,
        schema_name=sample.schema_name,
        schema_version=sample.schema_version,
        speed_parameter_name=sample.speed_parameter_name,
        flow_parameter_name=sample.flow_parameter_name,
        units=sample.units,
    )


def _multi_samples_for_mode(samples: list[FlexSample], radial_mode: str) -> list[FlexSample]:
    if radial_mode == "full":
        return samples
    grouped: dict[str | None, list[FlexSample]] = {}
    for sample in samples:
        grouped.setdefault(sample.section, []).append(sample)
    selected: list[FlexSample] = []
    for section_samples in grouped.values():
        ordered = sorted(section_samples, key=lambda item: item.xi)
        if len(ordered) <= 2:
            selected.extend(ordered)
        else:
            selected.extend((ordered[0], ordered[-1]))
    return selected


def load_multi_section_samples(input_path: str | Path, radial_mode: str) -> list[FlexSample]:
    """Explicitly adapt Multi-Section MAIN files into section-aware samples."""

    from project1.experiments.multi_section_adapter import (
        adapt_multi_section_file,
        flatten_multi_section_data,
    )
    from project1.services.dataset_schema import (
        INSTITUTE_FOUR_SECTION_SCHEMA,
        INSTITUTE_MULTI_SECTION_SCHEMA,
    )

    samples: list[FlexSample] = []
    accepted_schemas = {INSTITUTE_FOUR_SECTION_SCHEMA, INSTITUTE_MULTI_SECTION_SCHEMA}
    for file_path in iter_data_files(str(input_path)):
        meta = parse_metadata_from_path(str(file_path))
        if str(meta.get("station") or "MAIN").upper() != "MAIN":
            continue
        try:
            inspection = inspect_dataset_file(file_path)
        except (OSError, ValueError):
            continue
        if inspection.get("schema") not in accepted_schemas:
            continue
        canonical = adapt_multi_section_file(file_path)
        flattened = flatten_multi_section_data(canonical)
        valid_samples = [
            valid
            for sample in flattened
            for valid in [_copy_with_valid_outputs(sample)]
            if valid is not None
        ]
        samples.extend(_multi_samples_for_mode(valid_samples, radial_mode))
    return samples


def load_training_samples(input_path: str | Path, radial_mode: str) -> list[FlexSample]:
    """Schema-aware radial training dispatcher with stable Single behavior."""

    single_and_legacy = load_samples(str(input_path), radial_mode)
    multi = load_multi_section_samples(input_path, radial_mode)
    return [*single_and_legacy, *multi]
