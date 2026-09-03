"""Manifest-first registry and validation for saved radial models."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Iterable
import warnings

import joblib

from project1.experiments.model_partitioning import ModelPartition, safe_partition_name
from workbase.common.model_versioning import check_version_compatibility, load_model_metadata


@dataclass(frozen=True)
class ModelIdentity:
    dimension: str
    component: str
    stage: int
    section: str | None
    target: str
    station: str = "MAIN"

    def __post_init__(self) -> None:
        dimension = str(self.dimension).upper()
        if dimension not in {"2D", "3D"}:
            raise ValueError(f"unsupported radial model dimension: {dimension}")
        object.__setattr__(self, "dimension", dimension)
        object.__setattr__(self, "component", str(self.component).upper())
        object.__setattr__(self, "stage", int(self.stage))
        section = None if self.section is None else str(self.section).strip()
        object.__setattr__(self, "section", section or None)
        object.__setattr__(self, "target", str(self.target))
        object.__setattr__(self, "station", str(self.station).upper())

    @property
    def partition(self) -> ModelPartition:
        return ModelPartition(self.component, self.stage, self.section)


@dataclass(frozen=True)
class ModelRecord:
    identity: ModelIdentity
    model_name: str
    model_path: Path
    metrics: dict[str, float]
    manifest_entry: dict[str, object]
    legacy: bool = False


@dataclass(frozen=True)
class SelectedModel:
    record: ModelRecord
    model: object
    metadata: dict[str, object] | None


class ModelRegistry:
    """Discover models below one training output root.

    Structured manifests are authoritative.  Legacy folder/leaderboard fallback
    is allowed only for ``section=None`` Single-Section requests.
    """

    def __init__(
        self,
        model_root: str | Path,
        dimension: str,
        *,
        radial_mode: str = "full",
        legacy_partition_override: str | None = None,
    ) -> None:
        self.model_root = Path(model_root)
        self.dimension = str(dimension).upper()
        if self.dimension not in {"2D", "3D"}:
            raise ValueError(f"unsupported radial model dimension: {self.dimension}")
        self.radial_mode = str(radial_mode)
        if legacy_partition_override not in {None, "all"}:
            raise ValueError(
                "legacy_partition_override supports only the historical 'all' partition"
            )
        self.legacy_partition_override = legacy_partition_override
        self.models_dir_name = "models_2d" if self.dimension == "2D" else "models"
        self.manifest_path = self.model_root / self.models_dir_name / "models_manifest.json"
        self._leaderboard = self._load_leaderboard()
        self._manifest_records = self._load_manifest_records()

    def _load_json_rows(self, path: Path) -> list[dict[str, object]]:
        if not path.exists():
            return []
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError(f"expected a JSON list: {path}")
        return [dict(item) for item in payload if isinstance(item, dict)]

    def _load_leaderboard(self) -> list[dict[str, object]]:
        filename = (
            f"leaderboard_2d_{self.radial_mode}.json"
            if self.dimension == "2D"
            else f"leaderboard_{self.radial_mode}.json"
        )
        return self._load_json_rows(self.model_root / filename)

    def _resolve_manifest_path(self, raw_path: object) -> Path:
        candidate = (self.model_root / str(raw_path)).resolve()
        root = self.model_root.resolve()
        if candidate != root and root not in candidate.parents:
            raise ValueError(f"model manifest path escapes model root: {raw_path}")
        return candidate

    def _identity_from_row(self, row: dict[str, object]) -> ModelIdentity | None:
        required = {"model_dimension", "component", "stage", "target", "partition"}
        if not required.issubset(row):
            return None
        return ModelIdentity(
            dimension=str(row["model_dimension"]),
            component=str(row["component"]),
            stage=int(row["stage"]),
            section=None if row.get("section") in {None, ""} else str(row["section"]),
            target=str(row["target"]),
            station=str(row.get("station") or "MAIN"),
        )

    def _load_manifest_records(self) -> tuple[ModelRecord, ...]:
        records: list[ModelRecord] = []
        for row in self._load_json_rows(self.manifest_path):
            identity = self._identity_from_row(row)
            if identity is None or identity.dimension != self.dimension:
                continue
            declared_partition = str(row.get("partition"))
            if declared_partition != str(identity.partition):
                raise ValueError(
                    f"manifest partition mismatch: {declared_partition!r} != {identity.partition!s}"
                )
            if not row.get("path") or not row.get("model"):
                raise ValueError(f"incomplete model manifest entry: {row}")
            metrics = row.get("metrics") if isinstance(row.get("metrics"), dict) else {}
            records.append(
                ModelRecord(
                    identity=identity,
                    model_name=str(row["model"]),
                    model_path=self._resolve_manifest_path(row["path"]),
                    metrics={str(key): float(value) for key, value in metrics.items() if value is not None},
                    manifest_entry=row,
                )
            )
        return tuple(records)

    @property
    def uses_manifest(self) -> bool:
        return bool(self._manifest_records)

    def _legacy_records(self, partition: ModelPartition) -> tuple[ModelRecord, ...]:
        if partition.section is not None:
            return ()
        lookup_partition = self.legacy_partition_override or str(partition)
        records: list[ModelRecord] = []
        seen: set[tuple[str, str]] = set()
        for row in self._leaderboard:
            row_partition = str(row.get("partition") or "all")
            if row_partition != lookup_partition:
                continue
            target = str(row.get("target") or "")
            model_name = str(row.get("model") or "")
            if not target or not model_name or (model_name, target) in seen:
                continue
            seen.add((model_name, target))
            model_path = (
                self.model_root
                / self.models_dir_name
                / safe_partition_name(lookup_partition)
                / f"{model_name}_{target}.pkl"
            )
            if model_path.exists():
                records.append(
                    ModelRecord(
                        identity=ModelIdentity(
                            self.dimension,
                            partition.component,
                            partition.stage,
                            None,
                            target,
                        ),
                        model_name=model_name,
                        model_path=model_path,
                        metrics={
                            metric: float(row[metric])
                            for metric in ("rmse", "mape", "mae")
                            if row.get(metric) is not None
                        },
                        manifest_entry=row,
                        legacy=True,
                    )
                )
        return tuple(records)

    def _records_for_partition(self, partition: ModelPartition) -> tuple[ModelRecord, ...]:
        records = tuple(
            record
            for record in self._manifest_records
            if record.identity.partition == partition
        )
        if records:
            return records
        return self._legacy_records(partition)

    def available_targets(self, partition: ModelPartition | object) -> tuple[str, ...]:
        if not isinstance(partition, ModelPartition):
            partition = ModelPartition(
                str(getattr(partition, "component")),
                int(getattr(partition, "stage")),
                getattr(partition, "section", None),
            )
        records = self._records_for_partition(partition)
        targets: list[str] = []
        for record in records:
            if record.identity.target not in targets:
                targets.append(record.identity.target)
        return tuple(targets)

    def _leaderboard_metric(
        self,
        record: ModelRecord,
        selection_metric: str,
        selection_split_mode: str,
    ) -> float | None:
        candidates = [
            row
            for row in self._leaderboard
            if str(row.get("partition"))
            == str(
                record.manifest_entry.get("partition")
                if record.legacy
                else record.identity.partition
            )
            and str(row.get("target")) == record.identity.target
            and str(row.get("model")) == record.model_name
            and (
                selection_split_mode == "any"
                or str(row.get("split_mode")) == selection_split_mode
            )
        ]
        values = [float(row[selection_metric]) for row in candidates if row.get(selection_metric) is not None]
        if values:
            return min(values)
        value = record.metrics.get(selection_metric)
        return float(value) if value is not None else None

    def select(
        self,
        identity: ModelIdentity,
        *,
        selection_metric: str = "rmse",
        selection_split_mode: str = "random",
        model_name: str | None = None,
        schema_names: Iterable[str] = (),
        schema_versions: Iterable[str] = (),
    ) -> SelectedModel:
        if identity.dimension != self.dimension:
            raise ValueError(
                f"registry dimension {self.dimension} cannot serve {identity.dimension} request"
            )
        candidates = [
            record
            for record in self._records_for_partition(identity.partition)
            if record.identity == identity
        ]
        if model_name is not None:
            candidates = [record for record in candidates if record.model_name == model_name]
        if not candidates:
            raise LookupError(
                "no matching model for "
                f"{identity.dimension}/{identity.partition}/{identity.target}"
            )
        if model_name is None:
            scored = [
                (
                    self._leaderboard_metric(
                        record, selection_metric, selection_split_mode
                    ),
                    record,
                )
                for record in candidates
            ]
            usable = [(score, record) for score, record in scored if score is not None]
            if not usable:
                raise ValueError(
                    f"no {selection_metric} score for {identity.partition}/{identity.target}"
                )
            record = min(usable, key=lambda item: item[0])[1]
        else:
            record = candidates[0]
        if not record.model_path.exists():
            raise FileNotFoundError(f"registered model does not exist: {record.model_path}")
        metadata = load_model_metadata(record.model_path)
        if isinstance(metadata, dict) and metadata.get("dependencies"):
            compatible, compatibility_warnings = check_version_compatibility(metadata)
            if not compatible:
                raise RuntimeError(f"model dependency versions are incompatible: {record.model_path}")
            for message in compatibility_warnings:
                warnings.warn(message, UserWarning, stacklevel=3)
        self._validate_metadata(
            identity,
            metadata,
            schema_names=tuple(schema_names),
            schema_versions=tuple(schema_versions),
            legacy=record.legacy,
        )
        return SelectedModel(record=record, model=joblib.load(record.model_path), metadata=metadata)

    def _validate_metadata(
        self,
        identity: ModelIdentity,
        metadata: dict[str, object] | None,
        *,
        schema_names: tuple[str, ...],
        schema_versions: tuple[str, ...],
        legacy: bool,
    ) -> None:
        context = metadata.get("model_context") if isinstance(metadata, dict) else None
        if not isinstance(context, dict):
            if identity.section is not None:
                raise ValueError(
                    f"Multi-Section model metadata lacks model_context/section: {identity.partition}"
                )
            compatibility_label = "legacy Single" if legacy else "Single"
            warnings.warn(
                f"{compatibility_label} model metadata is incomplete: "
                f"{identity.partition}/{identity.target}",
                UserWarning,
                stacklevel=3,
            )
            return
        expected = {
            "model_dimension": identity.dimension,
            "component": identity.component,
            "stage": identity.stage,
            "station": identity.station,
            "section": identity.section,
            "partition": str(identity.partition),
            "target": identity.target,
            "inputs": (
                ["speed_parameter", "xi"]
                if identity.dimension == "2D"
                else ["speed_parameter", "flow_parameter", "xi"]
            ),
        }
        for field, value in expected.items():
            actual = context.get(field)
            if actual != value:
                raise ValueError(
                    f"model metadata {field} mismatch: expected {value!r}, got {actual!r}"
                )
        model_schema = context.get("schema_name")
        model_schemas = set(model_schema if isinstance(model_schema, list) else [model_schema])
        model_schemas.discard(None)
        if model_schemas and schema_names and not set(schema_names).issubset(model_schemas):
            raise ValueError(
                f"model metadata schema_name mismatch: {sorted(model_schemas)!r} "
                f"does not cover {sorted(set(schema_names))}"
            )
        model_version = context.get("schema_version")
        model_versions = set(model_version if isinstance(model_version, list) else [model_version])
        model_versions.discard(None)
        if model_versions and schema_versions and not set(schema_versions).issubset(model_versions):
            raise ValueError(
                "model metadata schema_version mismatch: "
                f"{sorted(model_versions)!r} does not cover {sorted(set(schema_versions))}"
            )
