"""Structured, additive metadata for domain-model training artifacts."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable


def _unique_values(values: Iterable[object]) -> list[object]:
    result: list[object] = []
    for value in values:
        if value is None or value == "":
            continue
        if value not in result:
            result.append(value)
    return result


def _one_or_many(values: Iterable[object]) -> object | None:
    unique = _unique_values(values)
    if not unique:
        return None
    if len(unique) == 1:
        return unique[0]
    return unique


def build_training_context(
    samples: list[object],
    *,
    model_dimension: str,
    partition: str,
    inputs: list[str],
    target: str,
    training_source_root: str | Path,
    sample_count: int,
) -> dict[str, object]:
    """Build self-describing context without changing the legacy metadata fields."""

    if not samples:
        raise ValueError("cannot build training metadata from an empty sample set")
    first = samples[0]
    source_files = sorted({str(getattr(sample, "source_path", "")) for sample in samples if getattr(sample, "source_path", "")})
    speed_names = [
        getattr(sample, "speed_parameter_source_name", None)
        or getattr(sample, "speed_parameter_name", None)
        for sample in samples
    ]
    flow_names = [
        getattr(sample, "flow_parameter_source_name", None)
        or getattr(sample, "flow_parameter_name", None)
        for sample in samples
    ]
    return {
        "model_dimension": str(model_dimension),
        "component": str(getattr(first, "component")),
        "stage": int(getattr(first, "stage")),
        "station": str(getattr(first, "station", "MAIN")).upper(),
        "section": getattr(first, "section", None),
        "partition": str(partition),
        "schema_name": _one_or_many(getattr(sample, "schema_name", None) for sample in samples),
        "schema_version": _one_or_many(getattr(sample, "schema_version", None) for sample in samples),
        "inputs": list(inputs),
        "target": str(target),
        "speed_parameter_source_name": _one_or_many(speed_names),
        "flow_parameter_source_name": _one_or_many(flow_names),
        "training_source_root": str(training_source_root),
        "training_sources": source_files,
        "sample_count": int(sample_count),
    }


def artifact_identity(context: dict[str, object]) -> dict[str, object]:
    """Return the stable identity fields shared by leaderboard and manifest rows."""

    keys = (
        "model_dimension",
        "component",
        "stage",
        "station",
        "section",
        "partition",
        "target",
    )
    return {key: context.get(key) for key in keys}
