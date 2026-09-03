"""Shared, file-I/O-free prediction core for 2D and 3D radial models."""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable

import numpy as np

from project1.experiments.model_partitioning import radial_partition
from project1.experiments.model_registry import ModelIdentity, ModelRegistry
from project1.experiments.prediction_results import PredictionResult
from project1.experiments.prediction_samples import PredictionSample


def _target_selection(targets: str | Iterable[str]) -> str | tuple[str, ...]:
    if isinstance(targets, str):
        if targets.strip().lower() != "auto":
            return (targets,)
        return "auto"
    return tuple(dict.fromkeys(str(target) for target in targets))


def _input_matrix(samples: list[PredictionSample], dimension: str) -> np.ndarray:
    if dimension == "2D":
        return np.asarray(
            [[sample.speed_parameter, sample.xi] for sample in samples],
            dtype=float,
        )
    missing = [sample.source_file for sample in samples if sample.flow_parameter is None]
    if missing:
        raise ValueError(
            "3D prediction requires flow_parameter in the structured sample; "
            f"missing for {missing[0]}"
        )
    return np.asarray(
        [
            [sample.speed_parameter, float(sample.flow_parameter), sample.xi]
            for sample in samples
        ],
        dtype=float,
    )


def predict_radial(
    samples: Iterable[PredictionSample],
    registry: ModelRegistry,
    *,
    dimension: str,
    targets: str | Iterable[str] = "auto",
    missing_target_policy: str = "error",
    selection_metric: str = "rmse",
    selection_split_mode: str = "random",
    model_name: str | None = None,
) -> list[PredictionResult]:
    """Predict normalized samples without reading paths or writing result files."""

    normalized_dimension = str(dimension).upper()
    if normalized_dimension not in {"2D", "3D"}:
        raise ValueError(f"unsupported radial prediction dimension: {normalized_dimension}")
    if registry.dimension != normalized_dimension:
        raise ValueError(
            f"registry dimension {registry.dimension} does not match requested dimension "
            f"{normalized_dimension}"
        )
    policy = str(missing_target_policy).lower()
    if policy not in {"error", "skip"}:
        raise ValueError(f"unsupported missing_target_policy: {policy}")
    sample_list = list(samples)
    if not sample_list:
        raise ValueError("no normalized prediction samples provided")

    grouped: dict[str, list[tuple[int, PredictionSample]]] = defaultdict(list)
    partitions = {}
    for sample_index, sample in enumerate(sample_list):
        if str(sample.station).upper() != "MAIN":
            raise ValueError(
                f"radial prediction accepts station=MAIN only: {sample.source_file}"
            )
        partition = radial_partition(sample)
        grouped[str(partition)].append((sample_index, sample))
        partitions[str(partition)] = partition

    requested = _target_selection(targets)
    indexed_results: list[tuple[int, int, PredictionResult]] = []
    for partition_key, indexed_samples in grouped.items():
        partition = partitions[partition_key]
        available = registry.available_targets(partition)
        if requested == "auto":
            selected_targets = available
        else:
            missing = [target for target in requested if target not in available]
            if missing and policy == "error":
                raise LookupError(
                    f"partition {partition_key} has no model for explicit target(s): "
                    + ", ".join(missing)
                )
            selected_targets = tuple(target for target in requested if target in available)
        if not selected_targets:
            raise LookupError(f"partition {partition_key} has no available radial model targets")

        partition_samples = [sample for _, sample in indexed_samples]
        x_array = _input_matrix(partition_samples, normalized_dimension)
        first = partition_samples[0]
        schema_names = {
            sample.schema_name
            for sample in partition_samples
            if sample.schema_name and sample.schema_name != "unclassified_prediction"
        }
        schema_versions = {
            sample.schema_version
            for sample in partition_samples
            if sample.schema_version and sample.schema_name != "unclassified_prediction"
        }
        for target_order, target in enumerate(selected_targets):
            identity = ModelIdentity(
                dimension=normalized_dimension,
                component=partition.component,
                stage=partition.stage,
                section=partition.section,
                target=target,
                station=first.station,
            )
            selected = registry.select(
                identity,
                selection_metric=selection_metric,
                selection_split_mode=selection_split_mode,
                model_name=model_name,
                schema_names=schema_names,
                schema_versions=schema_versions,
            )
            predictions = np.asarray(selected.model.predict(x_array), dtype=float).reshape(-1)
            if len(predictions) != len(partition_samples):
                raise ValueError(
                    f"model returned {len(predictions)} rows for {len(partition_samples)} samples"
                )
            for local_index, predicted_value in enumerate(predictions):
                sample_index, sample = indexed_samples[local_index]
                indexed_results.append(
                    (
                        sample_index,
                        target_order,
                        PredictionResult(
                            source_file=sample.source_file,
                            component=sample.component,
                            stage=sample.stage,
                            station=sample.station,
                            section=sample.section,
                            speed_parameter=sample.speed_parameter,
                            flow_parameter=sample.flow_parameter,
                            xi=sample.xi,
                            target=target,
                            predicted_value=float(predicted_value),
                            partition=partition_key,
                            model_identity=selected.record.identity,
                            model_name=selected.record.model_name,
                            model_path=str(selected.record.model_path),
                            schema_name=sample.schema_name,
                            schema_version=sample.schema_version,
                            sample_index=sample_index,
                            row_index=sample.row_index,
                            section_row_index=sample.section_row_index,
                        ),
                    )
                )
    indexed_results.sort(key=lambda item: (item[0], item[1]))
    return [result for _, _, result in indexed_results]
