from __future__ import annotations

from collections import defaultdict

import numpy as np
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import GroupKFold, KFold
from sklearn.neighbors import KNeighborsRegressor

from project1.experiments.benchmark_data import AnySample
from project1.experiments.model_partitioning import radial_partition


def to_arrays(samples: list[AnySample], target: str) -> tuple[np.ndarray, np.ndarray, list[AnySample]]:
    filtered_samples = [sample for sample in samples if target in sample.output_columns]
    x = np.array([[sample.rpm, sample.wcor, sample.xi] for sample in filtered_samples], dtype=float)
    y = np.array([sample.get_output(target) for sample in filtered_samples], dtype=float)
    return x, y, filtered_samples


def to_arrays_2d(samples: list[AnySample], target: str) -> tuple[np.ndarray, np.ndarray, list[AnySample]]:
    filtered_samples = [sample for sample in samples if target in sample.output_columns]
    x = np.array([[sample.rpm, sample.xi] for sample in filtered_samples], dtype=float)
    y = np.array([sample.get_output(target) for sample in filtered_samples], dtype=float)
    return x, y, filtered_samples


def partition_samples(samples: list[AnySample], partition_mode: str) -> dict[str, list[AnySample]]:
    groups: dict[str, list[AnySample]] = defaultdict(list)
    for sample in samples:
        if partition_mode in {"single", "none"}:
            key = "all"
        elif partition_mode == "family":
            key = sample.family
        elif partition_mode == "component":
            key = sample.component
        else:
            key = str(radial_partition(sample))
        groups[key].append(sample)
    return dict(groups)


def metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    denom = np.maximum(np.abs(y_true), 1e-6)
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(np.mean(np.abs((y_true - y_pred) / denom))),
        "r2": float(r2_score(y_true, y_pred)) if len(y_true) >= 2 else float("nan"),
    }


def group_labels(samples: list[AnySample], mode: str) -> np.ndarray:
    if mode == "condition":
        return condition_group_labels(samples)
    if mode == "rpm":
        return np.array([f"{sample.component}:{sample.stage}:{sample.rpm:.6f}" for sample in samples], dtype=object)
    if mode == "wcor":
        return np.array([f"{sample.component}:{sample.stage}:{sample.wcor:.6f}" for sample in samples], dtype=object)
    return np.array([f"{sample.component}:{sample.stage}" for sample in samples], dtype=object)


def _condition_pair(sample: AnySample) -> tuple[str, str]:
    # Current databases store decimal n/w metadata in paths or columns. Twelve
    # significant digits remove binary parsing noise without collapsing measured values.
    return (format(float(sample.rpm), ".12g"), format(float(sample.wcor), ".12g"))


def _explicit_condition_id(sample: AnySample) -> str:
    for field in ("condition_id", "state_id", "operating_condition_id"):
        value = getattr(sample, field, None)
        if value is not None and str(value).strip():
            return str(value).strip()
    return ""


def condition_group_labels(samples: list[AnySample]) -> np.ndarray:
    """Return one stable label per complete CFD operating condition/profile."""

    explicit_ids = [_explicit_condition_id(sample) for sample in samples]
    if explicit_ids and all(explicit_ids):
        return np.array(
            [
                f"{sample.component}:{sample.stage}:{str(sample.station).upper()}:id:{identifier}"
                for sample, identifier in zip(samples, explicit_ids)
            ],
            dtype=object,
        )

    pairs = [_condition_pair(sample) for sample in samples]
    sources = [str(getattr(sample, "source_file", "") or "").strip().replace("\\", "/").casefold() for sample in samples]
    source_pairs: dict[str, set[tuple[str, str]]] = defaultdict(set)
    pair_sources: dict[tuple[str, str], set[str]] = defaultdict(set)
    for source, pair in zip(sources, pairs):
        if source:
            source_pairs[source].add(pair)
            pair_sources[pair].add(source)

    labels: list[str] = []
    for sample, source, pair in zip(samples, sources, pairs):
        context = f"{sample.component}:{sample.stage}:{str(sample.station).upper()}"
        source_is_unique_profile = (
            bool(source)
            and len(source_pairs[source]) == 1
            and len(pair_sources[pair]) == 1
        )
        labels.append(
            f"{context}:source:{source}"
            if source_is_unique_profile
            else f"{context}:nw:{pair[0]}:{pair[1]}"
        )
    return np.array(labels, dtype=object)


def limit_samples_by_condition(
    samples: list[AnySample],
    max_samples: int,
    *,
    random_state: int = 42,
) -> list[AnySample]:
    """Apply a soft row cap without truncating any selected radial profile."""

    if max_samples < 1:
        raise ValueError("max_samples must be positive")
    if len(samples) <= max_samples:
        return samples
    groups = condition_group_labels(samples)
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(random_state)
    selected_groups: list[object] = []
    selected_count = 0
    for group in rng.permutation(unique_groups):
        group_count = int(np.sum(groups == group))
        if selected_count + group_count <= max_samples:
            selected_groups.append(group)
            selected_count += group_count
    if not selected_groups:
        selected_groups.append(min(unique_groups, key=lambda group: int(np.sum(groups == group))))
    selected = set(selected_groups)
    return [sample for sample, group in zip(samples, groups) if group in selected]


def condition_train_validation_test_split(
    samples: list[AnySample],
    *,
    validation_size: float = 0.2,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split row indices into three disjoint sets of complete conditions."""

    if not 0 < validation_size < 1 or not 0 < test_size < 1 or validation_size + test_size >= 1:
        raise ValueError("validation_size and test_size must be in (0, 1) and sum to less than 1")
    groups = condition_group_labels(samples)
    unique_groups = np.unique(groups)
    if len(unique_groups) < 3:
        raise ValueError("condition split requires at least three condition groups")
    shuffled_groups = np.random.default_rng(random_state).permutation(unique_groups)
    test_count = max(1, int(np.floor(len(unique_groups) * test_size)))
    validation_count = max(1, int(np.floor(len(unique_groups) * validation_size)))
    test_groups = set(shuffled_groups[:test_count])
    validation_groups = set(shuffled_groups[test_count:test_count + validation_count])
    train_groups = set(shuffled_groups[test_count + validation_count:])
    return tuple(
        np.flatnonzero(np.isin(groups, tuple(selected)))
        for selected in (train_groups, validation_groups, test_groups)
    )


def resolve_kfold_splits(sample_count: int, requested_splits: int) -> int:
    if sample_count < 2:
        return 0
    safe_cap = max(2, sample_count // 2)
    return max(2, min(requested_splits, sample_count, safe_cap))


def prepare_model_for_fold(model: object, train_size: int) -> object:
    candidate = clone(model)
    if isinstance(candidate, KNeighborsRegressor):
        candidate.set_params(n_neighbors=max(1, min(candidate.n_neighbors, train_size)))
        return candidate

    named_steps = getattr(candidate, "named_steps", None)
    if named_steps:
        updated_params: dict[str, int] = {}
        for step_name, step in named_steps.items():
            if isinstance(step, KNeighborsRegressor):
                updated_params[f"{step_name}__n_neighbors"] = max(1, min(step.n_neighbors, train_size))
        if updated_params:
            candidate.set_params(**updated_params)
    return candidate


def cv_splits(
    samples: list[AnySample],
    mode: str,
    requested_splits: int,
    *,
    random_state: int = 42,
):
    sample_count = len(samples)
    if sample_count < 2:
        return []
    if mode == "random":
        n_splits = resolve_kfold_splits(sample_count, requested_splits)
        if n_splits < 2:
            return []
        return KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(np.arange(sample_count))
    groups = group_labels(samples, mode)
    uniq = np.unique(groups)
    if len(uniq) < 2:
        return []
    n_splits = min(requested_splits, len(uniq))
    if n_splits < 2:
        return []
    if mode == "condition":
        group_splits = KFold(n_splits=n_splits, shuffle=True, random_state=random_state).split(uniq)
        return (
            (
                np.flatnonzero(np.isin(groups, uniq[train_groups])),
                np.flatnonzero(np.isin(groups, uniq[test_groups])),
            )
            for train_groups, test_groups in group_splits
        )
    return GroupKFold(n_splits=n_splits).split(np.arange(sample_count), groups=groups)
