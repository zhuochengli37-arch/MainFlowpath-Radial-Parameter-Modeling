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
    if mode == "rpm":
        return np.array([f"{sample.component}:{sample.stage}:{sample.rpm:.6f}" for sample in samples], dtype=object)
    if mode == "wcor":
        return np.array([f"{sample.component}:{sample.stage}:{sample.wcor:.6f}" for sample in samples], dtype=object)
    return np.array([f"{sample.component}:{sample.stage}" for sample in samples], dtype=object)


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


def cv_splits(samples: list[AnySample], mode: str, requested_splits: int):
    sample_count = len(samples)
    if sample_count < 2:
        return []
    if mode == "random":
        n_splits = resolve_kfold_splits(sample_count, requested_splits)
        if n_splits < 2:
            return []
        return KFold(n_splits=n_splits, shuffle=True, random_state=42).split(np.arange(sample_count))
    groups = group_labels(samples, mode)
    uniq = np.unique(groups)
    if len(uniq) < 2:
        return []
    n_splits = min(requested_splits, len(uniq))
    if n_splits < 2:
        return []
    return GroupKFold(n_splits=n_splits).split(np.arange(sample_count), groups=groups)
