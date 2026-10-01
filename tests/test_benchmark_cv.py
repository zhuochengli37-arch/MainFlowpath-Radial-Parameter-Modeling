from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
WORKBASE_SRC = ROOT / "workbase" / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(WORKBASE_SRC) not in sys.path:
    sys.path.insert(0, str(WORKBASE_SRC))

from project1.experiments.benchmark import Sample, _cv_splits, _resolve_kfold_splits
from project1.experiments.benchmark_partitioning import (
    condition_group_labels,
    condition_train_validation_test_split,
    limit_samples_by_condition,
)


def _sample(index: int, rpm: float | None = None) -> Sample:
    value = float(index)
    return Sample(
        component="CMP",
        family="CMP",
        station="MAIN",
        stage=1,
        rpm=value if rpm is None else rpm,
        wcor=0.1 + value,
        xi=0.01 * value,
        psi=1.0 + value,
        tsi=2.0 + value,
        mai=3.0 + value,
    )


def test_current_random_cv_split_cap_for_small_samples():
    assert _resolve_kfold_splits(6, 5) == 3
    assert _resolve_kfold_splits(10, 5) == 5
    assert _resolve_kfold_splits(5, 5) == 2


def test_current_random_cv_uses_adaptive_fold_cap():
    samples = [_sample(i) for i in range(6)]

    splits = list(_cv_splits(samples, "random", 5))

    assert len(splits) == 3


def test_current_group_cv_respects_requested_upper_bound():
    samples = [_sample(i, rpm=float(i // 2)) for i in range(8)]

    splits = list(_cv_splits(samples, "rpm", 5))

    assert len(splits) == 4


def _condition_profiles() -> list[Sample]:
    samples: list[Sample] = []
    for condition, point_count in enumerate((3, 4, 5, 3, 4, 5)):
        for radial_index in range(point_count):
            samples.append(
                Sample(
                    component="CMP",
                    family="CMP",
                    station="MAIN",
                    stage=1,
                    rpm=0.7 + condition * 0.05,
                    wcor=10.0 + condition,
                    xi=radial_index / max(1, point_count - 1),
                    psi=1.0,
                    tsi=2.0,
                    mai=3.0,
                    source_file=f"condition_{condition}.dat",
                )
            )
    return samples


def test_condition_cv_never_splits_one_radial_profile() -> None:
    samples = _condition_profiles()
    labels = condition_group_labels(samples)

    for train_idx, validation_idx in _cv_splits(samples, "condition", 3):
        train_groups = set(labels[train_idx])
        validation_groups = set(labels[validation_idx])
        assert train_groups.isdisjoint(validation_groups)
        for group in set(labels):
            profile_rows = set(np.flatnonzero(labels == group))
            assert profile_rows <= set(train_idx) or profile_rows <= set(validation_idx)


def test_condition_train_validation_test_groups_are_disjoint_and_complete() -> None:
    samples = _condition_profiles()
    labels = condition_group_labels(samples)
    train_idx, validation_idx, test_idx = condition_train_validation_test_split(
        samples, validation_size=0.2, test_size=0.2, random_state=17
    )
    split_indices = (set(train_idx), set(validation_idx), set(test_idx))
    split_groups = tuple(set(labels[list(indices)]) for indices in split_indices)

    assert split_groups[0].isdisjoint(split_groups[1])
    assert split_groups[0].isdisjoint(split_groups[2])
    assert split_groups[1].isdisjoint(split_groups[2])
    assert set().union(*split_indices) == set(range(len(samples)))
    for group in set(labels):
        profile_rows = set(np.flatnonzero(labels == group))
        assert sum(profile_rows <= indices for indices in split_indices) == 1


def test_condition_splits_are_reproducible_with_fixed_random_state() -> None:
    samples = _condition_profiles()

    first_cv = [(train.tolist(), test.tolist()) for train, test in _cv_splits(samples, "condition", 3, random_state=9)]
    second_cv = [(train.tolist(), test.tolist()) for train, test in _cv_splits(samples, "condition", 3, random_state=9)]
    first_three_way = tuple(
        split.tolist()
        for split in condition_train_validation_test_split(samples, random_state=9)
    )
    second_three_way = tuple(
        split.tolist()
        for split in condition_train_validation_test_split(samples, random_state=9)
    )

    assert first_cv == second_cv
    assert first_three_way == second_three_way


def test_condition_fallback_normalizes_floats_in_multi_condition_files() -> None:
    samples = [
        _sample(0, rpm=0.1 + 0.2),
        _sample(1, rpm=0.3),
        _sample(2, rpm=0.4),
    ]
    for sample in samples:
        object.__setattr__(sample, "source_file", "multi_condition.csv")
    object.__setattr__(samples[0], "wcor", 12.0)
    object.__setattr__(samples[1], "wcor", 12.0)
    object.__setattr__(samples[2], "wcor", 13.0)

    labels = condition_group_labels(samples)

    assert labels[0] == labels[1]
    assert labels[0] != labels[2]


def test_condition_sample_limit_keeps_selected_profiles_complete() -> None:
    samples = _condition_profiles()
    original_labels = condition_group_labels(samples)

    limited = limit_samples_by_condition(samples, max_samples=10, random_state=5)
    limited_labels = condition_group_labels(limited)

    assert 0 < len(limited) <= 10
    for group in set(limited_labels):
        assert np.sum(limited_labels == group) == np.sum(original_labels == group)
