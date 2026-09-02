from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WORKBASE_SRC = ROOT / "workbase" / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(WORKBASE_SRC) not in sys.path:
    sys.path.insert(0, str(WORKBASE_SRC))

from project1.experiments.benchmark import Sample, _cv_splits, _resolve_kfold_splits


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
