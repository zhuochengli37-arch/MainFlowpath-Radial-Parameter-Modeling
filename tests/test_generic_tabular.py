from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parent.parent
WORKBASE_SRC = ROOT / "workbase" / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(WORKBASE_SRC) not in sys.path:
    sys.path.insert(0, str(WORKBASE_SRC))

from workbase.common.generic_tabular import (
    _evaluate_model,
    _resolve_kfold_splits,
    _reserve_prediction_output_path,
    run_generic_benchmark,
    run_generic_predict,
)


def _write_rows(file_path: Path, rows: list[list[str]]) -> None:
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)


def test_generic_kfold_split_cap_for_small_samples():
    assert _resolve_kfold_splits(6, 5) == 3
    assert _resolve_kfold_splits(10, 5) == 5
    assert _resolve_kfold_splits(5, 5) == 2


def test_generic_knn_evaluation_adapts_neighbor_count():
    import numpy as np
    from sklearn.neighbors import KNeighborsRegressor

    x = np.array([[0.0], [0.2], [0.4], [0.6], [0.8], [1.0]], dtype=float)
    y = np.array([0.0, 0.4, 0.8, 1.2, 1.6, 2.0], dtype=float)

    metrics = _evaluate_model(KNeighborsRegressor(n_neighbors=5, weights="distance"), x, y, n_splits=5)

    assert metrics["folds"] == 3
    assert metrics["rmse"] >= 0.0


def test_generic_predict_accepts_sparse_prediction_rows(tmp_path: Path):
    train_file = tmp_path / "train" / "sample.csv"
    predict_file = tmp_path / "predict" / "sample.csv"
    output_dir = tmp_path / "output"
    results_dir = tmp_path / "results"

    _write_rows(
        train_file,
        [
            ["x", "y1", "y2"],
            ["0.0", "0.0", "1.0"],
            ["0.2", "0.4", "1.2"],
            ["0.4", "0.8", "1.4"],
            ["0.6", "1.2", "1.6"],
            ["0.8", "1.6", "1.8"],
            ["1.0", "2.0", "2.0"],
        ],
    )
    _write_rows(
        predict_file,
        [
            ["x", "y1", "y2"],
            ["0.5"],
            ["0.9"],
        ],
    )

    run_generic_benchmark(
        input_dim=1,
        input_path=train_file.parent,
        output_dir=output_dir,
        n_splits=3,
    )
    result = run_generic_predict(
        input_dim=1,
        input_path=predict_file.parent,
        model_output_dir=output_dir,
        output_dir=results_dir,
    )

    assert len(result["outputs"]) == 1

    output_file = Path(result["outputs"][0])
    rows = list(csv.DictReader(output_file.open("r", encoding="utf-8", newline="")))
    assert len(rows) == 2
    assert rows[0]["x"] == "0.5"
    assert rows[0]["y1"] == ""
    assert rows[0]["y2"] == ""
    assert "pred_y1" in rows[0]
    assert "pred_y2" in rows[0]


def test_generic_predict_fails_fast_on_version_mismatch(tmp_path: Path):
    train_file = tmp_path / "train" / "sample.csv"
    predict_file = tmp_path / "predict" / "sample.csv"
    output_dir = tmp_path / "output"
    results_dir = tmp_path / "results"

    _write_rows(
        train_file,
        [
            ["x", "y"],
            ["0.0", "0.0"],
            ["0.2", "0.4"],
            ["0.4", "0.8"],
            ["0.6", "1.2"],
            ["0.8", "1.6"],
            ["1.0", "2.0"],
        ],
    )
    _write_rows(
        predict_file,
        [
            ["x"],
            ["0.5"],
        ],
    )

    run_generic_benchmark(
        input_dim=1,
        input_path=train_file.parent,
        output_dir=output_dir,
        n_splits=3,
    )

    metadata_path = output_dir / "models" / "y__best.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    metadata["dependencies"]["scikit-learn"] = "9.9.9"
    metadata_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")

    with pytest.raises(ValueError, match="scikit-learn"):
        run_generic_predict(
            input_dim=1,
            input_path=predict_file.parent,
            model_output_dir=output_dir,
            output_dir=results_dir,
        )


def test_generic_prediction_output_uses_suffix_when_base_exists(tmp_path: Path):
    output_file = tmp_path / "results" / "sample_predictions.csv"
    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("locked elsewhere", encoding="utf-8")

    reserved = _reserve_prediction_output_path(output_file)

    assert reserved.name == "sample_predictions_1.csv"


def test_generic_benchmark_reports_effective_folds(tmp_path: Path):
    train_file = tmp_path / "train" / "sample.csv"
    output_dir = tmp_path / "output"

    _write_rows(
        train_file,
        [
            ["x", "y"],
            ["0.0", "0.0"],
            ["0.2", "0.4"],
            ["0.4", "0.8"],
            ["0.6", "1.2"],
            ["0.8", "1.6"],
            ["1.0", "2.0"],
        ],
    )

    result = run_generic_benchmark(
        input_dim=1,
        input_path=train_file.parent,
        output_dir=output_dir,
        n_splits=5,
    )

    assert result["requested_n_splits"] == 5
    assert result["effective_n_splits"] == {"y": 3}
