from __future__ import annotations

from pathlib import Path
from typing import Iterable
import csv

import numpy as np
from sklearn.multioutput import MultiOutputRegressor

from project1.experiments.benchmark_data import AnySample, FlexSample, load_samples
from project1.services.data_reader import read_tabular_file
from workbase.common.generic_tabular import get_generic_models

SUPPORTED_TABULAR_EXTENSIONS = (".txt", ".dat", ".csv")


def _iter_data_files(input_dir: str, extensions: Iterable[str] = SUPPORTED_TABULAR_EXTENSIONS) -> list[Path]:
    root = Path(input_dir)
    result: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in set(ext.lower() for ext in extensions):
            result.append(p)
    return result


def load_curve_samples(input_dir: str, radial_mode: str = "full") -> list[FlexSample]:
    return load_samples(input_dir, radial_mode)


def samples_to_arrays(samples: list[AnySample], targets: str | Iterable[str]) -> tuple[np.ndarray, np.ndarray]:
    x = np.array([[s.rpm, s.wcor, s.xi] for s in samples], dtype=float)
    if isinstance(targets, str):
        y = np.array([s.get_output(targets) for s in samples], dtype=float)
    else:
        y = np.column_stack([np.array([s.get_output(t) for s in samples], dtype=float) for t in targets])
    return x, y


def load_tabular_samples(
    input_dir: str,
    x_columns: str | Iterable[str],
    y_columns: str | Iterable[str],
    delimiter: str | None = None,
    extensions: Iterable[str] = SUPPORTED_TABULAR_EXTENSIONS,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    x_columns_list = [x_columns] if isinstance(x_columns, str) else list(x_columns)
    y_columns_list = [y_columns] if isinstance(y_columns, str) else list(y_columns)

    x_rows: list[list[float]] = []
    y_rows: list[list[float]] = []
    records: list[dict[str, object]] = []

    for file_path in _iter_data_files(input_dir, extensions=extensions):
        try:
            parsed = read_tabular_file(str(file_path), delimiter=delimiter)
        except ValueError:
            continue

        for row in parsed["rows"]:
            normalized = {str(key).strip().lower(): value for key, value in row.items()}
            try:
                x_values = [float(normalized[str(col).strip().lower()]) for col in x_columns_list]
                y_values = [float(normalized[str(col).strip().lower()]) for col in y_columns_list]
            except KeyError:
                continue
            except ValueError:
                continue

            x_rows.append(x_values)
            y_rows.append(y_values)
            records.append(
                {
                    "file": str(file_path),
                    "columns": parsed["columns"],
                    "x_columns": x_columns_list,
                    "y_columns": y_columns_list,
                }
            )

    if not x_rows:
        raise ValueError("no usable tabular samples found in input directory")

    x = np.array(x_rows, dtype=float)
    y = np.array(y_rows, dtype=float)
    if y.shape[1] == 1:
        y = y.reshape(-1)
    return x, y, records


def load_tabular_inputs(
    input_dir: str,
    x_columns: str | Iterable[str],
    delimiter: str | None = None,
    extensions: Iterable[str] = SUPPORTED_TABULAR_EXTENSIONS,
) -> tuple[np.ndarray, list[dict[str, object]]]:
    x_columns_list = [x_columns] if isinstance(x_columns, str) else list(x_columns)
    x_rows: list[list[float]] = []
    records: list[dict[str, object]] = []

    for file_path in _iter_data_files(input_dir, extensions=extensions):
        try:
            parsed = read_tabular_file(str(file_path), delimiter=delimiter)
        except ValueError:
            continue

        for row in parsed["rows"]:
            normalized = {str(key).strip().lower(): value for key, value in row.items()}
            try:
                x_values = [float(normalized[str(col).strip().lower()]) for col in x_columns_list]
            except KeyError:
                continue
            except ValueError:
                continue

            x_rows.append(x_values)
            records.append(
                {
                    "file": str(file_path),
                    "columns": parsed["columns"],
                    "x_columns": x_columns_list,
                }
            )

    if not x_rows:
        raise ValueError("no usable tabular input rows found in input directory")

    return np.array(x_rows, dtype=float), records


def predict_tabular_folder(
    train_dir: str,
    model_name: str,
    x_columns: str | Iterable[str],
    y_columns: str | Iterable[str],
    predict_dir: str | None = None,
    x_pred_columns: str | Iterable[str] | None = None,
    delimiter: str | None = None,
    output_dir: str | None = None,
    include_gpr: bool = False,
    multi_output: bool | None = None,
) -> dict[str, object]:
    x_train, y_train, train_records = load_tabular_samples(
        train_dir,
        x_columns,
        y_columns,
        delimiter=delimiter,
    )
    if x_pred_columns is None:
        x_pred_columns = x_columns
    x_predict, predict_records = load_tabular_inputs(
        predict_dir or train_dir,
        x_pred_columns,
        delimiter=delimiter,
    )

    if multi_output is None:
        multi_output = y_train.ndim == 2 and y_train.shape[1] > 1

    model = train_model(model_name, x_train, y_train, include_gpr=include_gpr, multi_output=multi_output)
    predictions = predict(model, x_predict)

    x_columns_list = [x_pred_columns] if isinstance(x_pred_columns, str) else list(x_pred_columns)
    if predictions.ndim == 1:
        pred_columns = ["prediction"]
    else:
        pred_columns = [f"prediction_{i}" for i in range(predictions.shape[1])]

    output_path = None
    if output_dir is not None:
        output_dir_path = Path(output_dir)
        output_dir_path.mkdir(parents=True, exist_ok=True)
        output_path = output_dir_path / f"predictions_{model_name}.csv"
        with output_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([*x_columns_list, *pred_columns])
            for x_row, pred_row in zip(x_predict.tolist(), predictions.tolist() if predictions.ndim > 1 else [[float(predictions[i])] for i in range(predictions.shape[0])]):
                writer.writerow([*x_row, *pred_row])

    return {
        "model": model_name,
        "train_dir": train_dir,
        "predict_dir": predict_dir or train_dir,
        "output_path": str(output_path) if output_path is not None else None,
        "predictions": predictions,
        "x_columns": x_columns_list,
        "pred_columns": pred_columns,
        "train_records": train_records,
        "predict_records": predict_records,
    }


def build_model(name: str, feature_count: int, include_gpr: bool = True, multi_output: bool = False):
    models = get_generic_models(feature_count, include_gpr=include_gpr)
    if name not in models:
        raise ValueError(f"unknown model name: {name}")
    model = models[name]
    if multi_output:
        return MultiOutputRegressor(model)
    return model


def train_model(name: str, x: np.ndarray, y: np.ndarray, include_gpr: bool = False, multi_output: bool = False):
    feature_count = int(x.shape[1]) if x.ndim == 2 else 1
    model = build_model(name, feature_count=feature_count, include_gpr=include_gpr, multi_output=multi_output)
    model.fit(x, y)
    return model


def predict(model, x_new: np.ndarray) -> np.ndarray:
    x_pred = np.array(x_new, dtype=float)
    if x_pred.ndim == 1:
        x_pred = x_pred.reshape(1, -1)
    return model.predict(x_pred)
