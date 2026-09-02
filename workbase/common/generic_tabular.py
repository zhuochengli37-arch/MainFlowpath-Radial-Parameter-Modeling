"""
通用表格训练/预测辅助模块。

设计约定：
- 输入数据必须带表头。
- 前 N 列作为输入列，剩余列作为输出列。
- 训练时自动遍历单文件或目录下的 .txt/.csv/.dat 文件。
- 预测时优先按训练阶段保存的 schema 取输入列。
"""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
from sklearn.svm import SVR

from project1.services.data_reader import read_tabular_file
from workbase.common.model_versioning import (
    check_version_compatibility,
    create_model_metadata,
    load_model_metadata,
    save_model_with_metadata,
)
from workbase.common.prediction_output import (
    find_matching_template_file,
    load_tabular_template,
    write_table_like_template,
)
from project1.modeling.factory import build_1d_models


SUPPORTED_SUFFIXES = {".txt", ".csv", ".dat"}


@dataclass
class GenericSchema:
    input_dim: int
    x_columns: list[str]
    y_columns: list[str]
    train_files: list[str]


def list_tabular_files(input_path: str | Path) -> list[Path]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"输入路径不存在: {path}")
    if path.is_file():
        return [path]

    files = sorted(
        file_path
        for file_path in path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(f"目录下未找到可用表格文件: {path}")
    return files


def _to_float_matrix(rows: list[dict[str, object]], columns: list[str]) -> np.ndarray:
    values: list[list[float]] = []
    for row in rows:
        try:
            values.append([float(row[column]) for column in columns])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"列 {columns} 中包含非数值数据，无法用于训练/预测") from exc
    return np.asarray(values, dtype=float)


def infer_schema(files: list[Path], input_dim: int) -> GenericSchema:
    first = read_tabular_file(str(files[0]))
    columns = [str(column) for column in first["columns"]]
    if len(columns) <= input_dim:
        raise ValueError(
            f"文件 {files[0].name} 列数不足。{input_dim}D 至少需要 {input_dim + 1} 列，"
            f"当前只有 {len(columns)} 列。"
        )

    x_columns = columns[:input_dim]
    y_columns = columns[input_dim:]

    for file_path in files[1:]:
        parsed = read_tabular_file(str(file_path))
        current_columns = [str(column) for column in parsed["columns"]]
        current_x_columns = current_columns[:input_dim]
        if current_x_columns != x_columns:
            raise ValueError(
                f"文件输入列不一致: {file_path.name} 的前 {input_dim} 列为 {current_x_columns}，"
                f"与首个文件 {files[0].name} 的输入列 {x_columns} 不一致。"
            )
        for target in current_columns[input_dim:]:
            if target not in y_columns:
                y_columns.append(target)

    return GenericSchema(
        input_dim=input_dim,
        x_columns=x_columns,
        y_columns=y_columns,
        train_files=[str(file_path) for file_path in files],
    )


def load_training_dataset(files: list[Path], schema: GenericSchema, max_samples: int | None = None) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    x_parts_by_target: dict[str, list[np.ndarray]] = {target: [] for target in schema.y_columns}
    y_parts: dict[str, list[np.ndarray]] = {target: [] for target in schema.y_columns}

    for file_path in files:
        parsed = read_tabular_file(str(file_path))
        rows = parsed["rows"]
        for target in schema.y_columns:
            if target not in parsed["columns"]:
                continue
            x_parts_by_target[target].append(_to_float_matrix(rows, schema.x_columns))
            y_parts[target].append(_to_float_matrix(rows, [target]).ravel())

    x_all_by_target: dict[str, np.ndarray] = {}
    y_all: dict[str, np.ndarray] = {}
    rng = np.random.default_rng(42)

    for target in schema.y_columns:
        if not x_parts_by_target[target]:
            continue
        x_all = np.vstack(x_parts_by_target[target])
        y_target = np.concatenate(y_parts[target])
        if max_samples and len(x_all) > max_samples:
            indices = np.sort(rng.choice(len(x_all), size=max_samples, replace=False))
            x_all = x_all[indices]
            y_target = y_target[indices]
        x_all_by_target[target] = x_all
        y_all[target] = y_target

    return x_all_by_target, y_all


def _build_generic_nd_models(input_dim: int, include_gpr: bool) -> dict[str, object]:
    models: dict[str, object] = {
        "linear_deg2": make_pipeline(
            StandardScaler(),
            PolynomialFeatures(2, include_bias=False),
            LinearRegression(),
        ),
        "ridge_deg2": make_pipeline(
            StandardScaler(),
            PolynomialFeatures(2, include_bias=False),
            Ridge(alpha=0.1),
        ),
        "spline_regression": make_pipeline(
            StandardScaler(),
            SplineTransformer(n_knots=5, degree=3, include_bias=False),
            Ridge(alpha=0.1),
        ),
        "knn_distance": make_pipeline(
            StandardScaler(),
            KNeighborsRegressor(n_neighbors=5, weights="distance"),
        ),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "lasso": make_pipeline(StandardScaler(), Lasso(alpha=0.01)),
        "elastic_net": make_pipeline(StandardScaler(), ElasticNet(alpha=0.01, l1_ratio=0.5)),
        "svr_rbf": make_pipeline(StandardScaler(), SVR(kernel="rbf", C=1.0, epsilon=0.1)),
    }

    try:
        import xgboost as xgb

        models["xgboost"] = xgb.XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbosity=0,
        )
    except ImportError:
        pass

    try:
        import lightgbm as lgb

        models["lightgbm"] = lgb.LGBMRegressor(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            random_state=42,
            verbosity=-1,
        )
    except ImportError:
        pass

    if include_gpr:
        kernel = ConstantKernel(1.0) * Matern(length_scale=[1.0] * input_dim, nu=1.5) + WhiteKernel(1e-5)
        models["gpr_matern"] = make_pipeline(
            StandardScaler(),
            GaussianProcessRegressor(kernel=kernel, normalize_y=True),
        )

    return models


def get_generic_models(input_dim: int, include_gpr: bool = False) -> dict[str, object]:
    if input_dim == 1:
        return build_1d_models(include_gpr=include_gpr)
    if input_dim in (2, 3):
        return _build_generic_nd_models(input_dim=input_dim, include_gpr=include_gpr)
    raise ValueError(f"暂不支持 {input_dim}D 通用建模")


def _resolve_kfold_splits(sample_count: int, requested_splits: int) -> int:
    if sample_count < 2:
        raise ValueError("At least 2 rows are required for cross-validation")
    safe_cap = max(2, sample_count // 2)
    return max(2, min(requested_splits, sample_count, safe_cap))


def _prepare_model_for_fold(model: object, train_size: int) -> object:
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


def _safe_r2_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if len(y_true) < 2:
        return float("nan")
    return float(r2_score(y_true, y_pred))


def _effective_folds_by_target(leaderboard: list[dict[str, Any]]) -> dict[str, int]:
    folds: dict[str, int] = {}
    for row in leaderboard:
        target = row.get("target")
        fold_count = row.get("folds")
        if not isinstance(target, str) or target in folds:
            continue
        if isinstance(fold_count, int):
            folds[target] = fold_count
    return folds


def _reserve_prediction_output_path(output_file: Path) -> Path:
    if not output_file.exists():
        return output_file

    stem = output_file.stem
    suffix = output_file.suffix
    for index in range(1, 1000):
        candidate = output_file.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError(f"unable to allocate prediction output path for {output_file}")


def _evaluate_model(model: object, x: np.ndarray, y: np.ndarray, n_splits: int) -> dict[str, float | int]:
    if len(x) < 2:
        raise ValueError("样本数不足，至少需要 2 行数据")

    actual_splits = _resolve_kfold_splits(len(x), n_splits)
    kf = KFold(n_splits=actual_splits, shuffle=True, random_state=42)
    rmses: list[float] = []
    maes: list[float] = []
    r2s: list[float] = []

    for train_idx, test_idx in kf.split(x):
        candidate = _prepare_model_for_fold(model, len(train_idx))
        candidate.fit(x[train_idx], y[train_idx])
        pred = candidate.predict(x[test_idx])
        rmses.append(float(np.sqrt(mean_squared_error(y[test_idx], pred))))
        maes.append(float(mean_absolute_error(y[test_idx], pred)))
        r2s.append(_safe_r2_score(y[test_idx], pred))

    return {
        "folds": actual_splits,
        "rmse": float(np.mean(rmses)),
        "mae": float(np.mean(maes)),
        "r2": float(np.nanmean(r2s)) if not np.all(np.isnan(r2s)) else float("nan"),
    }


def run_generic_benchmark(
    input_dim: int,
    input_path: str | Path,
    output_dir: str | Path,
    include_gpr: bool = False,
    max_samples: int | None = None,
    n_splits: int = 5,
) -> dict[str, Any]:
    files = list_tabular_files(input_path)
    schema = infer_schema(files, input_dim=input_dim)
    x_all_by_target, y_all = load_training_dataset(files, schema=schema, max_samples=max_samples)

    output_path = Path(output_dir)
    models_dir = output_path / "models"
    output_path.mkdir(parents=True, exist_ok=True)
    models_dir.mkdir(parents=True, exist_ok=True)

    schema_path = output_path / "schema.json"
    schema_path.write_text(
        json.dumps(
            {
                "input_dim": schema.input_dim,
                "x_columns": schema.x_columns,
                "y_columns": schema.y_columns,
                "train_files": schema.train_files,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    leaderboard: list[dict[str, Any]] = []
    saved_models: list[dict[str, str]] = []
    model_candidates = get_generic_models(input_dim=input_dim, include_gpr=include_gpr)

    for target in schema.y_columns:
        rows_for_target: list[dict[str, Any]] = []
        if target not in x_all_by_target:
            leaderboard.append(
                {
                    "target": target,
                    "model": None,
                    "rmse": None,
                    "mae": None,
                    "r2": None,
                    "error": "未找到包含该输出列的训练数据",
                }
            )
            continue
        x_target = x_all_by_target[target]
        y_target = y_all[target]

        for model_name, model in model_candidates.items():
            try:
                metrics = _evaluate_model(model, x_target, y_target, n_splits=n_splits)
                row = {"target": target, "model": model_name, **metrics}
                rows_for_target.append(row)
                leaderboard.append(row)
            except Exception as exc:
                leaderboard.append(
                    {
                        "target": target,
                        "model": model_name,
                        "rmse": None,
                        "mae": None,
                        "r2": None,
                        "error": str(exc),
                    }
                )

        valid_rows = [row for row in rows_for_target if row["rmse"] is not None]
        if not valid_rows:
            continue

        best_row = min(valid_rows, key=lambda item: item["rmse"])
        best_model = clone(model_candidates[best_row["model"]])
        best_model.fit(x_target, y_target)

        model_path = models_dir / f"{target}__best.pkl"
        metadata = create_model_metadata(
            model_name=best_row["model"],
            model_type=f"generic_{input_dim}d",
            input_type="__".join(schema.x_columns),
            data_path=Path(input_path),
            config={
                "input_dim": input_dim,
                "x_columns": schema.x_columns,
                "y_columns": schema.y_columns,
                "requested_n_splits": n_splits,
                "include_gpr": include_gpr,
                "max_samples": max_samples,
            },
            metrics=best_row,
            additional_info={"target": target, "schema_path": str(schema_path)},
        )
        save_model_with_metadata(best_model, model_path, metadata)
        saved_models.append(
            {
                "target": target,
                "model": best_row["model"],
                "model_path": str(model_path),
                "row_count": int(len(x_target)),
            }
        )

    leaderboard.sort(key=lambda item: (item["target"], item["rmse"] is None, item["rmse"] or float("inf"), item["model"]))
    leaderboard_path = output_path / "leaderboard.json"
    leaderboard_path.write_text(json.dumps(leaderboard, ensure_ascii=False, indent=2), encoding="utf-8")

    manifest_path = output_path / "models_manifest.json"
    manifest_path.write_text(json.dumps(saved_models, ensure_ascii=False, indent=2), encoding="utf-8")
    effective_n_splits = _effective_folds_by_target(leaderboard)

    return {
        "schema_path": str(schema_path),
        "leaderboard_path": str(leaderboard_path),
        "manifest_path": str(manifest_path),
        "saved_models": saved_models,
        "requested_n_splits": n_splits,
        "effective_n_splits": effective_n_splits,
        "row_count": int(sum(len(values) for values in x_all_by_target.values())),
    }


def load_schema(model_output_dir: str | Path) -> GenericSchema:
    schema_path = Path(model_output_dir) / "schema.json"
    if not schema_path.exists():
        raise FileNotFoundError(f"未找到 schema.json: {schema_path}")
    data = json.loads(schema_path.read_text(encoding="utf-8"))
    return GenericSchema(
        input_dim=int(data["input_dim"]),
        x_columns=[str(item) for item in data["x_columns"]],
        y_columns=[str(item) for item in data["y_columns"]],
        train_files=[str(item) for item in data.get("train_files", [])],
    )


def _coerce_prediction_value(value: str) -> object:
    text = value.strip()
    if not text:
        return text
    try:
        return float(text)
    except ValueError:
        return text


def _read_prediction_table(file_path: Path) -> tuple[list[str], list[dict[str, object]]]:
    raw = file_path.read_text(encoding="utf-8", errors="ignore")
    lines = [line for line in raw.splitlines() if line.strip()]
    if len(lines) < 2:
        raise ValueError("file has no usable tabular content")

    delimiter: str | None = None
    header_line = lines[0]
    if "," in header_line:
        delimiter = ","
    elif "\t" in header_line:
        delimiter = "\t"

    if delimiter is None:
        rows_data = [line.strip().replace("\t", " ").split() for line in lines]
    else:
        reader = csv.reader(lines, delimiter=delimiter)
        rows_data = [[cell.strip() for cell in row] for row in reader]

    if len(rows_data) < 2:
        raise ValueError("file has no usable tabular content")

    headers = [name for name in rows_data[0] if name]
    if not headers:
        raise ValueError("invalid header format")

    rows: list[dict[str, object]] = []
    for row in rows_data[1:]:
        values = [cell for cell in row if cell]
        if not values:
            continue
        record = {headers[idx]: _coerce_prediction_value(value) for idx, value in enumerate(values[: len(headers)])}
        rows.append(record)

    if not rows:
        raise ValueError("no numeric rows parsed")

    return headers, rows


def _read_prediction_features(file_path: Path, schema: GenericSchema) -> tuple[list[str], list[dict[str, object]], np.ndarray]:
    columns, rows = _read_prediction_table(file_path)

    if all(column in columns for column in schema.x_columns):
        x_columns = schema.x_columns
    else:
        if len(columns) < schema.input_dim:
            raise ValueError(f"文件 {file_path.name} 列数不足，无法提取 {schema.input_dim} 个输入列")
        x_columns = columns[: schema.input_dim]

    return columns, rows, _to_float_matrix(rows, x_columns)


def _build_train_template_root(train_files: list[str]) -> Path:
    paths = [Path(path) for path in train_files]
    if not paths:
        raise ValueError("schema has no train_files")
    if len(paths) == 1:
        return paths[0]
    return Path(os.path.commonpath([str(path) for path in paths]))


def _load_best_model_names(model_output_dir: Path) -> dict[str, str]:
    leaderboard_path = model_output_dir / "leaderboard.json"
    if not leaderboard_path.exists():
        raise FileNotFoundError(f"未找到 leaderboard.json: {leaderboard_path}")

    leaderboard = json.loads(leaderboard_path.read_text(encoding="utf-8"))
    best_by_target: dict[str, dict[str, Any]] = {}
    for row in leaderboard:
        if row.get("rmse") is None:
            continue
        target = str(row["target"])
        current = best_by_target.get(target)
        if current is None or float(row["rmse"]) < float(current["rmse"]):
            best_by_target[target] = row
    return {target: str(row["model"]) for target, row in best_by_target.items()}


def _load_prediction_model(model_path: Path) -> object:
    metadata = load_model_metadata(model_path)
    if metadata is not None:
        is_compatible, warnings = check_version_compatibility(metadata)
        if warnings:
            details = "; ".join(warnings)
            raise ValueError(f"妯″瀷鐗堟湰涓庡綋鍓嶇幆澧冧笉鍏煎: {details}")
        if not is_compatible:
            raise ValueError(f"妯″瀷鐗堟湰涓庡綋鍓嶇幆澧冧笉鍏煎: {model_path}")
    return joblib.load(model_path)


def run_generic_predict(
    input_dim: int,
    input_path: str | Path,
    model_output_dir: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    model_root = Path(model_output_dir)
    schema = load_schema(model_root)
    if schema.input_dim != input_dim:
        raise ValueError(f"模型目录是 {schema.input_dim}D，当前脚本是 {input_dim}D")

    files = list_tabular_files(input_path)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    input_root = Path(input_path)
    template_root = _build_train_template_root(schema.train_files)

    _load_best_model_names(model_root)
    models = {
        target: _load_prediction_model(model_root / "models" / f"{target}__best.pkl")
        for target in schema.y_columns
    }

    outputs: list[str] = []
    for file_path in files:
        columns, rows, x_values = _read_prediction_features(file_path, schema=schema)
        template_file = find_matching_template_file(
            template_root,
            predict_file=file_path,
            predict_root=input_root,
        )
        template = load_tabular_template(template_file)
        predictions = {target: model.predict(x_values) for target, model in models.items()}
        output_columns = list(template.columns)
        for target in schema.y_columns:
            pred_column = f"pred_{target}"
            if pred_column not in output_columns:
                output_columns.append(pred_column)

        result_rows: list[dict[str, object]] = []
        for idx, row in enumerate(rows):
            record: dict[str, object] = {}
            for column in output_columns:
                if column in row:
                    record[column] = row[column]
                elif column.startswith("pred_") and column[5:] in predictions:
                    value = predictions[column[5:]][idx]
                    record[column] = float(value) if np.isscalar(value) else value
                elif column in predictions:
                    record[column] = ""
                else:
                    record[column] = ""
            result_rows.append(record)

        if input_root.is_dir():
            relative = file_path.relative_to(input_root)
            output_file = output_path / relative.with_suffix(template.suffix)
        else:
            output_file = output_path / f"{file_path.stem}{template.suffix}"
        write_table_like_template(
            output_file,
            type(template)(source_path=template.source_path, columns=output_columns, delimiter=template.delimiter, suffix=template.suffix),
            result_rows,
        )
        outputs.append(str(output_file))

    summary_path = output_path / "prediction_manifest.json"
    summary_path.write_text(
        json.dumps(
            {
                "input_dim": input_dim,
                "x_columns": schema.x_columns,
                "y_columns": schema.y_columns,
                "outputs": outputs,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    return {"outputs": outputs, "manifest_path": str(summary_path)}
