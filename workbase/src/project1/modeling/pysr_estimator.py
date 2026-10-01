"""Small subprocess bridge to an isolated official PySR runtime."""

from __future__ import annotations

import json
import os
import subprocess
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class ExternalPySRRegressor(RegressorMixin, BaseEstimator):
    """Sklearn-compatible adapter; numeric data crosses environments as NPZ."""

    def __init__(
        self,
        python_executable: str | None = None,
        niterations: int = 2,
        populations: int = 2,
        population_size: int = 10,
        ncycles_per_iteration: int = 20,
        timeout_in_seconds: float = 120.0,
        binary_operators: tuple[str, ...] = ("+", "-", "*", "/"),
        unary_operators: tuple[str, ...] = (),
        constraints: dict[str, Any] | None = None,
        maxsize: int = 12,
        maxdepth: int = 6,
        model_selection: str = "best",
        random_state: int = 42,
        parallelism: str = "serial",
        precision: int = 64,
    ) -> None:
        self.python_executable = python_executable
        self.niterations = niterations
        self.populations = populations
        self.population_size = population_size
        self.ncycles_per_iteration = ncycles_per_iteration
        self.timeout_in_seconds = timeout_in_seconds
        self.binary_operators = binary_operators
        self.unary_operators = unary_operators
        self.constraints = constraints
        self.maxsize = maxsize
        self.maxdepth = maxdepth
        self.model_selection = model_selection
        self.random_state = random_state
        self.parallelism = parallelism
        self.precision = precision

    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[4]

    def _python_path(self) -> Path:
        configured = self.python_executable or os.environ.get("PROJECT1_PYSR_PYTHON")
        if configured:
            path = Path(configured).expanduser().resolve()
        else:
            environment = self._project_root() / ".venv-pysr"
            path = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not path.is_file():
            raise FileNotFoundError(
                "PySR isolated Python was not found. Create .venv-pysr or set "
                "PROJECT1_PYSR_PYTHON to its interpreter."
            )
        return path

    def _runner_environment(self, python_path: Path) -> dict[str, str]:
        environment = os.environ.copy()
        environment_root = python_path.parent.parent
        julia_depot = environment.get(
            "JULIA_DEPOT_PATH", str(environment_root / "julia_depot")
        )
        julia_project = self._project_root() / "envs" / "pysr-julia"
        julia_executable = (
            environment_root
            / "julia_project"
            / "pyjuliapkg"
            / "install"
            / "bin"
            / ("julia.exe" if os.name == "nt" else "julia")
        )
        environment.update(
            {
                "PYTHON_JULIAPKG_PROJECT": str(environment_root / "julia_project"),
                "PYTHON_JULIACALL_PROJECT": str(julia_project),
                "PYTHON_JULIACALL_EXE": str(julia_executable),
                "JULIA_DEPOT_PATH": julia_depot,
                "PYTHON_JULIACALL_THREADS": "auto",
                "PYTHON_JULIAPKG_OFFLINE": "yes",
                "JULIA_PKG_OFFLINE": "true",
            }
        )
        return environment

    def _invoke(self, arguments: list[str], operation: str) -> None:
        python_path = self._python_path()
        command = [str(python_path), str(Path(__file__).with_name("pysr_runner.py")), *arguments]
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=self._runner_environment(python_path),
            )
        except subprocess.CalledProcessError as exc:
            diagnostic = (exc.stderr or exc.stdout or str(exc)).strip()
            raise RuntimeError(f"PySR external {operation} failed: {diagnostic}") from exc

    def set_run_context(self, **context: Any) -> "ExternalPySRRegressor":
        self.run_context_ = context
        return self

    def set_prediction_context(self, **context: Any) -> "ExternalPySRRegressor":
        self.prediction_context_ = context
        return self

    def _search_config(self) -> dict[str, Any]:
        return {
            "niterations": self.niterations,
            "populations": self.populations,
            "population_size": self.population_size,
            "tournament_selection_n": min(5, self.population_size - 1),
            "ncycles_per_iteration": self.ncycles_per_iteration,
            "timeout_in_seconds": self.timeout_in_seconds,
            "binary_operators": list(self.binary_operators),
            "unary_operators": list(self.unary_operators),
            "constraints": self.constraints or {},
            "maxsize": self.maxsize,
            "maxdepth": self.maxdepth,
            "model_selection": self.model_selection,
            "random_state": self.random_state,
            "deterministic": True,
            "parallelism": self.parallelism,
            "precision": self.precision,
            "progress": False,
            "verbosity": 0,
        }

    def fit(self, x, y):
        x_array = np.asarray(x, dtype=float)
        y_array = np.asarray(y, dtype=float).reshape(-1)
        if self.population_size < 2:
            raise ValueError("PySR population_size must be at least 2")
        if x_array.ndim != 2 or x_array.shape[1] != 3 or len(x_array) != len(y_array):
            raise ValueError("PySR expects X.shape=(N, 3) ordered as [n, w, xi] and y.shape=(N,)")
        if not np.isfinite(x_array).all() or not np.isfinite(y_array).all():
            raise ValueError("PySR training data must contain only finite values")

        with tempfile.TemporaryDirectory(prefix="project1-pysr-") as temporary:
            root = Path(temporary)
            input_path = root / "train.npz"
            config_path = root / "config.json"
            artifact_dir = root / "artifact"
            response_path = root / "response.json"
            condition_ids = np.asarray(
                getattr(self, "run_context_", {}).get("train_condition_ids", []), dtype="U"
            )
            np.savez_compressed(input_path, X=x_array, y=y_array, condition_ids=condition_ids)
            config = {
                **self._search_config(),
                "feature_names": ["n", "w", "xi"],
                "context": getattr(self, "run_context_", {}),
            }
            config_path.write_text(json.dumps(config, ensure_ascii=False), encoding="utf-8")
            self._invoke(
                [
                    "train",
                    "--input", str(input_path),
                    "--config", str(config_path),
                    "--artifact-dir", str(artifact_dir),
                    "--response", str(response_path),
                ],
                "training",
            )
            self.symbolic_report_ = json.loads(response_path.read_text(encoding="utf-8"))
            self.model_artifacts_ = {
                str(path.relative_to(artifact_dir)): path.read_bytes()
                for path in artifact_dir.rglob("*")
                if path.is_file()
            }

        self.n_features_in_ = 3
        self.feature_names_in_ = np.asarray(["n", "w", "xi"], dtype=object)
        self.selected_equation_ = self.symbolic_report_["selected_equation"]
        self.complexity_ = self.symbolic_report_["selected_complexity"]
        self.equation_table_ = self.symbolic_report_["equation_table"]
        self.training_time_ = self.symbolic_report_["training_time_seconds"]
        return self

    def predict(self, x):
        check_is_fitted(self, ("model_artifacts_", "selected_equation_"))
        x_array = np.asarray(x, dtype=float)
        if x_array.ndim != 2 or x_array.shape[1] != 3:
            raise ValueError("PySR expects X.shape=(N, 3) ordered as [n, w, xi]")
        with tempfile.TemporaryDirectory(prefix="project1-pysr-") as temporary:
            root = Path(temporary)
            artifact_dir = root / "artifact"
            artifact_dir.mkdir()
            for relative_path, content in self.model_artifacts_.items():
                destination = artifact_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
            input_path = root / "predict.npz"
            output_path = root / "prediction.npz"
            condition_ids = np.asarray(
                getattr(self, "prediction_context_", {}).get("condition_ids", []), dtype="U"
            )
            np.savez_compressed(input_path, X=x_array, condition_ids=condition_ids)
            self._invoke(
                [
                    "predict",
                    "--input", str(input_path),
                    "--artifact-dir", str(artifact_dir),
                    "--output", str(output_path),
                ],
                "prediction",
            )
            with np.load(output_path, allow_pickle=False) as result:
                prediction = np.asarray(result["prediction"], dtype=float).reshape(-1)
                self.last_inference_time_ = float(result["inference_time_seconds"])
        if len(prediction) != len(x_array) or not np.isfinite(prediction).all():
            raise ValueError("PySR returned invalid predictions")
        self.symbolic_report_["last_inference_time_seconds"] = self.last_inference_time_
        return prediction
