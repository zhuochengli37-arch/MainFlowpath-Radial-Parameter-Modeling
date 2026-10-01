"""Small subprocess bridge to an isolated official TabPFN runtime."""

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


CHECKPOINT_FILENAME = "tabpfn-v3.5-fast-20260909.safetensors"


class ExternalTabPFNRegressor(RegressorMixin, BaseEstimator):
    """Sklearn-compatible adapter; numeric data crosses environments as NPZ."""

    def __init__(
        self,
        python_executable: str | None = None,
        checkpoint_path: str | None = None,
        n_estimators: int = 1,
        random_state: int = 42,
        fit_mode: str = "fit_preprocessors",
        n_preprocessing_jobs: int = 1,
        torch_num_threads: int = 1,
        timeout_seconds: float = 600.0,
    ) -> None:
        self.python_executable = python_executable
        self.checkpoint_path = checkpoint_path
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.fit_mode = fit_mode
        self.n_preprocessing_jobs = n_preprocessing_jobs
        self.torch_num_threads = torch_num_threads
        self.timeout_seconds = timeout_seconds

    @staticmethod
    def _project_root() -> Path:
        return Path(__file__).resolve().parents[4]

    def _python_path(self) -> Path:
        configured = self.python_executable or os.environ.get("PROJECT1_TABPFN_PYTHON")
        if configured:
            path = Path(configured).expanduser().resolve()
        else:
            environment = self._project_root() / ".venv-tabpfn"
            path = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
        if not path.is_file():
            raise FileNotFoundError(
                "TabPFN isolated Python was not found. Create .venv-tabpfn or set "
                "PROJECT1_TABPFN_PYTHON to its interpreter."
            )
        return path

    def _checkpoint(self, python_path: Path) -> Path:
        configured = self.checkpoint_path or os.environ.get("PROJECT1_TABPFN_CHECKPOINT")
        path = (
            Path(configured).expanduser().resolve()
            if configured
            else python_path.parent.parent / "model_cache" / CHECKPOINT_FILENAME
        )
        if not path.is_file():
            raise FileNotFoundError(
                f"TabPFN checkpoint was not found at {path}. Download the pinned "
                f"{CHECKPOINT_FILENAME} checkpoint before running the bridge."
            )
        return path

    def _runner_environment(self, python_path: Path) -> dict[str, str]:
        environment = os.environ.copy()
        environment_root = python_path.parent.parent
        environment.update(
            {
                "TABPFN_NO_BROWSER": "1",
                "TABPFN_MODEL_CACHE_DIR": str(environment_root / "model_cache"),
                "HF_HUB_OFFLINE": "1",
                "TRANSFORMERS_OFFLINE": "1",
                "SKB_DATA_DIRECTORY": str(environment_root / "skrub_data"),
                "OMP_NUM_THREADS": str(self.torch_num_threads),
                "MKL_NUM_THREADS": str(self.torch_num_threads),
                "OPENBLAS_NUM_THREADS": str(self.torch_num_threads),
            }
        )
        return environment

    def _invoke(self, arguments: list[str], operation: str) -> None:
        python_path = self._python_path()
        command = [
            str(python_path),
            str(Path(__file__).with_name("tabpfn_runner.py")),
            *arguments,
        ]
        try:
            subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                env=self._runner_environment(python_path),
                timeout=self.timeout_seconds,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(
                f"TabPFN external {operation} timed out after {self.timeout_seconds:g}s"
            ) from exc
        except subprocess.CalledProcessError as exc:
            diagnostic = (exc.stderr or exc.stdout or str(exc)).strip()
            raise RuntimeError(f"TabPFN external {operation} failed: {diagnostic}") from exc

    def set_run_context(self, **context: Any) -> "ExternalTabPFNRegressor":
        self.run_context_ = context
        return self

    def set_prediction_context(self, **context: Any) -> "ExternalTabPFNRegressor":
        self.prediction_context_ = context
        return self

    def fit(self, x, y):
        x_array = np.asarray(x, dtype=float)
        y_array = np.asarray(y, dtype=float).reshape(-1)
        if x_array.ndim != 2 or x_array.shape[1] != 3 or len(x_array) != len(y_array):
            raise ValueError(
                "TabPFN expects X.shape=(N, 3) ordered as [n, w, xi] and y.shape=(N,)"
            )
        if not np.isfinite(x_array).all() or not np.isfinite(y_array).all():
            raise ValueError("TabPFN training data must contain only finite values")

        python_path = self._python_path()
        checkpoint = self._checkpoint(python_path)
        context = getattr(self, "run_context_", {})
        condition_ids = np.asarray(context.get("train_condition_ids", []), dtype="U")
        if condition_ids.size and len(condition_ids) != len(x_array):
            raise ValueError("train_condition_ids must be empty or match the training rows")

        with tempfile.TemporaryDirectory(prefix="project1-tabpfn-") as temporary:
            root = Path(temporary)
            input_path = root / "train.npz"
            config_path = root / "config.json"
            artifact_dir = root / "artifact"
            response_path = root / "response.json"
            np.savez_compressed(input_path, X=x_array, y=y_array, condition_ids=condition_ids)
            config_path.write_text(
                json.dumps(
                    {
                        "feature_names": ["n", "w", "xi"],
                        "n_estimators": self.n_estimators,
                        "random_state": self.random_state,
                        "fit_mode": self.fit_mode,
                        "n_preprocessing_jobs": self.n_preprocessing_jobs,
                        "torch_num_threads": self.torch_num_threads,
                        "context": context,
                    },
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            self._invoke(
                [
                    "train",
                    "--input", str(input_path),
                    "--config", str(config_path),
                    "--checkpoint", str(checkpoint),
                    "--artifact-dir", str(artifact_dir),
                    "--response", str(response_path),
                ],
                "training",
            )
            self.foundation_report_ = json.loads(response_path.read_text(encoding="utf-8"))
            self.model_artifacts_ = {
                str(path.relative_to(artifact_dir)): path.read_bytes()
                for path in artifact_dir.rglob("*")
                if path.is_file()
            }

        self.n_features_in_ = 3
        self.feature_names_in_ = np.asarray(["n", "w", "xi"], dtype=object)
        self.training_time_ = self.foundation_report_["fit_context_time_seconds"]
        return self

    def predict(self, x):
        check_is_fitted(self, ("model_artifacts_", "foundation_report_"))
        x_array = np.asarray(x, dtype=float)
        if x_array.ndim != 2 or x_array.shape[1] != 3:
            raise ValueError("TabPFN expects X.shape=(N, 3) ordered as [n, w, xi]")
        if not np.isfinite(x_array).all():
            raise ValueError("TabPFN prediction data must contain only finite values")

        python_path = self._python_path()
        checkpoint = self._checkpoint(python_path)
        condition_ids = np.asarray(
            getattr(self, "prediction_context_", {}).get("condition_ids", []), dtype="U"
        )
        if condition_ids.size and len(condition_ids) != len(x_array):
            raise ValueError("prediction condition_ids must be empty or match the rows")

        with tempfile.TemporaryDirectory(prefix="project1-tabpfn-") as temporary:
            root = Path(temporary)
            artifact_dir = root / "artifact"
            artifact_dir.mkdir()
            for relative_path, content in self.model_artifacts_.items():
                destination = artifact_dir / relative_path
                destination.parent.mkdir(parents=True, exist_ok=True)
                destination.write_bytes(content)
            input_path = root / "predict.npz"
            output_path = root / "prediction.npz"
            np.savez_compressed(input_path, X=x_array, condition_ids=condition_ids)
            self._invoke(
                [
                    "predict",
                    "--input", str(input_path),
                    "--checkpoint", str(checkpoint),
                    "--artifact-dir", str(artifact_dir),
                    "--output", str(output_path),
                    "--torch-num-threads", str(self.torch_num_threads),
                ],
                "prediction",
            )
            with np.load(output_path, allow_pickle=False) as result:
                prediction = np.asarray(result["prediction"], dtype=float).reshape(-1)
                self.last_load_time_ = float(result["load_time_seconds"])
                self.last_inference_time_ = float(result["inference_time_seconds"])
                if "peak_rss_mib" in result.files:
                    self.last_peak_rss_mib_ = float(result["peak_rss_mib"])
        if len(prediction) != len(x_array) or not np.isfinite(prediction).all():
            raise ValueError("TabPFN returned invalid predictions")
        self.foundation_report_["last_load_time_seconds"] = self.last_load_time_
        self.foundation_report_["last_inference_time_seconds"] = self.last_inference_time_
        if hasattr(self, "last_peak_rss_mib_"):
            self.foundation_report_["last_prediction_peak_rss_mib"] = self.last_peak_rss_mib_
        return prediction
