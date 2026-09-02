"""
AIR2 Project1 配置加载器。

负责读取 YAML 配置文件，并提供带类型语义的配置访问入口。
未提供配置文件时，会回退到脚本级默认值。
"""

from datetime import datetime
from pathlib import Path
import shutil
from typing import Any, Optional


def _load_yaml(config_path: Path) -> dict:
    """加载 YAML 文件；如果缺少 PyYAML，则给出明确的错误提示。"""
    try:
        import yaml
    except ImportError:
        raise ImportError(
            "PyYAML is required for config file support. "
            "Install it with: pip install pyyaml"
        )
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _deep_get(data: dict, *keys, default=None) -> Any:
    """安全读取嵌套字典中的键。"""
    for key in keys:
        if not isinstance(data, dict):
            return default
        data = data.get(key, default)
    return data


class BenchmarkConfig:
    """`benchmark_config.yaml` 的类型化配置包装器。"""

    def __init__(self, data: dict):
        self._data = data

    # 1D / 2D / 3D benchmark 路径
    @property
    def benchmark_1d_input_path(self) -> str:
        return _deep_get(
            self._data,
            "benchmark_1d",
            "paths",
            "input",
            default="./data/current/1d/sample6/train/example1.txt",
        )

    @property
    def benchmark_1d_output_dir(self) -> str:
        return _deep_get(
            self._data,
            "benchmark_1d",
            "paths",
            "output",
            default="./data/current/1d/sample6/output",
        )

    @property
    def benchmark_2d_input_path(self) -> str:
        return _deep_get(
            self._data,
            "benchmark_2d",
            "paths",
            "input",
            default="./data/current/3d/DATACASE1/train",
        )

    @property
    def benchmark_2d_output_dir(self) -> str:
        return _deep_get(
            self._data,
            "benchmark_2d",
            "paths",
            "output",
            default="./data/current/3d/DATACASE1/output",
        )

    @property
    def benchmark_3d_input_path(self) -> str:
        return _deep_get(
            self._data,
            "benchmark_3d",
            "paths",
            "input",
            default="./data/current/3d/DATACASEVAR_31/train",
        )

    @property
    def benchmark_3d_output_dir(self) -> str:
        return _deep_get(
            self._data,
            "benchmark_3d",
            "paths",
            "output",
            default="./data/current/3d/DATACASEVAR_31/output",
        )

    # 数据集 schema
    @property
    def dataset_schema_mode(self) -> str:
        return _deep_get(self._data, "dataset_schema", "mode", default="auto")

    @property
    def schema_inputs(self) -> list[str]:
        values = _deep_get(self._data, "dataset_schema", "inputs", default=["rpm", "wcor", "xi"])
        return [str(item) for item in values]

    @property
    def schema_targets(self) -> list[str]:
        values = _deep_get(self._data, "dataset_schema", "targets", default=["psi", "tsi", "mai"])
        return [str(item) for item in values]

    @property
    def schema_metadata(self) -> list[str]:
        values = _deep_get(
            self._data,
            "dataset_schema",
            "metadata",
            default=["component", "stage", "database"],
        )
        return [str(item) for item in values]

    @property
    def schema_source_priority(self) -> dict[str, list[str]]:
        raw = _deep_get(self._data, "dataset_schema", "source_priority", default={})
        if not isinstance(raw, dict):
            return {}
        result: dict[str, list[str]] = {}
        for key, value in raw.items():
            if isinstance(value, list):
                result[str(key)] = [str(item) for item in value]
        return result

    @property
    def schema_aliases(self) -> dict[str, list[str]]:
        raw = _deep_get(self._data, "dataset_schema", "aliases", default={})
        if not isinstance(raw, dict):
            return {}
        result: dict[str, list[str]] = {}
        for key, value in raw.items():
            if isinstance(value, list):
                result[str(key)] = [str(item) for item in value]
        return result

    @property
    def schema_partition_mode(self) -> str:
        return _deep_get(self._data, "dataset_schema", "partition", "mode", default="config")

    @property
    def schema_partition_keys(self) -> list[str]:
        values = _deep_get(
            self._data,
            "dataset_schema",
            "partition",
            "keys",
            default=["component", "stage"],
        )
        return [str(item) for item in values]

    @property
    def missing_value_sentinels(self) -> list[float]:
        values = _deep_get(
            self._data,
            "dataset_schema",
            "missing_value_sentinels",
            default=[-999.0],
        )
        result: list[float] = []
        if not isinstance(values, list):
            values = [values]
        for item in values:
            try:
                result.append(float(item))
            except (TypeError, ValueError):
                continue
        return result or [-999.0]

    # 3D benchmark 配置
    @property
    def radial_mode(self) -> str:
        return _deep_get(self._data, "benchmark_3d", "radial_mode", default="mean")

    @property
    def partition_mode(self) -> str:
        return _deep_get(self._data, "benchmark_3d", "partition_mode", default="single")

    @property
    def max_samples_3d(self) -> int:
        return int(_deep_get(self._data, "benchmark_3d", "max_samples", default=10000))

    @property
    def include_gpr_3d(self) -> bool:
        return bool(_deep_get(self._data, "benchmark_3d", "include_gpr", default=False))

    @property
    def n_splits_3d(self) -> int:
        return int(_deep_get(self._data, "benchmark_3d", "n_splits", default=3))

    # 2D benchmark 配置
    @property
    def radial_mode_2d(self) -> str:
        return _deep_get(self._data, "benchmark_2d", "radial_mode", default=self.radial_mode)

    @property
    def partition_mode_2d(self) -> str:
        return _deep_get(self._data, "benchmark_2d", "partition_mode", default=self.partition_mode)

    @property
    def max_samples_2d(self) -> int:
        return int(_deep_get(self._data, "benchmark_2d", "max_samples", default=self.max_samples_3d))

    @property
    def include_gpr_2d(self) -> bool:
        return bool(_deep_get(self._data, "benchmark_2d", "include_gpr", default=self.include_gpr_3d))

    @property
    def n_splits_2d(self) -> int:
        return int(_deep_get(self._data, "benchmark_2d", "n_splits", default=3))

    # 1D benchmark 配置
    @property
    def n_splits(self) -> int:
        return int(_deep_get(self._data, "benchmark_1d", "n_splits", default=5))

    @property
    def n_splits_1d(self) -> int:
        return self.n_splits

    @property
    def max_samples_1d(self) -> int:
        return int(_deep_get(self._data, "benchmark_1d", "max_samples", default=10000))

    @property
    def include_gpr_1d(self) -> bool:
        return bool(_deep_get(self._data, "benchmark_1d", "include_gpr", default=False))

    # 1D / 2D / 3D prediction 路径
    @property
    def predict_1d_input_path(self) -> str:
        return _deep_get(
            self._data,
            "predict_1d",
            "paths",
            "input",
            default="./data/current/1d/sample6/predict/example1.txt",
        )

    @property
    def predict_1d_model_output_dir(self) -> str:
        return _deep_get(
            self._data,
            "predict_1d",
            "paths",
            "model_output",
            default="./data/current/1d/sample6/output",
        )

    @property
    def predict_1d_output_path(self) -> str:
        path = _deep_get(self._data, "predict_1d", "paths", "output", default="")
        return path if path else f"{self.predict_1d_model_output_dir}/results"

    @property
    def predict_2d_input_path(self) -> str:
        return _deep_get(
            self._data,
            "predict_2d",
            "paths",
            "input",
            default="./data/current/3d/DATACASE1/predict",
        )

    @property
    def predict_2d_model_output_dir(self) -> str:
        return _deep_get(
            self._data,
            "predict_2d",
            "paths",
            "model_output",
            default="./data/current/3d/DATACASE1/output",
        )

    @property
    def predict_2d_output_path(self) -> str:
        path = _deep_get(self._data, "predict_2d", "paths", "output", default="")
        return path if path else f"{self.predict_2d_model_output_dir}/results"

    @property
    def predict_2d_partition(self) -> Optional[str]:
        return _deep_get(self._data, "predict_2d", "partition", default=None)

    @property
    def predict_3d_input_path(self) -> str:
        return _deep_get(
            self._data,
            "predict_3d",
            "paths",
            "input",
            default="./data/current/3d/DATACASEVAR_31/predict",
        )

    @property
    def predict_3d_model_output_dir(self) -> str:
        return _deep_get(
            self._data,
            "predict_3d",
            "paths",
            "model_output",
            default="./data/current/3d/DATACASEVAR_31/output",
        )

    @property
    def predict_3d_output_path(self) -> str:
        path = _deep_get(self._data, "predict_3d", "paths", "output", default="")
        return path if path else f"{self.predict_3d_model_output_dir}/results"

    @property
    def predict_3d_partition(self) -> Optional[str]:
        return _deep_get(self._data, "predict_3d", "partition", default=None)

    @property
    def predict_3d_model_name(self) -> Optional[str]:
        return _deep_get(self._data, "predict_3d", "model_name", default=None)

    @property
    def predict_3d_selection_split_mode(self) -> str:
        return str(_deep_get(self._data, "predict_3d", "selection_split_mode", default="random")).lower()

    @property
    def predict_3d_selection_metric(self) -> str:
        return str(_deep_get(self._data, "predict_3d", "selection_metric", default="rmse")).lower()

    # 通用高级配置
    @property
    def random_state(self) -> int:
        return int(_deep_get(self._data, "advanced", "random_state", default=42))

    @property
    def log_dir(self) -> str:
        return _deep_get(self._data, "advanced", "log_dir", default="./logs")

    @property
    def save_models(self) -> bool:
        return bool(_deep_get(self._data, "output", "save_models", default=True))

    @property
    def generate_leaderboard(self) -> bool:
        return bool(_deep_get(self._data, "output", "generate_leaderboard", default=True))

    @property
    def generate_report(self) -> bool:
        return bool(_deep_get(self._data, "output", "generate_report", default=True))

    @classmethod
    def load(cls, config_path: str) -> "BenchmarkConfig":
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        return cls(_load_yaml(path))

    @classmethod
    def defaults(cls) -> "BenchmarkConfig":
        return cls({})

    def __repr__(self) -> str:
        return (
            f"BenchmarkConfig("
            f"benchmark_1d_input={self.benchmark_1d_input_path!r}, "
            f"predict_3d_input={self.predict_3d_input_path!r}, "
            f"radial_mode={self.radial_mode!r})"
        )


def load_config(config_path: Optional[str] = None) -> BenchmarkConfig:
    """加载 benchmark 配置；未提供路径时使用默认配置。"""
    if config_path is None:
        return BenchmarkConfig.defaults()
    return BenchmarkConfig.load(config_path)


class GenericConfig:
    """`generic_config.yaml` 的类型化配置包装器。"""

    def __init__(self, data: dict):
        self._data = data

    def benchmark_input_path(self, dim: int) -> str:
        return _deep_get(
            self._data,
            f"generic_benchmark_{dim}d",
            "paths",
            "input",
            default=f"./data/generic/{dim}d/train",
        )

    def benchmark_output_dir(self, dim: int) -> str:
        return _deep_get(
            self._data,
            f"generic_benchmark_{dim}d",
            "paths",
            "output",
            default=f"./data/generic/{dim}d/output",
        )

    def benchmark_max_samples(self, dim: int) -> Optional[int]:
        value = _deep_get(
            self._data,
            f"generic_benchmark_{dim}d",
            "max_samples",
            default=None,
        )
        return None if value in (None, "", "null") else int(value)

    def benchmark_n_splits(self, dim: int) -> int:
        return int(
            _deep_get(
                self._data,
                f"generic_benchmark_{dim}d",
                "n_splits",
                default=5,
            )
        )

    def benchmark_include_gpr(self, dim: int) -> bool:
        return bool(
            _deep_get(
                self._data,
                f"generic_benchmark_{dim}d",
                "include_gpr",
                default=False,
            )
        )

    def predict_input_path(self, dim: int) -> str:
        return _deep_get(
            self._data,
            f"generic_predict_{dim}d",
            "paths",
            "input",
            default=f"./data/generic/{dim}d/predict",
        )

    def predict_model_output_dir(self, dim: int) -> str:
        return _deep_get(
            self._data,
            f"generic_predict_{dim}d",
            "paths",
            "model_output",
            default=f"./data/generic/{dim}d/output",
        )

    def predict_output_dir(self, dim: int) -> str:
        return _deep_get(
            self._data,
            f"generic_predict_{dim}d",
            "paths",
            "output",
            default=f"./data/generic/{dim}d/results",
        )

    @property
    def log_dir(self) -> str:
        return _deep_get(self._data, "advanced", "log_dir", default="./logs")

    @classmethod
    def load(cls, config_path: str) -> "GenericConfig":
        path = Path(config_path)
        if not path.exists():
            raise FileNotFoundError(f"配置文件不存在: {path}")
        return cls(_load_yaml(path))

    @classmethod
    def defaults(cls) -> "GenericConfig":
        return cls({})


def load_generic_config(config_path: Optional[str] = None) -> GenericConfig:
    """加载通用流程配置；未提供路径时使用默认配置。"""
    if config_path is None:
        return GenericConfig.defaults()
    return GenericConfig.load(config_path)


def save_config_snapshot(
    config_path: str | Path,
    output_dir: str | Path,
    snapshot_prefix: str = "benchmark_config",
) -> dict[str, str]:
    """保存训练配置快照，同时写入最新副本和带时间戳的历史副本。"""
    config_file = Path(config_path)
    if not config_file.exists():
        raise FileNotFoundError(f"配置文件不存在，无法保存快照: {config_file}")

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    history_dir = output_path / "run_configs"
    history_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_path = history_dir / f"{snapshot_prefix}_{timestamp}.yaml"
    counter = 1
    while snapshot_path.exists():
        snapshot_path = history_dir / f"{snapshot_prefix}_{timestamp}_{counter}.yaml"
        counter += 1

    latest_path = output_path / "run_config_latest.yaml"
    shutil.copy2(config_file, snapshot_path)
    shutil.copy2(config_file, latest_path)

    return {
        "snapshot": str(snapshot_path),
        "latest": str(latest_path),
    }
