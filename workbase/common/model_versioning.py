"""
模型版本管理系统

功能：
1. 为每个保存的模型生成版本元数据
2. 记录模型训练时的配置、数据、依赖版本
3. 支持版本比较和兼容性检查
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Optional
import sys
import importlib.metadata


def get_package_version(package_name: str) -> str:
    """获取已安装包的版本号"""
    try:
        return importlib.metadata.version(package_name)
    except importlib.metadata.PackageNotFoundError:
        return "unknown"


def compute_data_hash(data_path: Path) -> str:
    """计算数据文件的哈希值（用于检测数据变化）"""
    if not data_path.exists():
        return "file_not_found"

    hasher = hashlib.sha256()
    if data_path.is_file():
        with open(data_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                hasher.update(chunk)
    elif data_path.is_dir():
        # 对目录中所有文件计算哈希
        for file_path in sorted(data_path.rglob('*')):
            if file_path.is_file():
                with open(file_path, 'rb') as f:
                    for chunk in iter(lambda: f.read(8192), b''):
                        hasher.update(chunk)

    return hasher.hexdigest()[:16]  # 取前16位


def create_model_metadata(
    model_name: str,
    model_type: str,
    input_type: str,
    data_path: Path,
    config: Dict[str, Any],
    metrics: Optional[Dict[str, float]] = None,
    additional_info: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    创建模型版本元数据

    参数:
        model_name: 模型名称（如 "ridge_deg2"）
        model_type: 模型类型（如 "1D" 或 "3D"）
        input_type: 输入类型（如 "xi" 或 "rpm_wcor_xi"）
        data_path: 训练数据路径
        config: 训练配置参数
        metrics: 模型评估指标（可选）
        additional_info: 其他附加信息（可选）

    返回:
        版本元数据字典
    """
    metadata = {
        "version": "1.0.0",  # 模型版本号
        "created_at": datetime.now().isoformat(),
        "model_info": {
            "name": model_name,
            "type": model_type,
            "input_type": input_type,
        },
        "data_info": {
            "path": str(data_path),
            "hash": compute_data_hash(data_path),
        },
        "training_config": config,
        "dependencies": {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "numpy": get_package_version("numpy"),
            "scikit-learn": get_package_version("scikit-learn"),
            "scipy": get_package_version("scipy"),
            "xgboost": get_package_version("xgboost"),
            "lightgbm": get_package_version("lightgbm"),
        },
    }

    if metrics:
        metadata["metrics"] = metrics

    if additional_info:
        metadata["additional_info"] = additional_info

    return metadata


def save_model_with_metadata(
    model: Any,
    model_path: Path,
    metadata: Dict[str, Any]
) -> None:
    """
    保存模型及其元数据

    参数:
        model: 训练好的模型对象
        model_path: 模型保存路径（.pkl）
        metadata: 模型元数据
    """
    import joblib

    # 保存模型
    joblib.dump(model, model_path)

    # 保存元数据到同名的 .json 文件
    metadata_path = model_path.with_suffix('.json')
    with open(metadata_path, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)


def load_model_metadata(model_path: Path) -> Optional[Dict[str, Any]]:
    """
    加载模型元数据

    参数:
        model_path: 模型文件路径（.pkl）

    返回:
        元数据字典，如果不存在则返回 None
    """
    metadata_path = model_path.with_suffix('.json')
    if not metadata_path.exists():
        return None

    with open(metadata_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def check_version_compatibility(
    model_metadata: Dict[str, Any],
    current_env: Optional[Dict[str, str]] = None
) -> tuple[bool, list[str]]:
    """
    检查模型版本与当前环境的兼容性

    参数:
        model_metadata: 模型元数据
        current_env: 当前环境信息（可选，默认自动获取）

    返回:
        (是否兼容, 警告信息列表)
    """
    if current_env is None:
        current_env = {
            "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
            "numpy": get_package_version("numpy"),
            "scikit-learn": get_package_version("scikit-learn"),
            "scipy": get_package_version("scipy"),
            "xgboost": get_package_version("xgboost"),
            "lightgbm": get_package_version("lightgbm"),
        }

    warnings = []
    is_compatible = True

    saved_deps = model_metadata.get("dependencies", {})

    # 检查 Python 版本（主版本号必须一致）
    saved_python = saved_deps.get("python", "unknown")
    current_python = current_env.get("python", "unknown")
    if saved_python.split('.')[0] != current_python.split('.')[0]:
        warnings.append(f"Python 主版本不匹配: 训练时 {saved_python}, 当前 {current_python}")
        is_compatible = False

    # 检查关键依赖版本
    critical_packages = ["numpy", "scikit-learn"]
    for pkg in critical_packages:
        saved_ver = saved_deps.get(pkg, "unknown")
        current_ver = current_env.get(pkg, "unknown")
        if saved_ver != current_ver and saved_ver != "unknown" and current_ver != "unknown":
            warnings.append(f"{pkg} 版本不同: 训练时 {saved_ver}, 当前 {current_ver}")

    return is_compatible, warnings


def compare_model_versions(
    model_path_1: Path,
    model_path_2: Path
) -> Dict[str, Any]:
    """
    比较两个模型版本的差异

    参数:
        model_path_1: 第一个模型路径
        model_path_2: 第二个模型路径

    返回:
        差异信息字典
    """
    meta1 = load_model_metadata(model_path_1)
    meta2 = load_model_metadata(model_path_2)

    if meta1 is None or meta2 is None:
        return {"error": "无法加载模型元数据"}

    comparison = {
        "model_1": str(model_path_1),
        "model_2": str(model_path_2),
        "created_at_diff": {
            "model_1": meta1.get("created_at"),
            "model_2": meta2.get("created_at"),
        },
        "data_hash_diff": {
            "model_1": meta1.get("data_info", {}).get("hash"),
            "model_2": meta2.get("data_info", {}).get("hash"),
            "same_data": meta1.get("data_info", {}).get("hash") == meta2.get("data_info", {}).get("hash"),
        },
        "config_diff": {},
        "metrics_diff": {},
    }

    # 比较配置差异
    config1 = meta1.get("training_config", {})
    config2 = meta2.get("training_config", {})
    all_keys = set(config1.keys()) | set(config2.keys())
    for key in all_keys:
        val1 = config1.get(key)
        val2 = config2.get(key)
        if val1 != val2:
            comparison["config_diff"][key] = {"model_1": val1, "model_2": val2}

    # 比较指标差异
    metrics1 = meta1.get("metrics", {})
    metrics2 = meta2.get("metrics", {})
    all_metrics = set(metrics1.keys()) | set(metrics2.keys())
    for metric in all_metrics:
        val1 = metrics1.get(metric)
        val2 = metrics2.get(metric)
        if val1 is not None and val2 is not None:
            comparison["metrics_diff"][metric] = {
                "model_1": val1,
                "model_2": val2,
                "improvement": val2 - val1 if isinstance(val1, (int, float)) and isinstance(val2, (int, float)) else None
            }

    return comparison
