"""
模型版本管理模块单元测试
"""

import pytest
import tempfile
import json
from pathlib import Path
from datetime import datetime
import sys

# 添加项目路径
ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from model_versioning import (
    create_model_metadata,
    save_model_with_metadata,
    load_model_metadata,
    check_version_compatibility,
    compare_model_versions,
    compute_data_hash
)


class TestCreateModelMetadata:
    """测试创建模型元数据"""

    def test_create_basic_metadata(self):
        """测试创建基本元数据"""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)

            metadata = create_model_metadata(
                model_name="ridge_deg2",
                model_type="1D",
                input_type="xi",
                data_path=data_path,
                config={"n_splits": 5, "include_gpr": False}
            )

            assert metadata["model_info"]["name"] == "ridge_deg2"
            assert metadata["model_info"]["type"] == "1D"
            assert metadata["model_info"]["input_type"] == "xi"
            assert metadata["training_config"]["n_splits"] == 5
            assert "version" in metadata
            assert "created_at" in metadata
            assert "dependencies" in metadata

    def test_create_metadata_with_metrics(self):
        """测试创建包含指标的元数据"""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)

            metrics = {
                "mae": 0.025,
                "rmse": 0.032,
                "r2": 0.991
            }

            metadata = create_model_metadata(
                model_name="xgboost",
                model_type="3D",
                input_type="rpm_wcor_xi",
                data_path=data_path,
                config={},
                metrics=metrics
            )

            assert metadata["metrics"]["mae"] == 0.025
            assert metadata["metrics"]["rmse"] == 0.032
            assert metadata["metrics"]["r2"] == 0.991

    def test_create_metadata_with_additional_info(self):
        """测试创建包含附加信息的元数据"""
        with tempfile.TemporaryDirectory() as temp_dir:
            data_path = Path(temp_dir)

            additional_info = {
                "component": "CMP",
                "model_category": "Ensemble / XGBoost"
            }

            metadata = create_model_metadata(
                model_name="xgboost",
                model_type="3D",
                input_type="rpm_wcor_xi",
                data_path=data_path,
                config={},
                additional_info=additional_info
            )

            assert metadata["additional_info"]["component"] == "CMP"
            assert metadata["additional_info"]["model_category"] == "Ensemble / XGBoost"


class TestSaveAndLoadMetadata:
    """测试保存和加载元数据"""

    def test_save_and_load_metadata(self):
        """测试保存和加载元数据"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            model_path = temp_path / "test_model.pkl"
            data_path = temp_path

            # 创建一个简单的模型对象（用字典模拟）
            model = {"type": "test_model", "params": [1, 2, 3]}

            # 创建元数据
            metadata = create_model_metadata(
                model_name="test_model",
                model_type="1D",
                input_type="xi",
                data_path=data_path,
                config={"test": True}
            )

            # 保存模型和元数据
            save_model_with_metadata(model, model_path, metadata)

            # 验证文件存在
            assert model_path.exists()
            assert model_path.with_suffix('.json').exists()

            # 加载元数据
            loaded_metadata = load_model_metadata(model_path)

            assert loaded_metadata is not None
            assert loaded_metadata["model_info"]["name"] == "test_model"
            assert loaded_metadata["training_config"]["test"] is True

    def test_load_nonexistent_metadata(self):
        """测试加载不存在的元数据"""
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "nonexistent.pkl"
            metadata = load_model_metadata(model_path)
            assert metadata is None


class TestCheckVersionCompatibility:
    """测试版本兼容性检查"""

    def test_compatible_versions(self):
        """测试兼容的版本"""
        metadata = {
            "dependencies": {
                "python": "3.9.0",
                "numpy": "1.24.0",
                "scikit-learn": "1.3.0"
            }
        }

        current_env = {
            "python": "3.9.5",
            "numpy": "1.24.0",
            "scikit-learn": "1.3.0"
        }

        is_compatible, warnings = check_version_compatibility(metadata, current_env)
        assert is_compatible is True
        assert len(warnings) == 0

    def test_incompatible_python_version(self):
        """测试不兼容的 Python 版本"""
        metadata = {
            "dependencies": {
                "python": "3.9.0",
                "numpy": "1.24.0",
                "scikit-learn": "1.3.0"
            }
        }

        current_env = {
            "python": "2.7.0",  # 主版本不同
            "numpy": "1.24.0",
            "scikit-learn": "1.3.0"
        }

        is_compatible, warnings = check_version_compatibility(metadata, current_env)
        assert is_compatible is False
        assert len(warnings) > 0
        assert any("Python" in w for w in warnings)

    def test_different_dependency_versions(self):
        """测试不同的依赖版本"""
        metadata = {
            "dependencies": {
                "python": "3.9.0",
                "numpy": "1.24.0",
                "scikit-learn": "1.3.0"
            }
        }

        current_env = {
            "python": "3.9.5",
            "numpy": "1.23.0",  # 版本不同
            "scikit-learn": "1.2.2"  # 版本不同
        }

        is_compatible, warnings = check_version_compatibility(metadata, current_env)
        # Python 主版本相同，所以兼容
        assert is_compatible is True
        # 但应该有警告
        assert len(warnings) > 0


class TestCompareModelVersions:
    """测试模型版本比较"""

    def test_compare_two_models(self):
        """测试比较两个模型"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # 创建第一个模型
            model1_path = temp_path / "model1.pkl"
            metadata1 = create_model_metadata(
                model_name="xgboost",
                model_type="3D",
                input_type="rpm_wcor_xi",
                data_path=temp_path,
                config={"n_splits": 5},
                metrics={"rmse": 0.032, "r2": 0.991}
            )
            save_model_with_metadata({"model": 1}, model1_path, metadata1)

            # 创建第二个模型
            model2_path = temp_path / "model2.pkl"
            metadata2 = create_model_metadata(
                model_name="xgboost",
                model_type="3D",
                input_type="rpm_wcor_xi",
                data_path=temp_path,
                config={"n_splits": 10},  # 配置不同
                metrics={"rmse": 0.028, "r2": 0.995}  # 指标不同
            )
            save_model_with_metadata({"model": 2}, model2_path, metadata2)

            # 比较两个模型
            comparison = compare_model_versions(model1_path, model2_path)

            assert "config_diff" in comparison
            assert "n_splits" in comparison["config_diff"]
            assert comparison["config_diff"]["n_splits"]["model_1"] == 5
            assert comparison["config_diff"]["n_splits"]["model_2"] == 10

            assert "metrics_diff" in comparison
            assert "rmse" in comparison["metrics_diff"]
            assert comparison["metrics_diff"]["rmse"]["improvement"] < 0  # 改进了


class TestComputeDataHash:
    """测试数据哈希计算"""

    def test_compute_file_hash(self):
        """测试计算文件哈希"""
        with tempfile.NamedTemporaryFile(mode='w', delete=False, encoding='utf-8') as f:
            f.write("test data content")
            temp_path = Path(f.name)

        try:
            hash1 = compute_data_hash(temp_path)
            assert len(hash1) == 16  # 前 16 位

            # 相同内容应该产生相同哈希
            hash2 = compute_data_hash(temp_path)
            assert hash1 == hash2
        finally:
            temp_path.unlink()

    def test_compute_directory_hash(self):
        """测试计算目录哈希"""
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)

            # 创建一些文件
            (temp_path / "file1.txt").write_text("content1", encoding='utf-8')
            (temp_path / "file2.txt").write_text("content2", encoding='utf-8')

            hash_value = compute_data_hash(temp_path)
            assert len(hash_value) == 16

    def test_compute_nonexistent_path_hash(self):
        """测试计算不存在路径的哈希"""
        hash_value = compute_data_hash(Path("nonexistent_path"))
        assert hash_value == "file_not_found"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
