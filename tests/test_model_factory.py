"""
模型工厂模块单元测试
"""

import pytest
from pathlib import Path
import sys

# 添加项目路径
ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from model_factory import build_1d_models, build_3d_models


class TestBuild1DModels:
    """测试构建1D模型"""

    def test_build_1d_models_without_gpr(self):
        """测试构建不包含GPR的1D模型"""
        models = build_1d_models(include_gpr=False)

        # 验证返回的是字典
        assert isinstance(models, dict)

        # 验证模型数量（13个模型，不包含GPR）
        assert len(models) >= 12

        # 验证关键模型存在
        assert "linear_deg2" in models
        assert "ridge_deg2" in models
        assert "xgboost" in models
        assert "lightgbm" in models
        assert "spline_regression" in models

        # 验证不包含GPR
        assert "gpr_matern" not in models

    def test_build_1d_models_with_gpr(self):
        """测试构建包含GPR的1D模型"""
        models = build_1d_models(include_gpr=True)

        # 验证包含GPR
        assert "gpr_matern" in models

    def test_1d_models_are_sklearn_compatible(self):
        """测试1D模型是否符合sklearn接口"""
        models = build_1d_models(include_gpr=False)

        for name, model in models.items():
            # 验证模型有fit和predict方法
            assert hasattr(model, 'fit'), f"模型 {name} 缺少 fit 方法"
            assert hasattr(model, 'predict'), f"模型 {name} 缺少 predict 方法"


class TestBuild3DModels:
    """测试构建3D模型"""

    def test_build_3d_models_without_gpr(self):
        """测试构建不包含GPR的3D模型"""
        models = build_3d_models(include_gpr=False)

        # 验证返回的是字典
        assert isinstance(models, dict)

        # 验证模型数量（19个模型，不包含GPR）
        assert len(models) >= 18

        # 验证关键模型存在
        assert "linear_deg2" in models
        assert "ridge_deg2" in models
        assert "xgboost" in models
        assert "lightgbm" in models
        assert "hierarchical_workline" in models
        assert "endpoint_hub_tip" in models
        assert "physics_clamped_ridge" in models

        # 验证不包含GPR
        assert "gpr_matern" not in models

    def test_build_3d_models_with_gpr(self):
        """测试构建包含GPR的3D模型"""
        models = build_3d_models(include_gpr=True)

        # 验证包含GPR
        assert "gpr_matern" in models

    def test_3d_models_are_sklearn_compatible(self):
        """测试3D模型是否符合sklearn接口"""
        models = build_3d_models(include_gpr=False)

        for name, model in models.items():
            # 验证模型有fit和predict方法
            assert hasattr(model, 'fit'), f"模型 {name} 缺少 fit 方法"
            assert hasattr(model, 'predict'), f"模型 {name} 缺少 predict 方法"

    def test_3d_has_more_models_than_1d(self):
        """测试3D模型数量多于1D模型（因为包含领域特定模型）"""
        models_1d = build_1d_models(include_gpr=False)
        models_3d = build_3d_models(include_gpr=False)

        # 3D应该包含更多模型（有领域特定模型）
        assert len(models_3d) > len(models_1d)

    def test_domain_specific_models_only_in_3d(self):
        """测试领域特定模型只在3D中存在"""
        models_1d = build_1d_models(include_gpr=False)
        models_3d = build_3d_models(include_gpr=False)

        # 领域特定模型
        domain_specific = ["hierarchical_workline", "endpoint_hub_tip", "physics_clamped_ridge"]

        for model_name in domain_specific:
            assert model_name not in models_1d, f"{model_name} 不应该在1D模型中"
            assert model_name in models_3d, f"{model_name} 应该在3D模型中"


class TestModelConsistency:
    """测试模型一致性"""

    def test_common_models_exist_in_both(self):
        """测试通用模型在1D和3D中都存在"""
        models_1d = build_1d_models(include_gpr=False)
        models_3d = build_3d_models(include_gpr=False)

        # 通用模型应该在两者中都存在
        common_models = [
            "linear_deg2", "ridge_deg2", "xgboost", "lightgbm",
            "spline_regression", "random_forest", "gradient_boosting"
        ]

        for model_name in common_models:
            assert model_name in models_1d, f"{model_name} 应该在1D模型中"
            assert model_name in models_3d, f"{model_name} 应该在3D模型中"

    def test_model_names_are_unique(self):
        """测试模型名称唯一性"""
        models_1d = build_1d_models(include_gpr=True)
        models_3d = build_3d_models(include_gpr=True)

        # 验证没有重复的键
        assert len(models_1d) == len(set(models_1d.keys()))
        assert len(models_3d) == len(set(models_3d.keys()))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
