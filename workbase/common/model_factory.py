"""
兼容层：旧路径 `workbase.common.model_factory` 继续可用。
正式实现已迁移到 `project1.modeling.factory`。
"""

from project1.modeling.factory import build_1d_models, build_3d_models, get_model_info

__all__ = ["build_1d_models", "build_3d_models", "get_model_info"]
