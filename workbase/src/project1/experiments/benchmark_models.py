"""
兼容层：历史路径 `project1.experiments.benchmark_models` 继续可用。
正式模型定义已迁移到 `project1.modeling.custom_models`。
"""

from project1.modeling.custom_models import (
    EndpointRegression,
    HierarchicalWorklineRegressor,
    LOWESSLikeRegressor,
    MODEL_CATEGORY,
    PhysicsClampedRegressor,
    PiecewiseRidgeRegressor,
    RBFRegressor,
)

__all__ = [
    "EndpointRegression",
    "HierarchicalWorklineRegressor",
    "LOWESSLikeRegressor",
    "MODEL_CATEGORY",
    "PhysicsClampedRegressor",
    "PiecewiseRidgeRegressor",
    "RBFRegressor",
]
