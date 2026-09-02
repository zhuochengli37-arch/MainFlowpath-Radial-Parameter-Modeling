from __future__ import annotations

from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures, SplineTransformer, StandardScaler
from sklearn.svm import SVR

from project1.modeling.custom_models import (
    EndpointRegression,
    HierarchicalWorklineRegressor,
    LOWESSLikeRegressor,
    PhysicsClampedRegressor,
    PiecewiseRidgeRegressor,
    RBFRegressor,
)


def build_1d_models(include_gpr: bool = False) -> dict[str, object]:
    import lightgbm as lgb
    import xgboost as xgb

    models = {
        "linear_deg2": make_pipeline(PolynomialFeatures(2, include_bias=False), LinearRegression()),
        "ridge_deg2": make_pipeline(PolynomialFeatures(2, include_bias=False), Ridge(alpha=0.1)),
        "spline_regression": make_pipeline(
            SplineTransformer(n_knots=5, degree=3, include_bias=False),
            Ridge(alpha=0.1),
        ),
        "knn_distance": KNeighborsRegressor(n_neighbors=5, weights="distance"),
        "random_forest": RandomForestRegressor(n_estimators=100, random_state=42),
        "gradient_boosting": GradientBoostingRegressor(n_estimators=100, random_state=42),
        "xgboost": xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, verbosity=0),
        "lightgbm": lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, verbosity=-1),
        "lasso": make_pipeline(PolynomialFeatures(2, include_bias=False), Lasso(alpha=0.01)),
        "elastic_net": make_pipeline(PolynomialFeatures(2, include_bias=False), ElasticNet(alpha=0.01, l1_ratio=0.5)),
        "svr_rbf": SVR(kernel="rbf", C=1.0, epsilon=0.1),
        "poly2": make_pipeline(PolynomialFeatures(degree=2), Ridge(alpha=1.0)),
        "poly3": make_pipeline(PolynomialFeatures(degree=3), Ridge(alpha=1.0)),
    }

    if include_gpr:
        from sklearn.gaussian_process.kernels import RBF

        models["gpr_rbf"] = GaussianProcessRegressor(kernel=RBF(), random_state=42)
        models["gpr_matern"] = GaussianProcessRegressor(kernel=Matern(), random_state=42)

    return models


def build_3d_models(include_gpr: bool = True) -> dict[str, object]:
    import lightgbm as lgb
    import xgboost as xgb

    models = {
        "linear_deg2": make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), LinearRegression()),
        "ridge_deg2": make_pipeline(StandardScaler(), PolynomialFeatures(2, include_bias=False), Ridge(alpha=0.1)),
        "piecewise_ridge": make_pipeline(StandardScaler(), PiecewiseRidgeRegressor(alpha=0.1)),
        "spline_regression": make_pipeline(
            StandardScaler(),
            SplineTransformer(n_knots=5, degree=3, include_bias=False),
            Ridge(alpha=0.1),
        ),
        "tensor_spline": make_pipeline(
            StandardScaler(),
            SplineTransformer(n_knots=4, degree=3, include_bias=False),
            PolynomialFeatures(2, include_bias=False),
            Ridge(alpha=0.1),
        ),
        "lowess_like": make_pipeline(StandardScaler(), LOWESSLikeRegressor(bandwidth=0.3)),
        "knn_distance": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5, weights="distance")),
        "rbf": make_pipeline(StandardScaler(), RBFRegressor(smoothing=0.01)),
        "hierarchical_workline": make_pipeline(StandardScaler(), HierarchicalWorklineRegressor(alpha=0.1)),
        "endpoint_hub_tip": make_pipeline(StandardScaler(), EndpointRegression(alpha=0.1)),
        "physics_clamped_ridge": make_pipeline(StandardScaler(), PhysicsClampedRegressor(alpha=0.1)),
        "random_forest": make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=42)),
        "gradient_boosting": make_pipeline(StandardScaler(), GradientBoostingRegressor(n_estimators=100, random_state=42)),
        "xgboost": make_pipeline(
            StandardScaler(),
            xgb.XGBRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, verbosity=0),
        ),
        "lightgbm": make_pipeline(
            StandardScaler(),
            lgb.LGBMRegressor(n_estimators=100, learning_rate=0.1, max_depth=6, random_state=42, verbosity=-1),
        ),
        "lasso": make_pipeline(StandardScaler(), Lasso(alpha=0.01)),
        "elastic_net": make_pipeline(StandardScaler(), ElasticNet(alpha=0.01, l1_ratio=0.5)),
        "svr_rbf": make_pipeline(StandardScaler(), SVR(kernel="rbf", C=1.0, epsilon=0.1)),
    }

    if include_gpr:
        gp_kernel = ConstantKernel(1.0) * Matern(length_scale=[1.0, 1.0, 1.0], nu=1.5) + WhiteKernel(1e-5)
        models["gpr_matern"] = make_pipeline(
            StandardScaler(),
            GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True),
        )

    return models


def get_model_info() -> dict:
    return {
        "1d_models": {
            "count": 13,
            "categories": [
                "多项式回归",
                "样条回归",
                "局部回归",
                "传统集成方法",
                "高级梯度提升",
                "正则化线性模型",
                "支持向量回归",
                "可选 GPR",
            ],
        },
        "3d_models": {
            "count": 19,
            "categories": [
                "RSM/多项式回归",
                "分段回归",
                "样条回归",
                "局部回归",
                "领域特定模型",
                "传统集成方法",
                "高级梯度提升",
                "正则化线性模型",
                "支持向量回归",
                "可选 GPR",
            ],
        },
    }
