from __future__ import annotations

from sklearn.ensemble import ExtraTreesRegressor, GradientBoostingRegressor, RandomForestRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, RBF, WhiteKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import ElasticNet, Lasso, LinearRegression, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
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
from project1.modeling.registry import MODEL_FAMILIES, list_model_metadata
from project1.modeling.optional_estimators import LazyImportRegressor, PyKANRegressor
from project1.modeling.pysr_estimator import ExternalPySRRegressor
from project1.modeling.tabpfn_estimator import ExternalTabPFNRegressor


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
        "gam": make_pipeline(
            StandardScaler(),
            LazyImportRegressor(
                "pygam",
                "LinearGAM",
                "gam",
                {"n_splines": 10, "lam": 0.6, "max_iter": 100},
            ),
        ),
        "lowess_like": make_pipeline(StandardScaler(), LOWESSLikeRegressor(bandwidth=0.3)),
        "knn_distance": make_pipeline(StandardScaler(), KNeighborsRegressor(n_neighbors=5, weights="distance")),
        "rbf": make_pipeline(StandardScaler(), RBFRegressor(smoothing=0.01)),
        "kernel_ridge_rbf": make_pipeline(
            StandardScaler(),
            KernelRidge(alpha=1.0, kernel="rbf", gamma=None),
        ),
        "hierarchical_workline": make_pipeline(StandardScaler(), HierarchicalWorklineRegressor(alpha=0.1)),
        "endpoint_hub_tip": make_pipeline(StandardScaler(), EndpointRegression(alpha=0.1)),
        "physics_clamped_ridge": make_pipeline(StandardScaler(), PhysicsClampedRegressor(alpha=0.1)),
        "random_forest": make_pipeline(StandardScaler(), RandomForestRegressor(n_estimators=100, random_state=42)),
        "extra_trees": make_pipeline(
            StandardScaler(),
            ExtraTreesRegressor(
                n_estimators=100,
                max_depth=None,
                min_samples_leaf=1,
                max_features=1.0,
                random_state=42,
            ),
        ),
        "catboost": LazyImportRegressor(
            "catboost",
            "CatBoostRegressor",
            "catboost",
            {
                "iterations": 200,
                "depth": 6,
                "learning_rate": 0.05,
                "loss_function": "RMSE",
                "random_seed": 42,
                "verbose": False,
                "allow_writing_files": False,
                "thread_count": 1,
            },
        ),
        "ebm": LazyImportRegressor(
            "interpret.glassbox",
            "ExplainableBoostingRegressor",
            "ebm",
            {
                "interactions": 0,
                "max_rounds": 500,
                "outer_bags": 8,
                "early_stopping_rounds": 50,
                "random_state": 42,
                "n_jobs": 1,
            },
        ),
        "ngboost": LazyImportRegressor(
            "ngboost",
            "NGBRegressor",
            "ngboost",
            {
                "n_estimators": 200,
                "learning_rate": 0.03,
                "random_state": 42,
                "verbose": False,
            },
        ),
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
        "mlp": make_pipeline(
            StandardScaler(),
            MLPRegressor(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                alpha=1e-4,
                learning_rate_init=1e-3,
                max_iter=500,
                random_state=42,
            ),
        ),
        "realmlp": LazyImportRegressor(
            "pytabkit",
            "RealMLP_TD_Regressor",
            "realmlp",
            {
                "n_epochs": 64,
                "batch_size": 256,
                "n_threads": 1,
                "random_state": 42,
                "device": "cpu",
                "verbosity": 0,
            },
            "phase2b",
        ),
        "tabm": LazyImportRegressor(
            "pytabkit",
            "TabM_D_Regressor",
            "tabm",
            {
                "n_epochs": 64,
                "patience": 10,
                "batch_size": 256,
                "tabm_k": 16,
                "d_block": 128,
                "n_blocks": 2,
                "compile_model": False,
                "n_threads": 1,
                "random_state": 42,
                "device": "cpu",
                "verbosity": 0,
            },
            "phase2b",
        ),
        "original_kan": make_pipeline(StandardScaler(), PyKANRegressor()),
        "resnet_mlp": LazyImportRegressor(
            "pytabkit",
            "Resnet_RTDL_D_Regressor",
            "resnet_mlp",
            {
                "max_epochs": 100,
                "batch_size": 256,
                "module_d": 128,
                "module_n_layers": 2,
                "n_threads": 1,
                "random_state": 42,
                "device": "cpu",
                "verbosity": 0,
            },
            "phase2b",
        ),
        "ft_transformer": LazyImportRegressor(
            "pytabkit",
            "FTT_D_Regressor",
            "ft_transformer",
            {
                "max_epochs": 100,
                "batch_size": 256,
                "module_d_token": 64,
                "module_n_layers": 2,
                "module_n_heads": 4,
                "n_threads": 1,
                "random_state": 42,
                "device": "cpu",
                "verbosity": 0,
            },
            "phase2b",
        ),
        "tabiclv2": LazyImportRegressor(
            "tabicl",
            "TabICLRegressor",
            "tabiclv2",
            {
                "n_estimators": 8,
                "batch_size": 1,
                "device": "cpu",
                "use_amp": False,
                "use_fa3": False,
                "offload_mode": False,
                "allow_auto_download": True,
                "checkpoint_version": "tabicl-regressor-v2-20260212.ckpt",
                "random_state": 42,
                "n_jobs": 1,
                "verbose": False,
            },
            "phase2c",
        ),
        "pysr": ExternalPySRRegressor(),
        "tabpfn": ExternalTabPFNRegressor(),
    }

    if include_gpr:
        gp_kernel = ConstantKernel(1.0) * Matern(length_scale=[1.0, 1.0, 1.0], nu=1.5) + WhiteKernel(1e-5)
        models["gpr_matern"] = make_pipeline(
            StandardScaler(),
            GaussianProcessRegressor(kernel=gp_kernel, normalize_y=True),
        )
        ard_kernel = ConstantKernel(1.0) * RBF(
            length_scale=[1.0, 1.0, 1.0],
            length_scale_bounds=(1e-2, 1e2),
        ) + WhiteKernel(1e-5)
        models["ard_gpr"] = make_pipeline(
            StandardScaler(),
            GaussianProcessRegressor(
                kernel=ard_kernel,
                normalize_y=True,
                random_state=42,
            ),
        )

    return models


def get_model_info() -> dict:
    implemented_3d = list_model_metadata(implementation_status="implemented")
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
            "count": len(implemented_3d),
            "registered_count": len(list_model_metadata()),
            "categories": list(MODEL_FAMILIES),
        },
    }
