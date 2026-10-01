"""
模型工厂模块单元测试
"""

import pytest
import numpy as np
import pickle
from pathlib import Path
import sys

# 添加项目路径
ROOT = Path(__file__).resolve().parent.parent
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from model_factory import build_1d_models, build_3d_models
from project1.experiments.benchmark_runners import _build_2d_models


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

        # 18 original non-GPR models + 15 implemented extension models.
        assert len(models) == 33

        # 验证关键模型存在
        assert "linear_deg2" in models
        assert "ridge_deg2" in models
        assert "xgboost" in models
        assert "lightgbm" in models
        assert "hierarchical_workline" in models
        assert "endpoint_hub_tip" in models
        assert "physics_clamped_ridge" in models
        assert "kernel_ridge_rbf" in models
        assert "extra_trees" in models
        assert {"gam", "catboost", "ebm", "ngboost", "mlp"} <= set(models)
        assert {
            "realmlp", "tabm", "original_kan", "resnet_mlp", "ft_transformer"
        } <= set(models)
        assert "tabiclv2" in models
        assert "pysr" in models
        assert "tabpfn" in models

        # 验证不包含GPR
        assert "gpr_matern" not in models
        assert "ard_gpr" not in models

    def test_build_3d_models_with_gpr(self):
        """测试构建包含GPR的3D模型"""
        models = build_3d_models(include_gpr=True)

        # 验证包含GPR
        assert len(models) == 35
        assert "gpr_matern" in models
        assert "ard_gpr" in models

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


@pytest.fixture
def direct_3d_smoke_data():
    x = np.array(
        [
            [n, w, xi]
            for n in (0.7, 0.85, 1.0)
            for w in (10.0, 12.0)
            for xi in (0.0, 0.5, 1.0)
        ],
        dtype=float,
    )
    y = 0.4 * x[:, 0] - 0.03 * x[:, 1] + np.sin(np.pi * x[:, 2])
    return x, y


@pytest.mark.parametrize(
    ("model_name", "include_gpr"),
    (("kernel_ridge_rbf", False), ("extra_trees", False), ("ard_gpr", True)),
)
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_phase_2a1_models_fit_and_predict(direct_3d_smoke_data, model_name, include_gpr):
    x, y = direct_3d_smoke_data
    model = build_3d_models(include_gpr=include_gpr)[model_name]

    model.fit(x, y)
    prediction = model.predict(x[:4])

    assert prediction.shape == (4,)
    assert np.all(np.isfinite(prediction))


def test_phase_2a1_model_defaults_are_configurable():
    models = build_3d_models(include_gpr=True)
    kernel_ridge = models["kernel_ridge_rbf"].named_steps["kernelridge"]
    extra_trees = models["extra_trees"].named_steps["extratreesregressor"]
    ard_gpr = models["ard_gpr"].named_steps["gaussianprocessregressor"]
    matern_gpr = models["gpr_matern"].named_steps["gaussianprocessregressor"]

    assert kernel_ridge.alpha == 1.0
    assert kernel_ridge.gamma is None
    assert extra_trees.n_estimators == 100
    assert extra_trees.max_depth is None
    assert extra_trees.min_samples_leaf == 1
    assert extra_trees.max_features == 1.0
    assert extra_trees.random_state == 42
    assert ard_gpr.kernel.k1.k2.__class__.__name__ == "RBF"
    assert len(ard_gpr.kernel.k1.k2.length_scale) == 3
    assert matern_gpr.kernel.k1.k2.__class__.__name__ == "Matern"

    models["kernel_ridge_rbf"].set_params(kernelridge__alpha=0.5, kernelridge__gamma=0.25)
    models["extra_trees"].set_params(extratreesregressor__max_depth=8)
    models["ard_gpr"].set_params(gaussianprocessregressor__random_state=7)


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_ard_gpr_learns_three_directional_scales_and_returns_uncertainty(
    direct_3d_smoke_data,
):
    x, y = direct_3d_smoke_data
    model = build_3d_models(include_gpr=True)["ard_gpr"].fit(x, y)

    prediction, standard_deviation = model.predict(x[:4], return_std=True)
    fitted_rbf = model.named_steps["gaussianprocessregressor"].kernel_.k1.k2

    assert len(fitted_rbf.length_scale) == 3
    assert prediction.shape == standard_deviation.shape == (4,)
    assert np.all(np.isfinite(standard_deviation))
    assert np.all(standard_deviation >= 0.0)


def test_phase_2a1_models_do_not_enter_2d_factory():
    assert {
        "kernel_ridge_rbf", "extra_trees", "ard_gpr", "gam", "catboost", "ebm",
        "ngboost", "mlp", "realmlp", "tabm", "original_kan", "resnet_mlp",
        "ft_transformer", "tabiclv2",
        "pysr", "tabpfn",
    }.isdisjoint(
        _build_2d_models(include_gpr=True)
    )


@pytest.mark.parametrize(
    ("model_name", "dependency"),
    (
        ("gam", "pygam"),
        ("ebm", "interpret"),
        ("catboost", "catboost"),
        ("ngboost", "ngboost"),
        ("mlp", None),
    ),
)
@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_remaining_phase_2a_models_fit_and_predict(
    direct_3d_smoke_data,
    model_name,
    dependency,
):
    if dependency:
        pytest.importorskip(dependency)
    x, y = direct_3d_smoke_data
    model = build_3d_models(include_gpr=False)[model_name]

    model.fit(x, y)
    prediction = model.predict(x[:4])

    assert prediction.shape == (4,)
    assert np.all(np.isfinite(prediction))


def test_remaining_phase_2a_defaults_are_exposed_and_configurable():
    models = build_3d_models(include_gpr=False)
    gam = models["gam"].named_steps["lazyimportregressor"]
    mlp = models["mlp"].named_steps["mlpregressor"]

    assert gam.model_kwargs == {"n_splines": 10, "lam": 0.6, "max_iter": 100}
    assert models["catboost"].model_kwargs["random_seed"] == 42
    assert models["ebm"].model_kwargs["random_state"] == 42
    assert models["ngboost"].model_kwargs["random_state"] == 42
    assert mlp.hidden_layer_sizes == (64, 32)
    assert mlp.random_state == 42

    models["gam"].set_params(lazyimportregressor__model_kwargs={"max_iter": 20})
    models["catboost"].set_params(model_kwargs={"iterations": 20})
    models["mlp"].set_params(mlpregressor__max_iter=20)


def test_optional_dependency_error_is_model_specific(monkeypatch, direct_3d_smoke_data):
    def missing_dependency(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("project1.modeling.optional_estimators.import_module", missing_dependency)
    x, y = direct_3d_smoke_data

    with pytest.raises(ImportError, match=r"gam.*pip install -e .\[phase2a\]"):
        build_3d_models(include_gpr=False)["gam"].fit(x, y)


@pytest.mark.parametrize(
    "model_name",
    ("realmlp", "tabm", "original_kan", "resnet_mlp", "ft_transformer"),
)
def test_phase_2b_models_fit_and_predict(direct_3d_smoke_data, model_name):
    pytest.importorskip("pytabkit" if model_name != "original_kan" else "kan")
    x, y = direct_3d_smoke_data
    model = build_3d_models(include_gpr=False)[model_name]
    tiny_kwargs = {
        "realmlp": {"n_epochs": 2, "batch_size": 16, "n_threads": 1, "random_state": 42, "device": "cpu", "verbosity": 0},
        "tabm": {"n_epochs": 2, "batch_size": 16, "tabm_k": 4, "d_block": 32, "n_blocks": 2, "compile_model": False, "n_threads": 1, "random_state": 42, "device": "cpu", "verbosity": 0},
        "resnet_mlp": {"max_epochs": 2, "batch_size": 16, "module_d": 32, "module_n_layers": 2, "n_threads": 1, "random_state": 42, "device": "cpu", "verbosity": 0},
        "ft_transformer": {"max_epochs": 2, "batch_size": 16, "module_d_token": 32, "module_n_layers": 2, "module_n_heads": 4, "n_threads": 1, "random_state": 42, "device": "cpu", "verbosity": 0},
    }
    if model_name == "original_kan":
        model.set_params(pykanregressor__hidden_width=8, pykanregressor__steps=2)
    else:
        model.set_params(model_kwargs=tiny_kwargs[model_name])

    model.fit(x, y)
    prediction = model.predict(x[:4])

    assert prediction.shape == (4,)
    assert np.all(np.isfinite(prediction))
    if model_name == "original_kan":
        restored_prediction = pickle.loads(pickle.dumps(model)).predict(x[:4])
        assert np.allclose(restored_prediction, prediction)


def test_phase_2b_defaults_are_cpu_seeded_and_configurable():
    models = build_3d_models(include_gpr=False)
    for name in ("realmlp", "tabm", "resnet_mlp", "ft_transformer"):
        assert models[name].model_kwargs["device"] == "cpu"
        assert models[name].model_kwargs["random_state"] == 42
        assert models[name].install_extra == "phase2b"
    kan = models["original_kan"].named_steps["pykanregressor"]
    assert kan.device == "cpu"
    assert kan.random_state == 42
    models["realmlp"].set_params(model_kwargs={"n_epochs": 2})
    models["original_kan"].set_params(pykanregressor__steps=2)


def test_phase_2b_optional_dependency_error_is_model_specific(
    monkeypatch, direct_3d_smoke_data
):
    def missing_dependency(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("project1.modeling.optional_estimators.import_module", missing_dependency)
    x, y = direct_3d_smoke_data

    with pytest.raises(ImportError, match=r"realmlp.*pip install -e .\[phase2b\]"):
        build_3d_models(include_gpr=False)["realmlp"].fit(x, y)


def test_tabiclv2_cached_cpu_fit_predict_and_serialization(direct_3d_smoke_data):
    pytest.importorskip("tabicl")
    hub = pytest.importorskip("huggingface_hub")
    checkpoint = hub.try_to_load_from_cache(
        "jingang/TabICL", "tabicl-regressor-v2-20260212.ckpt"
    )
    if not isinstance(checkpoint, str):
        pytest.skip("TabICLv2 regression checkpoint is not cached")
    x, y = direct_3d_smoke_data
    model = build_3d_models(include_gpr=False)["tabiclv2"]
    model.set_params(
        model_kwargs={
            "n_estimators": 1,
            "batch_size": 1,
            "device": "cpu",
            "use_amp": False,
            "use_fa3": False,
            "offload_mode": False,
            "allow_auto_download": False,
            "random_state": 42,
            "n_jobs": 1,
            "verbose": False,
        }
    )

    model.fit(x, y)
    prediction = model.predict(x[:4])
    restored = pickle.loads(pickle.dumps(model))

    assert prediction.shape == (4,)
    assert np.all(np.isfinite(prediction))
    assert np.allclose(restored.predict(x[:4]), prediction)


def test_tabiclv2_metadata_and_defaults_expose_weight_requirements():
    model = build_3d_models(include_gpr=False)["tabiclv2"]

    assert model.install_extra == "phase2c"
    assert model.model_kwargs["device"] == "cpu"
    assert model.model_kwargs["allow_auto_download"] is True
    assert model.model_kwargs["checkpoint_version"] == "tabicl-regressor-v2-20260212.ckpt"
    model.set_params(model_kwargs={"allow_auto_download": False})


def test_tabiclv2_optional_dependency_error_is_model_specific(
    monkeypatch, direct_3d_smoke_data
):
    def missing_dependency(name):
        raise ModuleNotFoundError(name)

    monkeypatch.setattr("project1.modeling.optional_estimators.import_module", missing_dependency)
    x, y = direct_3d_smoke_data

    with pytest.raises(ImportError, match=r"tabiclv2.*pip install -e .\[phase2c\]"):
        build_3d_models(include_gpr=False)["tabiclv2"].fit(x, y)


def test_ngboost_preserves_native_probability_interface(direct_3d_smoke_data):
    pytest.importorskip("ngboost")
    x, y = direct_3d_smoke_data
    model = build_3d_models(include_gpr=False)["ngboost"].fit(x, y)

    distribution = model.pred_dist(x[:4])

    assert np.asarray(distribution.loc).shape == (4,)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
