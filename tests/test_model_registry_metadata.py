from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WORKBASE_SRC = ROOT / "workbase" / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(WORKBASE_SRC) not in sys.path:
    sys.path.insert(0, str(WORKBASE_SRC))

from project1.modeling.factory import build_3d_models
from project1.modeling.registry import (
    MODEL_FAMILIES,
    MODEL_REGISTRY,
    list_model_metadata,
    validate_model_registry,
)


def test_registry_has_expected_candidate_counts() -> None:
    counts = Counter(model.candidate_status for model in MODEL_REGISTRY.values())

    assert len(MODEL_REGISTRY) == 46
    assert counts == {
        "existing": 19,
        "formal_candidate": 19,
        "secondary_candidate": 8,
    }


def test_registry_uses_the_approved_candidate_tiers() -> None:
    existing = {
        "linear_deg2", "ridge_deg2", "piecewise_ridge", "spline_regression",
        "tensor_spline", "lowess_like", "knn_distance", "rbf",
        "hierarchical_workline", "endpoint_hub_tip", "physics_clamped_ridge",
        "random_forest", "gradient_boosting", "xgboost", "lightgbm", "lasso",
        "elastic_net", "svr_rbf", "gpr_matern",
    }
    formal = {
        "gam", "kernel_ridge_rbf", "ard_gpr", "extra_trees", "catboost", "ebm",
        "ngboost", "mlp", "realmlp", "tabm", "tabr", "modern_nca", "xrfm",
        "tabpfn", "tabiclv2", "tabdpt", "original_kan", "pysr", "deeponet",
    }
    secondary = {
        "mars", "resnet_mlp", "ft_transformer", "grande", "dofen", "fastkan",
        "efficientkan", "sparse_variational_gp",
    }

    assert {
        model.name for model in MODEL_REGISTRY.values()
        if model.candidate_status == "existing"
    } == existing
    assert {
        model.name for model in MODEL_REGISTRY.values()
        if model.candidate_status == "formal_candidate"
    } == formal
    assert {
        model.name for model in MODEL_REGISTRY.values()
        if model.candidate_status == "secondary_candidate"
    } == secondary


def test_registry_uses_exactly_six_fixed_families() -> None:
    assert {model.family for model in MODEL_REGISTRY.values()} == set(MODEL_FAMILIES)
    assert len(MODEL_FAMILIES) == 6


def test_registry_metadata_is_complete() -> None:
    for key, model in MODEL_REGISTRY.items():
        assert key == model.name
        assert model.display_name
        assert model.subfamily
        assert model.capability_tags
        assert "regression" in model.capability_tags
        assert model.dependency_hints


def test_phase_2b_implementation_status_is_explicit() -> None:
    implemented_candidates = {
        "kernel_ridge_rbf", "extra_trees", "ard_gpr", "gam", "catboost", "ebm",
        "ngboost", "mlp", "realmlp", "tabm", "original_kan", "resnet_mlp",
        "ft_transformer", "tabiclv2", "pysr", "tabpfn",
    }
    future = [model for model in MODEL_REGISTRY.values() if model.candidate_status != "existing"]

    assert {
        model.name for model in future if model.implementation_status == "implemented"
    } == implemented_candidates
    assert sum(model.implementation_status == "not_implemented" for model in future) == 11
    assert MODEL_REGISTRY["kernel_ridge_rbf"].subfamily == "kernel_ridge_regression"
    assert MODEL_REGISTRY["extra_trees"].subfamily == "randomized_tree_ensemble"
    assert MODEL_REGISTRY["ard_gpr"].subfamily == "gaussian_process_ard"
    assert MODEL_REGISTRY["mars"].implementation_status == "not_implemented"
    assert MODEL_REGISTRY["sparse_variational_gp"].implementation_status == "not_implemented"
    for name in ("tabr", "modern_nca", "fastkan", "efficientkan"):
        assert MODEL_REGISTRY[name].implementation_status == "not_implemented"
    assert "requires_model_weights" in MODEL_REGISTRY["tabiclv2"].capability_tags
    assert "offline_after_cache" in MODEL_REGISTRY["tabiclv2"].capability_tags
    assert "requires_model_weights" in MODEL_REGISTRY["tabpfn"].capability_tags
    assert "offline_after_cache" in MODEL_REGISTRY["tabpfn"].capability_tags


def test_implemented_registry_matches_current_3d_factory() -> None:
    factory_names = build_3d_models(include_gpr=True)

    validate_model_registry(factory_names)
    assert {
        model.name for model in list_model_metadata(implementation_status="implemented")
    } == set(factory_names)
