"""Authoritative Direct-3D model candidate metadata."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


MODEL_FAMILIES = (
    "classical_regression",
    "kernel_and_local",
    "tree_and_boosting",
    "neural_tabular",
    "foundation_tabular",
    "specialized_structured",
)
CANDIDATE_STATUSES = ("existing", "formal_candidate", "secondary_candidate")
IMPLEMENTATION_STATUSES = ("implemented", "not_implemented")


@dataclass(frozen=True)
class ModelMetadata:
    name: str
    display_name: str
    family: str
    subfamily: str
    candidate_status: str
    implementation_status: str
    phase: int
    capability_tags: tuple[str, ...]
    dependency_hints: tuple[str, ...]


def _model(
    name: str,
    display_name: str,
    family: str,
    subfamily: str,
    candidate_status: str,
    phase: int,
    *,
    implementation_status: str | None = None,
    capabilities: tuple[str, ...] = ("pointwise_tabular",),
    dependencies: tuple[str, ...] = ("scikit-learn",),
) -> ModelMetadata:
    return ModelMetadata(
        name=name,
        display_name=display_name,
        family=family,
        subfamily=subfamily,
        candidate_status=candidate_status,
        implementation_status=(
            implementation_status
            or ("implemented" if candidate_status == "existing" else "not_implemented")
        ),
        phase=phase,
        capability_tags=("regression", *capabilities),
        dependency_hints=dependencies,
    )


_MODELS = (
    # Existing Direct-3D models (19)
    _model("linear_deg2", "Polynomial Linear Degree 2", "classical_regression", "polynomial", "existing", 0),
    _model("ridge_deg2", "Polynomial Ridge Degree 2", "classical_regression", "regularized_polynomial", "existing", 0),
    _model("piecewise_ridge", "Piecewise Ridge", "classical_regression", "piecewise_regression", "existing", 0),
    _model("spline_regression", "Spline Regression", "classical_regression", "spline", "existing", 0),
    _model("tensor_spline", "Tensor Spline", "classical_regression", "tensor_spline", "existing", 0),
    _model("lowess_like", "LOWESS-like Regression", "kernel_and_local", "local_smoothing", "existing", 0, capabilities=("pointwise_tabular", "local_method"), dependencies=("numpy",)),
    _model("knn_distance", "Distance-weighted KNN", "kernel_and_local", "nearest_neighbor", "existing", 0, capabilities=("pointwise_tabular", "local_method")),
    _model("rbf", "RBF Interpolation", "kernel_and_local", "radial_basis", "existing", 0, capabilities=("pointwise_tabular", "local_method"), dependencies=("scipy",)),
    _model("hierarchical_workline", "Hierarchical Workline", "specialized_structured", "workline_structure", "existing", 0, capabilities=("pointwise_tabular", "structured_radial")),
    _model("endpoint_hub_tip", "Hub/Tip Endpoint Regression", "specialized_structured", "radial_endpoint", "existing", 0, capabilities=("pointwise_tabular", "structured_radial")),
    _model("physics_clamped_ridge", "Physics-clamped Ridge", "specialized_structured", "physical_constraint", "existing", 0, capabilities=("pointwise_tabular", "bounded_output")),
    _model("random_forest", "Random Forest", "tree_and_boosting", "bagged_tree", "existing", 0),
    _model("gradient_boosting", "Gradient Boosting", "tree_and_boosting", "gradient_boosting", "existing", 0),
    _model("xgboost", "XGBoost", "tree_and_boosting", "gradient_boosting", "existing", 0, dependencies=("xgboost",)),
    _model("lightgbm", "LightGBM", "tree_and_boosting", "gradient_boosting", "existing", 0, dependencies=("lightgbm",)),
    _model("lasso", "Lasso", "classical_regression", "linear_regularization", "existing", 0),
    _model("elastic_net", "Elastic Net", "classical_regression", "linear_regularization", "existing", 0),
    _model("svr_rbf", "SVR-RBF", "kernel_and_local", "support_vector", "existing", 0),
    _model("gpr_matern", "Gaussian Process Matérn", "kernel_and_local", "gaussian_process", "existing", 0, capabilities=("pointwise_tabular", "probabilistic")),
    # Future Direct-3D candidates (27); research tier is explicit per entry.
    _model("gam", "GAM", "classical_regression", "generalized_additive", "formal_candidate", 2, implementation_status="implemented", dependencies=("pygam",)),
    _model("kernel_ridge_rbf", "Kernel Ridge-RBF", "kernel_and_local", "kernel_ridge_regression", "formal_candidate", 2, implementation_status="implemented"),
    _model("ard_gpr", "ARD-GPR", "kernel_and_local", "gaussian_process_ard", "formal_candidate", 2, implementation_status="implemented", capabilities=("pointwise_tabular", "probabilistic")),
    _model("extra_trees", "Extra Trees", "tree_and_boosting", "randomized_tree_ensemble", "formal_candidate", 2, implementation_status="implemented"),
    _model("catboost", "CatBoost", "tree_and_boosting", "gradient_boosting", "formal_candidate", 2, implementation_status="implemented", dependencies=("catboost",)),
    _model("ebm", "Explainable Boosting Machine", "tree_and_boosting", "interpretable_boosting", "formal_candidate", 2, implementation_status="implemented", capabilities=("pointwise_tabular", "interpretable"), dependencies=("interpret-core",)),
    _model("mlp", "MLP", "neural_tabular", "multilayer_perceptron", "formal_candidate", 2, implementation_status="implemented"),
    _model("mars", "MARS", "classical_regression", "adaptive_spline", "secondary_candidate", 2, dependencies=("sklearn-contrib-py-earth",)),
    _model("sparse_variational_gp", "Sparse/Variational GP", "kernel_and_local", "sparse_gaussian_process", "secondary_candidate", 2, capabilities=("pointwise_tabular", "probabilistic", "gpu_optional"), dependencies=("torch", "gpytorch")),
    _model("realmlp", "RealMLP", "neural_tabular", "modern_mlp", "formal_candidate", 3, implementation_status="implemented", capabilities=("pointwise_tabular", "gpu_optional"), dependencies=("pytabkit", "torch")),
    _model("tabm", "TabM", "neural_tabular", "ensemble_mlp", "formal_candidate", 3, implementation_status="implemented", capabilities=("pointwise_tabular", "gpu_optional"), dependencies=("pytabkit", "torch")),
    _model("tabr", "TabR", "neural_tabular", "retrieval_network", "formal_candidate", 3, capabilities=("pointwise_tabular", "local_method", "gpu_optional"), dependencies=("faiss", "torch")),
    _model("modern_nca", "ModernNCA", "neural_tabular", "neighborhood_component", "formal_candidate", 3, capabilities=("pointwise_tabular", "local_method", "gpu_optional"), dependencies=("talent", "torch")),
    _model("original_kan", "Original KAN", "neural_tabular", "kolmogorov_arnold", "formal_candidate", 3, implementation_status="implemented", capabilities=("pointwise_tabular", "gpu_optional"), dependencies=("pykan", "torch")),
    _model("ngboost", "NGBoost", "tree_and_boosting", "probabilistic_boosting", "formal_candidate", 2, implementation_status="implemented", capabilities=("pointwise_tabular", "probabilistic"), dependencies=("ngboost",)),
    _model("grande", "GRANDE", "tree_and_boosting", "differentiable_tree", "secondary_candidate", 4, capabilities=("pointwise_tabular", "gpu_optional"), dependencies=("official_implementation_tbd",)),
    _model("dofen", "DOFEN", "tree_and_boosting", "forest_embedding", "secondary_candidate", 4, capabilities=("pointwise_tabular", "gpu_optional"), dependencies=("official_implementation_tbd",)),
    _model("tabpfn", "TabPFN", "foundation_tabular", "prior_data_fitted_network", "formal_candidate", 5, implementation_status="implemented", capabilities=("pointwise_tabular", "pretrained", "pretrained_weights", "requires_model_weights", "network_first_use", "offline_after_cache", "gpu_recommended", "local_inference"), dependencies=("tabpfn", "torch", "huggingface-hub")),
    _model("tabiclv2", "TabICLv2", "foundation_tabular", "in_context_learning", "formal_candidate", 5, implementation_status="implemented", capabilities=("pointwise_tabular", "pretrained", "pretrained_weights", "requires_model_weights", "network_first_use", "offline_after_cache", "gpu_recommended"), dependencies=("tabicl", "torch", "huggingface-hub")),
    _model("tabdpt", "TabDPT", "foundation_tabular", "pretrained_transformer", "formal_candidate", 5, capabilities=("pointwise_tabular", "pretrained_weights", "gpu_recommended"), dependencies=("official_implementation_tbd",)),
    _model("xrfm", "xRFM", "specialized_structured", "kernel_feature_tree_hybrid", "formal_candidate", 6, dependencies=("official_implementation_tbd",)),
    _model("pysr", "PySR", "specialized_structured", "symbolic_regression", "formal_candidate", 6, implementation_status="implemented", capabilities=("pointwise_tabular", "interpretable", "symbolic_expression", "external_runtime", "offline_after_install"), dependencies=("pysr", "julia")),
    _model("deeponet", "DeepONet", "specialized_structured", "operator_learning", "formal_candidate", 6, capabilities=("operator_learning", "structured_radial", "gpu_optional"), dependencies=("official_implementation_tbd",)),
    # Additional Phase 3 neural candidates.
    _model("resnet_mlp", "ResNet-MLP", "neural_tabular", "residual_mlp", "secondary_candidate", 3, implementation_status="implemented", capabilities=("pointwise_tabular", "gpu_optional"), dependencies=("pytabkit", "skorch", "torch")),
    _model("ft_transformer", "FT-Transformer", "neural_tabular", "transformer", "secondary_candidate", 3, implementation_status="implemented", capabilities=("pointwise_tabular", "gpu_optional"), dependencies=("pytabkit", "skorch", "torch")),
    _model("fastkan", "FastKAN", "neural_tabular", "kolmogorov_arnold", "secondary_candidate", 3, capabilities=("pointwise_tabular", "gpu_optional"), dependencies=("fast-kan source", "torch")),
    _model("efficientkan", "EfficientKAN", "neural_tabular", "kolmogorov_arnold", "secondary_candidate", 3, capabilities=("pointwise_tabular", "gpu_optional"), dependencies=("efficient-kan source", "torch")),
)

MODEL_REGISTRY = {model.name: model for model in _MODELS}


def get_model_metadata(name: str) -> ModelMetadata:
    return MODEL_REGISTRY[name]


def list_model_metadata(
    *,
    candidate_status: str | None = None,
    implementation_status: str | None = None,
    family: str | None = None,
) -> tuple[ModelMetadata, ...]:
    return tuple(
        model
        for model in _MODELS
        if (candidate_status is None or model.candidate_status == candidate_status)
        and (implementation_status is None or model.implementation_status == implementation_status)
        and (family is None or model.family == family)
    )


def model_category_map(*, implemented_only: bool = False) -> dict[str, str]:
    models: Iterable[ModelMetadata] = _MODELS
    if implemented_only:
        models = (model for model in models if model.implementation_status == "implemented")
    return {model.name: f"{model.family} / {model.subfamily}" for model in models}


def validate_model_registry(implemented_names: Iterable[str] | None = None) -> None:
    errors: list[str] = []
    if len(MODEL_REGISTRY) != len(_MODELS):
        errors.append("duplicate model names")
    for model in _MODELS:
        if model.family not in MODEL_FAMILIES:
            errors.append(f"{model.name}: invalid family {model.family!r}")
        if model.candidate_status not in CANDIDATE_STATUSES:
            errors.append(f"{model.name}: invalid candidate_status")
        if model.implementation_status not in IMPLEMENTATION_STATUSES:
            errors.append(f"{model.name}: invalid implementation_status")
        if not model.subfamily or not model.capability_tags:
            errors.append(f"{model.name}: incomplete metadata")
    if implemented_names is not None:
        declared = {
            model.name for model in _MODELS if model.implementation_status == "implemented"
        }
        actual = set(implemented_names)
        if declared != actual:
            errors.append(
                f"implemented model mismatch: missing={sorted(actual - declared)}, "
                f"stale={sorted(declared - actual)}"
            )
    if errors:
        raise ValueError("; ".join(errors))


validate_model_registry()
