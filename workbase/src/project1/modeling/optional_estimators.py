from __future__ import annotations

from importlib import import_module
from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.utils.validation import check_is_fitted


class LazyImportRegressor(RegressorMixin, BaseEstimator):
    """Sklearn adapter that imports one optional regressor only when fitted."""

    def __init__(
        self,
        module: str,
        class_name: str,
        model_name: str,
        model_kwargs: dict[str, Any] | None = None,
        install_extra: str = "phase2a",
    ) -> None:
        self.module = module
        self.class_name = class_name
        self.model_name = model_name
        self.model_kwargs = model_kwargs
        self.install_extra = install_extra

    def fit(self, x, y):
        try:
            estimator_class = getattr(import_module(self.module), self.class_name)
        except (ImportError, OSError) as exc:
            raise ImportError(
                f"Direct-3D model '{self.model_name}' requires optional dependency "
                f"'{self.module}'. Install it with `pip install -e .[{self.install_extra}]`."
            ) from exc
        self.model_ = estimator_class(**(self.model_kwargs or {}))
        self.model_.fit(x, y)
        return self

    def predict(self, x):
        check_is_fitted(self, "model_")
        return self.model_.predict(x)

    def pred_dist(self, x):
        """Expose NGBoost's native predictive distribution when available."""

        check_is_fitted(self, "model_")
        if not hasattr(self.model_, "pred_dist"):
            raise AttributeError(f"Direct-3D model '{self.model_name}' has no pred_dist interface")
        return self.model_.pred_dist(x)


class PyKANRegressor(RegressorMixin, BaseEstimator):
    """Minimal sklearn wrapper for the official pykan training API."""

    def __init__(
        self,
        hidden_width: int = 16,
        grid: int = 3,
        spline_order: int = 3,
        steps: int = 20,
        optimizer: str = "LBFGS",
        regularization: float = 0.0,
        random_state: int = 42,
        device: str = "cpu",
    ) -> None:
        self.hidden_width = hidden_width
        self.grid = grid
        self.spline_order = spline_order
        self.steps = steps
        self.optimizer = optimizer
        self.regularization = regularization
        self.random_state = random_state
        self.device = device

    def _new_model(self, kan_class):
        return kan_class(
            width=[self.n_features_in_, self.hidden_width, 1],
            grid=self.grid,
            k=self.spline_order,
            seed=self.random_state,
            auto_save=False,
            symbolic_enabled=False,
            save_act=False,
            device=self.device,
        )

    def fit(self, x, y):
        try:
            torch = import_module("torch")
            kan_class = getattr(import_module("kan"), "KAN")
        except (ImportError, OSError) as exc:
            raise ImportError(
                "Direct-3D model 'original_kan' requires optional dependencies "
                "'torch' and 'pykan'. Install them with `pip install -e .[phase2b]`."
            ) from exc

        x_tensor = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)
        y_tensor = torch.as_tensor(np.asarray(y).reshape(-1, 1), dtype=torch.float32, device=self.device)
        self.n_features_in_ = x_tensor.shape[1]
        self.model_ = self._new_model(kan_class)
        dataset = {
            "train_input": x_tensor,
            "train_label": y_tensor,
            "test_input": x_tensor,
            "test_label": y_tensor,
        }
        self.model_.fit(
            dataset,
            opt=self.optimizer,
            steps=self.steps,
            lamb=self.regularization,
            log=max(1, self.steps),
        )
        return self

    def predict(self, x):
        check_is_fitted(self, "model_")
        torch = import_module("torch")
        x_tensor = torch.as_tensor(np.asarray(x), dtype=torch.float32, device=self.device)
        with torch.no_grad():
            return self.model_(x_tensor).detach().cpu().numpy().reshape(-1)

    def __getstate__(self):
        state = self.__dict__.copy()
        model = state.pop("model_", None)
        if model is not None:
            state["_model_state_dict"] = {
                name: tensor.detach().cpu() for name, tensor in model.state_dict().items()
            }
        return state

    def __setstate__(self, state):
        model_state = state.pop("_model_state_dict", None)
        self.__dict__.update(state)
        if model_state is not None:
            kan_class = getattr(import_module("kan"), "KAN")
            self.model_ = self._new_model(kan_class)
            self.model_.load_state_dict(model_state)
