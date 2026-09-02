from __future__ import annotations

import numpy as np
from scipy.interpolate import RBFInterpolator
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import PolynomialFeatures


MODEL_CATEGORY = {
    "linear_deg2": "工程回归 / 二次响应面",
    "ridge_deg2": "工程回归 / 二次响应面",
    "piecewise_ridge": "工程回归 / 分段岭回归",
    "spline_regression": "插值拟合 / 样条回归",
    "tensor_spline": "插值拟合 / 张量样条",
    "lowess_like": "插值拟合 / LOWESS",
    "knn_distance": "插值拟合 / 邻域距离",
    "rbf": "插值拟合 / RBF",
    "gpr_matern": "概率模型 / GPR-Kriging",
    "hierarchical_workline": "结构化模型 / 分层工作线",
    "endpoint_hub_tip": "结构化模型 / 端点-轮毂-叶尖",
    "physics_clamped_ridge": "物理约束 / 截断岭回归",
}


class RBFRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, smoothing: float = 0.01, kernel: str = "thin_plate_spline") -> None:
        self.smoothing = smoothing
        self.kernel = kernel
        self._rbf: RBFInterpolator | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "RBFRegressor":
        x_arr = np.asarray(x, dtype=float)
        y_arr = np.asarray(y, dtype=float).reshape(-1)
        self._rbf = RBFInterpolator(x_arr, y_arr, smoothing=self.smoothing, kernel=self.kernel)
        self.interpolator_ = self._rbf
        self.n_features_in_ = x_arr.shape[1]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self._rbf is None:
            raise ValueError("model not fitted")
        return self._rbf(np.asarray(x, dtype=float)).reshape(-1)


class PiecewiseRidgeRegressor:
    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.threshold_: float | None = None
        self.left_ = make_pipeline(PolynomialFeatures(2, include_bias=False), Ridge(alpha=alpha))
        self.right_ = make_pipeline(PolynomialFeatures(2, include_bias=False), Ridge(alpha=alpha))
        self.fallback_ = make_pipeline(PolynomialFeatures(2, include_bias=False), Ridge(alpha=alpha))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PiecewiseRidgeRegressor":
        self.threshold_ = float(np.median(x[:, 0]))
        mask = x[:, 0] <= self.threshold_
        self.fallback_.fit(x, y)
        if np.sum(mask) >= 5:
            self.left_.fit(x[mask], y[mask])
        if np.sum(~mask) >= 5:
            self.right_.fit(x[~mask], y[~mask])
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.threshold_ is None:
            raise ValueError("model not fitted")
        pred = np.zeros(x.shape[0], dtype=float)
        mask = x[:, 0] <= self.threshold_
        if np.sum(mask) > 0:
            try:
                pred[mask] = self.left_.predict(x[mask])
            except Exception:
                pred[mask] = self.fallback_.predict(x[mask])
        if np.sum(~mask) > 0:
            try:
                pred[~mask] = self.right_.predict(x[~mask])
            except Exception:
                pred[~mask] = self.fallback_.predict(x[~mask])
        return pred


class LOWESSLikeRegressor(BaseEstimator, RegressorMixin):
    def __init__(self, bandwidth: float = 0.25) -> None:
        self.bandwidth = bandwidth
        self.x_train: np.ndarray | None = None
        self.y_train: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LOWESSLikeRegressor":
        self.x_train = np.asarray(x, dtype=float)
        self.y_train = np.asarray(y, dtype=float).reshape(-1)
        self.x_train_ = self.x_train
        self.y_train_ = self.y_train
        self.n_features_in_ = self.x_train.shape[1]
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        if self.x_train is None or self.y_train is None:
            raise ValueError("model not fitted")
        xq = np.asarray(x, dtype=float)
        out = np.zeros(xq.shape[0], dtype=float)
        bw = max(self.bandwidth, 1e-6)
        for index in range(xq.shape[0]):
            distances = np.linalg.norm(self.x_train - xq[index], axis=1)
            scale = max(np.quantile(distances, 0.75), 1e-9)
            weights = np.exp(-((distances / (bw * scale + 1e-12)) ** 2))
            if np.sum(weights) <= 1e-12:
                out[index] = float(np.mean(self.y_train))
            else:
                out[index] = float(np.sum(weights * self.y_train) / np.sum(weights))
        return out


class HierarchicalWorklineRegressor:
    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.wcor_trend_ = make_pipeline(PolynomialFeatures(2, include_bias=False), Ridge(alpha=alpha))
        self.state_model_ = make_pipeline(PolynomialFeatures(2, include_bias=False), Ridge(alpha=alpha))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "HierarchicalWorklineRegressor":
        rpm = x[:, [0]]
        wcor = x[:, 1]
        xi = x[:, 2]
        self.wcor_trend_.fit(rpm, wcor)
        wcor_hat = self.wcor_trend_.predict(rpm)
        state_x = np.column_stack([rpm.reshape(-1), wcor_hat.reshape(-1), xi.reshape(-1)])
        self.state_model_.fit(state_x, y)
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        rpm = x[:, [0]]
        xi = x[:, 2]
        wcor_hat = self.wcor_trend_.predict(rpm)
        state_x = np.column_stack([rpm.reshape(-1), wcor_hat.reshape(-1), xi.reshape(-1)])
        return self.state_model_.predict(state_x)


class EndpointRegression:
    def __init__(self, alpha: float = 0.1) -> None:
        self.alpha = alpha
        self.hub_model_ = make_pipeline(PolynomialFeatures(2, include_bias=False), Ridge(alpha=alpha))
        self.tip_model_ = make_pipeline(PolynomialFeatures(2, include_bias=False), Ridge(alpha=alpha))
        self.global_model_ = make_pipeline(PolynomialFeatures(2, include_bias=False), Ridge(alpha=alpha))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "EndpointRegression":
        rpm = x[:, [0]]
        xi = x[:, 2]
        hub = xi <= 0.5
        self.global_model_.fit(rpm, y)
        if np.sum(hub) >= 3:
            self.hub_model_.fit(rpm[hub], y[hub])
        if np.sum(~hub) >= 3:
            self.tip_model_.fit(rpm[~hub], y[~hub])
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        rpm = x[:, [0]]
        xi = x[:, 2]
        hub = xi <= 0.5
        out = np.zeros(x.shape[0], dtype=float)
        if np.sum(hub) > 0:
            try:
                out[hub] = self.hub_model_.predict(rpm[hub])
            except Exception:
                out[hub] = self.global_model_.predict(rpm[hub])
        if np.sum(~hub) > 0:
            try:
                out[~hub] = self.tip_model_.predict(rpm[~hub])
            except Exception:
                out[~hub] = self.global_model_.predict(rpm[~hub])
        return out


class PhysicsClampedRegressor:
    def __init__(self, alpha: float = 0.1) -> None:
        self.model_ = make_pipeline(PolynomialFeatures(2, include_bias=False), Ridge(alpha=alpha))
        self.y_min_: float | None = None
        self.y_max_: float | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "PhysicsClampedRegressor":
        self.model_.fit(x, y)
        self.y_min_ = float(np.min(y))
        self.y_max_ = float(np.max(y))
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        pred = self.model_.predict(x)
        if self.y_min_ is None or self.y_max_ is None:
            return pred
        return np.clip(pred, self.y_min_, self.y_max_)
