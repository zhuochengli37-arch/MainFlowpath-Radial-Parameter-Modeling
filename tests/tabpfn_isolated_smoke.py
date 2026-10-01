"""Explicit offline smoke for the isolated official TabPFN bridge."""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "workbase" / "src"))

from project1.modeling.tabpfn_estimator import ExternalTabPFNRegressor


def main() -> None:
    checkpoint = ROOT / ".venv-tabpfn" / "model_cache" / (
        "tabpfn-v3.5-fast-20260909.safetensors"
    )
    x = np.array(
        [
            [n, w, xi]
            for n, w in ((0.7, 10.0), (0.85, 11.0), (1.0, 12.0))
            for xi in (0.0, 0.5, 1.0)
        ],
        dtype=float,
    )
    y = 0.4 * x[:, 0] - 0.03 * x[:, 1] + np.sin(np.pi * x[:, 2])
    x_test = np.array([[0.775, 10.5, 0.25], [0.925, 11.5, 0.75]])
    model = ExternalTabPFNRegressor(
        python_executable=sys.executable,
        checkpoint_path=str(checkpoint),
    )
    model.set_run_context(
        split_mode="condition",
        train_condition_ids=[f"n={row[0]}|w={row[1]}" for row in x],
        feature_order=["n", "w", "xi"],
    )
    model.fit(x, y)
    model.set_prediction_context(condition_ids=["test-a", "test-b"])
    prediction = model.predict(x_test)
    restored = pickle.loads(pickle.dumps(model))
    restored.set_prediction_context(condition_ids=["test-a", "test-b"])
    recovered_prediction = restored.predict(x_test)

    assert prediction.shape == (2,)
    assert np.isfinite(prediction).all()
    assert np.allclose(recovered_prediction, prediction, rtol=0.0, atol=1e-7)
    assert model.foundation_report_["device"] == "cpu"
    assert model.foundation_report_["offline_mode"] is True
    assert model.foundation_report_["checkpoint_sha256"] == (
        "14e0b81285bca7a6d1ef121169f27f651f76dff49cad6c0b8779d5c836018550"
    )
    print(
        json.dumps(
            {
                "offline_smoke": "passed",
                "prediction": prediction.tolist(),
                "recovery_max_abs_delta": float(
                    np.max(np.abs(recovered_prediction - prediction))
                ),
                "fit_context_time_seconds": model.training_time_,
                "load_time_seconds": restored.last_load_time_,
                "inference_time_seconds": restored.last_inference_time_,
                "fit_peak_rss_mib": model.foundation_report_["peak_rss_mib"],
                "prediction_peak_rss_mib": restored.last_peak_rss_mib_,
                "fit_state_size_bytes": model.foundation_report_["fit_state_size_bytes"],
                "versions": model.foundation_report_["versions"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
