"""Run explicitly with .venv-pysr; it is intentionally outside pytest collection."""

from __future__ import annotations

import json
import pickle
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "workbase" / "src"))

from project1.modeling.pysr_estimator import ExternalPySRRegressor


def load_probe(model_path: Path, data_path: Path) -> None:
    with model_path.open("rb") as stream:
        model = pickle.load(stream)
    arrays = np.load(data_path, allow_pickle=False)
    prediction = model.predict(arrays["X"])
    max_error = float(np.max(np.abs(prediction - arrays["prediction"])))
    assert max_error < 1e-12
    print(json.dumps({"new_process_restore": True, "max_prediction_error": max_error}))


def smoke() -> None:
    x = np.array(
        [[n, w, xi] for n in (0.8, 1.0, 1.2) for w in (0.2, 0.4) for xi in (0.0, 0.5, 1.0)],
        dtype=float,
    )
    y = 0.4 * x[:, 0] - 0.03 * x[:, 1] + x[:, 2] ** 2
    model = ExternalPySRRegressor(
        python_executable=sys.executable,
        niterations=1,
        populations=1,
        population_size=10,
        ncycles_per_iteration=10,
        timeout_in_seconds=60,
        maxsize=8,
        maxdepth=5,
    )
    model.set_run_context(
        target="synthetic_y",
        split_mode="condition",
        fold=0,
        train_condition_ids=[f"n={row[0]}|w={row[1]}" for row in x],
        feature_order=["n", "w", "xi"],
    )
    model.fit(x, y)
    prediction = model.predict(x)
    assert np.isfinite(prediction).all()
    assert model.selected_equation_
    assert model.complexity_ >= 1
    assert model.equation_table_

    with tempfile.TemporaryDirectory(prefix="project1-pysr-smoke-") as temporary:
        root = Path(temporary)
        model_path = root / "model.pkl"
        data_path = root / "data.npz"
        with model_path.open("wb") as stream:
            pickle.dump(model, stream)
        np.savez_compressed(data_path, X=x, prediction=prediction)
        subprocess.run(
            [sys.executable, __file__, "load", str(model_path), str(data_path)],
            check=True,
        )

    print(
        json.dumps(
            {
                "isolated_smoke": "passed",
                "finite_predictions": True,
                "feature_order": model.feature_names_in_.tolist(),
                "selected_equation": model.selected_equation_,
                "selected_complexity": model.complexity_,
                "equation_count": len(model.equation_table_),
                "versions": model.symbolic_report_["versions"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "load":
        load_probe(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        smoke()
