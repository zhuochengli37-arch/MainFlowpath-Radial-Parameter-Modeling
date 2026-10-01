"""Explicit minimal condition-level Direct-3D smoke for the TabPFN bridge."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "workbase" / "src"))

from project1.experiments import benchmark_runners
from project1.experiments.benchmark_data import FlexSample
from project1.modeling.tabpfn_estimator import ExternalTabPFNRegressor


def samples() -> list[FlexSample]:
    return [
        FlexSample(
            component="CMP",
            family="CMP",
            station="MAIN",
            database="SYNTHETIC",
            stage=1,
            rpm=0.7 + condition * 0.1,
            wcor=10.0 + condition,
            xi=xi,
            source_path=f"condition_{condition}.dat",
            outputs={"Cpt": 0.2 + condition * 0.1 + xi * 0.05},
        )
        for condition in range(4)
        for xi in (0.0, 0.5, 1.0)
    ]


def main() -> None:
    benchmark_samples = samples()
    original_cv_splits = benchmark_runners.cv_splits
    benchmark_runners.load_training_samples = lambda input_dir, radial_mode: benchmark_samples
    benchmark_runners._build_models = lambda include_gpr: {
        "tabpfn": ExternalTabPFNRegressor(python_executable=sys.executable)
    }

    def condition_only(items, mode, requested_splits):
        if mode != "condition":
            return []
        return list(original_cv_splits(items, mode, requested_splits))[:1]

    benchmark_runners.cv_splits = condition_only
    with tempfile.TemporaryDirectory(prefix="project1-tabpfn-benchmark-") as temporary:
        output = Path(temporary) / "output"
        result = benchmark_runners.run_benchmark(
            input_dir=str(Path(temporary) / "unused"),
            output_dir=str(output),
            radial_mode="full",
            include_gpr=False,
            max_samples=100,
            partition_mode="single",
            n_splits=2,
        )
        failures = json.loads(Path(result["failures_path"]).read_text(encoding="utf-8"))
        manifest = json.loads(Path(result["models_manifest_path"]).read_text(encoding="utf-8"))
        leaderboard = json.loads(Path(result["leaderboard_path"]).read_text(encoding="utf-8"))
        assert not failures
        assert leaderboard[0]["split_mode"] == "condition"
        assert manifest[0]["foundation_model"]["checkpoint_filename"].startswith("tabpfn-v3.5")
        model = joblib.load(output / manifest[0]["path"])
        prediction = model.predict(np.array([[0.9, 12.0, 0.5]]))
        assert prediction.shape == (1,) and np.isfinite(prediction).all()
        print(
            json.dumps(
                {
                    "benchmark_integration": "passed",
                    "condition_rows": len(leaderboard),
                    "failure_count": len(failures),
                    "checkpoint": manifest[0]["foundation_model"]["checkpoint_filename"],
                },
                ensure_ascii=False,
            )
        )


if __name__ == "__main__":
    main()
