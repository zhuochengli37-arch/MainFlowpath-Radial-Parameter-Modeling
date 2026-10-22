"""Recheck saved Phase B evidence against the real database, without training."""

from pathlib import Path
import json
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "workbase" / "src"))

from project1.experiments.benchmark_instance import formal_cases, formal_condition_id, inspect_formal_dataset, write_benchmark_json
from project1.experiments.benchmark_reporting import rebuild_formal_reports


def main():
    config = json.loads((ROOT / "config/hpt_formal_benchmark.json").read_text(encoding="utf-8"))
    frozen = json.loads((ROOT / config["instance_path"]).read_text(encoding="utf-8"))
    samples, current = inspect_formal_dataset(config["input_dir"])
    assert current["inventory_sha256"] == frozen["inventory_sha256"]
    assert current["split"] == frozen["split"]
    cases = formal_cases(samples)
    assert len(cases) == 40 and len(current["conditions"]) == 61
    assert all(c["subset_rows"] == {"train": 1480, "validation": 480, "test": 480} for c in current["cases"])
    negative = next(c for c in current["cases"] if c["case_id"] == "HPTB:S2:RO:MA")
    assert negative["negative_values"] == 66 and negative["negative_condition_count"] == 13
    base = ROOT / "data/output/benchmark/preflight/hpt_stage_b"
    checks = []
    for name, expected in (("ridge_coverage", 40), ("representative_smoke", 5)):
        output = base / name
        experiment = json.loads((output / "experiment.json").read_text(encoding="utf-8"))
        assert experiment["purpose"] == "stage_b_compatibility"
        before = {p.name: p.read_bytes() for p in (output / "results").glob("*.csv")}
        summary = rebuild_formal_reports(output)
        assert summary["passed_runs"] == summary["recorded_runs"] == expected and summary["failed_runs"] == 0
        assert before == {p.name: p.read_bytes() for p in (output / "results").glob("*.csv")}
        for path in sorted((output / "records").glob("*.json")):
            record = json.loads(path.read_text(encoding="utf-8"))
            model_path = output / record["metadata"]["model_path"]
            model = joblib.load(model_path)
            metadata = json.loads(model_path.with_suffix(".json").read_text(encoding="utf-8"))
            x, _, rows = cases[record["case_id"]]
            train = [i for i, row in enumerate(rows) if formal_condition_id(row) in frozen["split"]["conditions"]["train"]]
            assert metadata["model_context"]["sample_count"] == record["train_sample_count"] == len(train) == 1480
            assert metadata["model_context"]["flow_parameter_source_name"] == "PI"
            assert set(metadata["model_context"]["training_sources"]) == {rows[i].source_path for i in train}
            if hasattr(model, "named_steps") and "standardscaler" in model.named_steps:
                np.testing.assert_allclose(model.named_steps["standardscaler"].mean_, x[train].mean(axis=0))
            if record["model_key"] in {"gpr_matern", "ard_gpr"}:
                assert len(model.named_steps["gaussianprocessregressor"].X_train_) == 1480
            with np.load(output / record["prediction_path"]) as arrays:
                for subset in ("validation", "test"):
                    np.testing.assert_array_equal(model.predict(arrays[f"{subset}_x"]), arrays[f"{subset}_prediction"])
                    assert set(arrays[f"{subset}_condition_ids"]) == set(frozen["split"]["conditions"][subset])
                    assert len(arrays[f"{subset}_prediction"]) == 480
            checks.append({"run_id": record["run_id"], "status": "passed", "training_rows": 1480,
                           "fit_time": record["runtime"]["fit_time"], "restore_prediction_max_delta": 0.0})
    assert not (ROOT / config["output_dir"] / "experiment.json").exists()
    report = {"phase": "B", "status": "passed", "real_data_fits": 45, "cases": 40,
              "inventory_sha256": current["inventory_sha256"], "split_hash": current["split"]["sha256"],
              "condition_counts": current["split"]["condition_counts"], "database_unchanged": True,
              "negative_MA_rows_retained": 66, "negative_MA_conditions": 13,
              "record_and_table_recovery": "passed", "checks": checks,
              "formal_first_round_started": False, "selection_run": False, "sample_efficiency_run": False}
    write_benchmark_json(base / "stage_b_acceptance.json", report)
    print(json.dumps({key: value for key, value in report.items() if key != "checks"}, indent=2))


if __name__ == "__main__":
    main()
