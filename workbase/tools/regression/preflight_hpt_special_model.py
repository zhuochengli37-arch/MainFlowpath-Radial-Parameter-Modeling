"""One unchanged real-HPT fit at seed 42, followed by fresh-process restoration."""

from pathlib import Path
import argparse
import json
import subprocess
import sys

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "workbase" / "src"))

from project1.experiments.benchmark_instance import write_benchmark_json
from project1.experiments.benchmark_reporting import formal_metrics
from project1.experiments.benchmark_runners import run_formal_benchmark


def restore(output: Path) -> dict:
    paths = sorted((output / "records").glob("*.json"))
    assert len(paths) == 1
    record = json.loads(paths[0].read_text(encoding="utf-8"))
    assert record["status"] == "passed" and record["model_seed"] == 42
    assert record["train_sample_count"] == 1480
    frozen = json.loads((output / "config/frozen_instance.json").read_text(encoding="utf-8"))
    assert record["metadata"]["parameters"] == frozen["model_parameters"][record["model_key"]]
    model = joblib.load(output / record["metadata"]["model_path"])
    checks = {}
    with np.load(output / record["prediction_path"], allow_pickle=False) as arrays:
        for subset in ("validation", "test"):
            if hasattr(model, "set_prediction_context"):
                model.set_prediction_context(condition_ids=arrays[f"{subset}_condition_ids"].tolist(),
                                             evaluation_subset=subset, split_hash=record["split_hash"])
            prediction = np.asarray(model.predict(arrays[f"{subset}_x"])).reshape(-1)
            reference = arrays[f"{subset}_prediction"]
            np.testing.assert_allclose(prediction, reference, rtol=1e-6, atol=1e-8)
            assert len(prediction) == 480 and np.isfinite(prediction).all()
            recomputed = formal_metrics(arrays[f"{subset}_truth"], prediction, frozen["metric_protocol"])
            for name in frozen["metric_protocol"]["formal_metrics"]:
                actual, expected = recomputed[name], record["evaluations"][subset][name]
                if expected is None:
                    assert actual is None
                else:
                    np.testing.assert_allclose(actual, expected, rtol=1e-6, atol=1e-8)
            checks[subset] = {"rows": len(prediction), "max_abs_delta": float(np.max(np.abs(prediction - reference))),
                              "metrics": recomputed}
    report = {"status": "passed", "model_key": record["model_key"], "case_id": record["case_id"],
              "model_seed": 42, "training_rows": 1480, "new_process_restore": True,
              "factory_parameters_unchanged": True, "checks": checks,
              "runtime": record["runtime"], "instance_sha256": frozen["instance_sha256"],
              "split_hash": record["split_hash"], "restore_tolerance": {"rtol": 1e-6, "atol": 1e-8}}
    write_benchmark_json(output / "reports/restore_check.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--model", choices=("pysr", "tabpfn", "tabiclv2"))
    group.add_argument("--restore", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.restore:
        report = restore(args.restore.resolve())
        print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
        return
    output = (args.output or ROOT / "data/output/benchmark/preflight/hpt_stage_c" / args.model).resolve()
    print(f"Preflight {args.model}: HPTB:S2:RO:Cpt, seed=42, full frozen Train, output={output}", flush=True)
    summary = run_formal_benchmark(str(ROOT / "config/hpt_formal_benchmark.json"),
                                   model_keys=[args.model], case_keys=["HPTB:S2:RO:Cpt"],
                                   output_dir=str(output), compatibility_seed=42)
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    if summary["passed_runs"] != 1 or summary["failed_runs"]:
        raise SystemExit(1)
    subprocess.run([sys.executable, str(Path(__file__).resolve()), "--restore", str(output)], check=True)


if __name__ == "__main__":
    main()
