from __future__ import annotations

import csv
import json
from pathlib import Path
import sys
import os
import subprocess

import joblib
import numpy as np
import pytest
from sklearn.base import BaseEstimator, RegressorMixin

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "workbase" / "src"))

from project1.experiments.benchmark_data import FlexSample, load_training_samples
from project1.experiments.benchmark_instance import formal_cases, formal_condition_id, inspect_formal_dataset
from project1.experiments.benchmark_instance import formal_experiment_lock, load_formal_records
from project1.experiments.benchmark_reporting import formal_metrics, rebuild_formal_reports
from project1.experiments.benchmark_runners import run_benchmark, run_benchmark_2d, run_formal_benchmark, seed_formal_model
from project1.modeling.factory import build_3d_models
from project1.services.meta_parser import parse_metadata_from_path
from project1.experiments.prediction_samples import load_prediction_batch


class RecordingRegressor(RegressorMixin, BaseEstimator):
    fits = []

    def __init__(self, name="spy", random_state=42, fail=False):
        self.name = name
        self.random_state = random_state
        self.fail = fail

    def fit(self, x, y):
        type(self).fits.append((self.name, len(x), x.copy(), y.copy(), self.random_state))
        if self.fail:
            raise RuntimeError("intentional fit failure")
        self.mean_ = float(np.mean(y))
        return self

    def predict(self, x):
        return np.full(len(x), self.mean_)


def make_database(tmp_path, conditions=6, points=3, stages=(1, 2), sections=("SI", "SO", "RI", "RO"), targets=("Cpt", "Cps", "Ctt", "Cts", "Ma")):
    root = tmp_path / "database"
    for stage in stages:
        for condition in range(conditions):
            n, pi = 0.5 + (condition % 3) * 0.1, 1.2 + condition * 0.13
            path = root / "DATABASE_HPTB" / f"STAGE_{stage}" / f"CNC_{n}" / f"PI_{pi}.dat"
            path.parent.mkdir(parents=True, exist_ok=True)
            columns = [f"{target}_{section}" for section in sections for target in ("xi", *targets)]
            rows = []
            for xi in np.linspace(0.01, 0.99, points):
                values = [value for index, section in enumerate(sections)
                          for value in [xi, *(n + pi * 0.1 + xi * 0.2 + stage * 0.01 + index * 0.02 - (0.9 if t == "Ma" else 0) for t in targets)]]
                rows.append(" ".join(str(v) for v in values))
            path.write_text(" ".join(columns) + "\n" + "\n".join(rows) + "\n", encoding="utf-8")
    configuration = {"dataset": "fixture_PI", "input_dir": str(root),
                     "instance_path": str(tmp_path / "instance.json"), "output_dir": str(tmp_path / "official"),
                     "feature_order": ["n", "PI", "xi"], "second_parameter_source_name": "PI",
                     "second_parameter_definition": "original pressure ratio, no conversion",
                     "expected_inventory": {"conditions": conditions, "main_files": conditions * len(stages),
                                            "cases": len(stages) * len(sections) * len(targets), "profile_points": [points]}}
    config = tmp_path / "config.json"
    config.write_text(json.dumps(configuration), encoding="utf-8")
    return root, config


@pytest.mark.parametrize("prefix,value", [("PI", 6.59374000638067), ("pi", 1.23e-4), ("W25COR", 27.998), ("PHI", 0.42)])
def test_pi_and_original_flow_metadata(prefix, value):
    metadata = parse_metadata_from_path(f"DATABASE_HPTB/STAGE_1/CNC_0.8/{prefix}_{value}.dat")
    assert metadata["flow_parameter"] == value
    assert metadata["wcor"] == value
    assert metadata["flow_parameter_name"].upper() == prefix.upper()
    assert metadata["speed_parameter"] == 0.8


def test_pi_enters_second_dimension_and_all_40_cases_share_split(tmp_path):
    root, _ = make_database(tmp_path)
    samples, instance = inspect_formal_dataset(str(root))
    assert len(formal_cases(samples)) == 40
    assert instance["split"]["condition_counts"] == {"train": 4, "validation": 1, "test": 1}
    sets = [set(ids) for ids in instance["split"]["conditions"].values()]
    assert all(a.isdisjoint(b) for i, a in enumerate(sets) for b in sets[i + 1:])
    for x, _, subset in formal_cases(samples).values():
        np.testing.assert_array_equal(x[:, 1], [float(Path(s.source_path).stem[3:]) for s in subset])
        assert {formal_condition_id(s) for s in subset} == set.union(*sets)
    assert all(case["subset_rows"] == {"train": 12, "validation": 3, "test": 3} for case in instance["cases"])
    batch = load_prediction_batch(root, require_flow=True)
    assert len(batch.samples) == len(samples)
    assert all(s.flow_parameter == float(Path(s.source_file).stem[3:]) for s in batch.samples)
    ma_case = next(c for c in instance["cases"] if c["case_id"] == "HPTB:S2:RO:MA")
    assert ma_case["negative_values"] > 0


def test_formal_five_metrics_and_protocol_tolerance():
    protocol = json.loads((ROOT / "data/output/benchmark/config/metric_definitions.json").read_text(encoding="utf-8"))
    result = formal_metrics(np.array([1., 3.]), np.array([2., 2.]), protocol)
    assert result["RMSE"] == result["MAE"] == 1
    assert result["MAPE"] == pytest.approx(2 / 3)
    assert result["R2"] == 0
    assert result["NRMSE_range"] == 0.5
    constant = formal_metrics(np.array([1., 1.]), np.array([0., 1.]), protocol)
    assert constant["NRMSE_range"] is None
    assert constant["NRMSE_range_reason"] == "constant_or_near_constant_target"
    negative = formal_metrics(np.array([-1., 0., 1.]), np.zeros(3), protocol)
    assert negative["MAPE"] == pytest.approx(2 / 3)
    with pytest.raises(ValueError, match="finite"):
        formal_metrics(np.ones(2), np.array([np.nan, 1]), protocol)


def test_every_stochastic_seed_reaches_estimator_configuration_without_mutating_factory():
    models = build_3d_models(True)
    protocol = json.loads((ROOT / "data/output/benchmark/config/stochastic_repeat_protocol.json").read_text(encoding="utf-8"))
    for key, classification in protocol["model_classification"].items():
        if classification["stochasticity_type"] == "deterministic":
            continue
        original = repr(models[key].get_params(deep=True))
        seeded = seed_formal_model(models[key], 3407, 1480)
        parameters = seeded.get_params(deep=True)
        actual = [value for name, value in parameters.items() if name.rsplit("__", 1)[-1] in {"random_state", "random_seed"}]
        actual += [value for name, kwargs in parameters.items() if name.rsplit("__", 1)[-1] == "model_kwargs" and isinstance(kwargs, dict)
                   for field, value in kwargs.items() if field in {"random_state", "random_seed"}]
        assert actual and all(value == 3407 for value in actual), key
        assert repr(models[key].get_params(deep=True)) == original


def test_formal_train_only_save_restore_and_report_recovery(tmp_path):
    root, config = make_database(tmp_path)
    prepared = run_formal_benchmark(str(config), prepare_only=True)
    assert prepared["planned_fits"] == 2600
    output = tmp_path / "compatibility"
    summary = run_formal_benchmark(str(config), model_keys=["ridge_deg2", "random_forest"],
                                   case_keys=["HPTB:S2:RO:MA"], output_dir=str(output))
    assert summary["planned_fits"] == summary["passed_runs"] == 4
    assert summary["purpose"] == "stage_b_compatibility"
    records = [json.loads(p.read_text(encoding="utf-8")) for p in (output / "records").glob("*.json")]
    assert sorted(r["model_seed"] for r in records if r["model_key"] == "random_forest") == [42, 2026, 3407]
    split = json.loads((tmp_path / "instance.json").read_text(encoding="utf-8"))["split"]["conditions"]
    all_samples = load_training_samples(str(root), "full")
    x, y, samples = formal_cases(all_samples)["HPTB:S2:RO:MA"]
    train = [i for i, s in enumerate(samples) if formal_condition_id(s) in split["train"]]
    for record in records:
        assert record["train_sample_count"] == 12
        assert set(record["train_condition_ids"]) == set(split["train"])
        saved = joblib.load(output / record["metadata"]["model_path"])
        if record["model_key"] == "ridge_deg2":
            np.testing.assert_allclose(saved.named_steps["standardscaler"].mean_, x[train].mean(axis=0))
        with np.load(output / record["prediction_path"]) as predictions:
            np.testing.assert_allclose(saved.predict(predictions["test_x"]), predictions["test_prediction"])
            assert set(predictions["test_condition_ids"]) == set(split["test"])
        metadata = json.loads((output / record["metadata"]["metadata_path"]).read_text(encoding="utf-8"))
        assert metadata["model_context"]["flow_parameter_source_name"] == "PI"
        assert set(metadata["model_context"]["training_sources"]) == {s.source_path for i, s in enumerate(samples) if i in train}
    aggregate_path = output / "results/model_accuracy_aggregate.csv"
    original_table = aggregate_path.read_bytes()
    aggregate_path.unlink()
    assert rebuild_formal_reports(output) == summary
    assert aggregate_path.read_bytes() == original_table
    with aggregate_path.open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    for row in rows:
        values = [r["evaluations"][row["evaluation_subset"]]["RMSE"] for r in records if r["model_key"] == row["model_key"]]
        assert float(row["RMSE_mean"]) == pytest.approx(np.mean(values))
        if len(values) > 1:
            assert float(row["RMSE_sample_std"]) == pytest.approx(np.std(values, ddof=1))
    with pytest.raises(FileExistsError):
        run_formal_benchmark(str(config), model_keys=["ridge_deg2"], output_dir=str(output))


def test_frozen_instance_rejects_changed_data(tmp_path):
    root, config = make_database(tmp_path)
    run_formal_benchmark(str(config), prepare_only=True)
    source = next(root.rglob("*.dat"))
    source.write_text(source.read_text(encoding="utf-8") + "\n")
    with pytest.raises(ValueError, match="frozen instance differs"):
        run_formal_benchmark(str(config), prepare_only=True)


def test_single_seed_preflight_preserves_frozen_three_seed_protocol(tmp_path):
    _, config = make_database(tmp_path)
    run_formal_benchmark(str(config), prepare_only=True)
    instance_path = tmp_path / "instance.json"
    before = instance_path.read_bytes()
    output = tmp_path / "single_seed"
    summary = run_formal_benchmark(str(config), model_keys=["random_forest"], case_keys=["HPTB:S2:RO:MA"],
                                   output_dir=str(output), compatibility_seed=42)
    assert summary["passed_runs"] == summary["planned_fits"] == 1
    assert summary["purpose"] == "preflight_single_seed"
    assert instance_path.read_bytes() == before
    assert json.loads(instance_path.read_text(encoding="utf-8"))["model_seeds"]["random_forest"] == [42, 2026, 3407]


def test_single_seed_is_not_an_official_schedule_override(tmp_path):
    _, config = make_database(tmp_path)
    run_formal_benchmark(str(config), prepare_only=True)
    with pytest.raises(ValueError, match="explicit compatibility"):
        run_formal_benchmark(str(config), compatibility_seed=42)
    assert not (tmp_path / "official/experiment.json").exists()


def test_failed_seed_is_retained_and_not_aggregated_as_success(tmp_path, monkeypatch):
    _, config = make_database(tmp_path)
    models = build_3d_models(True)
    models["random_forest"] = RecordingRegressor("random_forest", fail=True)
    monkeypatch.setattr("project1.experiments.benchmark_runners._build_models", lambda include_gpr: models)
    run_formal_benchmark(str(config), prepare_only=True)
    output = tmp_path / "failed"
    result = run_formal_benchmark(str(config), model_keys=["random_forest"], case_keys=["HPTB:S1:SI:Cpt"], output_dir=str(output))
    assert result["failed_runs"] == result["recorded_runs"] == 3
    with (output / "results/model_accuracy_aggregate.csv").open(encoding="utf-8-sig") as stream:
        rows = list(csv.DictReader(stream))
    assert all(row["RMSE_mean"] == "" and row["status"] == "incomplete" for row in rows)


def test_formal_gprs_receive_all_1480_rows(tmp_path, monkeypatch):
    _, config = make_database(tmp_path, conditions=61, points=40, stages=(2,), sections=("RO",), targets=("Ma",))
    models = build_3d_models(True)
    for key in ("gpr_matern", "ard_gpr"):
        models[key] = RecordingRegressor(key)
    monkeypatch.setattr("project1.experiments.benchmark_runners._build_models", lambda include_gpr: models)
    RecordingRegressor.fits = []
    run_formal_benchmark(str(config), prepare_only=True)
    summary = run_formal_benchmark(str(config), model_keys=["gpr_matern", "ard_gpr"], output_dir=str(tmp_path / "gprs"))
    assert summary["passed_runs"] == 2
    assert [(key, size) for key, size, *_ in RecordingRegressor.fits] == [("gpr_matern", 1480), ("ard_gpr", 1480)]
    frozen = json.loads((tmp_path / "instance.json").read_text(encoding="utf-8"))
    for _, _, x, _, _ in RecordingRegressor.fits:
        assert {f"{row[0]:.12g}|{row[1]:.12g}" for row in x} == set(frozen["split"]["conditions"]["train"])


@pytest.mark.parametrize("runner,keys", [(run_benchmark, ["gpr_matern", "ard_gpr"]), (run_benchmark_2d, ["gpr_matern"])])
def test_cv_gprs_keep_every_training_fold_row(tmp_path, monkeypatch, runner, keys):
    samples = [FlexSample("CMP", "compressor", "MAIN", "DATABASE_CMP", 1, 0.5 + i // 40 * 0.01,
                          0.1 + i // 40 * 0.001, (i % 40) / 39, str(tmp_path / f"{i//40}.dat"), {"Cpt": 1 + i * 0.001})
               for i in range(1200)]
    monkeypatch.setattr("project1.experiments.benchmark_runners.load_training_samples", lambda *args: samples)
    monkeypatch.setattr("project1.experiments.benchmark_runners._build_models", lambda include_gpr: {key: RecordingRegressor(key) for key in keys})
    monkeypatch.setattr("project1.experiments.benchmark_runners.cv_splits", lambda *args: [(np.arange(900), np.arange(900, 1200))])
    RecordingRegressor.fits = []
    runner(str(tmp_path), str(tmp_path / "cv"), "full", True, 10000, "multi", 3)
    for key in keys:
        sizes = [size for name, size, *_ in RecordingRegressor.fits if name == key]
        assert sizes == ([900] * (4 if runner is run_benchmark else 2)) + [1200]


def test_formal_cli_forwards_explicit_options(monkeypatch):
    import project1.cli as cli
    calls = []
    monkeypatch.setattr(cli, "run_workflow", lambda name, argv, **kwargs: calls.append((name, argv)) or 0)
    assert cli.main(["current", "benchmark-3d", "--formal-config", "config/hpt_formal_benchmark.json", "--prepare-only"]) == 0
    assert calls == [("current_benchmark_3d", ["--formal-config", "config/hpt_formal_benchmark.json", "--prepare-only"])]


@pytest.fixture
def resumable_fixture(tmp_path, monkeypatch):
    root, config = make_database(tmp_path)
    models = build_3d_models(True)
    models["random_forest"] = RecordingRegressor("random_forest")
    monkeypatch.setattr("project1.experiments.benchmark_runners._build_models", lambda include_gpr: models)
    RecordingRegressor.fits = []
    run_formal_benchmark(str(config), prepare_only=True)
    return root, config, tmp_path / "resumable", models


def start_fixture(config, output):
    return run_formal_benchmark(str(config), model_keys=["random_forest"],
                                case_keys=["HPTB:S2:RO:MA"], output_dir=str(output))


def read_records(output):
    experiment = json.loads((output / "experiment.json").read_text(encoding="utf-8"))
    return load_formal_records(output, experiment)


def test_partial_interrupt_explicit_resume_no_refit_or_double_count(resumable_fixture, monkeypatch):
    _, config, output, _ = resumable_fixture
    original_fit = RecordingRegressor.fit

    def interrupt_second(self, x, y):
        if self.random_state == 2026:
            raise KeyboardInterrupt("simulated interruption")
        return original_fit(self, x, y)

    monkeypatch.setattr(RecordingRegressor, "fit", interrupt_second)
    with pytest.raises(KeyboardInterrupt):
        start_fixture(config, output)
    records = read_records(output)
    passed_id = next(k for k, v in records.items() if v["status"] == "passed")
    interrupted_id = next(k for k, v in records.items() if v["status"] == "interrupted")
    saved = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in
             [output / "records" / f"{passed_id}.json", output / records[passed_id]["metadata"]["model_path"]]}
    summary = json.loads((output / "reports/benchmark_summary.json").read_text(encoding="utf-8"))
    assert summary["passed_runs"] == summary["incomplete_runs"] == summary["missing_runs"] == 1
    assert not summary["complete"]
    with (output / "results/model_accuracy_aggregate.csv").open(encoding="utf-8-sig") as stream:
        assert all(row["RMSE_mean"] == "" for row in csv.DictReader(stream))
    with pytest.raises(RuntimeError, match="explicit --retry-runs"):
        run_formal_benchmark(str(config), output_dir=str(output), resume=True)
    assert len(RecordingRegressor.fits) == 1
    monkeypatch.setattr(RecordingRegressor, "fit", original_fit)
    summary = run_formal_benchmark(str(config), output_dir=str(output), resume=True, retry_runs=[interrupted_id])
    assert summary["complete"] and summary["recorded_runs"] == summary["passed_runs"] == 3
    assert [fit[-1] for fit in RecordingRegressor.fits] == [42, 2026, 3407]
    for path, (content, timestamp) in saved.items():
        assert path.read_bytes() == content and path.stat().st_mtime_ns == timestamp
    history = list((output / "history" / interrupted_id).glob("*/records/*.json"))
    assert len(history) == 1 and json.loads(history[0].read_text())["status"] == "interrupted"
    assert read_records(output)[interrupted_id]["attempt"] == 2
    assert run_formal_benchmark(str(config), output_dir=str(output), resume=True) == summary
    assert len(RecordingRegressor.fits) == 3
    assert rebuild_formal_reports(output) == summary
    with (output / "results/model_accuracy.csv").open(encoding="utf-8-sig") as stream:
        assert len(list(csv.DictReader(stream))) == 6
    records = list(read_records(output).values())
    with (output / "results/model_accuracy_aggregate.csv").open(encoding="utf-8-sig") as stream:
        for row in csv.DictReader(stream):
            values = [r["evaluations"][row["evaluation_subset"]]["RMSE"] for r in records]
            assert float(row["RMSE_mean"]) == pytest.approx(np.mean(values))
            assert float(row["RMSE_sample_std"]) == pytest.approx(np.std(values, ddof=1))


@pytest.mark.parametrize("artifact", ["model_path", "metadata_path", "prediction_path", "record"])
def test_corrupt_artifacts_require_explicit_retry_preserve_history(resumable_fixture, artifact):
    _, config, output, _ = resumable_fixture
    start_fixture(config, output)
    run_id, record = next(iter(read_records(output).items()))
    if artifact == "record":
        path = output / "records" / f"{run_id}.json"
        record["evaluations"]["test"]["RMSE"] += 1
        path.write_text(json.dumps(record), encoding="utf-8")
    else:
        path = output / (record["prediction_path"] if artifact == "prediction_path" else record["metadata"][artifact])
        path.write_bytes(b"corrupted artifact")
    corrupted_bytes = path.read_bytes()
    with pytest.raises(RuntimeError, match="explicit --retry-runs"):
        run_formal_benchmark(str(config), output_dir=str(output), resume=True)
    assert len(RecordingRegressor.fits) == 3
    summary = rebuild_formal_reports(output)
    assert summary["passed_runs"] == 2 and summary["incomplete_runs"] == 1 and not summary["complete"]
    result = run_formal_benchmark(str(config), output_dir=str(output), resume=True, retry_runs=[run_id])
    assert result["complete"] and result["passed_runs"] == 3 and len(RecordingRegressor.fits) == 4
    copies = list((output / "history" / run_id).glob("*/" + path.relative_to(output).as_posix()))
    assert len(copies) == 1 and copies[0].read_bytes() == corrupted_bytes


def test_failure_not_silently_skipped_and_manual_retry_uses_same_seed(resumable_fixture, monkeypatch):
    _, config, output, _ = resumable_fixture
    original_fit = RecordingRegressor.fit

    def fail_second(self, x, y):
        if self.random_state == 2026:
            raise RuntimeError("intentional second-seed failure")
        return original_fit(self, x, y)

    monkeypatch.setattr(RecordingRegressor, "fit", fail_second)
    summary = start_fixture(config, output)
    assert summary["failed_runs"] == 1 and not summary["complete"]
    run_id = next(k for k, v in read_records(output).items() if v["status"] == "failed")
    with pytest.raises(RuntimeError, match="explicit --retry-runs"):
        run_formal_benchmark(str(config), output_dir=str(output), resume=True)
    assert len(RecordingRegressor.fits) == 2
    monkeypatch.setattr(RecordingRegressor, "fit", original_fit)
    result = run_formal_benchmark(str(config), output_dir=str(output), resume=True, retry_runs=[run_id])
    assert result["complete"] and result["passed_runs"] == 3
    assert sorted(fit[-1] for fit in RecordingRegressor.fits) == [42, 2026, 3407]
    historical = next((output / "history" / run_id).glob("*/records/*.json"))
    assert "intentional second-seed failure" in json.loads(historical.read_text())["reason"]


@pytest.mark.parametrize("change", ["data", "split", "parameters", "runtime", "saved_plan", "saved_instance"])
def test_resume_rejects_identity_drift_before_fitting(resumable_fixture, monkeypatch, change):
    root, config, output, models = resumable_fixture
    start_fixture(config, output)
    if change == "data":
        path = next(root.rglob("*.dat"))
        path.write_bytes(path.read_bytes() + b"\n")
    elif change == "split":
        path = config.parent / "instance.json"
        frozen = json.loads(path.read_text(encoding="utf-8"))
        frozen["split"]["conditions"]["train"].reverse()
        path.write_text(json.dumps(frozen), encoding="utf-8")
    elif change == "parameters":
        models["random_forest"].set_params(random_state=1234)
    elif change == "runtime":
        monkeypatch.setattr("project1.experiments.benchmark_instance.formal_runtime_identity", lambda *args: {"changed_environment_or_weights": True})
    elif change == "saved_plan":
        path = output / "experiment.json"
        experiment = json.loads(path.read_text(encoding="utf-8"))
        experiment["run_plan"][0]["model_seed"] = 1234
        path.write_text(json.dumps(experiment), encoding="utf-8")
    else:
        path = output / "config/frozen_instance.json"
        frozen = json.loads(path.read_text(encoding="utf-8"))
        frozen["dataset"] = "wrong"
        path.write_text(json.dumps(frozen), encoding="utf-8")
    with pytest.raises(ValueError, match="differs"):
        run_formal_benchmark(str(config), output_dir=str(output), resume=True)
    assert len(RecordingRegressor.fits) == 3


def test_resume_disallows_scope_changes_retries_of_success_and_duplicate_records(resumable_fixture):
    _, config, output, _ = resumable_fixture
    start_fixture(config, output)
    run_id = next(iter(read_records(output)))
    with pytest.raises(ValueError, match="do not supply"):
        run_formal_benchmark(str(config), output_dir=str(output), resume=True, model_keys=["random_forest"])
    with pytest.raises(ValueError, match="require --resume"):
        run_formal_benchmark(str(config), output_dir=str(output), retry_runs=[run_id])
    with pytest.raises(ValueError, match="retry is allowed only"):
        run_formal_benchmark(str(config), output_dir=str(output), resume=True, retry_runs=[run_id])
    original = output / "records" / f"{run_id}.json"
    (output / "records/duplicate.json").write_bytes(original.read_bytes())
    with pytest.raises(ValueError, match="duplicate/noncanonical"):
        run_formal_benchmark(str(config), output_dir=str(output), resume=True)
    assert len(RecordingRegressor.fits) == 3


def test_active_writer_and_crash_marker_need_manual_acknowledgement(tmp_path):
    output = tmp_path / "lock_test"
    script = ("from pathlib import Path; from project1.experiments.benchmark_instance import formal_experiment_lock; "
              "import sys\nwith formal_experiment_lock(Path(sys.argv[1]), resume=False):\n print('locked', flush=True)\n sys.stdin.read()")
    environment = {**os.environ, "PYTHONPATH": str(ROOT / "workbase/src") + os.pathsep + str(ROOT)}
    process = subprocess.Popen([sys.executable, "-c", script, str(output)], stdin=subprocess.PIPE,
                               stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=environment)
    try:
        assert process.stdout.readline().strip() == "locked"
        with pytest.raises(RuntimeError, match="writer is active"):
            with formal_experiment_lock(output, resume=True, recover_lock=True):
                pytest.fail("active lock was bypassed")
        process.kill()
        process.communicate(timeout=20)
        with pytest.raises(RuntimeError, match="unclean benchmark lock"):
            with formal_experiment_lock(output, resume=True):
                pytest.fail("unclean marker was silently reused")
        with formal_experiment_lock(output, resume=True, recover_lock=True):
            assert len(list((output / "history/locks").glob("*.json"))) == 1
        with formal_experiment_lock(output, resume=True):
            pass
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate(timeout=20)


def test_resume_cli_forwards_explicit_flags(monkeypatch):
    import project1.cli as cli
    calls = []
    monkeypatch.setattr(cli, "run_workflow", lambda name, argv, **kwargs: calls.append(argv) or 0)
    options = ["--formal-config", "config/hpt_formal_benchmark.json", "--resume", "--retry-runs", "run_1", "run_2", "--recover-lock"]
    assert cli.main(["current", "benchmark-3d", *options]) == 0
    assert calls == [options]


def test_interruption_during_save_does_not_commit_success(resumable_fixture, monkeypatch):
    _, config, output, _ = resumable_fixture
    original_save = np.savez_compressed

    def interrupt_save(path, **arrays):
        Path(path).write_bytes(b"unfinished prediction archive")
        raise KeyboardInterrupt("interrupted while saving predictions")

    monkeypatch.setattr(np, "savez_compressed", interrupt_save)
    with pytest.raises(KeyboardInterrupt):
        start_fixture(config, output)
    run_id, record = next(iter(read_records(output).items()))
    assert record["status"] == "interrupted" and record["failure_stage"] == "save"
    assert list((output / "models").rglob("*.pkl"))
    assert json.loads((output / "reports/benchmark_summary.json").read_text())["passed_runs"] == 0
    with pytest.raises(RuntimeError, match="explicit --retry-runs"):
        run_formal_benchmark(str(config), output_dir=str(output), resume=True)
    monkeypatch.setattr(np, "savez_compressed", original_save)
    result = run_formal_benchmark(str(config), output_dir=str(output), resume=True, retry_runs=[run_id])
    assert result["complete"] and result["passed_runs"] == 3
    staged_history = next((output / "history" / run_id).glob("*/predictions/*.staging.npz"))
    assert staged_history.read_bytes() == b"unfinished prediction archive"


def test_ambiguous_lock_is_preserved(tmp_path):
    output = tmp_path / "ambiguous"
    output.mkdir()
    path = output / ".benchmark.lock"
    path.write_bytes(b"broken owner record")
    with pytest.raises(RuntimeError, match="ambiguous benchmark lock"):
        with formal_experiment_lock(output, resume=True, recover_lock=True):
            pytest.fail("ambiguous marker was automatically deleted")
    assert path.read_bytes() == b"broken owner record"


def test_first_run_does_not_overwrite_unrecognized_output(resumable_fixture):
    _, config, output, _ = resumable_fixture
    output.mkdir()
    evidence = output / "unrecognized.txt"
    evidence.write_text("preserve me", encoding="utf-8")
    with pytest.raises(FileExistsError, match="first-run output must be empty"):
        start_fixture(config, output)
    assert evidence.read_text(encoding="utf-8") == "preserve me"
    assert not RecordingRegressor.fits


def test_runtime_identity_tracks_isolated_packages_and_weight_bytes(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from project1.experiments.benchmark_instance import formal_runtime_identity

    checkpoint = tmp_path / "checkpoint.bin"
    checkpoint.write_bytes(b"original pretrained weights")
    adapter = SimpleNamespace(_python_path=lambda: Path(sys.executable),
                              _runner_environment=lambda python: dict(os.environ),
                              _checkpoint=lambda python: checkpoint)
    packages = {"python": "isolated python", "packages": {"tabpfn": "pinned"}}
    monkeypatch.setattr(subprocess, "run", lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(packages)))
    original = formal_runtime_identity({"tabpfn": adapter}, ROOT)
    checkpoint.write_bytes(b"different pretrained weights")
    changed_weights = formal_runtime_identity({"tabpfn": adapter}, ROOT)
    assert changed_weights["tabpfn"]["checkpoint_sha256"] != original["tabpfn"]["checkpoint_sha256"]
    packages["packages"]["tabpfn"] = "different version"
    changed_environment = formal_runtime_identity({"tabpfn": adapter}, ROOT)
    assert changed_environment["tabpfn"]["environment"] != changed_weights["tabpfn"]["environment"]


def test_source_identity_ignores_git_newline_conversion(tmp_path):
    from project1.experiments.benchmark_instance import formal_runtime_identity

    source = tmp_path / "workbase/common/model_versioning.py"
    source.parent.mkdir(parents=True)
    source.write_bytes(b"first line\nsecond line\n")
    original = formal_runtime_identity({}, tmp_path)
    source.write_bytes(b"first line\r\nsecond line\r\n")
    assert formal_runtime_identity({}, tmp_path) == original
    source.write_bytes(b"different implementation\n")
    assert formal_runtime_identity({}, tmp_path) != original
