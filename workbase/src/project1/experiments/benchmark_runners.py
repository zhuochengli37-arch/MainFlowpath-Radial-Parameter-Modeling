from __future__ import annotations

import json
import csv
import hashlib
import platform
from pathlib import Path
from time import perf_counter

import numpy as np
from tqdm import tqdm

from project1.experiments.benchmark_data import (
    PartitionTargetPlan,
    SCHEMA_CONFIG,
    load_training_samples,
    resolve_partition_targets,
    schema_input_type,
)
from project1.experiments.benchmark_partitioning import (
    condition_group_labels,
    cv_splits,
    limit_samples_by_condition,
    metrics,
    partition_samples,
    prepare_model_for_fold,
    to_arrays,
    to_arrays_2d,
)
from project1.experiments.benchmark_reporting import (
    build_aggregate_rows,
    build_benchmark_report,
    write_partition_leaderboards,
    write_utf8_bom_text,
)
from project1.experiments.model_partitioning import safe_partition_name
from project1.experiments.training_metadata import artifact_identity, build_training_context
from project1.modeling.custom_models import MODEL_CATEGORY
from project1.modeling.factory import build_3d_models
from workbase.common.model_versioning import create_model_metadata, save_model_with_metadata


def _formal_parameters(model: object) -> dict:
    def describe(value):
        if hasattr(value, "get_params"):
            parameters = value.get_params(deep=False)
        elif type(value).__module__.startswith("project1.modeling"):
            parameters = vars(value)
        else:
            return str(value)
        return {"class": f"{type(value).__module__}.{type(value).__qualname__}", "parameters": parameters}

    # Some defaults (e.g. XGBoost's missing=NaN) are parameter symbols, not metric values.
    return json.loads(json.dumps(model.get_params(deep=True), default=describe, sort_keys=True), parse_constant=str)


def seed_formal_model(model: object, seed: int, train_size: int) -> object:
    """Clone the factory definition; change only its existing random-seed fields."""
    candidate = prepare_model_for_fold(model, train_size)
    parameters = candidate.get_params(deep=True)
    updates = {key: seed for key in parameters if key.rsplit("__", 1)[-1] in {"random_state", "random_seed"}}
    for key, value in parameters.items():
        if key.rsplit("__", 1)[-1] == "model_kwargs" and isinstance(value, dict):
            seeded = {name: seed if name in {"random_state", "random_seed"} else item for name, item in value.items()}
            updates[key] = seeded
    candidate.set_params(**updates)
    return candidate


def run_formal_benchmark(config_path: str, *, prepare_only: bool = False,
                         model_keys: list[str] | None = None, case_keys: list[str] | None = None,
                         output_dir: str | None = None, compatibility_seed: int | None = None,
                         resume: bool = False, retry_runs: list[str] | None = None,
                         recover_lock: bool = False) -> dict[str, object]:
    """Existing Direct-3D models on one frozen Train/Validation/Test instance; no CV/refit."""
    from project1.experiments.benchmark_data import PROJECT_ROOT
    from project1.experiments.benchmark_instance import (
        formal_cases, formal_condition_id, freeze_or_verify_instance, inspect_formal_dataset,
        json_fingerprint, write_benchmark_json, file_sha256, formal_experiment_lock,
        formal_runtime_identity, formal_run_paths, load_formal_records,
        write_formal_record, archive_formal_attempt,
    )
    from project1.experiments.benchmark_reporting import formal_metrics, rebuild_formal_reports
    from project1.modeling.registry import validate_model_registry
    from workbase.common.model_versioning import get_package_version

    if (retry_runs or recover_lock) and not resume:
        raise ValueError("--retry-runs and --recover-lock require --resume")
    if resume and (prepare_only or model_keys is not None or case_keys is not None or compatibility_seed is not None):
        raise ValueError("resume reads the original plan; do not supply prepare/model/case/seed overrides")
    configuration = json.loads(Path(config_path).read_text(encoding="utf-8"))
    contract_root = PROJECT_ROOT / "data" / "output" / "benchmark"
    metric_protocol = json.loads((contract_root / "config" / "metric_definitions.json").read_text(encoding="utf-8"))
    repeats = json.loads((contract_root / "config" / "stochastic_repeat_protocol.json").read_text(encoding="utf-8"))
    with (contract_root / "config" / "model_taxonomy.csv").open(encoding="utf-8-sig", newline="") as stream:
        taxonomy = {row["model_key"]: row for row in csv.DictReader(stream)}
    models = _build_models(include_gpr=True)
    validate_model_registry(models)
    if set(models) != set(taxonomy) or set(models) != set(repeats["model_classification"]):
        raise ValueError("formal factory, taxonomy and stochasticity protocol differ")
    model_parameters = {key: _formal_parameters(model) for key, model in models.items()}
    samples, instance = inspect_formal_dataset(configuration["input_dir"], split_seed=42,
                                               validation_size=0.2, test_size=0.2)
    cases = formal_cases(samples)
    expected = configuration.get("expected_inventory", {})
    observed = {"conditions": len(instance["conditions"]), "main_files": sum(f["station"] == "MAIN" for f in instance["files"]),
                "cases": len(cases), "profile_points": sorted({n for c in instance["cases"] for n in c["profile_row_counts"]})}
    for key, value in expected.items():
        if observed.get(key) != value:
            raise ValueError(f"formal inventory {key}: expected {value}, got {observed.get(key)}")
    source_names = {s.flow_parameter_name for s in samples}
    if source_names != {configuration["second_parameter_source_name"]}:
        raise ValueError(f"unexpected second-parameter meaning: {source_names}")
    schedule = {}
    for key in models:
        kind = repeats["model_classification"][key]["stochasticity_type"]
        rule = repeats["initial_screening"][kind]
        schedule[key] = rule.get("seeds", repeats["fixed_seeds"]["initial_screening"])
    instance.update(dataset=configuration["dataset"], feature_order=configuration["feature_order"],
                    second_parameter_definition=configuration["second_parameter_definition"],
                    model_parameters=model_parameters, model_seeds=schedule,
                    metric_protocol=metric_protocol, repeat_protocol=repeats,
                    taxonomy=taxonomy, configuration=configuration,
                    environment={key: get_package_version(key) for key in
                                 ("numpy", "scipy", "scikit-learn", "pandas", "torch", "xgboost", "lightgbm",
                                  "pygam", "interpret-core", "catboost", "ngboost", "pytabkit", "pykan", "skorch", "tabicl")},
                    training_policy={"global_subsampling": False, "gpr_subsampling": False,
                                     "training_set": "all frozen train conditions", "full_data_refit": False,
                                     "test_used_for_fit_or_selection": False})
    instance_path = PROJECT_ROOT / configuration["instance_path"]
    if not prepare_only and not instance_path.exists():
        raise FileNotFoundError(f"prepare the formal instance first: {instance_path}")
    frozen = freeze_or_verify_instance(instance_path, instance)
    if prepare_only:
        return {"status": "prepared", "instance_path": str(instance_path), "instance_sha256": frozen["instance_sha256"],
                "inventory": observed, "condition_counts": instance["split"]["condition_counts"],
                "planned_fits": len(cases) * sum(len(seeds) for seeds in schedule.values())}

    output = Path(output_dir or configuration["output_dir"]).resolve()
    saved_experiment = None
    if resume:
        saved_experiment = json.loads((output / "experiment.json").read_text(encoding="utf-8"))
        if saved_experiment.get("resume_format") != 1:
            raise ValueError("experiment predates transactional resume; preserve it and inspect manually")
        model_keys, case_keys = saved_experiment["model_keys"], saved_experiment["case_keys"]
        if saved_experiment["purpose"] == "preflight_single_seed":
            compatibility_seed = next(iter(saved_experiment["model_seeds"].values()))[0]
    selected_models = list(models) if model_keys is None else list(dict.fromkeys(model_keys))
    selected_cases = list(cases) if case_keys is None else list(dict.fromkeys(case_keys))
    if not selected_models or not selected_cases or set(selected_models) - set(models) or set(selected_cases) - set(cases):
        raise ValueError("unknown or empty formal model/case selection")
    partial = saved_experiment["purpose"] != "initial_screening" if resume else (model_keys is not None or case_keys is not None)
    if partial and output_dir is None:
        raise ValueError("explicit model/case compatibility checks require a separate --output directory")
    if partial and output == Path(configuration["output_dir"]).resolve():
        raise ValueError("compatibility output must differ from the official first-round output")
    if compatibility_seed is not None and (not partial or any(compatibility_seed not in schedule[key] for key in selected_models)):
        raise ValueError("a prescribed single seed is allowed only for an explicit compatibility check")
    run_schedule = {key: [compatibility_seed] if compatibility_seed is not None else schedule[key] for key in selected_models}
    if not resume and (output / "experiment.json").exists():
        raise FileExistsError(f"experiment output already exists; records are preserved: {output}")
    planned = sum(len(run_schedule[key]) for key in selected_models) * len(selected_cases)
    purpose = "preflight_single_seed" if compatibility_seed is not None else ("stage_b_compatibility" if partial else "initial_screening")
    experiment = {"experiment_id": output.name, "purpose": purpose,
                  "planned_fits": planned, "model_keys": selected_models, "case_keys": selected_cases,
                  "inventory_sha256": frozen["inventory_sha256"], "split_hash": frozen["split"]["sha256"],
                  "instance_sha256": frozen["instance_sha256"], "instance_path": str(instance_path),
                  "model_seeds": run_schedule,
                  "hardware": {"platform": platform.platform(), "processor": platform.processor()},
                  "training_policy": frozen["training_policy"]}
    run_plan = [{"run_id": f"{case_id.replace(':', '_')}__{key}__seed_{seed}", "case_id": case_id,
                 "model_key": key, "model_seed": seed, "repeat_index": repeat_index}
                for case_id in selected_cases for key in selected_models
                for repeat_index, seed in enumerate(run_schedule[key])]
    if len({task["run_id"] for task in run_plan}) != planned:
        raise ValueError("formal run IDs are not unique")
    experiment.update(resume_format=1, run_plan=run_plan,
                      runtime_identity=formal_runtime_identity({key: models[key] for key in selected_models}, PROJECT_ROOT))
    experiment["contract_sha256"] = json_fingerprint(experiment)
    if resume and experiment != saved_experiment:
        raise ValueError("resume experiment/configuration/runtime identity differs from the original")
    with formal_experiment_lock(output, resume=resume, recover_lock=recover_lock):
        if resume:
            if json.loads((output / "experiment.json").read_text(encoding="utf-8")) != experiment:
                raise ValueError("experiment changed while acquiring writer lock")
            if json.loads((output / "config/frozen_instance.json").read_text(encoding="utf-8")) != frozen:
                raise ValueError("saved frozen instance differs")
            records = load_formal_records(output, experiment)
            issues = {key: value for key, value in records.items() if value["status"] != "passed"}
            authorized = set(retry_runs or [])
            if authorized - set(issues):
                raise ValueError(f"retry is allowed only for failed/interrupted/corrupt run IDs: {sorted(authorized - set(issues))}")
            write_benchmark_json(output / "reports/resume_issues.json",
                                 [{"run_id": key, "status": value["status"], "reason": value.get("reason", "interrupted before commit")}
                                  for key, value in issues.items()])
            if set(issues) - authorized:
                rebuild_formal_reports(output)
                raise RuntimeError(f"explicit --retry-runs authorization required: {sorted(set(issues) - authorized)}")
        else:
            if any(p.name != ".benchmark.lock" for p in output.iterdir()):
                raise FileExistsError(f"first-run output must be empty; preserve and inspect: {output}")
            write_benchmark_json(output / "config/frozen_instance.json", frozen)
            write_benchmark_json(output / "experiment.json", experiment)
            records, authorized = {}, set()
        try:
            splits = {name: set(ids) for name, ids in frozen["split"]["conditions"].items()}
            for case_id in selected_cases:
                x, y, case_samples = cases[case_id]
                partition, target = case_id.rsplit(":", 1)
                condition_ids = np.array([formal_condition_id(s) for s in case_samples])
                indices = {name: np.flatnonzero(np.isin(condition_ids, list(ids))) for name, ids in splits.items()}
                train = indices["train"]
                train_samples = [case_samples[i] for i in train]
                for key in selected_models:
                    for repeat_index, seed in enumerate(run_schedule[key]):
                        run_id = f"{case_id.replace(':', '_')}__{key}__seed_{seed}"
                        previous = records.get(run_id)
                        if previous and previous["status"] == "passed":
                            continue
                        task = {"run_id": run_id, "case_id": case_id, "model_key": key,
                                "model_seed": seed, "repeat_index": repeat_index}
                        if previous:
                            archive_formal_attempt(output, task)
                        record = {"experiment_id": experiment["experiment_id"], "run_id": run_id, "case_id": case_id,
                                  "experiment_contract_sha256": experiment["contract_sha256"],
                                  "attempt": previous.get("attempt", 1) + 1 if previous else 1,
                                  "model_key": key, "family": taxonomy[key]["registry_family"],
                                  "paper_family_code": taxonomy[key]["paper_family_code"],
                                  "component": case_samples[0].component, "stage": case_samples[0].stage,
                                  "section": case_samples[0].section, "target": target, "dataset": frozen["dataset"],
                                  "split_hash": frozen["split"]["sha256"], "split_seed": 42,
                                  **{f"{name}_condition_hash": value for name, value in frozen["split"]["condition_hashes"].items()},
                                  "stochasticity_type": repeats["model_classification"][key]["stochasticity_type"],
                                  "model_seed": seed, "repeat_index": repeat_index, "scheduled_runs": len(run_schedule[key]),
                                  "train_sample_count": len(train), "train_condition_ids": sorted(splits["train"]),
                                  "evaluations": {}, "runtime": {"fit_time": None, "validation_inference_time": None,
                                  "test_inference_time": None, "memory": None, "memory_note": "not measured by common runner",
                                  "hardware": experiment["hardware"]}, "status": "running", "failure_stage": "factory"}
                        write_formal_record(output, record)
                        stage = "factory"
                        try:
                            candidate = seed_formal_model(models[key], seed, len(train))
                            record["metadata"] = {"implementation_status": "implemented", "version": "factory_snapshot",
                                                  "parameters": _formal_parameters(candidate),
                                                  "feature_order": frozen["feature_order"], "target_definition": target}
                            if key in {"pysr", "tabpfn"}:
                                candidate.set_run_context(partition=partition, target=target, split_mode="formal_train",
                                                          model_seed=seed, train_condition_ids=condition_ids[train].tolist(),
                                                          feature_order=frozen["feature_order"], split_hash=record["split_hash"])
                            stage = "fit"
                            started = perf_counter()
                            candidate.fit(x[train], y[train])
                            record["runtime"]["fit_time"] = perf_counter() - started
                            predictions = {}
                            for subset in ("validation", "test"):
                                index = indices[subset]
                                if key in {"pysr", "tabpfn"}:
                                    candidate.set_prediction_context(condition_ids=condition_ids[index].tolist(),
                                                                     evaluation_subset=subset, split_hash=record["split_hash"])
                                stage = f"{subset}_predict"
                                started = perf_counter()
                                prediction = np.asarray(candidate.predict(x[index]), dtype=float).reshape(-1)
                                record["runtime"][f"{subset}_inference_time"] = perf_counter() - started
                                stage = f"{subset}_metrics"
                                record["evaluations"][subset] = formal_metrics(y[index], prediction, metric_protocol)
                                predictions.update({f"{subset}_x": x[index], f"{subset}_truth": y[index],
                                                    f"{subset}_prediction": prediction,
                                                    f"{subset}_condition_ids": condition_ids[index]})
                            stage = "save"
                            model_path, metadata_path, prediction_path = formal_run_paths(output, record)
                            model_path.parent.mkdir(parents=True, exist_ok=True)
                            context = build_training_context(train_samples, model_dimension="3D", partition=partition,
                                                             inputs=["speed_parameter", "flow_parameter", "xi"], target=target,
                                                             training_source_root=configuration["input_dir"], sample_count=len(train))
                            context.update(feature_order=frozen["feature_order"], split_hash=record["split_hash"],
                                           inventory_sha256=frozen["inventory_sha256"], model_seed=seed)
                            additional = {name: getattr(candidate, attr) for name, attr in
                                          (("symbolic_model", "symbolic_report_"), ("foundation_model", "foundation_report_"))
                                          if hasattr(candidate, attr)}
                            metadata = create_model_metadata(key, "3D", "_".join(frozen["feature_order"]), instance_path,
                                                             config={"model_seed": seed, "parameters": record["metadata"]["parameters"],
                                                                     "train_samples": len(train), "split_hash": record["split_hash"]},
                                                             model_context=context, additional_info=additional)
                            temporary_model = model_path.with_name(model_path.stem + ".staging.pkl")
                            save_model_with_metadata(candidate, temporary_model, metadata)
                            temporary_model.replace(model_path)
                            temporary_model.with_suffix(".json").replace(metadata_path)
                            prediction_path.parent.mkdir(parents=True, exist_ok=True)
                            temporary_prediction = prediction_path.with_name(prediction_path.stem + ".staging.npz")
                            np.savez_compressed(temporary_prediction, **predictions)
                            temporary_prediction.replace(prediction_path)
                            record["metadata"].update(model_path=model_path.relative_to(output).as_posix(),
                                                       metadata_path=model_path.with_suffix(".json").relative_to(output).as_posix(),
                                                       **additional)
                            record["artifact"] = {**artifact_identity(context), "model": key,
                                                  "path": model_path.relative_to(output).as_posix(), "model_seed": seed,
                                                  "train_samples": len(train), "sample_count": len(train), "metrics": {}}
                            record["artifact_sha256"] = {p.relative_to(output).as_posix(): file_sha256(p)
                                                         for p in (model_path, metadata_path, prediction_path)}
                            record.update(status="passed", failure_stage=None, prediction_path=prediction_path.relative_to(output).as_posix())
                        except Exception as exc:
                            record.update(status="failed", failure_stage=stage, exception_type=type(exc).__name__, reason=str(exc) or repr(exc))
                            if stage == "fit":
                                record["runtime"]["fit_time"] = perf_counter() - started
                            elif stage.endswith("_predict"):
                                record["runtime"][stage.replace("_predict", "_inference_time")] = perf_counter() - started
                        except BaseException as exc:
                            record.update(status="interrupted", failure_stage=stage,
                                          exception_type=type(exc).__name__, reason=str(exc) or repr(exc))
                            raise
                        finally:
                            write_formal_record(output, record)

        finally:
            summary = rebuild_formal_reports(output)
        return summary


def _build_models(include_gpr: bool = True) -> dict[str, object]:
    return build_3d_models(include_gpr=include_gpr)


def _build_2d_models(include_gpr: bool = True) -> dict[str, object]:
    direct_3d_only = {
        "kernel_ridge_rbf", "extra_trees", "ard_gpr", "gam", "catboost", "ebm",
        "ngboost", "mlp", "realmlp", "tabm", "original_kan", "resnet_mlp",
        "ft_transformer", "tabiclv2", "pysr", "tabpfn",
    }
    return {
        name: model
        for name, model in _build_models(include_gpr=include_gpr).items()
        if name not in direct_3d_only
    }


def _resolve_target_plans(
    partitions: list[tuple[str, list[object]]],
    target_selection: str | list[str] | None = None,
    missing_target_policy: str | None = None,
) -> tuple[dict[str, PartitionTargetPlan], tuple[str, ...]]:
    plans: dict[str, PartitionTargetPlan] = {}
    ordered_targets: list[str] = []
    for partition_key, subset in partitions:
        try:
            plan = resolve_partition_targets(
                subset,
                targets=(
                    SCHEMA_CONFIG.schema_target_selection
                    if target_selection is None
                    else target_selection
                ),
                missing_target_policy=(
                    SCHEMA_CONFIG.missing_target_policy
                    if missing_target_policy is None
                    else missing_target_policy
                ),
            )
        except ValueError as exc:
            raise ValueError(f"partition {partition_key!r}: {exc}") from exc
        plans[partition_key] = plan
        for target in plan.selected_targets:
            if target not in ordered_targets:
                ordered_targets.append(target)
    return plans, tuple(ordered_targets)


def _write_target_plan(
    output: Path,
    model_dimension: str,
    partition_list: list[tuple[str, list[object]]],
    target_plans: dict[str, PartitionTargetPlan],
) -> Path:
    rows = []
    for partition_key, subset in partition_list:
        plan = target_plans[partition_key]
        first = subset[0]
        rows.append(
            {
                "model_dimension": model_dimension,
                "component": first.component,
                "stage": int(first.stage),
                "station": str(getattr(first, "station", "MAIN")).upper(),
                "section": getattr(first, "section", None),
                "partition": partition_key,
                "sample_count": len(subset),
                "available_outputs": list(plan.available_outputs),
                "selected_targets": list(plan.selected_targets),
                "skipped_targets": plan.skipped_targets,
                "usable_samples_by_target": {
                    target: sum(target in sample.output_columns for sample in subset)
                    for target in plan.selected_targets
                },
                "training_skips": {
                    target: "fewer than 2 valid samples after missing-value filtering"
                    for target in plan.selected_targets
                    if sum(target in sample.output_columns for sample in subset) < 2
                },
            }
        )
    path = output / f"training_target_plan_{model_dimension.lower()}.json"
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def run_benchmark(
    input_dir: str,
    output_dir: str,
    radial_mode: str,
    include_gpr: bool,
    max_samples: int,
    partition_mode: str,
    n_splits: int = 3,
) -> dict[str, object]:
    print("\n" + "=" * 70)
    print("Benchmark started")
    print("=" * 70)

    print("\n[1/7] Loading data...")
    samples = load_training_samples(input_dir, radial_mode)
    if len(samples) < 8:
        raise ValueError("not enough usable samples for benchmark")
    if len(samples) > max_samples:
        samples = limit_samples_by_condition(samples, max_samples, random_state=42)
    print(f"Loaded {len(samples)} samples")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print("\n[2/7] Building model candidates...")
    models = _build_models(include_gpr=include_gpr)
    print(f"Loaded {len(models)} models")

    split_modes = ("condition", "random", "rpm", "wcor")
    rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []

    print("\n[3/7] Partitioning data...")
    partitions = partition_samples(samples, partition_mode)
    partition_list = [(key, subset) for key, subset in partitions.items() if len(subset) >= 8]
    target_plans, targets = _resolve_target_plans(partition_list)
    if not targets:
        raise ValueError("no usable output targets found after partition target resolution")
    print(f"Valid partitions: {len(partition_list)}")
    print(f"Selected targets across partitions: {list(targets)}")
    target_plan_path = _write_target_plan(output, "3D", partition_list, target_plans)

    total_work = 0
    for partition_key, subset in partition_list:
        for target in target_plans[partition_key].selected_targets:
            _, _, target_samples = to_arrays(subset, target)
            if len(target_samples) < 3:
                continue
            for split_mode in split_modes:
                if list(cv_splits(target_samples, split_mode, n_splits)):
                    total_work += len(models)

    print("\n[4/7] Running cross-validation...")
    print(f"Estimated tasks: {total_work}")
    completed_work = 0
    with tqdm(total=total_work, desc="models", unit="model") as pbar:
        for partition_key, subset in partition_list:
            for target in target_plans[partition_key].selected_targets:
                x, y, target_samples = to_arrays(subset, target)
                if len(target_samples) < 3:
                    continue
                for split_mode in split_modes:
                    splits = list(cv_splits(target_samples, split_mode, n_splits))
                    if not splits:
                        continue
                    for name, model in models.items():
                        fold_metrics: list[dict[str, float]] = []
                        failed = False
                        for fold, (train_idx, test_idx) in enumerate(splits):
                            if len(test_idx) == 0 or len(train_idx) == 0:
                                continue
                            x_train, y_train = x[train_idx], y[train_idx]
                            x_test, y_test = x[test_idx], y[test_idx]
                            stage = "factory"
                            try:
                                candidate = prepare_model_for_fold(model, len(train_idx))
                                if name in {"pysr", "tabpfn"}:
                                    condition_ids = condition_group_labels(target_samples)
                                    candidate.set_run_context(
                                        partition=str(partition_key),
                                        target=target,
                                        split_mode=split_mode,
                                        fold=fold,
                                        train_condition_ids=condition_ids[train_idx].tolist(),
                                        feature_order=["n", "w", "xi"],
                                    )
                                stage = "fit"
                                candidate.fit(x_train, y_train)
                                if name in {"pysr", "tabpfn"}:
                                    candidate.set_prediction_context(
                                        partition=str(partition_key),
                                        target=target,
                                        split_mode=split_mode,
                                        fold=fold,
                                        condition_ids=condition_ids[test_idx].tolist(),
                                    )
                                stage = "predict"
                                pred = candidate.predict(x_test)
                                stage = "metric"
                                fold_metrics.append(metrics(y_test, pred))
                            except Exception as exc:
                                failures.append(
                                    {
                                        "model_key": name,
                                        "status": "failed",
                                        "exception_type": type(exc).__name__,
                                        "reason": str(exc) or repr(exc),
                                        "stage": stage,
                                        "partition": partition_key,
                                        "target": target,
                                        "split_mode": split_mode,
                                        "fold": fold,
                                    }
                                )
                                failed = True
                                break
                        if failed:
                            pbar.update(1)
                            continue
                        if not fold_metrics:
                            continue
                        completed_work += 1
                        context = build_training_context(
                            target_samples,
                            model_dimension="3D",
                            partition=partition_key,
                            inputs=["speed_parameter", "flow_parameter", "xi"],
                            target=target,
                            training_source_root=input_dir,
                            sample_count=len(target_samples),
                        )
                        rows.append(
                            {
                                **artifact_identity(context),
                                "partition": partition_key,
                                "partition_size": len(target_samples),
                                "target": target,
                                "split_mode": split_mode,
                                "model": name,
                                "path": f"models/{safe_partition_name(partition_key)}/{name}_{target}.pkl",
                                "schema_name": context.get("schema_name"),
                                "schema_version": context.get("schema_version"),
                                "model_category": MODEL_CATEGORY.get(name, "other"),
                                "folds": len(fold_metrics),
                                "mae": float(np.mean([item["mae"] for item in fold_metrics])),
                                "rmse": float(np.mean([item["rmse"] for item in fold_metrics])),
                                "mape": float(np.mean([item["mape"] for item in fold_metrics])),
                                "r2": float(np.mean([item["r2"] for item in fold_metrics])),
                            }
                        )
                        pbar.update(1)

    print("\n[5/7] Training final models...")
    models_dir = output / "models"
    models_dir.mkdir(parents=True, exist_ok=True)

    saved_models_info: list[dict[str, object]] = []
    for partition_key, subset in partition_list:
        safe_partition = safe_partition_name(partition_key)
        partition_models_dir = models_dir / safe_partition
        partition_models_dir.mkdir(parents=True, exist_ok=True)

        for target in target_plans[partition_key].selected_targets:
            x, y, target_samples = to_arrays(subset, target)
            if len(target_samples) < 2:
                continue
            for name, model in models.items():
                stage = "factory"
                try:
                    final_model = prepare_model_for_fold(model, len(x))
                    if name in {"pysr", "tabpfn"}:
                        final_model.set_run_context(
                            partition=str(partition_key),
                            target=target,
                            split_mode="full_training_data",
                            fold=None,
                            train_condition_ids=condition_group_labels(target_samples).tolist(),
                            feature_order=["n", "w", "xi"],
                        )
                    stage = "fit"
                    final_model.fit(x, y)
                    symbolic_info = getattr(final_model, "symbolic_report_", None)
                    foundation_info = getattr(final_model, "foundation_report_", None)
                    if name in {"pysr", "tabpfn"}:
                        final_model.set_prediction_context(
                            partition=str(partition_key),
                            target=target,
                            split_mode="artifact_check",
                            fold=None,
                            condition_ids=condition_group_labels(target_samples[:1]).tolist(),
                        )
                        stage = "predict"
                        final_model.predict(x[:1])
                        foundation_info = getattr(final_model, "foundation_report_", None)
                    if name == "pysr":
                        symbolic_info = final_model.symbolic_report_
                        for row in rows:
                            if (
                                row["partition"] == partition_key
                                and row["target"] == target
                                and row["model"] == name
                            ):
                                row["selected_equation"] = symbolic_info["selected_equation"]
                                row["selected_complexity"] = symbolic_info["selected_complexity"]
                    best_result = next(
                        (
                            row
                            for row in rows
                            if row["partition"] == partition_key
                            and row["target"] == target
                            and row["model"] == name
                            and row["split_mode"] == "condition"
                        ),
                        None,
                    )
                    metrics_payload = (
                        {
                            "mae": float(best_result["mae"]),
                            "rmse": float(best_result["rmse"]),
                            "mape": float(best_result["mape"]),
                            "r2": float(best_result["r2"]),
                        }
                        if best_result
                        else None
                    )
                    model_context = build_training_context(
                        target_samples,
                        model_dimension="3D",
                        partition=partition_key,
                        inputs=["speed_parameter", "flow_parameter", "xi"],
                        target=target,
                        training_source_root=input_dir,
                        sample_count=len(target_samples),
                    )
                    model_context["available_outputs"] = list(target_plans[partition_key].available_outputs)
                    model_context["selected_targets"] = list(target_plans[partition_key].selected_targets)
                    metadata = create_model_metadata(
                        model_name=name,
                        model_type="3D",
                        input_type=schema_input_type(((0, "rpm"), (1, "wcor"), (2, "xi"))),
                        data_path=Path(input_dir),
                        config={
                            "partition_mode": partition_mode,
                            "partition_key": str(partition_key),
                            "radial_mode": radial_mode,
                            "requested_n_splits": n_splits,
                            "target": target,
                            "train_samples": len(target_samples),
                            "include_gpr": include_gpr,
                        },
                        metrics=metrics_payload,
                        additional_info={
                            "component": subset[0].component if subset else None,
                            "model_category": MODEL_CATEGORY.get(name, "other"),
                            **({"symbolic_model": symbolic_info} if symbolic_info else {}),
                            **({"foundation_model": foundation_info} if foundation_info else {}),
                        },
                        model_context=model_context,
                    )
                    model_path = partition_models_dir / f"{name}_{target}.pkl"
                    stage = "save"
                    save_model_with_metadata(final_model, model_path, metadata)
                    saved_models_info.append(
                        {
                            **artifact_identity(model_context),
                            "model": name,
                            "path": str(model_path.relative_to(output)),
                            "train_samples": len(target_samples),
                            "sample_count": len(target_samples),
                            "metrics": metrics_payload,
                            **({"symbolic_model": symbolic_info} if symbolic_info else {}),
                            **({"foundation_model": foundation_info} if foundation_info else {}),
                        }
                    )
                except Exception as exc:
                    failures.append(
                        {
                            "model_key": name,
                            "status": "failed",
                            "exception_type": type(exc).__name__,
                            "reason": str(exc) or repr(exc),
                            "stage": stage,
                            "partition": partition_key,
                            "target": target,
                        }
                    )
                    rows[:] = [
                        row for row in rows
                        if not (
                            row["partition"] == partition_key
                            and row["target"] == target
                            and row["model"] == name
                        )
                    ]

    models_manifest_path = models_dir / "models_manifest.json"
    models_manifest_path.write_text(json.dumps(saved_models_info, ensure_ascii=False, indent=2), encoding="utf-8")
    failures_path = output / "benchmark_failures_3d.json"
    failures_path.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[6/7] Writing leaderboard...")
    leaderboard = sorted(rows, key=lambda item: (item["partition"], item["target"], item["split_mode"], item["rmse"]))
    leaderboard_path = output / f"leaderboard_{radial_mode}.json"
    leaderboard_path.write_text(json.dumps(leaderboard, ensure_ascii=False, indent=2), encoding="utf-8")
    write_partition_leaderboards(leaderboard, output, radial_mode)

    aggregate_rows = build_aggregate_rows(rows, MODEL_CATEGORY)
    aggregate_path = output / f"leaderboard_{radial_mode}_aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    component_source_rows = [
        {
            "partition": row["partition"],
            "target": row["target"],
            "split_mode": row["split_mode"],
            "model": row["model"],
            "model_category": row["model_category"],
            "mae": row["mae"],
            "rmse": row["rmse"],
            "mape": row["mape"],
            "r2": row["r2"],
            "partition_size": row["partition_size"],
            "folds": row["folds"],
        }
        for row in rows
    ]

    report_path = output / f"benchmark_report_{radial_mode}.md"
    report_content = build_benchmark_report(
        samples=samples,
        input_dir=input_dir,
        partition_mode=partition_mode,
        targets=targets,
        component_source_rows=component_source_rows,
        model_category=MODEL_CATEGORY,
        radial_mode=radial_mode,
    )
    write_utf8_bom_text(report_path, report_content)

    print("\n[7/7] Done")
    return {
        "sample_count": len(samples),
        "model_count": len(models),
        "saved_models_count": len(saved_models_info),
        "leaderboard_path": str(leaderboard_path),
        "aggregate_path": str(aggregate_path),
        "report_path": str(report_path),
        "models_dir": str(models_dir),
        "models_manifest_path": str(models_manifest_path),
        "failures_path": str(failures_path),
        "failure_count": len(failures),
        "target_plan_path": str(target_plan_path),
        "completed_work": completed_work,
    }


def run_benchmark_2d(
    input_dir: str,
    output_dir: str,
    radial_mode: str,
    include_gpr: bool,
    max_samples: int,
    partition_mode: str,
    n_splits: int = 3,
) -> dict[str, object]:
    print("\n" + "=" * 70)
    print("2D Benchmark started (inputs: rpm, xi)")
    print("=" * 70)

    print("\n[1/7] Loading data...")
    samples = load_training_samples(input_dir, radial_mode)
    if len(samples) < 8:
        raise ValueError("not enough usable samples for 2D benchmark")
    if len(samples) > max_samples:
        rng = np.random.default_rng(42)
        idx = rng.choice(np.arange(len(samples)), size=max_samples, replace=False)
        samples = [samples[i] for i in sorted(idx)]
    print(f"Loaded {len(samples)} samples")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print("\n[2/7] Building model candidates...")
    models = _build_2d_models(include_gpr=include_gpr)
    print(f"Loaded {len(models)} models")

    split_modes = ("random", "rpm")
    rows: list[dict[str, object]] = []

    print("\n[3/7] Partitioning data...")
    partitions = partition_samples(samples, partition_mode)
    partition_list = [(key, subset) for key, subset in partitions.items() if len(subset) >= 8]
    target_plans, targets = _resolve_target_plans(partition_list)
    if not targets:
        raise ValueError("no usable output targets found after partition target resolution")
    print(f"Valid partitions: {len(partition_list)}")
    print(f"Selected targets across partitions: {list(targets)}")
    target_plan_path = _write_target_plan(output, "2D", partition_list, target_plans)

    total_work = 0
    for partition_key, subset in partition_list:
        for target in target_plans[partition_key].selected_targets:
            _, _, target_samples = to_arrays_2d(subset, target)
            if len(target_samples) < 3:
                continue
            for split_mode in split_modes:
                if list(cv_splits(target_samples, split_mode, n_splits)):
                    total_work += len(models)

    print("\n[4/7] Running cross-validation...")
    print(f"Estimated tasks: {total_work}")
    with tqdm(total=total_work, desc="models", unit="model") as pbar:
        for partition_key, subset in partition_list:
            for target in target_plans[partition_key].selected_targets:
                x, y, target_samples = to_arrays_2d(subset, target)
                if len(target_samples) < 3:
                    continue
                for split_mode in split_modes:
                    splits = list(cv_splits(target_samples, split_mode, n_splits))
                    if not splits:
                        continue
                    for name, model in models.items():
                        fold_metrics: list[dict[str, float]] = []
                        for train_idx, test_idx in splits:
                            if len(test_idx) == 0 or len(train_idx) == 0:
                                continue
                            x_train, y_train = x[train_idx], y[train_idx]
                            x_test, y_test = x[test_idx], y[test_idx]
                            try:
                                candidate = prepare_model_for_fold(model, len(train_idx))
                                candidate.fit(x_train, y_train)
                                pred = candidate.predict(x_test)
                            except Exception:
                                continue
                            fold_metrics.append(metrics(y_test, pred))
                        if not fold_metrics:
                            continue
                        context = build_training_context(
                            target_samples,
                            model_dimension="2D",
                            partition=partition_key,
                            inputs=["speed_parameter", "xi"],
                            target=target,
                            training_source_root=input_dir,
                            sample_count=len(target_samples),
                        )
                        rows.append(
                            {
                                **artifact_identity(context),
                                "partition": partition_key,
                                "partition_size": len(target_samples),
                                "target": target,
                                "split_mode": split_mode,
                                "model": name,
                                "path": f"models_2d/{safe_partition_name(partition_key)}/{name}_{target}.pkl",
                                "schema_name": context.get("schema_name"),
                                "schema_version": context.get("schema_version"),
                                "model_category": MODEL_CATEGORY.get(name, "other"),
                                "folds": len(fold_metrics),
                                "mae": float(np.mean([item["mae"] for item in fold_metrics])),
                                "rmse": float(np.mean([item["rmse"] for item in fold_metrics])),
                                "mape": float(np.mean([item["mape"] for item in fold_metrics])),
                                "r2": float(np.mean([item["r2"] for item in fold_metrics])),
                            }
                        )
                        pbar.update(1)

    leaderboard = sorted(rows, key=lambda item: (item["partition"], item["target"], item["split_mode"], item["rmse"]))
    leaderboard_path = output / f"leaderboard_2d_{radial_mode}.json"
    leaderboard_path.write_text(json.dumps(leaderboard, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n[5/7] Training final models...")
    models_dir = output / "models_2d"
    models_dir.mkdir(parents=True, exist_ok=True)
    saved_models = 0
    saved_models_info: list[dict[str, object]] = []
    for partition_key, subset in partition_list:
        safe_partition = safe_partition_name(partition_key)
        partition_models_dir = models_dir / safe_partition
        partition_models_dir.mkdir(parents=True, exist_ok=True)
        for target in target_plans[partition_key].selected_targets:
            x, y, target_samples = to_arrays_2d(subset, target)
            if len(target_samples) < 2:
                continue
            for name, model in models.items():
                try:
                    final_model = prepare_model_for_fold(model, len(x))
                    final_model.fit(x, y)
                    best_result = next(
                        (
                            row
                            for row in rows
                            if row["partition"] == partition_key
                            and row["target"] == target
                            and row["model"] == name
                            and row["split_mode"] == "random"
                        ),
                        None,
                    )
                    metrics_payload = (
                        {
                            "mae": float(best_result["mae"]),
                            "rmse": float(best_result["rmse"]),
                            "mape": float(best_result["mape"]),
                            "r2": float(best_result["r2"]),
                        }
                        if best_result
                        else None
                    )
                    model_context = build_training_context(
                        target_samples,
                        model_dimension="2D",
                        partition=partition_key,
                        inputs=["speed_parameter", "xi"],
                        target=target,
                        training_source_root=input_dir,
                        sample_count=len(target_samples),
                    )
                    model_context["available_outputs"] = list(target_plans[partition_key].available_outputs)
                    model_context["selected_targets"] = list(target_plans[partition_key].selected_targets)
                    metadata = create_model_metadata(
                        model_name=name,
                        model_type="2D",
                        input_type=schema_input_type(((0, "rpm"), (2, "xi"))),
                        data_path=Path(input_dir),
                        config={
                            "partition_mode": partition_mode,
                            "partition_key": str(partition_key),
                            "radial_mode": radial_mode,
                            "requested_n_splits": n_splits,
                            "target": target,
                            "train_samples": len(target_samples),
                            "include_gpr": include_gpr,
                        },
                        metrics=metrics_payload,
                        model_context=model_context,
                    )
                    model_path = partition_models_dir / f"{name}_{target}.pkl"
                    save_model_with_metadata(final_model, model_path, metadata)
                    saved_models += 1
                    saved_models_info.append(
                        {
                            **artifact_identity(model_context),
                            "model": name,
                            "path": str(model_path.relative_to(output)),
                            "train_samples": len(target_samples),
                            "sample_count": len(target_samples),
                            "metrics": metrics_payload,
                        }
                    )
                except Exception:
                    continue
    models_manifest_path = models_dir / "models_manifest.json"
    models_manifest_path.write_text(json.dumps(saved_models_info, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {saved_models} models to models_2d/")

    print("\n[6/7] Writing leaderboards...")
    aggregate_rows = build_aggregate_rows(rows, MODEL_CATEGORY)
    aggregate_path = output / f"leaderboard_2d_{radial_mode}_aggregate.json"
    aggregate_path.write_text(json.dumps(aggregate_rows, ensure_ascii=False, indent=2), encoding="utf-8")

    component_source_rows = [
        {
            "partition": row["partition"],
            "target": row["target"],
            "split_mode": row["split_mode"],
            "model": row["model"],
            "model_category": row["model_category"],
            "mae": row["mae"],
            "rmse": row["rmse"],
            "mape": row["mape"],
            "r2": row["r2"],
            "partition_size": row["partition_size"],
            "folds": row["folds"],
        }
        for row in rows
    ]

    report_path = output / f"benchmark_report_2d_{radial_mode}.md"
    report_content = build_benchmark_report(
        samples=samples,
        input_dir=input_dir,
        partition_mode=partition_mode,
        targets=targets,
        component_source_rows=component_source_rows,
        model_category=MODEL_CATEGORY,
        radial_mode=f"2d_{radial_mode}",
    )
    write_utf8_bom_text(report_path, report_content)

    print("\n[7/7] Done")
    return {
        "sample_count": len(samples),
        "model_count": len(models),
        "leaderboard_path": str(leaderboard_path),
        "aggregate_path": str(aggregate_path),
        "report_path": str(report_path),
        "models_dir": str(models_dir),
        "models_manifest_path": str(models_manifest_path),
        "target_plan_path": str(target_plan_path),
    }
