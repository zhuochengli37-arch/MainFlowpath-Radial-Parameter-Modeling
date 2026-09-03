from __future__ import annotations

import json
from pathlib import Path

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
    cv_splits,
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


def _build_models(include_gpr: bool = True) -> dict[str, object]:
    return build_3d_models(include_gpr=include_gpr)


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
        rng = np.random.default_rng(42)
        idx = rng.choice(np.arange(len(samples)), size=max_samples, replace=False)
        samples = [samples[i] for i in sorted(idx)]
    print(f"Loaded {len(samples)} samples")

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    print("\n[2/7] Building model candidates...")
    models = _build_models(include_gpr=include_gpr)
    print(f"Loaded {len(models)} models")

    split_modes = ("random", "rpm", "wcor")
    rows: list[dict[str, object]] = []

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
                        for train_idx, test_idx in splits:
                            if len(test_idx) == 0 or len(train_idx) == 0:
                                continue
                            x_train, y_train = x[train_idx], y[train_idx]
                            x_test, y_test = x[test_idx], y[test_idx]
                            if name == "gpr_matern" and len(train_idx) > 800:
                                rng = np.random.default_rng(42)
                                selected = rng.choice(train_idx, size=800, replace=False)
                                x_train, y_train = x[selected], y[selected]
                            try:
                                candidate = prepare_model_for_fold(model, len(train_idx))
                                candidate.fit(x_train, y_train)
                                pred = candidate.predict(x_test)
                            except Exception:
                                continue
                            fold_metrics.append(metrics(y_test, pred))
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
                        },
                        model_context=model_context,
                    )
                    model_path = partition_models_dir / f"{name}_{target}.pkl"
                    save_model_with_metadata(final_model, model_path, metadata)
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
    models = _build_models(include_gpr=include_gpr)
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
                            if name == "gpr_matern" and len(train_idx) > 800:
                                rng = np.random.default_rng(42)
                                selected = rng.choice(train_idx, size=800, replace=False)
                                x_train, y_train = x[selected], y[selected]
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
