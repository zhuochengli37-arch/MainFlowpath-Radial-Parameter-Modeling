from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import json

import pytest
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression

from project1.experiments.benchmark_data import (
    SCHEMA_CONFIG,
    FlexSample,
    load_multi_section_samples,
    load_operating_conditions,
    resolve_partition_targets,
)
from project1.experiments.benchmark_partitioning import partition_samples, to_arrays, to_arrays_2d
from project1.experiments.benchmark_runners import run_benchmark, run_benchmark_2d
from project1.experiments.model_partitioning import (
    radial_partition,
    workline_partition,
)
from project1.modeling.factory import build_3d_models


FIXTURE_ROOT = Path(__file__).parent / "fixtures" / "institute_four"


class AlwaysFailRegressor(RegressorMixin, BaseEstimator):
    def fit(self, x, y):
        raise RuntimeError("intentional model failure")

    def predict(self, x):
        raise RuntimeError("unreachable")


def test_training_config_uses_dynamic_targets_with_strict_structural_checks():
    assert SCHEMA_CONFIG.schema_target_selection == "auto"
    assert SCHEMA_CONFIG.missing_target_policy == "error"


def _sample(
    *,
    component: str = "CMP",
    stage: int = 1,
    section: str | None = None,
    outputs: dict[str, float] | None = None,
    source_file: str = "case.dat",
    source_fields: dict[str, str] | None = None,
) -> FlexSample:
    actual_outputs = outputs or {"Cpt": 1.0}
    return FlexSample(
        component=component,
        family="compressor",
        station="MAIN",
        database=f"DATABASE_{component}",
        stage=stage,
        rpm=0.8,
        wcor=20.0,
        xi=0.5,
        source_path=source_file,
        outputs=actual_outputs,
        section=section,
        schema="institute_multi_section" if section else "institute_single_section",
        source_fields=source_fields or {"xi": "xi", **{name: name for name in actual_outputs}},
    )


def test_shared_partition_contract_preserves_single_and_separates_multi_sections():
    single = SimpleNamespace(component="CMP", stage=1, section=None)
    ri = SimpleNamespace(component="CMP", stage=1, section="RI")
    ro = SimpleNamespace(component="CMP", stage=1, section="RO")
    custom = SimpleNamespace(component="CMP", stage=1, section="MIXING_PLANE")

    assert str(workline_partition(single)) == "CMP:S1"
    assert str(workline_partition(ri)) == "CMP:S1"
    assert str(workline_partition(ro)) == "CMP:S1"
    assert str(radial_partition(single)) == "CMP:S1"
    assert str(radial_partition(ri)) == "CMP:S1:RI"
    assert str(radial_partition(ro)) == "CMP:S1:RO"
    assert str(radial_partition(custom)) == "CMP:S1:MIXING_PLANE"
    assert radial_partition(single).safe_name == "CMP_S1"
    assert radial_partition(ri).safe_name == "CMP_S1_RI"


def test_operating_condition_loader_emits_one_main_record_per_source_file():
    conditions = load_operating_conditions(FIXTURE_ROOT)
    radial_samples = load_multi_section_samples(FIXTURE_ROOT, radial_mode="full")

    assert len(conditions) == 4
    assert {condition.component for condition in conditions} == {"FAN", "CMP", "HPTB", "LPTB"}
    assert {condition.station for condition in conditions} == {"MAIN"}
    assert all(condition.section is None for condition in conditions)
    assert all(condition.schema_name == "institute_multi_section" for condition in conditions)
    assert all(condition.schema_version == "v1" for condition in conditions)
    assert {condition.speed_parameter_source_name for condition in conditions} == {"CNC"}
    assert {condition.flow_parameter_source_name for condition in conditions} == {
        "W2COR",
        "W25COR",
        "W4COR",
        "W5COR",
    }
    assert len(radial_samples) > len(conditions)
    assert len({condition.source_file for condition in conditions}) == len(conditions)


def test_multi_radial_loader_flattens_main_sections_without_boundary_files():
    samples = load_multi_section_samples(FIXTURE_ROOT, radial_mode="full")

    assert samples
    assert {sample.station for sample in samples} == {"MAIN"}
    assert {sample.component for sample in samples} == {"FAN", "CMP", "HPTB", "LPTB"}
    assert {sample.section for sample in samples} == {"RI", "RO", "SI", "SO"}
    assert not any("INLET" in sample.source_file or "OUTLET" in sample.source_file for sample in samples)
    assert {sample.xi for sample in samples if sample.component == "CMP" and sample.section == "RI"} == {
        0.01,
        0.1,
    }


def test_dynamic_targets_support_auto_explicit_and_unknown_alpha():
    samples = [
        _sample(
            section="CUSTOM",
            outputs={"Cpt": 1.0, "Alpha": 2.0},
            source_fields={"xi": "xi_CUSTOM", "Cpt": "Cpt_CUSTOM", "Alpha": "Alpha_CUSTOM"},
        )
    ]

    auto = resolve_partition_targets(samples, targets="auto")
    explicit = resolve_partition_targets(samples, targets=["Alpha"])

    assert auto.available_outputs == ("Cpt", "Alpha")
    assert auto.selected_targets == ("Cpt", "Alpha")
    assert explicit.available_outputs == ("Cpt", "Alpha")
    assert explicit.selected_targets == ("Alpha",)


def test_explicit_missing_target_is_not_silently_ignored():
    with pytest.raises(ValueError, match="MA"):
        resolve_partition_targets([_sample(section="RI")], targets=["MA"])


def test_different_partitions_can_have_different_available_outputs():
    compressor = resolve_partition_targets(
        [_sample(component="CMP", section="RI", outputs={"Cpt": 1.0, "Vz": 2.0, "Rho": 3.0})],
        targets="auto",
    )
    turbine = resolve_partition_targets(
        [_sample(component="HPTB", section="RI", outputs={"Cpt": 1.0, "MA": 0.2})],
        targets="auto",
    )

    assert compressor.selected_targets == ("Cpt", "Vz", "Rho")
    assert turbine.selected_targets == ("Cpt", "MA")


def test_same_partition_structural_mismatch_is_explicit_in_strict_mode():
    first = _sample(
        section="RI",
        outputs={"Cpt": 1.0, "Alpha": 2.0},
        source_file="first.dat",
    )
    second = _sample(section="RI", outputs={"Cpt": 1.1}, source_file="second.dat")

    with pytest.raises(ValueError, match="inconsistent available outputs"):
        resolve_partition_targets([first, second], targets="auto", missing_target_policy="error")

    plan = resolve_partition_targets([first, second], targets="auto", missing_target_policy="skip")
    assert plan.selected_targets == ("Cpt",)
    assert plan.skipped_targets == {"Alpha": "not available in every source of the partition"}


def test_radial_partitioning_and_existing_array_builders_support_independent_sections():
    ri = _sample(section="RI", outputs={"Alpha": 2.0}, source_file="ri.dat")
    ro = _sample(section="RO", outputs={"Alpha": 3.0}, source_file="ro.dat")
    ri.xi = 0.15
    ro.xi = 0.85

    partitions = partition_samples([ri, ro], "multi")

    assert set(partitions) == {"CMP:S1:RI", "CMP:S1:RO"}
    x_2d, _, _ = to_arrays_2d(partitions["CMP:S1:RI"], "Alpha")
    x_3d, _, _ = to_arrays(partitions["CMP:S1:RO"], "Alpha")
    assert x_2d.tolist() == [[0.8, 0.15]]
    assert x_3d.tolist() == [[0.8, 20.0, 0.85]]


def test_single_and_legacy_default_target_discovery_is_unchanged(tmp_path: Path):
    single = tmp_path / "DATABASE_CMP" / "STAGE_1" / "CNC_0.8" / "W25COR_20.dat"
    single.parent.mkdir(parents=True)
    single.write_text(
        "xi Cpt Ctt Cps Cts MA\n0.0 1.0 1.1 0.9 1.0 -999.0\n1.0 1.2 1.3 1.0 1.1 0.4\n",
        encoding="utf-8",
    )
    samples = __import__(
        "project1.experiments.benchmark_data",
        fromlist=["load_samples"],
    ).load_samples(str(tmp_path), "full")

    plan = resolve_partition_targets(samples, targets="auto")
    assert plan.available_outputs == ("Cpt", "Ctt", "Cps", "Cts", "MA")
    assert plan.selected_targets == plan.available_outputs
    assert "MA" not in samples[0].output_columns
    assert "MA" in samples[1].output_columns


def _runner_samples() -> list[FlexSample]:
    samples: list[FlexSample] = []
    for section, target, offset in (("RI", "Alpha", 0.0), ("RO", "Cpt", 1.0)):
        for index in range(8):
            sample = _sample(
                section=section,
                outputs={target: offset + index * 0.1},
                source_file=f"{section}_{index}.dat",
                source_fields={"xi": f"xi_{section}", target: f"{target}_{section}"},
            )
            sample.speed_parameter = 0.5 + index * 0.05
            sample.flow_parameter = 10.0 + index
            sample.xi = 0.02 + index * 0.1
            samples.append(sample)
    return samples


@pytest.mark.parametrize(
    ("runner", "model_dir", "dimension"),
    ((run_benchmark_2d, "models_2d", "2D"), (run_benchmark, "models", "3D")),
)
def test_multi_training_reuses_runner_and_saves_section_aware_metadata(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner,
    model_dir: str,
    dimension: str,
):
    samples = _runner_samples()
    monkeypatch.setattr(
        "project1.experiments.benchmark_runners.load_training_samples",
        lambda input_dir, radial_mode: samples,
    )
    monkeypatch.setattr(
        "project1.experiments.benchmark_runners._build_models",
        lambda include_gpr: {"linear": LinearRegression()},
    )
    output = tmp_path / dimension.lower()

    result = runner(
        input_dir=str(tmp_path / "unused"),
        output_dir=str(output),
        radial_mode="full",
        include_gpr=False,
        max_samples=100,
        partition_mode="multi",
        n_splits=2,
    )

    ri_model = output / model_dir / "CMP_S1_RI" / "linear_Alpha.pkl"
    ro_model = output / model_dir / "CMP_S1_RO" / "linear_Cpt.pkl"
    assert ri_model.exists()
    assert ro_model.exists()
    assert ri_model != ro_model
    ri_metadata = json.loads(ri_model.with_suffix(".json").read_text(encoding="utf-8"))
    assert ri_metadata["model_context"]["model_dimension"] == dimension
    assert ri_metadata["model_context"]["section"] == "RI"
    assert ri_metadata["model_context"]["partition"] == "CMP:S1:RI"
    assert ri_metadata["model_context"]["target"] == "Alpha"
    assert ri_metadata["model_context"]["schema_name"] == "institute_multi_section"
    assert ri_metadata["model_context"]["schema_version"] == "v1"
    assert ri_metadata["model_context"]["inputs"] == (
        ["speed_parameter", "xi"]
        if dimension == "2D"
        else ["speed_parameter", "flow_parameter", "xi"]
    )
    assert ri_metadata["model_context"]["training_sources"]
    manifest = json.loads(Path(result["models_manifest_path"]).read_text(encoding="utf-8"))
    assert {(row["partition"], row["target"]) for row in manifest} == {
        ("CMP:S1:RI", "Alpha"),
        ("CMP:S1:RO", "Cpt"),
    }
    assert {row["model_dimension"] for row in manifest} == {dimension}
    leaderboard = json.loads(Path(result["leaderboard_path"]).read_text(encoding="utf-8"))
    split_modes = {row["split_mode"] for row in leaderboard}
    if dimension == "3D":
        assert "condition" in split_modes
    else:
        assert "condition" not in split_modes
    target_plan = json.loads(Path(result["target_plan_path"]).read_text(encoding="utf-8"))
    assert {(row["partition"], tuple(row["selected_targets"])) for row in target_plan} == {
        ("CMP:S1:RI", ("Alpha",)),
        ("CMP:S1:RO", ("Cpt",)),
    }


@pytest.mark.parametrize(
    ("runner", "model_dir"),
    ((run_benchmark_2d, "models_2d"), (run_benchmark, "models")),
)
def test_single_section_runner_keeps_existing_partition_path(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    runner,
    model_dir: str,
):
    samples = []
    for index in range(8):
        sample = _sample(section=None, outputs={"Cpt": 1.0 + index * 0.1}, source_file=f"{index}.dat")
        sample.speed_parameter = 0.5 + index * 0.05
        sample.flow_parameter = 10.0 + index
        sample.xi = 0.02 + index * 0.1
        samples.append(sample)
    monkeypatch.setattr(
        "project1.experiments.benchmark_runners.load_training_samples",
        lambda input_dir, radial_mode: samples,
    )
    monkeypatch.setattr(
        "project1.experiments.benchmark_runners._build_models",
        lambda include_gpr: {"linear": LinearRegression()},
    )
    output = tmp_path / model_dir

    result = runner(
        input_dir=str(tmp_path / "unused"),
        output_dir=str(output),
        radial_mode="full",
        include_gpr=False,
        max_samples=100,
        partition_mode="multi",
        n_splits=2,
    )

    manifest = json.loads(Path(result["models_manifest_path"]).read_text(encoding="utf-8"))
    assert {row["partition"] for row in manifest} == {"CMP:S1"}
    assert {row["section"] for row in manifest} == {None}
    assert (output / model_dir / "CMP_S1" / "linear_Cpt.pkl").exists()
    assert not (output / model_dir / "CMP_S1_None").exists()


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_phase_2a1_models_run_through_direct_3d_benchmark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    samples: list[FlexSample] = []
    for condition in range(4):
        for xi in (0.0, 1.0):
            sample = _sample(
                section=None,
                outputs={"Cpt": 1.0 + condition * 0.1 + xi * 0.05},
                source_file=f"condition_{condition}.dat",
            )
            sample.speed_parameter = 0.7 + condition * 0.1
            sample.flow_parameter = 10.0 + condition
            sample.xi = xi
            samples.append(sample)
    models = build_3d_models(include_gpr=True)
    phase_models = {
        name: models[name]
        for name in ("kernel_ridge_rbf", "extra_trees", "ard_gpr")
    }
    monkeypatch.setattr(
        "project1.experiments.benchmark_runners.load_training_samples",
        lambda input_dir, radial_mode: samples,
    )
    monkeypatch.setattr(
        "project1.experiments.benchmark_runners._build_models",
        lambda include_gpr: phase_models,
    )

    result = run_benchmark(
        input_dir=str(tmp_path / "unused"),
        output_dir=str(tmp_path / "output"),
        radial_mode="full",
        include_gpr=True,
        max_samples=100,
        partition_mode="multi",
        n_splits=2,
    )

    manifest = json.loads(Path(result["models_manifest_path"]).read_text(encoding="utf-8"))
    leaderboard = json.loads(Path(result["leaderboard_path"]).read_text(encoding="utf-8"))
    assert result["model_count"] == 3
    assert {row["model"] for row in manifest} == set(phase_models)
    assert {
        row["model"] for row in leaderboard if row["split_mode"] == "condition"
    } == set(phase_models)


@pytest.mark.filterwarnings("ignore::sklearn.exceptions.ConvergenceWarning")
def test_remaining_phase_2a_models_run_through_direct_3d_benchmark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    for dependency in ("pygam", "interpret", "catboost", "ngboost"):
        pytest.importorskip(dependency)
    samples: list[FlexSample] = []
    for condition in range(6):
        for xi in (0.0, 0.5, 1.0):
            sample = _sample(
                section=None,
                outputs={"Cpt": 1.0 + condition * 0.1 + xi * 0.05},
                source_file=f"condition_{condition}.dat",
            )
            sample.speed_parameter = 0.7 + condition * 0.05
            sample.flow_parameter = 10.0 + condition
            sample.xi = xi
            samples.append(sample)
    models = build_3d_models(include_gpr=False)
    models["gam"].set_params(
        lazyimportregressor__model_kwargs={"n_splines": 5, "max_iter": 20}
    )
    models["catboost"].set_params(
        model_kwargs={
            "iterations": 10,
            "depth": 3,
            "random_seed": 42,
            "verbose": False,
            "allow_writing_files": False,
            "thread_count": 1,
        }
    )
    models["ebm"].set_params(
        model_kwargs={
            "interactions": 0,
            "max_rounds": 20,
            "outer_bags": 1,
            "early_stopping_rounds": 5,
            "random_state": 42,
            "n_jobs": 1,
        }
    )
    models["ngboost"].set_params(
        model_kwargs={
            "n_estimators": 10,
            "learning_rate": 0.05,
            "random_state": 42,
            "verbose": False,
        }
    )
    models["mlp"].set_params(
        mlpregressor__hidden_layer_sizes=(8,),
        mlpregressor__max_iter=50,
    )
    phase_models = {
        name: models[name]
        for name in ("gam", "catboost", "ebm", "ngboost", "mlp")
    }
    monkeypatch.setattr(
        "project1.experiments.benchmark_runners.load_training_samples",
        lambda input_dir, radial_mode: samples,
    )
    monkeypatch.setattr(
        "project1.experiments.benchmark_runners._build_models",
        lambda include_gpr: phase_models,
    )

    result = run_benchmark(
        input_dir=str(tmp_path / "unused"),
        output_dir=str(tmp_path / "output"),
        radial_mode="full",
        include_gpr=False,
        max_samples=100,
        partition_mode="multi",
        n_splits=2,
    )

    manifest = json.loads(Path(result["models_manifest_path"]).read_text(encoding="utf-8"))
    leaderboard = json.loads(Path(result["leaderboard_path"]).read_text(encoding="utf-8"))
    assert result["model_count"] == 5
    assert {row["model"] for row in manifest} == set(phase_models)
    assert {
        row["model"] for row in leaderboard if row["split_mode"] == "condition"
    } == set(phase_models)


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_phase_2b_models_run_through_direct_3d_benchmark(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    pytest.importorskip("pytabkit")
    pytest.importorskip("kan")
    samples: list[FlexSample] = []
    for condition in range(6):
        for xi in (0.0, 0.5, 1.0):
            sample = _sample(
                outputs={"Cpt": 1.0 + condition * 0.1 + xi * 0.05},
                source_file=f"condition_{condition}.dat",
            )
            sample.speed_parameter = 0.7 + condition * 0.05
            sample.flow_parameter = 10.0 + condition
            sample.xi = xi
            samples.append(sample)
    models = build_3d_models(include_gpr=False)
    models["realmlp"].set_params(model_kwargs={"n_epochs": 1, "batch_size": 16, "n_threads": 1, "random_state": 42, "device": "cpu", "verbosity": 0})
    models["tabm"].set_params(model_kwargs={"n_epochs": 1, "batch_size": 16, "tabm_k": 4, "d_block": 32, "n_blocks": 2, "compile_model": False, "n_threads": 1, "random_state": 42, "device": "cpu", "verbosity": 0})
    models["resnet_mlp"].set_params(model_kwargs={"max_epochs": 1, "batch_size": 16, "module_d": 32, "module_n_layers": 2, "n_threads": 1, "random_state": 42, "device": "cpu", "verbosity": 0})
    models["ft_transformer"].set_params(model_kwargs={"max_epochs": 1, "batch_size": 16, "module_d_token": 32, "module_n_layers": 2, "module_n_heads": 4, "n_threads": 1, "random_state": 42, "device": "cpu", "verbosity": 0})
    models["original_kan"].set_params(pykanregressor__hidden_width=8, pykanregressor__steps=1)
    phase_models = {
        name: models[name]
        for name in ("realmlp", "tabm", "original_kan", "resnet_mlp", "ft_transformer")
    }
    monkeypatch.setattr(
        "project1.experiments.benchmark_runners.load_training_samples",
        lambda input_dir, radial_mode: samples,
    )
    monkeypatch.setattr(
        "project1.experiments.benchmark_runners._build_models",
        lambda include_gpr: phase_models,
    )
    monkeypatch.setattr(
        "project1.experiments.benchmark_runners.cv_splits",
        lambda rows, mode, n_splits: [(list(range(12)), list(range(12, 18)))] if mode == "condition" else [],
    )

    result = run_benchmark(
        input_dir=str(tmp_path / "unused"),
        output_dir=str(tmp_path / "output"),
        radial_mode="full",
        include_gpr=False,
        max_samples=100,
        partition_mode="multi",
        n_splits=2,
    )

    manifest = json.loads(Path(result["models_manifest_path"]).read_text(encoding="utf-8"))
    leaderboard = json.loads(Path(result["leaderboard_path"]).read_text(encoding="utf-8"))
    assert result["failure_count"] == 0
    assert {row["model"] for row in manifest} == set(phase_models)
    assert {row["model"] for row in leaderboard} == set(phase_models)


def test_tabiclv2_runs_through_direct_3d_benchmark_from_offline_cache(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    pytest.importorskip("tabicl")
    hub = pytest.importorskip("huggingface_hub")
    checkpoint = hub.try_to_load_from_cache(
        "jingang/TabICL", "tabicl-regressor-v2-20260212.ckpt"
    )
    if not isinstance(checkpoint, str):
        pytest.skip("TabICLv2 regression checkpoint is not cached")
    samples: list[FlexSample] = []
    for condition in range(6):
        for xi in (0.0, 0.5, 1.0):
            sample = _sample(
                outputs={"Cpt": 1.0 + condition * 0.1 + xi * 0.05},
                source_file=f"condition_{condition}.dat",
            )
            sample.speed_parameter = 0.7 + condition * 0.05
            sample.flow_parameter = 10.0 + condition
            sample.xi = xi
            samples.append(sample)
    model = build_3d_models(include_gpr=False)["tabiclv2"]
    model.set_params(
        model_kwargs={
            "n_estimators": 1,
            "batch_size": 1,
            "device": "cpu",
            "use_amp": False,
            "use_fa3": False,
            "offload_mode": False,
            "allow_auto_download": False,
            "random_state": 42,
            "n_jobs": 1,
            "verbose": False,
        }
    )
    monkeypatch.setattr(
        "project1.experiments.benchmark_runners.load_training_samples",
        lambda input_dir, radial_mode: samples,
    )
    monkeypatch.setattr(
        "project1.experiments.benchmark_runners._build_models",
        lambda include_gpr: {"tabiclv2": model},
    )
    monkeypatch.setattr(
        "project1.experiments.benchmark_runners.cv_splits",
        lambda rows, mode, n_splits: [(list(range(12)), list(range(12, 18)))] if mode == "condition" else [],
    )

    result = run_benchmark(
        input_dir=str(tmp_path / "unused"),
        output_dir=str(tmp_path / "output"),
        radial_mode="full",
        include_gpr=False,
        max_samples=100,
        partition_mode="multi",
        n_splits=2,
    )

    manifest = json.loads(Path(result["models_manifest_path"]).read_text(encoding="utf-8"))
    leaderboard = json.loads(Path(result["leaderboard_path"]).read_text(encoding="utf-8"))
    assert result["failure_count"] == 0
    assert {row["model"] for row in manifest} == {"tabiclv2"}
    assert {row["model"] for row in leaderboard} == {"tabiclv2"}


def test_direct_3d_benchmark_records_model_failures_without_stopping(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
):
    samples = []
    for condition in range(4):
        for xi in (0.0, 1.0):
            sample = _sample(
                outputs={"Cpt": 1.0 + condition * 0.1 + xi * 0.05},
                source_file=f"condition_{condition}.dat",
            )
            sample.speed_parameter = 0.7 + condition * 0.1
            sample.flow_parameter = 10.0 + condition
            sample.xi = xi
            samples.append(sample)
    monkeypatch.setattr(
        "project1.experiments.benchmark_runners.load_training_samples",
        lambda input_dir, radial_mode: samples,
    )
    monkeypatch.setattr(
        "project1.experiments.benchmark_runners._build_models",
        lambda include_gpr: {"linear": LinearRegression(), "broken": AlwaysFailRegressor()},
    )
    monkeypatch.setattr(
        "project1.experiments.benchmark_runners.cv_splits",
        lambda rows, mode, n_splits: [(list(range(4)), list(range(4, 8)))] if mode == "condition" else [],
    )

    result = run_benchmark(
        input_dir=str(tmp_path / "unused"),
        output_dir=str(tmp_path / "output"),
        radial_mode="full",
        include_gpr=False,
        max_samples=100,
        partition_mode="multi",
        n_splits=2,
    )

    failures = json.loads(Path(result["failures_path"]).read_text(encoding="utf-8"))
    leaderboard = json.loads(Path(result["leaderboard_path"]).read_text(encoding="utf-8"))
    manifest = json.loads(Path(result["models_manifest_path"]).read_text(encoding="utf-8"))
    assert failures
    assert {"model_key", "status", "exception_type", "reason", "stage"} <= failures[0].keys()
    assert all(row["model_key"] == "broken" for row in failures)
    assert {row["model"] for row in leaderboard} == {"linear"}
    assert {row["model"] for row in manifest} == {"linear"}
