"""Freeze a real-data inventory and reuse one existing condition split across cases."""

from __future__ import annotations

from collections import Counter, defaultdict
import hashlib
import json
import csv
from contextlib import contextmanager
from datetime import datetime, timezone
import importlib.metadata
import os
from pathlib import Path
import shutil
import socket
import subprocess
import sys
from uuid import uuid4

import numpy as np

from project1.experiments.benchmark_data import iter_data_files, load_training_samples, resolve_partition_targets
from project1.experiments.benchmark_partitioning import condition_train_validation_test_split, partition_samples, to_arrays
from project1.services.meta_parser import parse_metadata_from_path


def json_fingerprint(value: object) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode("utf-8")).hexdigest()


def write_benchmark_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@contextmanager
def formal_experiment_lock(output: Path, *, resume: bool, recover_lock: bool = False):
    """OS lock excludes writers; a crash marker additionally requires human acknowledgement."""
    output.mkdir(parents=True, exist_ok=True)
    path = output / ".benchmark.lock"
    with path.open("a+b") as stream:
        try:
            stream.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            raise RuntimeError(f"benchmark writer is active; cannot lock {path}") from exc
        try:
            stream.seek(0)
            previous = stream.read()
            if previous:
                try:
                    owner = json.loads(previous)
                except (ValueError, UnicodeDecodeError) as exc:
                    raise RuntimeError(f"ambiguous benchmark lock; inspect manually: {path}") from exc
                if not isinstance(owner, dict):
                    raise RuntimeError(f"ambiguous benchmark lock; inspect manually: {path}")
                if owner.get("state") != "closed":
                    if not (resume and recover_lock and owner.get("state") == "active"):
                        raise RuntimeError(f"unclean benchmark lock; inspect PID/host and child processes, then use --resume --recover-lock: {path}")
                    write_benchmark_json(output / "history" / "locks" / f"{uuid4().hex}.json", owner)
            elif recover_lock:
                raise RuntimeError("--recover-lock requires an existing unclean lock to inspect")
            owner = {"pid": os.getpid(), "host": socket.gethostname(), "state": "active",
                     "started_at": datetime.now(timezone.utc).isoformat()}

            def store_owner():
                stream.seek(0)
                stream.truncate()
                stream.write(json.dumps(owner).encode("utf-8"))
                stream.flush()
                os.fsync(stream.fileno())

            store_owner()
            try:
                yield
            finally:
                owner.update(state="closed", closed_at=datetime.now(timezone.utc).isoformat())
                store_owner()
        finally:
            stream.seek(0)
            if os.name == "nt":
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


def formal_runtime_identity(models: dict[str, object], root: Path) -> dict:
    """Read local environments and weights only; never import/train/download a model."""
    probe = ("import importlib.metadata as m,json,sys; "
             "print(json.dumps({'python':sys.version,'packages':"
             "{d.metadata['Name'].lower():d.version for d in m.distributions()}}))")
    identity = {"main": {"python": sys.version, "executable": str(Path(sys.executable).resolve()),
                         "packages": {d.metadata["Name"].lower(): d.version for d in importlib.metadata.distributions()}},
                "thread_environment": {key: os.environ.get(key) for key in
                                       ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")}}
    for key in ("pysr", "tabpfn"):
        if key not in models:
            continue
        model = models[key]
        python = model._python_path()
        environment = model._runner_environment(python)
        result = subprocess.run([str(python), "-c", probe], capture_output=True, text=True, check=True, env=environment)
        identity[key] = {"executable": str(python), "environment": json.loads(result.stdout)}
        if key == "tabpfn":
            checkpoint = model._checkpoint(python)
            identity[key]["checkpoint_sha256"] = file_sha256(checkpoint)
        else:
            identity[key]["julia_executable_sha256"] = file_sha256(Path(environment["PYTHON_JULIACALL_EXE"]))
            identity[key]["julia_project"] = {name: file_sha256(root / "envs/pysr-julia" / name)
                                               for name in ("Project.toml", "Manifest.toml")}
    if "tabiclv2" in models:
        from huggingface_hub import try_to_load_from_cache
        kwargs = models["tabiclv2"].get_params()["model_kwargs"]
        checkpoint = kwargs.get("checkpoint_path") or try_to_load_from_cache(
            "jingang/TabICL", kwargs["checkpoint_version"])
        if not isinstance(checkpoint, str) or not Path(checkpoint).is_file():
            raise FileNotFoundError("stage the TabICLv2 checkpoint locally before starting/resuming")
        identity["tabiclv2"] = {"checkpoint_version": kwargs["checkpoint_version"],
                                "checkpoint_sha256": file_sha256(Path(checkpoint))}
    sources = [*sorted((root / "workbase/src/project1/modeling").glob("*.py")),
               *sorted((root / "workbase/src/project1/experiments").glob("benchmark_*.py")),
               root / "workbase/common/model_versioning.py"]
    # Git normalizes source newlines; this does not change the model implementation.
    identity["implementation_sha256"] = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_text(encoding="utf-8").encode("utf-8")).hexdigest()
        for p in sources}
    return identity


def formal_run_paths(output: Path, task: dict) -> list[Path]:
    from project1.experiments.model_partitioning import safe_partition_name

    partition, target = task["case_id"].rsplit(":", 1)
    model = output / "models" / safe_partition_name(partition) / f"{task['model_key']}_{target}_seed_{task['model_seed']}.pkl"
    return [model, model.with_suffix(".json"), output / "predictions" / f"{task['run_id']}.npz"]


def write_formal_record(output: Path, record: dict) -> None:
    record.pop("record_sha256", None)
    record["record_sha256"] = json_fingerprint(record)
    write_benchmark_json(output / "records" / f"{record['run_id']}.json", record)


def load_formal_records(output: Path, experiment: dict) -> dict[str, dict]:
    """One canonical record per planned run; damaged successes never count as complete."""
    plan = {t["run_id"]: t for t in experiment.get("run_plan", [])}
    records = {}
    for path in sorted((output / "records").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        run_id = record["run_id"]
        if run_id in records or path.name != f"{run_id}.json":
            raise ValueError(f"duplicate/noncanonical run record: {path}")
        if plan:
            if run_id not in plan or any(record.get(k) != v for k, v in plan[run_id].items()):
                raise ValueError(f"record does not match frozen run plan: {path}")
            if record.get("experiment_contract_sha256") != experiment["contract_sha256"]:
                raise ValueError(f"record experiment identity differs: {path}")
            if record["status"] == "passed":
                try:
                    body = {k: v for k, v in record.items() if k != "record_sha256"}
                    if record.get("record_sha256") != json_fingerprint(body):
                        raise ValueError("canonical record checksum differs")
                    paths = formal_run_paths(output, record)
                    expected = {p.relative_to(output).as_posix() for p in paths}
                    if set(record.get("artifact_sha256", {})) != expected:
                        raise ValueError("required artifact checksums missing")
                    recorded_paths = [record["metadata"]["model_path"], record["metadata"]["metadata_path"], record["prediction_path"]]
                    if recorded_paths != [p.relative_to(output).as_posix() for p in paths]:
                        raise ValueError("artifact paths differ from run identity")
                    for artifact in paths:
                        if file_sha256(artifact) != record["artifact_sha256"][artifact.relative_to(output).as_posix()]:
                            raise ValueError(f"artifact checksum differs: {artifact}")
                    if set(record["evaluations"]) != {"validation", "test"}:
                        raise ValueError("incomplete evaluation record")
                except (OSError, ValueError, KeyError) as exc:
                    record = {**record, "status": "corrupt", "failure_stage": "integrity", "reason": str(exc)}
        records[run_id] = record
    return records


def archive_formal_attempt(output: Path, task: dict) -> None:
    """Preserve the entire previous attempt before replacing any canonical artifact."""
    archive = output / "history" / task["run_id"] / uuid4().hex
    paths = [output / "records" / f"{task['run_id']}.json", *formal_run_paths(output, task)]
    paths += [p.with_name(p.stem + ".staging" + p.suffix) for p in formal_run_paths(output, task)]
    for path in paths:
        if path.exists():
            destination = archive / path.relative_to(output)
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, destination)
    write_benchmark_json(archive / "retry_authorization.json", {
        "run_id": task["run_id"], "authorized_at": datetime.now(timezone.utc).isoformat(),
        "policy": "explicit --retry-runs; replaces the canonical attempt without adding a seed or choosing best results"})


def formal_condition_id(sample: object) -> str:
    # Match the existing condition fallback's twelve-significant-digit contract.
    return f"{sample.rpm:.12g}|{sample.wcor:.12g}"


def formal_cases(samples: list[object]) -> dict[str, tuple[np.ndarray, np.ndarray, list[object]]]:
    cases = {}
    for partition, subset in sorted(partition_samples(samples, "multi").items()):
        for target in resolve_partition_targets(subset, targets="auto", missing_target_policy="error").selected_targets:
            x, y, usable = to_arrays(subset, target)
            if len(usable) != len(subset) or not np.isfinite(x).all() or not np.isfinite(y).all():
                raise ValueError(f"incomplete/non-finite formal case: {partition}:{target}")
            points = [(formal_condition_id(s), s.xi) for s in usable]
            if len(set(points)) != len(points):
                raise ValueError(f"duplicate condition/xi: {partition}:{target}")
            cases[f"{partition}:{target}"] = (x, y, usable)
    if not cases:
        raise ValueError("no usable formal benchmark cases")
    return cases


def inspect_formal_dataset(input_dir: str, *, split_seed: int = 42, validation_size: float = 0.2,
                           test_size: float = 0.2) -> tuple[list[object], dict[str, object]]:
    root = Path(input_dir).resolve()
    inventory = []
    inventory_hasher = hashlib.sha256()
    for path in iter_data_files(str(root)):
        relative = path.relative_to(root).as_posix()
        digest = hashlib.sha256(path.read_bytes()).digest()
        inventory_hasher.update(relative.encode("utf-8") + b"\0" + digest)
        meta = parse_metadata_from_path(str(path))
        inventory.append({"path": relative, "sha256": digest.hex(), "station": meta["station"],
                          "stage": meta["stage"], "n": meta["speed_parameter"],
                          "second_parameter": meta["flow_parameter"],
                          "second_parameter_name": meta["flow_parameter_name"]})
    samples = load_training_samples(str(root), "full")
    cases = formal_cases(samples)
    first_samples = next(iter(cases.values()))[2]
    indices = condition_train_validation_test_split(first_samples, validation_size=validation_size,
                                                     test_size=test_size, random_state=split_seed)
    splits = {name: sorted({formal_condition_id(first_samples[i]) for i in index})
              for name, index in zip(("train", "validation", "test"), indices)}
    universe = set().union(*(set(values) for values in splits.values()))
    condition_sources = defaultdict(set)
    for sample in samples:
        condition_sources[formal_condition_id(sample)].add(Path(sample.source_path).relative_to(root).as_posix())
    conditions = []
    for identifier in sorted(universe):
        source = next(s for s in first_samples if formal_condition_id(s) == identifier)
        conditions.append({"condition_id": identifier, "n": source.rpm, "second_parameter": source.wcor,
                           "second_parameter_name": source.flow_parameter_name,
                           "subset": next(name for name, ids in splits.items() if identifier in ids),
                           "source_files": sorted(condition_sources[identifier])})
    case_inventory = []
    for case_id, (_, y, subset) in cases.items():
        counts = Counter(formal_condition_id(s) for s in subset)
        if set(counts) != universe:
            raise ValueError(f"formal cases do not share one condition universe: {case_id}")
        first = subset[0]
        case_inventory.append({"case_id": case_id, "component": first.component, "stage": first.stage,
                               "section": first.section, "target": case_id.rsplit(":", 1)[1],
                               "sample_count": len(subset), "condition_count": len(counts),
                               "profile_row_counts": sorted(set(counts.values())),
                               "subset_rows": {name: sum(counts[c] for c in ids) for name, ids in splits.items()},
                               "target_min": float(y.min()), "target_max": float(y.max()),
                               "negative_values": int(np.sum(y < 0)),
                               "negative_condition_count": len({formal_condition_id(s) for s, value in zip(subset, y) if value < 0})})
    instance = {"dataset_root_hint": str(root), "inventory_sha256": inventory_hasher.hexdigest(),
                "files": inventory, "sample_count": len(samples), "conditions": conditions,
                "cases": case_inventory,
                "split": {"split_seed": split_seed, "validation_size": validation_size, "test_size": test_size,
                          "condition_key": "n|second_parameter (.12g)", "conditions": splits,
                          "condition_counts": {name: len(ids) for name, ids in splits.items()},
                          "condition_hashes": {name: json_fingerprint(ids) for name, ids in splits.items()},
                          "sha256": json_fingerprint(splits)},
                "sample_efficiency": {"instantiated": False, "count_rounding_policy": "pending"}}
    return samples, instance


def freeze_or_verify_instance(path: Path, instance: dict[str, object]) -> dict[str, object]:
    payload = {**instance, "instance_sha256": json_fingerprint(instance)}
    if path.exists():
        saved = json.loads(path.read_text(encoding="utf-8"))
        if saved != payload:
            raise ValueError(f"frozen instance differs from current data/configuration: {path}")
    else:
        write_benchmark_json(path, payload)
    with path.with_suffix(".csv").open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=("condition_id", "n", "second_parameter", "second_parameter_name", "subset", "source_files"))
        writer.writeheader()
        for condition in instance["conditions"]:
            writer.writerow({**condition, "source_files": json.dumps(condition["source_files"], ensure_ascii=False)})
    return payload
