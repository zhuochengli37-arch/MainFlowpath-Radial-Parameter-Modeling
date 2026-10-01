"""Runner executed by the isolated official TabPFN Python environment."""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import importlib.metadata
import json
import os
import platform
import time
from pathlib import Path

import numpy as np
import torch
from tabpfn import TabPFNRegressor
from tabpfn.constants import ModelVersion


MODEL_ID = "Prior-Labs/tabpfn_3_5"
MODEL_VERSION = "v3.5-fast"
CHECKPOINT_FILENAME = "tabpfn-v3.5-fast-20260909.safetensors"
CHECKPOINT_REVISION = "06bf2ba35c80a92a3b9abb436b99cf49e7a0365e"
CHECKPOINT_SHA256 = "14e0b81285bca7a6d1ef121169f27f651f76dff49cad6c0b8779d5c836018550"
WEIGHTS_LICENSE = "tabpfn-3-5-license-v1.0 (non-commercial/non-production)"


def _version(package: str) -> str:
    return importlib.metadata.version(package)


def _windows_peak_rss_mib() -> float | None:
    if os.name != "nt":
        return None
    try:
        from ctypes import wintypes

        class ProcessMemoryCounters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
                ("PrivateUsage", ctypes.c_size_t),
            ]

        counters = ProcessMemoryCounters()
        counters.cb = ctypes.sizeof(counters)
        get_process = ctypes.windll.kernel32.GetCurrentProcess
        get_process.restype = wintypes.HANDLE
        process = get_process()
        get_memory = ctypes.windll.psapi.GetProcessMemoryInfo
        get_memory.argtypes = [
            wintypes.HANDLE,
            ctypes.POINTER(ProcessMemoryCounters),
            wintypes.DWORD,
        ]
        get_memory.restype = wintypes.BOOL
        if not get_memory(
            process, ctypes.byref(counters), counters.cb
        ):
            return None
        return counters.PeakWorkingSetSize / (1024 * 1024)
    except (AttributeError, OSError):
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _data_identity(x: np.ndarray, y: np.ndarray, condition_ids: np.ndarray) -> str:
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(x).tobytes())
    digest.update(np.ascontiguousarray(y).tobytes())
    digest.update("\0".join(condition_ids.tolist()).encode("utf-8"))
    return digest.hexdigest()


def _validate_checkpoint(path: Path) -> None:
    if path.name != CHECKPOINT_FILENAME or _sha256(path) != CHECKPOINT_SHA256:
        raise ValueError(f"TabPFN checkpoint identity mismatch: {path}")


def train(
    input_path: Path,
    config_path: Path,
    checkpoint: Path,
    artifact_dir: Path,
    response_path: Path,
) -> None:
    arrays = np.load(input_path, allow_pickle=False)
    x = np.asarray(arrays["X"], dtype=float)
    y = np.asarray(arrays["y"], dtype=float)
    condition_ids = np.asarray(arrays["condition_ids"], dtype="U")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    context = config.pop("context")
    feature_names = config.pop("feature_names")
    torch.set_num_threads(config.pop("torch_num_threads"))
    _validate_checkpoint(checkpoint)

    # Keep the official fitted-state artifact relocatable within another prepared env.
    original_directory = Path.cwd()
    os.chdir(checkpoint.parent)
    try:
        model = TabPFNRegressor.create_default_for_version(
            ModelVersion.V3_5_FAST,
            model_path=checkpoint.name,
            device="cpu",
            show_progress_bar=False,
            **config,
        )
        started = time.perf_counter()
        model.fit(x, y)
        fit_time = time.perf_counter() - started
        artifact_dir.mkdir()
        fit_state = artifact_dir / "model.tabpfn_fit"
        model.save_fit_state(fit_state)
    finally:
        os.chdir(original_directory)

    metadata_context = dict(context)
    metadata_context.pop("train_condition_ids", None)
    metadata = {
        "model_id": MODEL_ID,
        "model_version": MODEL_VERSION,
        "checkpoint_filename": CHECKPOINT_FILENAME,
        "checkpoint_revision": CHECKPOINT_REVISION,
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_size_bytes": checkpoint.stat().st_size,
        "checkpoint_cache_path": str(checkpoint),
        "code_license": "Apache-2.0",
        "weights_license": WEIGHTS_LICENSE,
        "feature_names": feature_names,
        "input_scaling": "none externally; official TabPFN preprocessing is internal",
        "target_preprocessing": (
            "official internal regression target transforms; predictions are returned "
            "in the original target space"
        ),
        "training_sample_count": len(x),
        "training_condition_count": len(np.unique(condition_ids)) if condition_ids.size else None,
        "training_context_sha256": _data_identity(x, y, condition_ids),
        "context": metadata_context,
        "fit_context_time_seconds": fit_time,
        "fit_state_size_bytes": fit_state.stat().st_size,
        "configuration": config,
        "device": "cpu",
        "cuda_available": torch.cuda.is_available(),
        "torch_num_threads": torch.get_num_threads(),
        "process_count": 1,
        "peak_rss_mib": _windows_peak_rss_mib(),
        "environment_identifier": ".venv-tabpfn",
        "offline_mode": os.environ.get("HF_HUB_OFFLINE") == "1",
        "versions": {
            "python": platform.python_version(),
            "tabpfn": _version("tabpfn"),
            "torch": torch.__version__,
            "numpy": _version("numpy"),
            "scipy": _version("scipy"),
            "pandas": _version("pandas"),
            "scikit-learn": _version("scikit-learn"),
        },
    }
    (artifact_dir / "project1_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    response_path.write_text(json.dumps(metadata, ensure_ascii=False), encoding="utf-8")


def predict(
    input_path: Path,
    checkpoint: Path,
    artifact_dir: Path,
    output_path: Path,
    torch_num_threads: int,
) -> None:
    arrays = np.load(input_path, allow_pickle=False)
    x = np.asarray(arrays["X"], dtype=float)
    torch.set_num_threads(torch_num_threads)
    _validate_checkpoint(checkpoint)
    original_directory = Path.cwd()
    os.chdir(checkpoint.parent)
    try:
        started = time.perf_counter()
        model = TabPFNRegressor.load_from_fit_state(
            artifact_dir / "model.tabpfn_fit", device="cpu"
        )
        load_time = time.perf_counter() - started
        started = time.perf_counter()
        prediction = np.asarray(model.predict(x), dtype=float).reshape(-1)
        inference_time = time.perf_counter() - started
    finally:
        os.chdir(original_directory)
    if len(prediction) != len(x) or not np.isfinite(prediction).all():
        raise ValueError("TabPFN returned invalid predictions")
    np.savez_compressed(
        output_path,
        prediction=prediction,
        load_time_seconds=load_time,
        inference_time_seconds=inference_time,
        peak_rss_mib=_windows_peak_rss_mib(),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="command", required=True)
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument("--input", type=Path, required=True)
    train_parser.add_argument("--config", type=Path, required=True)
    train_parser.add_argument("--checkpoint", type=Path, required=True)
    train_parser.add_argument("--artifact-dir", type=Path, required=True)
    train_parser.add_argument("--response", type=Path, required=True)
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument("--input", type=Path, required=True)
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    predict_parser.add_argument("--artifact-dir", type=Path, required=True)
    predict_parser.add_argument("--output", type=Path, required=True)
    predict_parser.add_argument("--torch-num-threads", type=int, required=True)
    args = parser.parse_args()
    if args.command == "train":
        train(args.input, args.config, args.checkpoint, args.artifact_dir, args.response)
    else:
        predict(
            args.input,
            args.checkpoint,
            args.artifact_dir,
            args.output,
            args.torch_num_threads,
        )


if __name__ == "__main__":
    main()
