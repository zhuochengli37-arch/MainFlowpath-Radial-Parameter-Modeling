from __future__ import annotations

import os
import runpy
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable, Iterator


PROJECT_ROOT = Path(__file__).resolve().parents[3]
WORKBASE_SRC = PROJECT_ROOT / "workbase" / "src"

_WORKFLOW_MODULES: dict[str, str] = {
    "current_benchmark_1d": "workbase.current_scripts.benchmark_1d",
    "current_predict_1d": "workbase.current_scripts.predict_1d",
    "current_benchmark_2d": "workbase.current_scripts.benchmark_2d",
    "current_workline_pipeline": "workbase.current_scripts.workline_pipeline",
    "current_predict_2d": "workbase.current_scripts.predict_2d",
    "current_benchmark_3d": "workbase.current_scripts.benchmark_3d",
    "current_predict_3d": "workbase.current_scripts.predict_3d",
    "generic_benchmark_1d": "workbase.generic_scripts.generic_benchmark_1d",
    "generic_predict_1d": "workbase.generic_scripts.generic_predict_1d",
    "generic_benchmark_2d": "workbase.generic_scripts.generic_benchmark_2d",
    "generic_predict_2d": "workbase.generic_scripts.generic_predict_2d",
    "generic_benchmark_3d": "workbase.generic_scripts.generic_benchmark_3d",
    "generic_predict_3d": "workbase.generic_scripts.generic_predict_3d",
}


def workflow_names() -> tuple[str, ...]:
    return tuple(_WORKFLOW_MODULES)


def resolve_workflow_module(workflow_name: str) -> str:
    try:
        return _WORKFLOW_MODULES[workflow_name]
    except KeyError as exc:
        available = ", ".join(sorted(_WORKFLOW_MODULES))
        raise ValueError(f"unknown workflow '{workflow_name}', available: {available}") from exc


def ensure_project_paths() -> None:
    for candidate in (PROJECT_ROOT, WORKBASE_SRC):
        candidate_str = str(candidate)
        if candidate_str not in sys.path:
            sys.path.insert(0, candidate_str)


@contextmanager
def project_root_context(argv: Iterable[str] | None = None) -> Iterator[None]:
    ensure_project_paths()
    original_cwd = Path.cwd()
    original_argv = sys.argv[:]
    try:
        os.chdir(PROJECT_ROOT)
        if argv is not None:
            sys.argv = list(argv)
        yield
    finally:
        sys.argv = original_argv
        os.chdir(original_cwd)


def run_workflow(
    workflow_name: str,
    argv: Iterable[str] | None = None,
    prog_name: str | None = None,
) -> int:
    module_name = resolve_workflow_module(workflow_name)
    forwarded = list(argv or ())
    display_name = prog_name or workflow_name
    with project_root_context([display_name, *forwarded]):
        try:
            runpy.run_module(module_name, run_name="__main__", alter_sys=False)
        except SystemExit as exc:
            if exc.code is None:
                return 0
            if isinstance(exc.code, int):
                return exc.code
            raise
    return 0
