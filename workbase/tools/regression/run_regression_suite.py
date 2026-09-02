from __future__ import annotations

import argparse
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
PYTHON_EXE = PROJECT_ROOT / ".venv" / "Scripts" / "python.exe"


@dataclass(frozen=True)
class Step:
    name: str
    script_path: str
    group: str


CURRENT_STEPS = [
    Step("current_benchmark_1d", "workbase/current_scripts/benchmark_1d.py", "current"),
    Step("current_predict_1d", "workbase/current_scripts/predict_1d.py", "current"),
    Step("current_benchmark_2d", "workbase/current_scripts/benchmark_2d.py", "current"),
    Step("current_predict_2d", "workbase/current_scripts/predict_2d_with_workline.py", "current"),
    Step("current_benchmark_3d", "workbase/current_scripts/benchmark_3d.py", "current"),
    Step("current_predict_3d", "workbase/current_scripts/predict_3d.py", "current"),
]

GENERIC_STEPS = [
    Step("generic_benchmark_1d", "workbase/generic_scripts/generic_benchmark_1d.py", "generic"),
    Step("generic_predict_1d", "workbase/generic_scripts/generic_predict_1d.py", "generic"),
    Step("generic_benchmark_2d", "workbase/generic_scripts/generic_benchmark_2d.py", "generic"),
    Step("generic_predict_2d", "workbase/generic_scripts/generic_predict_2d.py", "generic"),
    Step("generic_benchmark_3d", "workbase/generic_scripts/generic_benchmark_3d.py", "generic"),
    Step("generic_predict_3d", "workbase/generic_scripts/generic_predict_3d.py", "generic"),
]

ALL_STEPS = CURRENT_STEPS + GENERIC_STEPS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="运行 current/generic 1D/2D/3D 回归测试套件")
    parser.add_argument(
        "--group",
        choices=["all", "current", "generic"],
        default="all",
        help="选择要运行的工作流分组",
    )
    parser.add_argument(
        "--fail-fast",
        action="store_true",
        help="遇到第一个失败步骤后立即停止",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="仅打印可用步骤，不实际执行",
    )
    return parser.parse_args()


def select_steps(group: str) -> list[Step]:
    if group == "current":
        return CURRENT_STEPS
    if group == "generic":
        return GENERIC_STEPS
    return ALL_STEPS


def ensure_python() -> Path:
    if PYTHON_EXE.exists():
        return PYTHON_EXE
    return Path(sys.executable)


def run_step(step: Step, python_exe: Path) -> tuple[bool, float]:
    script = PROJECT_ROOT / step.script_path
    command = [str(python_exe), str(script)]
    print("=" * 78)
    print(f"[运行] {step.name}")
    print(f"      {' '.join(command)}")
    started = time.perf_counter()
    completed = subprocess.run(command, cwd=PROJECT_ROOT)
    elapsed = time.perf_counter() - started
    ok = completed.returncode == 0
    status = "成功" if ok else "失败"
    print(f"[{status}] {step.name} ({elapsed:.1f}s)")
    return ok, elapsed


def main() -> int:
    args = parse_args()
    python_exe = ensure_python()
    steps = select_steps(args.group)

    if args.list:
        print(f"Python: {python_exe}")
        for step in steps:
            print(f"{step.name}: {step.script_path}")
        return 0

    print(f"项目根目录: {PROJECT_ROOT}")
    print(f"Python: {python_exe}")
    print(f"所选分组: {args.group}")
    print(f"总步骤数: {len(steps)}")

    failures: list[tuple[Step, float]] = []
    successes: list[tuple[Step, float]] = []

    for step in steps:
        ok, elapsed = run_step(step, python_exe)
        if ok:
            successes.append((step, elapsed))
            continue
        failures.append((step, elapsed))
        if args.fail_fast:
            break

    print("=" * 78)
    print("回归测试汇总")
    print(f"  成功: {len(successes)}")
    print(f"  失败: {len(failures)}")
    print(f"  总计: {len(successes) + len(failures)}")
    if successes:
        print("成功步骤:")
        for step, elapsed in successes:
            print(f"  - {step.name} ({elapsed:.1f}s)")
    if failures:
        print("失败步骤:")
        for step, elapsed in failures:
            print(f"  - {step.name} ({elapsed:.1f}s)")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
