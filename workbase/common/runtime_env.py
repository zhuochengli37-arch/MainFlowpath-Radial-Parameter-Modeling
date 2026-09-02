from __future__ import annotations

import os
import sys
from pathlib import Path


_REEXEC_ENV_VAR = "AIR2_PROJECT_VENV_REEXEC"
_PROJECT_VENV_HINT = ".project_venv_python"


def _default_project_venv_python(project_root: str | Path) -> Path:
    root = Path(project_root).resolve()
    if os.name == "nt":
        return root / ".venv" / "Scripts" / "python.exe"
    return root / ".venv" / "bin" / "python"


def _configured_project_venv_python(project_root: str | Path) -> Path | None:
    root = Path(project_root).resolve()
    hint_file = root / _PROJECT_VENV_HINT
    if not hint_file.exists():
        return None

    try:
        configured = hint_file.read_text(encoding="utf-8-sig").strip()
    except OSError:
        return None

    if not configured:
        return None

    configured_path = Path(configured)
    if not configured_path.is_absolute():
        configured_path = (root / configured_path).resolve(strict=False)

    return configured_path


def project_venv_python(project_root: str | Path) -> Path:
    configured = _configured_project_venv_python(project_root)
    if configured is not None:
        return configured
    return _default_project_venv_python(project_root)


def _normalized_path(path: str | Path) -> str:
    return os.path.normcase(str(Path(path).resolve(strict=False)))


def should_reexec_into_project_venv(
    project_root: str | Path,
    current_executable: str | Path | None = None,
) -> bool:
    current = current_executable or sys.executable
    return _normalized_path(current) != _normalized_path(project_venv_python(project_root))


def ensure_project_venv(project_root: str | Path) -> Path:
    expected_python = project_venv_python(project_root)
    if not expected_python.exists():
        raise RuntimeError(f"未找到项目虚拟环境解释器: {expected_python}")

    if not should_reexec_into_project_venv(project_root):
        return expected_python

    if os.environ.get(_REEXEC_ENV_VAR) == "1":
        raise RuntimeError(
            "重新切换到项目虚拟环境失败。"
            f"期望解释器: {expected_python}。当前解释器: {sys.executable}"
        )

    os.environ[_REEXEC_ENV_VAR] = "1"
    print(f"正在切换到项目虚拟环境解释器: {expected_python}", file=sys.stderr)
    os.execv(str(expected_python), [str(expected_python), *sys.argv])
    raise AssertionError("os.execv 按预期不应返回")
