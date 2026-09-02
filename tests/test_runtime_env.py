from __future__ import annotations

from pathlib import Path
import sys

from workbase.common.runtime_env import project_venv_python, should_reexec_into_project_venv


def test_project_venv_python_points_to_repo_venv(tmp_path: Path):
    project_root = tmp_path / "repo"
    expected = project_root / ".venv" / ("Scripts" if sys.platform.startswith("win") else "bin")
    expected = expected / ("python.exe" if sys.platform.startswith("win") else "python")

    assert project_venv_python(project_root) == expected


def test_should_reexec_into_project_venv_is_false_for_matching_interpreter(tmp_path: Path):
    project_root = tmp_path / "repo"
    current = project_venv_python(project_root)

    assert not should_reexec_into_project_venv(project_root, current_executable=current)


def test_should_reexec_into_project_venv_is_true_for_non_project_interpreter(tmp_path: Path):
    project_root = tmp_path / "repo"
    other_python = project_root / "python.exe"

    assert should_reexec_into_project_venv(project_root, current_executable=other_python)


def test_project_venv_python_uses_configured_hint_when_present(tmp_path: Path):
    project_root = tmp_path / "repo"
    project_root.mkdir()
    configured = tmp_path / "custom-venv" / ("Scripts" if sys.platform.startswith("win") else "bin")
    configured = configured / ("python.exe" if sys.platform.startswith("win") else "python")
    (project_root / ".project_venv_python").write_text(str(configured), encoding="utf-8")

    assert project_venv_python(project_root) == configured


def test_should_reexec_into_project_venv_respects_configured_hint(tmp_path: Path):
    project_root = tmp_path / "repo"
    project_root.mkdir()
    configured = tmp_path / "custom-venv" / ("Scripts" if sys.platform.startswith("win") else "bin")
    configured = configured / ("python.exe" if sys.platform.startswith("win") else "python")
    (project_root / ".project_venv_python").write_text(str(configured), encoding="utf-8")

    assert not should_reexec_into_project_venv(project_root, current_executable=configured)
