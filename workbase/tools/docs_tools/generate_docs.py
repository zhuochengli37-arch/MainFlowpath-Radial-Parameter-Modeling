"""
项目文档生成工具。

该工具会扫描仓库结构、提取 Python API 信息，
并刷新 README 中的项目结构区块。
"""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def generate_directory_tree(
    root_path: Path,
    prefix: str = "",
    ignore_patterns: Optional[List[str]] = None,
    max_depth: int = 5,
    current_depth: int = 0,
) -> List[str]:
    """返回以 ``root_path`` 为根目录的文本树。"""
    if ignore_patterns is None:
        ignore_patterns = [
            ".git",
            "__pycache__",
            ".venv",
            "venv",
            ".pytest_cache",
            "*.pyc",
            ".DS_Store",
            "node_modules",
            ".vscode",
            ".idea",
            "*.egg-info",
            "dist",
            "build",
        ]

    if current_depth >= max_depth:
        return []

    try:
        items = sorted(root_path.iterdir(), key=lambda item: (not item.is_dir(), item.name))
    except PermissionError:
        return []

    lines: List[str] = []
    visible_items = []
    for item in items:
        should_ignore = False
        for pattern in ignore_patterns:
            if pattern.startswith("*") and item.name.endswith(pattern[1:]):
                should_ignore = True
                break
            if item.name == pattern:
                should_ignore = True
                break
        if not should_ignore:
            visible_items.append(item)

    for index, item in enumerate(visible_items):
        is_last = index == len(visible_items) - 1
        branch = "└── " if is_last else "├── "
        lines.append(f"{prefix}{branch}{item.name}")
        if item.is_dir():
            extension = "    " if is_last else "│   "
            lines.extend(
                generate_directory_tree(
                    item,
                    prefix + extension,
                    ignore_patterns,
                    max_depth,
                    current_depth + 1,
                )
            )

    return lines


def extract_docstring(node: ast.AST) -> Optional[str]:
    """返回受支持 AST 节点的 docstring。"""
    if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return ast.get_docstring(node)
    return None


def _collect_top_level_functions(tree: ast.Module) -> List[Dict[str, Any]]:
    functions: List[Dict[str, Any]] = []
    for node in tree.body:
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        functions.append(
            {
                "name": node.name,
                "docstring": extract_docstring(node),
                "args": [arg.arg for arg in node.args.args],
                "line": node.lineno,
            }
        )
    return functions


def _collect_classes(tree: ast.Module) -> List[Dict[str, Any]]:
    classes: List[Dict[str, Any]] = []
    for node in tree.body:
        if not isinstance(node, ast.ClassDef):
            continue

        methods = []
        for item in node.body:
            if not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            methods.append(
                {
                    "name": item.name,
                    "docstring": extract_docstring(item),
                    "args": [arg.arg for arg in item.args.args],
                    "line": item.lineno,
                }
            )

        classes.append(
            {
                "name": node.name,
                "docstring": extract_docstring(node),
                "methods": methods,
                "line": node.lineno,
            }
        )
    return classes


def parse_python_file(file_path: Path) -> Dict[str, Any]:
    """解析 Python 文件并返回模块、函数和类的元数据。"""
    try:
        source = file_path.read_text(encoding="utf-8")
        tree = ast.parse(source)
    except Exception as exc:
        return {"file": str(file_path), "error": str(exc)}

    return {
        "file": str(file_path),
        "module_docstring": extract_docstring(tree),
        "functions": _collect_top_level_functions(tree),
        "classes": _collect_classes(tree),
    }


def generate_api_docs(src_dir: Path, output_file: Path, title: str = "API Reference") -> None:
    """为 ``src_dir`` 下的 Python 源码生成 Markdown API 参考。"""
    lines = [f"# {title}", "", "> 根据项目源码自动生成。", "", "---", ""]

    for py_file in sorted(src_dir.rglob("*.py")):
        if py_file.name == "__init__.py" or py_file.name.startswith("test_"):
            continue

        relative_path = py_file.relative_to(src_dir)
        module_info = parse_python_file(py_file)
        if "error" in module_info:
            continue

        lines.append(f"## {relative_path.as_posix()}")
        lines.append("")

        if module_info["module_docstring"]:
            lines.append(module_info["module_docstring"])
            lines.append("")

        if module_info["functions"]:
            lines.append("### 函数")
            lines.append("")
            for func in module_info["functions"]:
                signature = ", ".join(func["args"])
                lines.append(f"#### `{func['name']}({signature})`")
                lines.append("")
                if func["docstring"]:
                    lines.append(func["docstring"])
                    lines.append("")
                lines.append(f"*定义于第 {func['line']} 行*")
                lines.append("")

        if module_info["classes"]:
            lines.append("### 类")
            lines.append("")
            for cls in module_info["classes"]:
                lines.append(f"#### `{cls['name']}`")
                lines.append("")
                if cls["docstring"]:
                    lines.append(cls["docstring"])
                    lines.append("")

                if cls["methods"]:
                    lines.append("**方法**")
                    lines.append("")
                    for method in cls["methods"]:
                        signature = ", ".join(method["args"])
                        lines.append(f"- `{method['name']}({signature})`")
                        if method["docstring"]:
                            first_line = method["docstring"].splitlines()[0]
                            lines.append(f"  - {first_line}")
                    lines.append("")

                lines.append(f"*定义于第 {cls['line']} 行*")
                lines.append("")

        lines.append("---")
        lines.append("")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    output_file.write_text("\n".join(lines), encoding="utf-8")


def generate_module_summary(script_dirs: List[Path], src_dir: Path) -> Dict[str, List[Dict[str, str]]]:
    """为入口脚本和核心包生成轻量模块摘要。"""
    summary: Dict[str, List[Dict[str, str]]] = {
        "scripts": [],
        "core_modules": [],
        "services": [],
    }

    project_root = src_dir.parents[1]
    for script_dir in script_dirs:
        if not script_dir.exists():
            continue

        for py_file in sorted(script_dir.glob("*.py")):
            if py_file.name.startswith("_"):
                continue
            module_info = parse_python_file(py_file)
            description = module_info.get("module_docstring") or "没有模块 docstring。"
            summary["scripts"].append(
                {
                    "name": py_file.relative_to(project_root).as_posix(),
                    "description": description.splitlines()[0],
                }
            )

    experiments_dir = src_dir / "project1" / "experiments"
    if experiments_dir.exists():
        for py_file in sorted(experiments_dir.glob("*.py")):
            if py_file.name == "__init__.py":
                continue
            module_info = parse_python_file(py_file)
            description = module_info.get("module_docstring") or "没有模块 docstring。"
            summary["core_modules"].append(
                {
                    "name": f"experiments/{py_file.name}",
                    "description": description.splitlines()[0],
                }
            )

    services_dir = src_dir / "project1" / "services"
    if services_dir.exists():
        for py_file in sorted(services_dir.glob("*.py")):
            if py_file.name == "__init__.py":
                continue
            module_info = parse_python_file(py_file)
            description = module_info.get("module_docstring") or "没有模块 docstring。"
            summary["services"].append(
                {
                    "name": f"services/{py_file.name}",
                    "description": description.splitlines()[0],
                }
            )

    return summary


def update_readme_with_structure(readme_path: Path, project_root: Path) -> None:
    """如果 README 中存在标记，则替换其中的项目结构区块。"""
    if not readme_path.exists():
        print(f"未找到 README: {readme_path}")
        return

    tree_lines = generate_directory_tree(
        project_root,
        max_depth=3,
        ignore_patterns=[
            ".git",
            "__pycache__",
            ".venv",
            "venv",
            ".pytest_cache",
            "*.pyc",
            ".DS_Store",
            "node_modules",
            ".vscode",
            ".idea",
            "*.egg-info",
            "dist",
            "build",
            "logs",
            "offline_packages",
        ],
    )
    tree_text = "\n".join(tree_lines)
    content = readme_path.read_text(encoding="utf-8")

    start_marker = "<!-- PROJECT_STRUCTURE_START -->"
    end_marker = "<!-- PROJECT_STRUCTURE_END -->"
    if start_marker not in content or end_marker not in content:
        print("未找到 README 结构标记。")
        return

    start_idx = content.index(start_marker) + len(start_marker)
    end_idx = content.index(end_marker)
    new_structure = f"\n```\n{tree_text}\n```\n"
    new_content = content[:start_idx] + new_structure + content[end_idx:]
    readme_path.write_text(new_content, encoding="utf-8")
    print(f"已更新 README 结构: {readme_path}")


if __name__ == "__main__":
    root = Path(__file__).resolve().parents[3]
    workbase = root / "workbase"
    script_dirs = [
        root / "scripts",
        workbase / "current_scripts",
        workbase / "generic_scripts",
        workbase / "tools" / "docs_tools",
    ]
    src_dir = workbase / "src"
    docs_dir = root / "docs"

    print("=" * 70)
    print("项目文档生成工具")
    print("=" * 70)

    print("\n[1/3] 正在生成 API 参考...")
    api_doc_path = docs_dir / "API_REFERENCE.md"
    generate_api_docs(src_dir, api_doc_path, title="AIR2 Project1 - API Reference")
    print(f"  已写入 {api_doc_path}")

    print("\n[2/3] 正在生成模块摘要...")
    summary = generate_module_summary(script_dirs, src_dir)
    summary_path = docs_dir / "MODULE_SUMMARY.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"  已写入 {summary_path}")

    print("\n[3/3] 正在更新 README 结构...")
    update_readme_with_structure(root / "README.md", root)

    print("\n完成。")
