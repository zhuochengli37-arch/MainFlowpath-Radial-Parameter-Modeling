from __future__ import annotations

from dataclasses import dataclass
from math import isnan
from numbers import Real
from pathlib import Path

from project1.services.data_reader import read_tabular_file


SUPPORTED_SUFFIXES = {".txt", ".csv", ".dat"}


@dataclass(frozen=True)
class TabularTemplate:
    source_path: Path
    columns: list[str]
    delimiter: str
    suffix: str


def list_template_files(input_path: str | Path) -> list[Path]:
    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(f"template path does not exist: {path}")
    if path.is_file():
        return [path]
    files = sorted(
        file_path
        for file_path in path.rglob("*")
        if file_path.is_file() and file_path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if not files:
        raise FileNotFoundError(f"no template files found under: {path}")
    return files


def detect_tabular_delimiter(file_path: str | Path) -> str:
    raw = Path(file_path).read_text(encoding="utf-8", errors="ignore")
    lines = [line for line in raw.splitlines() if line.strip()]
    if not lines:
        return "whitespace"
    header = lines[0]
    if "," in header and "\t" not in header:
        return ","
    if "\t" in header:
        return "\t"
    return "whitespace"


def load_tabular_template(file_path: str | Path) -> TabularTemplate:
    path = Path(file_path)
    parsed = read_tabular_file(str(path))
    return TabularTemplate(
        source_path=path,
        columns=[str(column) for column in parsed["columns"]],
        delimiter=detect_tabular_delimiter(path),
        suffix=path.suffix or ".txt",
    )


def find_matching_template_file(
    template_root: str | Path,
    *,
    predict_file: str | Path | None = None,
    predict_root: str | Path | None = None,
) -> Path:
    root = Path(template_root)
    files = list_template_files(root)
    if root.is_file():
        return files[0]

    predict_path = Path(predict_file) if predict_file is not None else None
    predict_base = Path(predict_root) if predict_root is not None else None

    if predict_path is not None and predict_base is not None:
        try:
            relative = predict_path.relative_to(predict_base)
            candidate = root / relative
            if candidate.exists() and candidate.suffix.lower() in SUPPORTED_SUFFIXES:
                return candidate
        except ValueError:
            pass

    if predict_path is not None:
        same_name = [file_path for file_path in files if file_path.name == predict_path.name]
        if len(same_name) == 1:
            return same_name[0]

    return files[0]


def write_table_like_template(
    output_file: str | Path,
    template: TabularTemplate,
    rows: list[dict[str, object]],
) -> None:
    path = Path(output_file)
    path.parent.mkdir(parents=True, exist_ok=True)

    lines = [_join_values(template.columns, template.delimiter)]
    for row in rows:
        values = [_format_value(row.get(column, "")) for column in template.columns]
        lines.append(_join_values(values, template.delimiter))

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _join_values(values: list[str], delimiter: str) -> str:
    if delimiter == "whitespace":
        return " ".join(values)
    return delimiter.join(values)


def _format_value(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, Real):
        numeric = float(value)
        if isnan(numeric):
            return "NaN"
        return f"{numeric:.15g}"
    return str(value)
