from pathlib import Path
import csv


def _split_tokens(line: str) -> list[str]:
    return [token for token in line.strip().replace("\t", " ").split(" ") if token]


def _coerce_value(value: str) -> object:
    text = value.strip()
    if not text:
        return text
    try:
        return float(text)
    except ValueError:
        return text


def _parse_tabular_lines(lines: list[str], delimiter: str | None = None, min_columns: int = 1) -> dict[str, object]:
    if len(lines) < 2:
        raise ValueError("file has no usable tabular content")

    if delimiter is None:
        header_line = lines[0]
        if "," in header_line and "\t" not in header_line:
            delimiter = ","
        else:
            delimiter = "whitespace"

    if delimiter == ",":
        reader = csv.reader(lines, delimiter=delimiter)
        rows_iter = list(reader)
    else:
        rows_iter = [_split_tokens(line) for line in lines]

    headers = rows_iter[0]
    if len(headers) < min_columns:
        raise ValueError("invalid header format")

    rows: list[dict[str, object]] = []
    for cols in rows_iter[1:]:
        if len(cols) < len(headers):
            continue
        values = [_coerce_value(item) for item in cols[: len(headers)]]
        rows.append({name: values[idx] for idx, name in enumerate(headers)})

    if not rows:
        raise ValueError("no numeric rows parsed")

    return {"columns": headers, "rows": rows, "row_count": len(rows)}


def read_curve_file(file_path: str) -> dict[str, object]:
    path = Path(file_path)
    raw = path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    # Prediction-side curve files may contain only a single logical input column
    # such as `xi`, while training-side curve files contain multiple columns.
    return _parse_tabular_lines(lines, min_columns=1)


def read_tabular_file(file_path: str, delimiter: str | None = None) -> dict[str, object]:
    path = Path(file_path)
    raw = path.read_text(encoding="utf-8", errors="ignore")
    lines = [ln for ln in raw.splitlines() if ln.strip()]
    if len(lines) < 2:
        raise ValueError("file has no usable tabular content")

    if delimiter is None:
        header_line = lines[0]
        if "," in header_line:
            delimiter = ","
        elif "\t" in header_line:
            delimiter = "\t"

    if delimiter is None:
        rows_data = [_split_tokens(line) for line in lines]
    else:
        import csv

        reader = csv.reader(lines, delimiter=delimiter)
        rows_data = [[cell.strip() for cell in row if cell.strip()] for row in reader]

    if len(rows_data) < 2:
        raise ValueError("file has no usable tabular content")

    headers = [name for name in rows_data[0] if name]
    if len(headers) < 1:
        raise ValueError("invalid header format")

    rows: list[dict[str, object]] = []
    for row in rows_data[1:]:
        if len(row) < len(headers):
            continue
        values = [_coerce_value(item) for item in row[: len(headers)]]
        rows.append({name: values[idx] for idx, name in enumerate(headers)})

    if not rows:
        raise ValueError("no numeric rows parsed")

    return {"columns": headers, "rows": rows, "row_count": len(rows)}
