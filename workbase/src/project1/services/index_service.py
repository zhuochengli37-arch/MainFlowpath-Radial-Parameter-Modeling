import json
from pathlib import Path

from project1.services.data_reader import read_curve_file
from project1.services.meta_parser import parse_metadata_from_path


def _iter_data_files(input_dir: str) -> list[Path]:
    root = Path(input_dir)
    files: list[Path] = []
    for p in root.rglob("*"):
        suffix = p.suffix.lower()
        if p.name.lower().endswith(".bak"):
            continue
        if p.is_file() and suffix in {".txt", ".dat"}:
            files.append(p)
    return files


def build_manifest(input_dir: str, output_dir: str) -> dict[str, object]:
    files = _iter_data_files(input_dir)
    records: list[dict[str, object]] = []
    errors: list[dict[str, str]] = []

    for file_path in files:
        path_str = str(file_path)
        metadata = parse_metadata_from_path(path_str)
        try:
            parsed = read_curve_file(path_str)
            records.append(
                {
                    **metadata,
                    "columns": parsed["columns"],
                    "row_count": parsed["row_count"],
                }
            )
        except ValueError as exc:
            errors.append({"path": path_str, "reason": str(exc)})

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    manifest_path = out / "manifest.json"
    manifest = {
        "input_dir": str(Path(input_dir)),
        "total_files": len(files),
        "indexed_files": len(records),
        "failed_files": len(errors),
        "records": records,
        "errors": errors,
    }
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    manifest["manifest_path"] = str(manifest_path)
    return manifest
