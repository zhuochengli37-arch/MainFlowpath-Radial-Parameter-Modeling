import os
from pathlib import Path


def write_report(
    output_dir: str,
    processed_files: int,
    indexed_files: int,
    failed_files: int,
    manifest_path: str,
) -> Path:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report_file = out_path / "report.txt"
    report_file.write_text(
        (
            f"processed_files={processed_files}{os.linesep}"
            f"indexed_files={indexed_files}{os.linesep}"
            f"failed_files={failed_files}{os.linesep}"
            f"manifest_path={manifest_path}{os.linesep}"
        ),
        encoding="utf-8",
    )
    return report_file
