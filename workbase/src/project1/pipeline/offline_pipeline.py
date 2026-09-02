from pathlib import Path

from project1.models.task_result import TaskResult
from project1.services.index_service import build_manifest
from project1.services.query_service import query_records
from project1.services.report_service import write_report


def run_pipeline(input_dir: str, output_dir: str) -> tuple[int, TaskResult | None]:
    input_path = Path(input_dir)
    if not input_path.exists():
        return 1, None

    manifest = build_manifest(input_dir, output_dir)

    # Minimal query sample for quick verification in first phase.
    _ = query_records(manifest, component="FAN", stage=3)

    processed_files = int(manifest.get("total_files", 0))
    indexed_files = int(manifest.get("indexed_files", 0))
    failed_files = int(manifest.get("failed_files", 0))
    manifest_path = str(manifest.get("manifest_path", ""))

    report_path = write_report(
        output_dir,
        processed_files=processed_files,
        indexed_files=indexed_files,
        failed_files=failed_files,
        manifest_path=manifest_path,
    )
    return (
        0,
        TaskResult(
            processed_files=processed_files,
            indexed_files=indexed_files,
            failed_files=failed_files,
            manifest_path=manifest_path,
            report_path=str(report_path),
        ),
    )
