from dataclasses import dataclass


@dataclass(frozen=True)
class TaskResult:
    processed_files: int
    indexed_files: int
    failed_files: int
    manifest_path: str
    report_path: str
