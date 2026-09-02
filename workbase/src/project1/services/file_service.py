from pathlib import Path


def list_input_files(input_dir: str) -> list[Path]:
    base = Path(input_dir)
    if not base.exists():
        return []
    return [p for p in base.glob("*") if p.is_file()]
