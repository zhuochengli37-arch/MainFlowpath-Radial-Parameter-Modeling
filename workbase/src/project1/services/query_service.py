import json
from pathlib import Path


def _approx_equal(a: float | None, b: float | None, tol: float = 1e-9) -> bool:
    if a is None or b is None:
        return False
    return abs(a - b) <= tol


def _to_float(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def load_manifest(manifest_path: str) -> dict[str, object]:
    content = Path(manifest_path).read_text(encoding="utf-8")
    return json.loads(content)


def query_records(
    manifest: dict[str, object],
    component: str | None = None,
    stage: int | None = None,
    rpm: float | None = None,
    wcor: float | None = None,
    phi: float | None = None,
) -> list[dict[str, object]]:
    records = manifest.get("records", [])
    if not isinstance(records, list):
        return []

    result: list[dict[str, object]] = []
    for item in records:
        if not isinstance(item, dict):
            continue
        if component is not None and str(item.get("component", "")).upper() != component.upper():
            continue
        if stage is not None and item.get("stage") != stage:
            continue
        if rpm is not None and not _approx_equal(_to_float(item.get("rpm")), rpm):
            continue
        # `phi` kept as backward-compatible alias for `wcor`.
        expect_wcor = wcor if wcor is not None else phi
        if expect_wcor is not None and not _approx_equal(_to_float(item.get("wcor")), expect_wcor):
            if not _approx_equal(_to_float(item.get("phi")), expect_wcor):
                continue
        result.append(item)
    return result
