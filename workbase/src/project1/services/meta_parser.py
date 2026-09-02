from pathlib import Path
import re

_IGNORED_COMPONENT_PARTS = {
    "data",
    "input",
    "output",
    "current",
    "generic",
    "offline_data",
    "train",
    "predict",
    "results",
    "models",
    "models_2d",
    "workline_models",
    "run_configs",
    "1d",
    "2d",
    "3d",
}


def _to_float(value: str) -> float | None:
    try:
        return float(value)
    except ValueError:
        return None


def _to_int(value: str) -> int | None:
    try:
        return int(value)
    except ValueError:
        return None


def _split_component_station(raw_component: str) -> tuple[str, str]:
    normalized = str(raw_component).strip().upper()
    if normalized.endswith("_INLET"):
        return normalized[: -len("_INLET")], "INLET"
    if normalized.endswith("_OUTLET"):
        return normalized[: -len("_OUTLET")], "OUTLET"
    return normalized, "MAIN"


def _component_before_stage(parts: tuple[str, ...]) -> str | None:
    for index, part in enumerate(parts):
        upper = part.upper()
        if not (upper.startswith("STAGE_") or upper.startswith("STAGE-")):
            continue
        for back_index in range(index - 1, -1, -1):
            candidate = parts[back_index]
            candidate_upper = candidate.upper()
            if candidate_upper.startswith(("DATABASE_", "RPM_", "CNC_", "CNF_", "STAGE_")):
                continue
            if candidate.lower() in _IGNORED_COMPONENT_PARTS:
                continue
            if _to_float(candidate) is not None:
                continue
            return candidate
    return None


def _rpm_after_stage(parts: tuple[str, ...]) -> float | None:
    for index, part in enumerate(parts):
        upper = part.upper()
        if not (upper.startswith("STAGE_") or upper.startswith("STAGE-")):
            continue
        if index + 1 >= len(parts):
            continue
        candidate = parts[index + 1]
        candidate_upper = candidate.upper()
        parsed_prefixed = _prefixed_value(candidate, ("RPM_", "CNC_", "CNF_"))
        if parsed_prefixed is not None:
            return parsed_prefixed
        if candidate_upper.startswith(("STAGE_", "DATABASE_")):
            continue
        return _to_float(candidate)
    return None


def _prefixed_value(text: str, prefixes: tuple[str, ...]) -> float | None:
    upper = text.upper()
    for prefix in prefixes:
        if upper.startswith(prefix):
            return _to_float(text[len(prefix) :])
    return None


def _wcor_from_stem(stem: str) -> float | None:
    value = _prefixed_value(stem, ("PHI_", "WCOR_"))
    if value is not None:
        return value
    match = re.match(r"W\d*COR[_-](.+)", stem, flags=re.IGNORECASE)
    if match:
        return _to_float(match.group(1))
    return _to_float(stem)


def parse_metadata_from_path(file_path: str) -> dict[str, object | None]:
    p = Path(file_path)
    component: str | None = None
    database: str | None = None
    station: str | None = None
    stage: int | None = None
    rpm: float | None = None
    wcor: float | None = None

    for part in p.parts:
        upper = part.upper()
        if upper.startswith("DATABASE_"):
            database = part
            component, station = _split_component_station(part.split("_", 1)[1])
        elif upper.startswith("STAGE_") or upper.startswith("STAGE-"):
            stage_token = re.split(r"[_-]", part, maxsplit=1)[1]
            stage = _to_int(stage_token)
        else:
            parsed_rpm = _prefixed_value(part, ("RPM_", "CNC_", "CNF_"))
            if parsed_rpm is not None:
                rpm = parsed_rpm

    if component is None:
        raw_component = _component_before_stage(p.parts)
        if raw_component is not None:
            component, station = _split_component_station(raw_component)
            if database is None:
                database = f"DATABASE_{raw_component}"

    if rpm is None:
        rpm = _rpm_after_stage(p.parts)

    stem = p.stem
    wcor = _wcor_from_stem(stem)

    # Fallback for "component-stage-rpm-wcor.txt/dat".
    if component is None or stage is None or rpm is None or wcor is None:
        items = stem.split("-")
        if len(items) >= 4:
            fallback_component, fallback_station = _split_component_station(items[0])
            component = component or fallback_component
            station = station or fallback_station
            stage = stage if stage is not None else _to_int(items[1])
            rpm = rpm if rpm is not None else _to_float(items[2])
            wcor = wcor if wcor is not None else _to_float(items[3])

    text = p.as_posix()
    if stage is None:
        match = re.search(r"STAGE[_-](\d+)", text, flags=re.IGNORECASE)
        if match:
            stage = _to_int(match.group(1))
    if rpm is None:
        match = re.search(r"(?:RPM|CNC|CNF)[_-]([0-9.]+)", text, flags=re.IGNORECASE)
        if match:
            rpm = _to_float(match.group(1))
    if wcor is None:
        match = re.search(r"W(?:\d+)?COR[_-]([0-9.]+)", text, flags=re.IGNORECASE)
        if match:
            wcor = _to_float(match.group(1))
    if wcor is None:
        match = re.search(r"PHI[_-]([0-9.]+)", text, flags=re.IGNORECASE)
        if match:
            wcor = _to_float(match.group(1))

    return {
        "component": component,
        "database": database,
        "station": station or "MAIN",
        "stage": stage,
        "rpm": rpm,
        "wcor": wcor,
        "phi": wcor,
        "ext": p.suffix.lower(),
        "path": str(p),
    }
