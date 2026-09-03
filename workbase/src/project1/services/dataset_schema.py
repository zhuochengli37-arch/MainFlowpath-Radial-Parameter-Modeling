"""Dataset schema identification for domain-specific flowpath databases."""

from __future__ import annotations

from collections.abc import Iterable

from project1.services.section_registry import discover_sections


LEGACY_VALIDATION_SCHEMA = "legacy_validation"
INSTITUTE_SINGLE_SECTION_SCHEMA = "institute_single_section"
INSTITUTE_FOUR_SECTION_SCHEMA = "institute_four_section"
INSTITUTE_MULTI_SECTION_SCHEMA = "institute_multi_section"
INSTITUTE_SCHEMA_VERSION = "v1"

INSTITUTE_SINGLE_SECTION_FIELDS = frozenset({"xi", "Cpt", "Ctt", "Cps", "Cts", "MA"})
INSTITUTE_FOUR_SECTION_NAMES = frozenset({"RI", "RO", "SI", "SO"})
INSTITUTE_COMMON_AERO_FIELDS = frozenset({"xi", "Cpt", "Ctt", "Cps", "Cts"})


def _normalize_column(column: object) -> str:
    return str(column).strip().replace("\ufeff", "")


def detect_dataset_schema(columns: Iterable[object], station: str = "MAIN") -> str:
    """Classify an actual header while keeping station and section semantics separate."""

    normalized = [_normalize_column(column) for column in columns]
    try:
        sections = discover_sections(normalized)
    except ValueError:
        sections = ()
    if sections:
        section_names = {descriptor.name for descriptor in sections}
        if len(sections) == 4 and section_names == INSTITUTE_FOUR_SECTION_NAMES:
            # Compatibility identity used by the already released adapter/tests.
            return INSTITUTE_FOUR_SECTION_SCHEMA
        return INSTITUTE_MULTI_SECTION_SCHEMA

    field_set = set(normalized)
    if INSTITUTE_SINGLE_SECTION_FIELDS.issubset(field_set):
        return INSTITUTE_SINGLE_SECTION_SCHEMA

    # Four-section DATABASE boundary files use unsuffixed single-section
    # headers. Their INLET/OUTLET station is path metadata; it must not be
    # converted into a RI/RO/SI/SO section.
    normalized_station = str(station).strip().upper()
    is_boundary = normalized_station in {"INLET", "OUTLET"}
    is_unsuffixed_institute_header = INSTITUTE_COMMON_AERO_FIELDS.issubset(field_set) and (
        {"Vz", "Rho"}.issubset(field_set)
        or bool({"Ma", "MA"}.intersection(field_set))
    )
    if is_boundary and is_unsuffixed_institute_header:
        return INSTITUTE_SINGLE_SECTION_SCHEMA

    return LEGACY_VALIDATION_SCHEMA


def is_current_proxy_schema(schema: str) -> bool:
    """Return whether the schema is supported by the current single-section models."""

    return schema in {LEGACY_VALIDATION_SCHEMA, INSTITUTE_SINGLE_SECTION_SCHEMA}


def schema_identity(schema: str) -> tuple[str, str]:
    """Return the canonical schema name/version without removing legacy identifiers."""

    if schema in {INSTITUTE_FOUR_SECTION_SCHEMA, INSTITUTE_MULTI_SECTION_SCHEMA}:
        return INSTITUTE_MULTI_SECTION_SCHEMA, INSTITUTE_SCHEMA_VERSION
    return schema, INSTITUTE_SCHEMA_VERSION
