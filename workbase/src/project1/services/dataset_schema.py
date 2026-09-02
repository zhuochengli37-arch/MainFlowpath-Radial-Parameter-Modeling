"""Dataset schema identification for domain-specific flowpath databases."""

from __future__ import annotations

from collections.abc import Iterable


LEGACY_VALIDATION_SCHEMA = "legacy_validation"
INSTITUTE_SINGLE_SECTION_SCHEMA = "institute_single_section"
INSTITUTE_FOUR_SECTION_SCHEMA = "institute_four_section"

INSTITUTE_SINGLE_SECTION_FIELDS = frozenset({"xi", "Cpt", "Ctt", "Cps", "Cts", "MA"})
INSTITUTE_FOUR_SECTION_NAMES = frozenset({"RI", "RO", "SI", "SO"})
INSTITUTE_COMMON_AERO_FIELDS = frozenset({"xi", "Cpt", "Ctt", "Cps", "Cts"})


def _normalize_column(column: object) -> str:
    return str(column).strip().replace("\ufeff", "")


def _section_suffix(column: str) -> str | None:
    if "_" not in column:
        return None
    suffix = column.rsplit("_", 1)[1].upper()
    return suffix if suffix in INSTITUTE_FOUR_SECTION_NAMES else None


def detect_dataset_schema(columns: Iterable[object]) -> str:
    """Classify a parsed header without interpreting component-specific units."""

    normalized = [_normalize_column(column) for column in columns]
    section_suffixes = {
        suffix
        for column in normalized
        for suffix in [_section_suffix(column)]
        if suffix is not None
    }
    if section_suffixes == INSTITUTE_FOUR_SECTION_NAMES:
        return INSTITUTE_FOUR_SECTION_SCHEMA

    field_set = set(normalized)
    if INSTITUTE_COMMON_AERO_FIELDS.issubset(field_set) and (
        {"Vz", "Rho"}.issubset(field_set) or "Ma" in field_set
    ):
        return INSTITUTE_FOUR_SECTION_SCHEMA

    if INSTITUTE_SINGLE_SECTION_FIELDS.issubset(field_set):
        return INSTITUTE_SINGLE_SECTION_SCHEMA

    return LEGACY_VALIDATION_SCHEMA


def is_current_proxy_schema(schema: str) -> bool:
    """Return whether the schema is supported by the current single-section models."""

    return schema in {LEGACY_VALIDATION_SCHEMA, INSTITUTE_SINGLE_SECTION_SCHEMA}
