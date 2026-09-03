"""Compatibility entry points for the Institute four-section DATABASE.

The parsing implementation lives in :mod:`multi_section_adapter`.  The fixed
RI/RO/SI/SO tuple below is used only to validate the named four-section API and
to preserve its historical display order; it is not a core parsing constraint.
"""

from __future__ import annotations

from pathlib import Path

from project1.experiments.benchmark_data import FlexSample
from project1.experiments.multi_section_adapter import (
    MultiSectionAdapterError,
    adapt_multi_section_file,
    audit_multi_section_database,
    flatten_multi_section_data,
    multi_section_layout,
)


CANONICAL_SECTION_ORDER = ("RI", "RO", "SI", "SO")
FourSectionAdapterError = MultiSectionAdapterError


def four_section_layout(columns: list[object]) -> dict[str, object]:
    """Return the legacy layout shape for an exact RI/RO/SI/SO dataset."""

    layout = multi_section_layout(columns)
    present = set(layout.section_names)
    expected = set(CANONICAL_SECTION_ORDER)
    missing = expected - present
    unexpected = present - expected
    if missing:
        raise FourSectionAdapterError(
            "missing section(s): " + ", ".join(sorted(missing))
        )
    if unexpected:
        raise FourSectionAdapterError(
            "unexpected section(s): " + ", ".join(sorted(unexpected))
        )
    return {
        "source_section_order": tuple(
            section.source_name for section in layout.sections
        ),
        "sections": {
            section.name: dict(section.source_fields)
            for section in layout.sections
        },
    }


def adapt_four_section_file(file_path: str | Path) -> list[FlexSample]:
    """Adapt the formal four-section file through the generic core."""

    try:
        data = adapt_multi_section_file(file_path)
    except MultiSectionAdapterError as exc:
        if "is not a multi-section MAIN file" in str(exc):
            station = str(exc).split(" ", 2)[1]
            raise FourSectionAdapterError(
                f"station {station} is not a four-section MAIN file"
            ) from exc
        raise
    present = set(data.sections)
    expected = set(CANONICAL_SECTION_ORDER)
    if present != expected:
        missing = expected - present
        unexpected = present - expected
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(sorted(missing)))
        if unexpected:
            details.append("unexpected " + ", ".join(sorted(unexpected)))
        raise FourSectionAdapterError("four-section layout mismatch: " + "; ".join(details))
    return flatten_multi_section_data(data, section_order=CANONICAL_SECTION_ORDER)


def audit_four_section_database(input_dir: str | Path) -> dict[str, object]:
    """Compatibility name for the generic adapter-level DATABASE audit."""

    return audit_multi_section_database(
        input_dir,
        expected_sections=set(CANONICAL_SECTION_ORDER),
    )
