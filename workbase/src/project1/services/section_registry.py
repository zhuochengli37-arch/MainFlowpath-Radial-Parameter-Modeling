"""Name discovery and alias registries for sectioned tabular data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Mapping


DEFAULT_SECTION_ALIASES: dict[str, str] = {
    "ri": "RI",
    "ro": "RO",
    "si": "SI",
    "so": "SO",
    "rotor_inlet": "RI",
    "rotor_outlet": "RO",
    "stator_inlet": "SI",
    "stator_outlet": "SO",
}

DEFAULT_VARIABLE_ALIASES: dict[str, str] = {
    "xi": "xi",
    "cpt": "Cpt",
    "cps": "Cps",
    "ctt": "Ctt",
    "cts": "Cts",
    "ma": "MA",
    "vz": "Vz",
    "rho": "Rho",
}


@dataclass(frozen=True)
class SectionDescriptor:
    """A discovered source section and its optional canonical alias."""

    name: str
    source_name: str


def _clean_name(value: object) -> str:
    return str(value).strip().replace("\ufeff", "")


def _merged_aliases(
    defaults: Mapping[str, str],
    additions: Mapping[str, str] | None,
) -> dict[str, str]:
    merged = {str(alias).casefold(): str(canonical) for alias, canonical in defaults.items()}
    if additions:
        merged.update(
            {str(alias).casefold(): str(canonical) for alias, canonical in additions.items()}
        )
    return merged


def canonical_section_name(
    source_name: str,
    aliases: Mapping[str, str] | None = None,
) -> str:
    """Normalize a known alias while preserving an unknown valid name."""

    cleaned = _clean_name(source_name)
    registry = _merged_aliases(DEFAULT_SECTION_ALIASES, aliases)
    return registry.get(cleaned.casefold(), cleaned)


def canonical_variable_name(
    source_name: str,
    aliases: Mapping[str, str] | None = None,
) -> str:
    """Normalize spelling only; this function never performs physical conversion."""

    cleaned = _clean_name(source_name)
    registry = _merged_aliases(DEFAULT_VARIABLE_ALIASES, aliases)
    return registry.get(cleaned.casefold(), cleaned)


def discover_sections(
    columns: Iterable[object],
    radial_variable: str = "xi",
    section_aliases: Mapping[str, str] | None = None,
    variable_aliases: Mapping[str, str] | None = None,
) -> tuple[SectionDescriptor, ...]:
    """Discover sections from `<radial variable>_<section>` columns in source order."""

    radial_name = canonical_variable_name(radial_variable, variable_aliases)
    discovered: list[SectionDescriptor] = []
    canonical_names: dict[str, str] = {}

    for raw_column in columns:
        column = _clean_name(raw_column)
        source_variable, separator, source_section = column.partition("_")
        if not separator or not source_section:
            continue
        if canonical_variable_name(source_variable, variable_aliases) != radial_name:
            continue
        canonical = canonical_section_name(source_section, section_aliases)
        canonical_key = canonical.casefold()
        if canonical_key in canonical_names:
            previous = canonical_names[canonical_key]
            raise ValueError(
                f"duplicate section {canonical!r} from source names "
                f"{previous!r} and {source_section!r}"
            )
        canonical_names[canonical_key] = source_section
        discovered.append(SectionDescriptor(name=canonical, source_name=source_section))

    return tuple(discovered)


def split_section_column(
    column: object,
    sections: Iterable[SectionDescriptor],
) -> tuple[str, SectionDescriptor] | None:
    """Split `<variable>_<section>` using discovered source names, longest first."""

    cleaned = _clean_name(column)
    ordered = sorted(sections, key=lambda item: len(item.source_name), reverse=True)
    folded = cleaned.casefold()
    for descriptor in ordered:
        suffix = "_" + descriptor.source_name
        if not folded.endswith(suffix.casefold()):
            continue
        source_variable = cleaned[: -len(suffix)]
        if source_variable:
            return source_variable, descriptor
    return None
