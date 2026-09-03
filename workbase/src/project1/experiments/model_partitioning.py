"""Shared model partition identities for workline and radial models."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ModelPartition:
    """A structured partition with backward-compatible string serialization."""

    component: str
    stage: int
    section: str | None = None

    def __post_init__(self) -> None:
        component = str(self.component).strip().upper()
        if not component:
            raise ValueError("partition component must not be empty")
        section = None if self.section is None else str(self.section).strip()
        if section == "":
            section = None
        object.__setattr__(self, "component", component)
        object.__setattr__(self, "stage", int(self.stage))
        object.__setattr__(self, "section", section)

    @property
    def key(self) -> str:
        parts = [self.component, f"S{self.stage}"]
        if self.section is not None:
            parts.append(self.section)
        return ":".join(parts)

    @property
    def safe_name(self) -> str:
        return self.key.replace(" ", "_").replace(":", "_")

    def as_dict(self) -> dict[str, object]:
        return {
            "component": self.component,
            "stage": self.stage,
            "section": self.section,
            "partition": self.key,
        }

    def __str__(self) -> str:
        return self.key


def workline_partition(sample: object) -> ModelPartition:
    """Return the 1D partition. Section is intentionally ignored."""

    return ModelPartition(
        component=str(getattr(sample, "component")),
        stage=int(getattr(sample, "stage")),
    )


def radial_partition(sample: object) -> ModelPartition:
    """Return the 2D/3D partition, adding section only when it is present."""

    return ModelPartition(
        component=str(getattr(sample, "component")),
        stage=int(getattr(sample, "stage")),
        section=getattr(sample, "section", None),
    )


def safe_partition_name(partition: str | ModelPartition) -> str:
    """Encode a logical partition using the established model-folder rule."""

    return str(partition).replace(" ", "_").replace(":", "_")
