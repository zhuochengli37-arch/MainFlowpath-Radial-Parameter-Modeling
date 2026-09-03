"""Canonical in-memory results returned by the radial prediction core."""

from __future__ import annotations

from dataclasses import asdict, dataclass

from project1.experiments.model_registry import ModelIdentity


@dataclass(frozen=True)
class PredictionResult:
    source_file: str
    component: str
    stage: int
    station: str
    section: str | None
    speed_parameter: float
    flow_parameter: float | None
    xi: float
    target: str
    predicted_value: float
    partition: str
    model_identity: ModelIdentity
    model_name: str
    model_path: str
    schema_name: str
    schema_version: str
    sample_index: int
    row_index: int
    section_row_index: int

    def as_dict(self) -> dict[str, object]:
        payload = asdict(self)
        payload["model_identity"] = {
            "dimension": self.model_identity.dimension,
            "component": self.model_identity.component,
            "stage": self.model_identity.stage,
            "station": self.model_identity.station,
            "section": self.model_identity.section,
            "partition": str(self.model_identity.partition),
            "target": self.model_identity.target,
        }
        return payload
