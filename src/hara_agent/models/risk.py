from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any
from .common import EvidenceValue


@dataclass
class RiskAssessment:
    assessment_id: str
    scenario_id: str
    severity: EvidenceValue[str]
    exposure: EvidenceValue[str]
    controllability: EvidenceValue[str]
    asil: EvidenceValue[str]
    malfunction_id: str = ""
    hazardous_event: str = ""
    potential_harm: str = ""
    exposure_tf: str = ""
    safety_goal_id: str = ""
    branch_evaluations: list[dict[str, Any]] = field(default_factory=list)
    branch_summary: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.assessment_id or not self.scenario_id:
            raise ValueError("RiskAssessment必须具有assessment_id并关联scenario_id")

    def to_dict(self):
        result = asdict(self)
        for key in ("severity", "exposure", "controllability", "asil"):
            result[key] = getattr(self, key).to_dict()
        return result
