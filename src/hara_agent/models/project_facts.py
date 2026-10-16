from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .common import FactProvenance, ReviewStatus, SourceRef


PHYSICAL_SEMANTIC_FIELDS = (
    "analysis_value_semantic", "time_role", "impact_time_id", "trajectory_id",
    "branch_id", "object_id", "unit", "assessment_reference", "model_valid_until_s",
)
# Neutral quantities accepted by the existing trajectory model, not rating rules.
PHYSICAL_QUANTITY_UNITS = {
    "longitudinal_acceleration_mps2": ("m/s²", "m/s^2", "m/s2"),
    "travel_to_impact_m": ("m",), "relative_distance_m": ("m",),
    "model_valid_until_s": ("s",),
}


def validated_physical_semantics(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict) or set(value) - set(PHYSICAL_SEMANTIC_FIELDS):
        raise ValueError("INVALID_PHYSICAL_SEMANTICS: unknown fields or non-object")
    for key, item in value.items():
        if key == "model_valid_until_s":
            if isinstance(item, bool) or not isinstance(item, (int, float)) or not math.isfinite(item) or item < 0:
                raise ValueError("INVALID_PHYSICAL_SEMANTICS: horizon must be a finite nonnegative number")
        elif not isinstance(item, str) or not item.strip():
            raise ValueError("INVALID_PHYSICAL_SEMANTICS: labels must be nonempty strings")
    return dict(value)


class ConstraintOperator(str, Enum):
    LT = "LT"
    LE = "LE"
    EQ = "EQ"
    GE = "GE"
    GT = "GT"
    RANGE = "RANGE"


class ProjectFactOutputType(str, Enum):
    SPEED_ENVELOPE = "SPEED_ENVELOPE"
    RISK_FACT = "RISK_FACT"


@dataclass(frozen=True)
class RiskFact:
    """A source-grounded project/scenario fact independent from any template."""

    fact_id: str
    parameter: str
    value: str | float
    unit: str = ""
    context: dict[str, str] = field(default_factory=dict)
    source_refs: list[SourceRef] = field(default_factory=list)
    provenance: FactProvenance = FactProvenance.LLM_INFERENCE
    approval: ReviewStatus = ReviewStatus.PENDING
    produced_by: str = "bounded_semantic_interpretation"
    physical_semantics: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.fact_id.strip() or not self.parameter.strip():
            raise ValueError("RiskFact requires fact_id and parameter")
        if isinstance(self.value, bool) or not isinstance(
            self.value, (str, int, float)
        ):
            raise ValueError("RiskFact.value must be a string or number")
        if isinstance(self.value, str) and not self.value.strip():
            raise ValueError("RiskFact.value must not be blank")
        if isinstance(self.value, (int, float)) and not math.isfinite(
            float(self.value)
        ):
            raise ValueError("RiskFact.value must be finite")
        if not self.source_refs:
            raise ValueError("RiskFact requires an exact SourceRef")
        _validate_context(self.context)
        validated_physical_semantics(self.physical_semantics)
        if (
            self.approval is ReviewStatus.FINALIZED
            and self.provenance is FactProvenance.LLM_INFERENCE
        ):
            raise ValueError(
                "FINALIZED RiskFact cannot rely on LLM inference alone"
            )


@dataclass(frozen=True)
class MethodRiskFactBinding:
    """Template-specific mapping from a reusable RiskFact to Method IR."""

    source_fact_id: str
    target_fact_type: str
    method_contract_hash: str
    source_refs: list[SourceRef] = field(default_factory=list)
    provenance: FactProvenance = FactProvenance.DERIVED
    approval: ReviewStatus = ReviewStatus.PENDING
    binding_method: str = "automatic_exact"

    def __post_init__(self) -> None:
        if not self.source_fact_id.strip() or not self.target_fact_type.strip():
            raise ValueError(
                "MethodRiskFactBinding requires source and target identities"
            )
        if len(self.method_contract_hash) != 64 or any(
            char not in "0123456789abcdefABCDEF" for char in self.method_contract_hash
        ):
            raise ValueError(
                "MethodRiskFactBinding requires a 64-character template hash"
            )
        if not self.source_refs:
            raise ValueError("MethodRiskFactBinding requires an exact SourceRef")
        if self.approval is ReviewStatus.FINALIZED and self.provenance not in {
            FactProvenance.METHOD_CONTRACT,
            FactProvenance.HUMAN_CONFIRMATION,
        }:
            raise ValueError(
                "FINALIZED MethodRiskFactBinding requires deterministic or human authority"
            )


def _validate_context(context: dict[str, str]) -> None:
    if not isinstance(context, dict):
        raise ValueError("Project Fact context must be an object")
    if any(not isinstance(key, str) or not isinstance(value, str) for key, value in context.items()):
        raise ValueError("Project Fact context must contain only string keys and values")


def risk_fact_from_dict(value: dict[str, Any]) -> RiskFact:
    payload = dict(value)
    payload["source_refs"] = [
        item if isinstance(item, SourceRef) else SourceRef(**item)
        for item in payload.get("source_refs", [])
    ]
    payload["provenance"] = FactProvenance(
        payload.get("provenance", FactProvenance.LLM_INFERENCE.value)
    )
    payload["approval"] = ReviewStatus(
        payload.get("approval", ReviewStatus.PENDING.value)
    )
    return RiskFact(**payload)


def method_risk_fact_binding_from_dict(
    value: dict[str, Any],
) -> MethodRiskFactBinding:
    payload = dict(value)
    payload["source_refs"] = [
        item if isinstance(item, SourceRef) else SourceRef(**item)
        for item in payload.get("source_refs", [])
    ]
    payload["provenance"] = FactProvenance(
        payload.get("provenance", FactProvenance.DERIVED.value)
    )
    payload["approval"] = ReviewStatus(
        payload.get("approval", ReviewStatus.PENDING.value)
    )
    return MethodRiskFactBinding(**payload)
