from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
import re
from typing import Any
from copy import deepcopy

from hara_agent.models import (
    FactProvenance,
    ItemDefinitionFacts,
    ReviewStatus,
    SpeedEnvelope,
    SourceRef,
)


def effective_project_policy(policy: dict, item_sha256: str = "") -> dict:
    """Scope project decisions to their recorded input, without changing methods.

    Missing historical identity is not evidence of membership. Numerical and
    conditional reference methods remain available; only project-only decisions
    and location conventions are filtered. The compiled authority stays intact.
    """
    effective = deepcopy(policy)
    bound = bool(item_sha256 and item_sha256 in policy.get("bound_item_sha256", ()))
    if not bound:
        effective["project_scope"] = ""
        effective["engineering_decisions"] = {}
        effective.pop("driver_location_convention", None)
    effective.pop("decisions", None)  # Historical narrative, not executable policy.
    return effective


class ProjectFactResolutionError(ValueError):
    """Raised when context-specific project facts cannot be resolved safely."""


class ProjectContextResolutionStatus(str, Enum):
    RESOLVED = "RESOLVED"
    RESOLVED_AGGREGATE_FALLBACK = "RESOLVED_AGGREGATE_FALLBACK"
    UNRESOLVED_PROJECT_CONTEXT = "UNRESOLVED_PROJECT_CONTEXT"


@dataclass(frozen=True)
class SpeedResolutionResult:
    operating_mode: str
    requested_fact: str
    resolved_value: float | None
    resolution_source: str
    provenance: FactProvenance
    source_refs: tuple[SourceRef, ...]
    approval: ReviewStatus
    fallback_used: bool
    resolution_status: ProjectContextResolutionStatus
    matched_envelope: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["provenance"] = self.provenance.value
        result["approval"] = self.approval.value
        result["resolution_status"] = self.resolution_status.value
        return result


class UnresolvedProjectContextError(ProjectFactResolutionError):
    def __init__(self, operating_mode: str, reason: str):
        self.operating_mode = operating_mode
        self.reason = reason
        self.resolution_status = ProjectContextResolutionStatus.UNRESOLVED_PROJECT_CONTEXT
        super().__init__(
            f"UNRESOLVED_PROJECT_CONTEXT operating_mode={operating_mode!r}: {reason}"
        )
    def to_dict(self) -> dict[str, Any]:
        return {
            "operating_mode": self.operating_mode,
            "requested_fact": "ego_speed_kph",
            "resolution_status": self.resolution_status.value,
            "reason": self.reason,
            "fallback_used": False,
        }


def canonical_operating_mode(value: str) -> str:
    """Normalize representation only; semantic aliases belong to input data."""

    return re.sub(
        r"_+", "_", value.strip().casefold().replace("-", "_").replace(" ", "_")
    ).strip("_")


class ProjectFactResolver:
    """Resolve mode-specific facts; aggregate resolution is explicit opt-in."""

    def resolve_speed_envelope(
        self,
        facts: ItemDefinitionFacts,
        operating_mode: str,
        *,
        allow_aggregate_fallback: bool = False,
    ) -> SpeedEnvelope:
        requested = canonical_operating_mode(operating_mode)
        if not requested:
            raise UnresolvedProjectContextError(operating_mode, "operating_mode must not be empty")
        matches = [
            item for item in facts.speed_envelopes
            if canonical_operating_mode(item.operating_mode) == requested
        ]
        if len(matches) == 1:
            return matches[0]
        if len(matches) > 1:
            raise UnresolvedProjectContextError(
                operating_mode,
                f"ambiguous speed envelope for operating mode {operating_mode!r}"
            )
        if not allow_aggregate_fallback:
            available_envelopes = sorted({
                item.operating_mode for item in facts.speed_envelopes
                if item.operating_mode.strip()
            })
            declared_modes = sorted({
                item for item in facts.operating_modes if item.strip()
            })
            raise UnresolvedProjectContextError(
                operating_mode,
                f"no speed envelope for operating mode {operating_mode!r}; "
                "aggregate fallback was not authorized; "
                f"available_speed_envelope_modes={available_envelopes}; "
                f"declared_operating_modes={declared_modes}; "
                "use an exact source-defined operating mode or provide an explicit "
                "project speed input with provenance"
            )
        if len(facts.speed_envelopes) > 1:
            raise UnresolvedProjectContextError(
                operating_mode,
                "aggregate fallback cannot collapse multiple contextual speed envelopes",
            )
        if facts.speed_min_kph is None or facts.speed_max_kph is None:
            raise UnresolvedProjectContextError(
                operating_mode,
                f"no aggregate speed envelope available for operating mode {operating_mode!r}"
            )
        return SpeedEnvelope(
            operating_mode=operating_mode,
            speed_min_kph=facts.speed_min_kph,
            speed_max_kph=facts.speed_max_kph,
            condition="explicit aggregate speed fallback",
            sources=list(facts.sources),
            provenance=FactProvenance.DERIVED,
            status=ReviewStatus.PENDING,
        )
    def resolve_speed_kph(
        self,
        facts: ItemDefinitionFacts,
        operating_mode: str,
        *,
        allow_aggregate_fallback: bool = False,
    ) -> float:
        envelope = self.resolve_speed_envelope(
            facts,
            operating_mode,
            allow_aggregate_fallback=allow_aggregate_fallback,
        )
        if envelope.speed_max_kph is None:
            raise ProjectFactResolutionError(
                f"speed envelope for operating mode {operating_mode!r} has no maximum"
            )
        return float(envelope.speed_max_kph)

    def resolve_speed_context(
        self,
        facts: ItemDefinitionFacts,
        operating_mode: str,
        *,
        allow_aggregate_fallback: bool = False,
        allow_missing: bool = False,
    ) -> SpeedResolutionResult:
        # A draft candidate may carry an unknown operational envelope to the
        # existing hypothesis producer. This is not a numeric fallback and
        # must not absorb a mismatched, ambiguous or partially specified bound.
        if (allow_missing and canonical_operating_mode(operating_mode)
                and not facts.speed_envelopes
                and facts.speed_min_kph is None and facts.speed_max_kph is None):
            return SpeedResolutionResult(
                operating_mode=canonical_operating_mode(operating_mode),
                requested_fact="ego_speed_kph", resolved_value=None,
                resolution_source="NoSourceSpeedEnvelope",
                provenance=FactProvenance.PROJECT_INPUT, source_refs=(),
                approval=ReviewStatus.PENDING, fallback_used=False,
                resolution_status=ProjectContextResolutionStatus.UNRESOLVED_PROJECT_CONTEXT,
                matched_envelope={},
            )
        envelope = self.resolve_speed_envelope(
            facts,
            operating_mode,
            allow_aggregate_fallback=allow_aggregate_fallback,
        )
        if envelope.speed_max_kph is None:
            raise UnresolvedProjectContextError(
                operating_mode,
                "matched speed envelope has no conservative upper bound",
            )
        fallback_used = not any(item is envelope for item in facts.speed_envelopes)
        return SpeedResolutionResult(
            operating_mode=canonical_operating_mode(operating_mode),
            requested_fact="ego_speed_kph",
            resolved_value=float(envelope.speed_max_kph),
            resolution_source=(
                "AggregateSpeedEnvelope" if fallback_used else "SpeedEnvelope"
            ),
            provenance=envelope.provenance,
            source_refs=tuple(envelope.sources),
            approval=envelope.status,
            fallback_used=fallback_used,
            resolution_status=(
                ProjectContextResolutionStatus.RESOLVED_AGGREGATE_FALLBACK
                if fallback_used else ProjectContextResolutionStatus.RESOLVED
            ),
            matched_envelope={
                "operating_mode": envelope.operating_mode,
                "speed_min_kph": envelope.speed_min_kph,
                "speed_max_kph": envelope.speed_max_kph,
                "condition": envelope.condition,
                "unit": envelope.unit,
            },
        )

    @staticmethod
    def explicit_project_speed(
        operating_mode: str,
        speed_kph: float,
        *,
        sources: list[SourceRef] | None = None,
    ) -> SpeedResolutionResult:
        if not canonical_operating_mode(operating_mode):
            raise UnresolvedProjectContextError(
                operating_mode, "explicit speed requires a structured operating_mode"
            )
        if speed_kph < 0:
            raise ValueError("speed_kph must not be negative")
        refs = tuple(sources or [])
        return SpeedResolutionResult(
            operating_mode=canonical_operating_mode(operating_mode),
            requested_fact="ego_speed_kph",
            resolved_value=float(speed_kph),
            resolution_source="ExplicitProjectInput",
            provenance=FactProvenance.PROJECT_INPUT,
            source_refs=refs,
            approval=(
                ReviewStatus.FINALIZED if refs else ReviewStatus.PENDING
            ),
            fallback_used=False,
            resolution_status=ProjectContextResolutionStatus.RESOLVED,
            matched_envelope={
                "operating_mode": operating_mode,
                "speed_min_kph": speed_kph,
                "speed_max_kph": speed_kph,
                "condition": "explicit project speed override",
                "unit": "km/h",
            },
        )
