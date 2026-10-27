from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from hara_agent.models.common import ReviewStatus
from hara_agent.models.evidence import EvidenceKind

from .causal_graph import CausalEdge, CausalGraph, CausalNode, CausalNodeType, CausalRelation
from .evidence_binding import EvidenceBinding


SCENARIO_CAUSAL_ASSESSMENT_VERSION = "scenario-causal-assessment-v4"


class CausalBreakpoint(str, Enum):
    M_TO_B = "M_TO_B"
    B_TO_I = "B_TO_I"
    I_TO_H = "I_TO_H"
    H_TO_HARM = "H_TO_HARM"
    NONE = "NONE"


class CausalAssessmentStatus(str, Enum):
    VALIDATED = "VALIDATED"
    PENDING_CAUSAL_EVIDENCE = "PENDING_CAUSAL_EVIDENCE"
    CAUSAL_GAP = "CAUSAL_GAP"
    PENDING_BREAKPOINT = "PENDING_BREAKPOINT"


@dataclass(frozen=True)
class RiskDimensionChange:
    dimension: str
    evidence_refs: tuple[str, ...]
    reason: str

    def __post_init__(self) -> None:
        if not self.dimension.strip() or not self.reason.strip():
            raise ValueError("RiskDimensionChange requires dimension and reason")
        if not self.evidence_refs or any(not item.strip() for item in self.evidence_refs):
            raise ValueError("RiskDimensionChange requires evidence_refs")
        if len(self.evidence_refs) != len(set(self.evidence_refs)):
            raise ValueError("RiskDimensionChange.evidence_refs must be unique")

    def to_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "evidence_refs": list(self.evidence_refs),
            "reason": self.reason,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "RiskDimensionChange":
        return cls(
            dimension=str(value["dimension"]),
            evidence_refs=tuple(str(item) for item in value.get("evidence_refs", [])),
            reason=str(value["reason"]),
        )


@dataclass(frozen=True)
class ScenarioCausalAssessment:
    scenario_id: str
    causal_graph: CausalGraph
    causal_chain: tuple[str, ...]
    breakpoint: CausalBreakpoint | None
    evidence_bindings: tuple[EvidenceBinding, ...]
    unsupported_links: tuple[str, ...]
    risk_dimension_changes: tuple[RiskDimensionChange, ...]
    hazardous_event: str = ""
    potential_harm: str = ""
    provenance: tuple[str, ...] = ()
    review_status: ReviewStatus = ReviewStatus.PENDING
    contract_version: str = SCENARIO_CAUSAL_ASSESSMENT_VERSION
    status: CausalAssessmentStatus = field(init=False)

    def __post_init__(self) -> None:
        if not self.scenario_id.strip():
            raise ValueError("ScenarioCausalAssessment.scenario_id must not be empty")
        if self.contract_version != SCENARIO_CAUSAL_ASSESSMENT_VERSION:
            raise ValueError(
                f"unsupported Scenario causal contract: {self.contract_version}"
            )
        if len(self.causal_chain) != len(set(self.causal_chain)):
            raise ValueError("Scenario causal_chain node IDs must be unique")
        known_nodes = {item.node_id for item in self.causal_graph.nodes}
        if any(item not in known_nodes for item in self.causal_chain):
            raise ValueError("Scenario causal_chain references an unknown node")
        edge_ids = {item.edge_id for item in self.causal_graph.edges}
        binding_ids = [item.edge_id for item in self.evidence_bindings]
        if len(binding_ids) != len(set(binding_ids)):
            raise ValueError("Scenario evidence bindings must be unique by edge_id")
        if any(item not in edge_ids for item in binding_ids):
            raise ValueError("Scenario evidence binding references an unknown edge")
        object.__setattr__(self, "status", self._evaluate_status())

    @property
    def is_validated(self) -> bool:
        return self.status is CausalAssessmentStatus.VALIDATED

    def _evaluate_status(self) -> CausalAssessmentStatus:
        if self.breakpoint is None:
            return CausalAssessmentStatus.PENDING_BREAKPOINT
        if not self._has_complete_domain_neutral_path():
            return CausalAssessmentStatus.CAUSAL_GAP
        binding_by_edge = {item.edge_id: item for item in self.evidence_bindings}
        for edge in self.causal_graph.edges:
            binding = binding_by_edge.get(edge.edge_id)
            if (
                not edge.evidence_refs
                or binding is None
                or not binding.evidence_refs
                or tuple(edge.evidence_refs) != tuple(binding.evidence_refs)
            ):
                return CausalAssessmentStatus.PENDING_CAUSAL_EVIDENCE
            if binding.status in {ReviewStatus.REJECTED, ReviewStatus.NOT_APPLICABLE}:
                return CausalAssessmentStatus.PENDING_CAUSAL_EVIDENCE
            if binding.basis_type is EvidenceKind.ASSUMPTION:
                return CausalAssessmentStatus.CAUSAL_GAP
        if self.breakpoint is not CausalBreakpoint.NONE or self.unsupported_links:
            return CausalAssessmentStatus.CAUSAL_GAP
        # v4 stops the semantic stage at Hazardous Event.  The optional harm
        # node is attached later by the deterministic risk stage.  Keep the
        # five-node check for explicitly restored legacy-shaped objects so
        # old in-memory fixtures remain inspectable, but never require harm
        # from a four-node v4 causal result.
        if not self.hazardous_event.strip():
            return CausalAssessmentStatus.CAUSAL_GAP
        has_harm_node = any(
            item.node_type is CausalNodeType.HARM for item in self.causal_graph.nodes
        )
        if has_harm_node and not self.potential_harm.strip():
            return CausalAssessmentStatus.CAUSAL_GAP
        return CausalAssessmentStatus.VALIDATED

    def _has_complete_domain_neutral_path(self) -> bool:
        node_by_id = {item.node_id: item for item in self.causal_graph.nodes}
        expected = (
            CausalNodeType.MALFUNCTION,
            CausalNodeType.SYSTEM_BEHAVIOR_CHANGE,
            CausalNodeType.OPERATIONAL_CONSEQUENCE,
            CausalNodeType.HAZARD,
        )
        if any(item.node_type is CausalNodeType.HARM for item in node_by_id.values()):
            expected += (CausalNodeType.HARM,)
        actual = tuple(node_by_id[item].node_type for item in self.causal_chain)
        return actual == expected and self.causal_graph.has_path(self.causal_chain)

    def with_harm(
        self,
        *,
        potential_harm: str,
        evidence_refs: tuple[str, ...],
        source_refs: tuple[Any, ...] = (),
        basis_type: EvidenceKind = EvidenceKind.APPROVED_RULE,
        review_status: ReviewStatus | None = None,
    ) -> "ScenarioCausalAssessment":
        """Bind the current S-derived harm, replacing a prior scoring tail."""

        if not potential_harm.strip() or not evidence_refs:
            raise ValueError("Deterministic harm attachment requires value and evidence")
        # Legacy five-node checkpoints may already contain a harm label. Keep
        # their M→B→I→H evidence, but do not retain an obsolete H→Harm result.
        old_harm_ids = {item.node_id for item in self.causal_graph.nodes
                        if item.node_type is CausalNodeType.HARM}
        chain = tuple(identity for identity in self.causal_chain if identity not in old_harm_ids)
        bindings = tuple(item for item in self.evidence_bindings if item.edge_id != "H_TO_HARM")
        old_harm_refs = {ref for item in self.evidence_bindings if item.edge_id == "H_TO_HARM"
                         for ref in item.evidence_refs}
        retained_refs = {ref for item in bindings for ref in item.evidence_refs}
        provenance = tuple(ref for ref in self.provenance if ref not in old_harm_refs or ref in retained_refs)
        hazard_id = chain[-1]
        harm_id = f"{self.scenario_id}:harm"
        harm_node = CausalNode(
            harm_id, CausalNodeType.HARM, potential_harm,
            tuple(evidence_refs),
        )
        edge = CausalEdge(
            "H_TO_HARM", hazard_id, harm_id, CausalRelation.CAUSES,
            f"{self.hazardous_event} is associated with {potential_harm}",
            tuple(evidence_refs),
        )
        binding = EvidenceBinding(
            "H_TO_HARM", tuple(evidence_refs), basis_type,
            tuple(source_refs),
            review_status or ReviewStatus.PENDING,
        )
        graph = CausalGraph(
            nodes=(*(n for n in self.causal_graph.nodes if n.node_id not in old_harm_ids), harm_node),
            edges=(*(e for e in self.causal_graph.edges if e.target not in old_harm_ids), edge),
        )
        return ScenarioCausalAssessment(
            scenario_id=self.scenario_id,
            causal_graph=graph,
            causal_chain=(*chain, harm_id),
            breakpoint=self.breakpoint,
            evidence_bindings=(*bindings, binding),
            unsupported_links=tuple(
                item for item in self.unsupported_links if item != "H_TO_HARM"
            ),
            risk_dimension_changes=self.risk_dimension_changes,
            hazardous_event=self.hazardous_event,
            potential_harm=potential_harm,
            provenance=tuple(dict.fromkeys((*provenance, *evidence_refs))),
            review_status=(
                self.review_status
                if review_status is None else review_status
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario_id": self.scenario_id,
            "causal_graph": self.causal_graph.to_dict(),
            "causal_chain": list(self.causal_chain),
            "breakpoint": self.breakpoint.value if self.breakpoint is not None else None,
            "evidence_bindings": [item.to_dict() for item in self.evidence_bindings],
            "unsupported_links": list(self.unsupported_links),
            "risk_dimension_changes": [
                item.to_dict() for item in self.risk_dimension_changes
            ],
            "hazardous_event": self.hazardous_event,
            "potential_harm": self.potential_harm,
            "status": self.status.value,
            "review_status": self.review_status.value,
            "provenance": list(self.provenance),
            "contract_version": self.contract_version,
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "ScenarioCausalAssessment":
        breakpoint = value.get("breakpoint")
        result = cls(
            scenario_id=str(value["scenario_id"]),
            causal_graph=CausalGraph.from_dict(value["causal_graph"]),
            causal_chain=tuple(str(item) for item in value.get("causal_chain", [])),
            breakpoint=CausalBreakpoint(breakpoint) if breakpoint is not None else None,
            evidence_bindings=tuple(
                EvidenceBinding.from_dict(item)
                for item in value.get("evidence_bindings", [])
            ),
            unsupported_links=tuple(
                str(item) for item in value.get("unsupported_links", [])
            ),
            risk_dimension_changes=tuple(
                RiskDimensionChange.from_dict(item)
                for item in value.get("risk_dimension_changes", [])
            ),
            hazardous_event=str(value.get("hazardous_event", "")),
            potential_harm=str(value.get("potential_harm", "")),
            provenance=tuple(str(item) for item in value.get("provenance", [])),
            review_status=ReviewStatus(
                value.get("review_status", ReviewStatus.PENDING.value)
            ),
            contract_version=str(
                value.get("contract_version", SCENARIO_CAUSAL_ASSESSMENT_VERSION)
            ),
        )
        serialized_status = value.get("status")
        if serialized_status is not None and serialized_status != result.status.value:
            raise ValueError("Scenario causal status does not match deterministic validation")
        return result
