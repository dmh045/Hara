from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, is_dataclass
from enum import Enum
from typing import Any

from hara_agent.contracts.causal_graph import (
    CausalEdge, CausalGraph, CausalNode, CausalNodeType, CausalRelation,
)
from hara_agent.contracts.evidence_binding import EvidenceBinding
from hara_agent.contracts.scenario_causal_assessment import (
    SCENARIO_CAUSAL_ASSESSMENT_VERSION, CausalBreakpoint,
    RiskDimensionChange, ScenarioCausalAssessment,
)
from hara_agent.models import (
    EvidenceKind, EvidenceRecord, FactProvenance, MalfunctionCandidate,
    ReviewStatus, ScenarioCandidate, SourceRef,
)
from hara_agent.services.analysis.scenario_physics import (
    DerivedPhysicsType, derive_scenario_physics, has_analysis_assumption_lineage,
    source_is_accepted_for,
    has_conditional_lineage,
)

from .scenario_contract import RISK_DIMENSION_VALUES, SCENARIO_CONTRACT_VERSION


SCENARIO_ASSESSMENT_CONTRACT_VERSION = SCENARIO_CAUSAL_ASSESSMENT_VERSION
_VALIDATED_ANALYTICAL_ATOM_BASIS = "VALIDATED_ANALYTICAL_SCENARIO_ATOM"


def _conditional_evidence(record, malfunction_id=None, scenario_id=None):
    """A verified condition can support a conditional proof, not fact approval."""
    if isinstance(record, EvidenceRecord):
        metadata = {**record.metadata, "provenance": record.provenance.value,
            "approval": record.approval_status.value, "source_refs": [vars(s) for s in record.source_refs]}
    else:
        metadata = record
    if not isinstance(metadata, dict) or not has_conditional_lineage(metadata):
        return False
    scope = metadata.get("analysis_assumption_scope", metadata.get("applicable_scope", {}))
    return source_is_accepted_for(metadata,
        malfunction_id=malfunction_id or scope.get("malfunction_id", ""),
        scenario_id=scenario_id or scope.get("scenario_id", ""))


# v1 hop basis values and Registry evidence kinds intentionally share values,
# while kind and provenance remain separate axes.
EvidenceBasisType = EvidenceKind


def _is_validated_analytical_scenario_atom(value: EvidenceRecord | dict[str, Any]) -> bool:
    if isinstance(value, EvidenceRecord):
        metadata = value.metadata
        approval = value.approval_status.value
        sources = value.source_refs
        provenance = value.provenance.value
    else:
        metadata = value
        approval = str(value.get("approval_status", ""))
        sources = value.get("source_refs", [])
        provenance = str(value.get("provenance", ""))
    return (
        provenance == FactProvenance.SCENARIO_DEFINED.value
        and approval == ReviewStatus.FINALIZED.value
        and bool(sources)
        and metadata.get("selection_basis") == _VALIDATED_ANALYTICAL_ATOM_BASIS
    )


class ScenarioEvidenceErrorCode(str, Enum):
    INVALID_BREAKPOINT = "INVALID_BREAKPOINT"
    CAUSAL_BREAKPOINT_MISMATCH = "CAUSAL_BREAKPOINT_MISMATCH"
    INVALID_CAUSAL_CHAIN_SHAPE = "INVALID_CAUSAL_CHAIN_SHAPE"
    REQUIRED_HOP_MISSING = "REQUIRED_HOP_MISSING"
    EMPTY_HOP_CLAIM = "EMPTY_HOP_CLAIM"
    INVALID_EVIDENCE_REF_SHAPE = "INVALID_EVIDENCE_REF_SHAPE"
    INVALID_BASIS_TYPE = "INVALID_BASIS_TYPE"
    UNRESOLVED_EVIDENCE_REF = "UNRESOLVED_EVIDENCE_REF"
    MISSING_EVIDENCE_REF = "MISSING_EVIDENCE_REF"
    DIRECT_FACT_KIND_MISMATCH = "DIRECT_FACT_KIND_MISMATCH"
    DERIVED_PHYSICS_KIND_MISMATCH = "DERIVED_PHYSICS_KIND_MISMATCH"
    APPROVED_RULE_KIND_MISMATCH = "APPROVED_RULE_KIND_MISMATCH"
    ASSUMPTION_IN_POSITIVE_CHAIN = "ASSUMPTION_IN_POSITIVE_CHAIN"
    INVALID_RISK_DIMENSION_SHAPE = "INVALID_RISK_DIMENSION_SHAPE"
    UNKNOWN_RISK_DIMENSION = "UNKNOWN_RISK_DIMENSION"
    DUPLICATE_RISK_DIMENSION = "DUPLICATE_RISK_DIMENSION"
    INVALID_RISK_DIMENSION_EVIDENCE = "INVALID_RISK_DIMENSION_EVIDENCE"
    MISSING_RISK_DIMENSION_REASON = "MISSING_RISK_DIMENSION_REASON"
    CAUSAL_FALSE_WITH_DIMENSIONS = "CAUSAL_FALSE_WITH_DIMENSIONS"
    CAUSAL_TRUE_WITHOUT_DIMENSIONS = "CAUSAL_TRUE_WITHOUT_DIMENSIONS"
    SELF_REFERENTIAL_CAUSAL_EVIDENCE = "SELF_REFERENTIAL_CAUSAL_EVIDENCE"


class ScenarioEvidenceContractError(ValueError):
    """A Scenario result cites missing or impermissible causal evidence."""

    def __init__(self, message: str, *, code: ScenarioEvidenceErrorCode, hop: str,
                 reason: str, malfunction_id: str, scenario_id: str,
                 semantic_fingerprint: str, invalid_evidence_refs: list[str],
                 basis_type: str, claim: str):
        super().__init__(message)
        self.code = code
        self.hop = hop
        self.reason = reason
        self.malfunction_id = malfunction_id
        self.scenario_id = scenario_id
        self.semantic_fingerprint = semantic_fingerprint
        self.invalid_evidence_refs = list(invalid_evidence_refs)
        self.basis_type = basis_type
        self.claim = claim


class FactRegistry:
    """Typed evidence store with fail-closed namespaces and collisions."""

    def __init__(self):
        self._records: dict[str, EvidenceRecord] = {}
        self._duplicate_ref_count = 0

    @property
    def records(self) -> tuple[EvidenceRecord, ...]:
        return tuple(self._records.values())

    def register(self, record: EvidenceRecord) -> None:
        if record.evidence_ref in self._records:
            self._duplicate_ref_count += 1
            raise DuplicateEvidenceRefError(record.evidence_ref)
        self._records[record.evidence_ref] = record

    def extend(self, records) -> None:
        for record in records:
            self.register(record)

    def to_prompt_dict(self) -> dict[str, dict[str, Any]]:
        def value_of(value: Any) -> Any:
            return asdict(value) if is_dataclass(value) else value

        return {
            ref: {
                "value": value_of(record.value),
                "kind": record.kind.value,
                "provenance": record.provenance.value,
                "approval_status": record.approval_status.value,
                **{
                    key: record.metadata[key]
                    for key in (
                        "evidence_role", "rule_id", "method_source_hash",
                        "source_rule_id", "semantic_fingerprint",
                    ) if key in record.metadata
                },
                **(
                    {
                        "derivation_type": record.metadata.get("derivation_type"),
                        "inputs": record.metadata.get("inputs", []),
                    }
                    if record.kind is EvidenceKind.DERIVED_PHYSICS else {}
                ),
                **(
                    {"evidence_role": record.metadata["evidence_role"]}
                    if record.metadata.get("evidence_role") else {}
                ),
            }
            for ref, record in self._records.items()
        }

    def resolve(self, evidence_ref: str) -> dict[str, Any] | None:
        record = self._records.get(evidence_ref)
        return self._record_dict(record) if record else None

    def resolve_record(self, evidence_ref: str) -> EvidenceRecord | None:
        return self._records.get(evidence_ref)

    def snapshot(self, *, include_values: bool = False) -> dict[str, Any]:
        return {
            "records": [
                record.to_dict(include_value=include_values)
                for record in self._records.values()
            ],
            "diagnostics": {
                "record_count": len(self._records),
                "duplicate_ref_count": self._duplicate_ref_count,
            },
        }

    @staticmethod
    def _record_dict(record: EvidenceRecord) -> dict[str, Any]:
        return {
            "value": record.value,
            "kind": record.kind.value,
            "provenance": record.provenance.value,
            "approval_status": record.approval_status.value,
            "source_refs": [
                {
                    "source_type": item.source_type,
                    "source_id": item.source_id,
                    "location": item.location,
                    "excerpt": item.excerpt,
                }
                for item in record.source_refs
            ],
            **record.metadata,
        }


DEFAULT_CAUSAL_EVIDENCE_BUDGET = 10


@dataclass(frozen=True)
class CausalEvidenceSelection:
    """A compact, causal-only view over the complete FactRegistry.

    The complete records remain available to the validator and audit dump.  This
    object is deliberately presentation-only: it contains no source excerpts,
    provenance, or method metadata for the LLM prompt.
    """

    selected_refs: tuple[str, ...]
    mandatory_evidence_refs: tuple[str, ...]
    selected_context_evidence_refs: tuple[str, ...]
    candidate_evidence_count: int
    dropped_evidence_count: int
    evidence_selection_reason: str
    compact_view: str

    @property
    def selected_evidence_count(self) -> int:
        return len(self.selected_context_evidence_refs)

    @property
    def total_prompt_evidence_count(self) -> int:
        return len(self.selected_refs)

    @property
    def compact_chars(self) -> int:
        return len(self.compact_view)

    @property
    def compact_tokens(self) -> int:
        # A stable audit estimate is sufficient here; provider tokenizers are
        # provider-specific and must not influence semantic selection.
        return (self.compact_chars + 3) // 4


class CausalEvidenceSelector:
    """Select bounded causal context without changing evidence authority."""

    _P1_TERMS = (
        "object_position", "object_speed", "object_type", "longitudinal_direction",
        "closing_speed", "initial_speed", "impact_time", "speed_value_semantic",
        "relative_distance", "relative_speed", "ttc", "delta_v",
        "impact_speed", "ego_speed", "collision", "traffic_object", "road_user",
        "scenario_object", "stopping_margin", "geometry", "braking_distance",
    )
    _P2_TERMS = (
        "driver_in_vehicle", "direct_control", "remote_intervention",
        "other_road_user_avoidance", "vehicle_stability", "intervention",
        "control_available", "avoidance_possible",
    )
    _P3_TERMS = (
        "operating_mode", "road", "weather", "surface", "ego_action",
        "ego_dynamics", "dynamics", "location", "driver_position",
        "operating_scenario", "vehicle_state",
    )
    _P4_TERMS = (
        "exposure", "e_z", "e_f", "asil", "ftti", "severity",
        "controllability", "combination",
    )

    def __init__(self, *, max_evidence: int = DEFAULT_CAUSAL_EVIDENCE_BUDGET):
        if not isinstance(max_evidence, int) or isinstance(max_evidence, bool):
            raise ValueError("causal evidence budget must be an integer")
        if max_evidence <= 0:
            raise ValueError("causal evidence budget must be greater than zero")
        self.max_evidence = max_evidence

    @classmethod
    def _rank(cls, record: EvidenceRecord) -> tuple[int, str] | None:
        ref = record.evidence_ref
        lower = ref.casefold()
        metadata = record.metadata if isinstance(record.metadata, dict) else {}
        # Malfunction semantics are already presented in the fixed prompt
        # header; repeating MF records in every item is not useful causal
        # context and would invite self-referential proof.
        if record.namespace == "MF":
            return None
        validated_atom = _is_validated_analytical_scenario_atom(record)
        if (
            record.provenance is FactProvenance.SCENARIO_DEFINED
            or has_analysis_assumption_lineage(metadata)
        ) and not validated_atom and not _conditional_evidence(record):
            return None
        if record.namespace == "METHOD" and metadata.get("causal_relevance") is not True:
            return None
        if record.namespace == "APPROVED_RULE" and metadata.get("causal_relevance") is not True:
            return None
        # Exposure/S/E/C/ASIL/FTTI records are downstream method context.  An
        # explicitly causal approved rule is the only opt-in exception.
        if metadata.get("causal_relevance") is not True and any(
            term in lower for term in cls._P4_TERMS
        ):
            return None
        for priority, terms, reason in (
            (1, cls._P1_TERMS, "P1 direct physical/interaction evidence"),
            (2, cls._P2_TERMS, "P2 intervention/control context"),
            (3, cls._P3_TERMS, "P3 scenario operating context"),
        ):
            if any(term in lower for term in terms):
                return priority, reason
        if metadata.get("causal_relevance") is True:
            return 3, "approved rule explicitly marked causal_relevance"
        return None

    @staticmethod
    def _compact_value(value: Any) -> str:
        if is_dataclass(value):
            value = asdict(value)
        return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)

    @staticmethod
    def _mandatory_records(registry: FactRegistry) -> list[EvidenceRecord]:
        """Return finalized malfunction definition anchors in stable order."""

        records = []
        for ref in ("MF.description", "MF.functional_effect"):
            record = registry.resolve_record(ref)
            if record is not None and record.approval_status is ReviewStatus.FINALIZED:
                records.append(record)
        return records

    @staticmethod
    def _semantic_key(record: EvidenceRecord) -> str:
        """Identify a derived canonical copy of a direct Scenario fact."""

        metadata = record.metadata if isinstance(record.metadata, dict) else {}
        inputs = metadata.get("inputs", [])
        if (
            record.kind is EvidenceKind.DERIVED_PHYSICS
            and metadata.get("derivation_type") == "CANONICAL_INPUT_NORMALIZATION"
            and len(inputs) == 1
            and str(inputs[0]).startswith("SCN.")
        ):
            return str(inputs[0]).casefold()
        return record.evidence_ref.casefold()

    def select(self, registry: FactRegistry) -> CausalEvidenceSelection:
        mandatory = self._mandatory_records(registry)
        ranked: list[tuple[int, str, EvidenceRecord, str]] = []
        for record in registry.records:
            ranked_value = self._rank(record)
            if ranked_value is None:
                continue
            priority, reason = ranked_value
            ranked.append((priority, record.evidence_ref, record, reason))
        ranked.sort(key=lambda item: (
            item[0],
            self._semantic_key(item[2]),
            0 if item[2].kind is EvidenceKind.DIRECT_FACT else 1,
            item[1],
        ))
        deduplicated: list[tuple[int, str, EvidenceRecord, str]] = []
        seen_semantic_keys: set[str] = set()
        for item in ranked:
            semantic_key = self._semantic_key(item[2])
            if semantic_key in seen_semantic_keys:
                continue
            seen_semantic_keys.add(semantic_key)
            deduplicated.append(item)
        selected = deduplicated[:self.max_evidence]
        mandatory_refs = tuple(record.evidence_ref for record in mandatory)
        selected_context_refs = tuple(item[1] for item in selected)
        selected_refs = tuple(dict.fromkeys(mandatory_refs + selected_context_refs))
        compact_view = "\n".join(
            f"{record.evidence_ref} [kind={record.kind.value}] = {self._compact_value(record.value)}"
            for record in mandatory
        )
        if compact_view and selected:
            compact_view += "\n"
        compact_view += "\n".join(
            f"{ref} [kind={record.kind.value}] = {self._compact_value(record.value)}"
            for _, ref, record, _ in selected
        )
        reason_counts: dict[str, int] = {}
        for _, _, _, reason in selected:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
        if mandatory:
            reason_counts["mandatory finalized malfunction anchors"] = len(mandatory)
        reason = "; ".join(
            f"{name}: {count}"
            for name, count in sorted(reason_counts.items())
        ) or "no P1-P3 causal evidence selected"
        return CausalEvidenceSelection(
            selected_refs=selected_refs,
            mandatory_evidence_refs=mandatory_refs,
            selected_context_evidence_refs=selected_context_refs,
            candidate_evidence_count=len(ranked),
            dropped_evidence_count=max(0, len(ranked) - len(selected)),
            evidence_selection_reason=reason,
            compact_view=compact_view,
        )


class DuplicateEvidenceRefError(ValueError):
    def __init__(self, evidence_ref: str):
        self.evidence_ref = evidence_ref
        super().__init__(f"duplicate evidence_ref rejected: {evidence_ref}")


def build_fact_registry(
    malfunction: MalfunctionCandidate, scenario: ScenarioCandidate,
    project_registry: FactRegistry | None = None,
) -> FactRegistry:
    registry = FactRegistry()
    if project_registry is not None:
        # Pair-scoped project risk facts are valid only for the exact
        # malfunction/Scenario pair that produced them.  In particular, a
        # fact attached to another analytical Scenario must not enter either
        # the prompt view or the deterministic validator for this child.
        def applies_to_current_pair(record: EvidenceRecord) -> bool:
            metadata = record.metadata if isinstance(record.metadata, dict) else {}
            context = metadata.get("context", {})
            if not isinstance(context, dict) or not context:
                return True
            expected = {
                "malfunction_id": malfunction.malfunction_id,
                "scenario_id": scenario.scenario_id,
            }
            for key, current in expected.items():
                scoped = str(context.get(key, "")).strip()
                if scoped and scoped.casefold() != str(current).strip().casefold():
                    return False
            return True

        registry.extend(
            record for record in project_registry.records
            if applies_to_current_pair(record)
        )
    malfunction_provenance = (
        FactProvenance.DERIVED
        if malfunction.sources else FactProvenance.LLM_INFERENCE
    )
    for name, value in (
        ("description", malfunction.description),
        ("functional_effect", malfunction.functional_effect),
        ("vehicle_level_hazard", malfunction.vehicle_level_hazard),
    ):
        registry.register(EvidenceRecord(
            f"MF.{name}", value, EvidenceKind.DIRECT_FACT,
            malfunction_provenance, malfunction.status, tuple(malfunction.sources),
            {
                "extracted_by": "LLM",
                **(
                    {"evidence_role": "UPSTREAM_CAUSAL_CLAIM"}
                    if name == "vehicle_level_hazard" else {}
                ),
            },
        ))
    for key, value in sorted(scenario.facts.items()):
        if (
            value is None
            or (isinstance(value, str) and not value.strip())
            or (isinstance(value, (list, tuple, dict, set)) and not value)
            or key.endswith("_source_status")
            or key in {
                "engineering_status", "review_reason", "method_scenario_dimensions",
            }
        ):
            continue
        metadata = scenario.fact_provenance.get(key, {})
        if not isinstance(metadata, dict):
            metadata = {}
        provenance = FactProvenance(
            metadata.get("provenance", FactProvenance.LLM_INFERENCE.value)
        )
        approval = ReviewStatus(metadata.get("approval", scenario.status.value))
        sources = tuple(
            item if isinstance(item, SourceRef) else SourceRef(**item)
            for item in metadata.get("source_refs", [])
        )
        fact_metadata = {name: item for name, item in metadata.items() if name not in {
            "provenance", "approval", "source_refs",
        }}
        if scenario.semantic_fingerprint:
            fact_metadata.setdefault("semantic_fingerprint", scenario.semantic_fingerprint)
        registry.register(EvidenceRecord(
            f"SCN.{key}", value, EvidenceKind.DIRECT_FACT,
            provenance, approval, sources, fact_metadata,
        ))
    registry.extend(derive_scenario_physics(scenario))
    # Exposure metadata is pre-causal context. It is method-sourced evidence,
    # not a final E judgement, and is therefore explicitly attached to the
    # atomic Scenario candidate.
    for item in scenario.exposure_context:
        if not isinstance(item, dict):
            continue
        atom_id = str(item.get("atom_id", "")).strip()
        if not atom_id:
            continue
        for key in ("e_z", "e_f"):
            value = item.get(key)
            if value in (None, ""):
                continue
            registry.register(EvidenceRecord(
                f"SCN.exposure.{atom_id}.{key}", value,
                EvidenceKind.APPROVED_RULE, FactProvenance.APPROVED_RULE,
                ReviewStatus.FINALIZED,
                tuple(source for source in scenario.sources if isinstance(source, SourceRef)),
                {
                    "atom_id": atom_id,
                    "dimension": item.get("dimension", []),
                    "source_rule_id": item.get("source_rule_id", ""),
                    "coupling": item.get("coupling", []),
                    "method_source_hash": item.get("method_source_hash", ""),
                    "semantic_fingerprint": scenario.semantic_fingerprint,
                },
            ))
    return registry


def validate_evidence_contract(
    *,
    malfunction: MalfunctionCandidate,
    scenario: ScenarioCandidate,
    item: dict[str, Any],
    registry: FactRegistry,
    prompt_version: str,
    batch: str,
    split_path: str,
    split_depth: int,
) -> tuple[list[str], dict[str, Any]]:
    causal = item["causally_relevant"]
    breakpoint_raw = item.get("breakpoint")
    try:
        breakpoint = CausalBreakpoint(breakpoint_raw)
    except (ValueError, TypeError) as exc:
        raise _error(malfunction, scenario, prompt_version, batch, split_path, split_depth,
                     code=ScenarioEvidenceErrorCode.INVALID_BREAKPOINT,
                     hop="NONE", claim="", basis_type="", invalid_refs=[],
                     reason=f"invalid breakpoint={breakpoint_raw!r}") from exc
    if causal != (breakpoint is CausalBreakpoint.NONE):
        raise _error(malfunction, scenario, prompt_version, batch, split_path, split_depth,
                     code=ScenarioEvidenceErrorCode.CAUSAL_BREAKPOINT_MISMATCH,
                     hop="NONE", claim="", basis_type="", invalid_refs=[],
                     reason=f"causal/breakpoint mismatch causal={causal} breakpoint={breakpoint.value}")

    chain = item.get("causal_chain")
    if not isinstance(chain, dict):
        shape = (
            f"list_length={len(chain)}"
            if isinstance(chain, list)
            else f"actual_type={type(chain).__name__}"
        )
        raise _error(malfunction, scenario, prompt_version, batch, split_path, split_depth,
                     code=ScenarioEvidenceErrorCode.INVALID_CAUSAL_CHAIN_SHAPE,
                     hop="causal_chain", claim="", basis_type="", invalid_refs=[],
                     reason=f"causal_chain must be object; {shape}")
    # v4 semantic assessment ends at Hazardous Event. A supplied v3-shaped
    # payload is validated as a legacy compatibility input, but v4 prompts
    # never request or generate the optional fourth hop.
    legacy_harm = "h_to_harm" in chain or bool(item.get("potential_harm", ""))
    hop_names = ("m_to_b", "b_to_i", "i_to_h", "h_to_harm") if legacy_harm else (
        "m_to_b", "b_to_i", "i_to_h"
    )
    breakpoint_index = {
        CausalBreakpoint.M_TO_B: 0, CausalBreakpoint.B_TO_I: 1,
        CausalBreakpoint.I_TO_H: 2,
        CausalBreakpoint.H_TO_HARM: 3 if legacy_harm else 2,
        CausalBreakpoint.NONE: 3 if legacy_harm else 2,
    }[breakpoint]
    assumption_hops = 0
    for index, hop_name in enumerate(hop_names):
        hop = chain.get(hop_name)
        required = causal or index <= breakpoint_index
        if not isinstance(hop, dict):
            if required:
                raise _error(malfunction, scenario, prompt_version, batch, split_path, split_depth,
                             code=ScenarioEvidenceErrorCode.REQUIRED_HOP_MISSING,
                             hop=hop_name, claim="", basis_type="", invalid_refs=[],
                             reason="required hop missing")
            continue
        claim = str(hop.get("claim", "")).strip()
        refs = hop.get("evidence_refs")
        basis_raw = hop.get("basis_type")
        try:
            basis = EvidenceBasisType(basis_raw)
        except (ValueError, TypeError) as exc:
            raise _error(malfunction, scenario, prompt_version, batch, split_path, split_depth,
                         code=ScenarioEvidenceErrorCode.INVALID_BASIS_TYPE,
                         hop=hop_name, claim=claim, basis_type=str(basis_raw), invalid_refs=[],
                         reason="invalid basis_type") from exc
        if not claim:
            raise _error(malfunction, scenario, prompt_version, batch, split_path, split_depth,
                         code=ScenarioEvidenceErrorCode.EMPTY_HOP_CLAIM,
                         hop=hop_name, claim=claim, basis_type=basis.value, invalid_refs=[],
                         reason="hop claim must be non-empty")
        if not isinstance(refs, list) or any(not isinstance(ref, str) for ref in refs):
            raise _error(malfunction, scenario, prompt_version, batch, split_path, split_depth,
                         code=ScenarioEvidenceErrorCode.INVALID_EVIDENCE_REF_SHAPE,
                         hop=hop_name, claim=claim, basis_type=basis.value, invalid_refs=[],
                         reason="evidence_refs must be an array of strings")
        invalid_refs = [ref for ref in refs if registry.resolve(ref) is None]
        if invalid_refs:
            raise _error(malfunction, scenario, prompt_version, batch, split_path, split_depth,
                         code=ScenarioEvidenceErrorCode.UNRESOLVED_EVIDENCE_REF,
                         hop=hop_name, claim=claim, basis_type=basis.value,
                         invalid_refs=invalid_refs, reason="unresolved evidence refs")
        if basis is not EvidenceBasisType.ASSUMPTION and not refs:
            raise _error(malfunction, scenario, prompt_version, batch, split_path, split_depth,
                         code=ScenarioEvidenceErrorCode.MISSING_EVIDENCE_REF,
                         hop=hop_name, claim=claim, basis_type=basis.value, invalid_refs=[],
                         reason="supported hop requires evidence refs")
        resolved = [registry.resolve(ref) for ref in refs]
        if causal and any(
            fact is not None
            and (
                fact.get("provenance") == FactProvenance.SCENARIO_DEFINED.value
                or has_analysis_assumption_lineage(fact)
            )
            and not _is_validated_analytical_scenario_atom(fact)
            and not (scenario.analysis_instance.get("physical_hypothesis")
                     and _conditional_evidence(fact, malfunction.malfunction_id, scenario.scenario_id))
            and not (fact.get("kind") == EvidenceKind.DERIVED_PHYSICS.value
                     and scenario.analysis_instance.get("physical_input_production", {}).get("producer")
                         == "AnalyticalPhysicsInstantiationService.produce_inputs"
                     and scenario.facts.get("fault_motion_model")
                     and source_is_accepted_for(scenario.fact_provenance.get("fault_motion_model", {}),
                         malfunction_id=malfunction.malfunction_id, scenario_id=scenario.scenario_id)
                     and source_is_accepted_for(fact, malfunction_id=malfunction.malfunction_id,
                                               scenario_id=scenario.scenario_id))
            for fact in resolved
        ):
            raise _error(
                malfunction, scenario, prompt_version, batch, split_path,
                split_depth,
                code=ScenarioEvidenceErrorCode.ASSUMPTION_IN_POSITIVE_CHAIN,
                hop=hop_name, claim=claim, basis_type=basis.value,
                invalid_refs=refs,
                reason=(
                    "unvalidated SCENARIO_DEFINED analysis assumptions cannot prove "
                    "a positive causal hop"
                ),
            )
        if causal and hop_name in ({"i_to_h", "h_to_harm"} if legacy_harm else {"i_to_h"}) and refs and all(
            fact.get("evidence_role") == "UPSTREAM_CAUSAL_CLAIM"
            for fact in resolved
        ):
            raise _error(
                malfunction, scenario, prompt_version, batch, split_path,
                split_depth,
                code=ScenarioEvidenceErrorCode.SELF_REFERENTIAL_CAUSAL_EVIDENCE,
                hop=hop_name, claim=claim, basis_type=basis.value,
                invalid_refs=refs,
                reason=(
                    "an upstream hazard claim cannot be the sole evidence for "
                    "a downstream hazard or harm transition"
                ),
            )
        if basis is EvidenceBasisType.DIRECT_FACT and any(
            fact["kind"] != "DIRECT_FACT" for fact in resolved
        ):
            raise _error(malfunction, scenario, prompt_version, batch, split_path, split_depth,
                         code=ScenarioEvidenceErrorCode.DIRECT_FACT_KIND_MISMATCH,
                         hop=hop_name, claim=claim, basis_type=basis.value, invalid_refs=refs,
                         reason="DIRECT_FACT refs must resolve to direct facts")
        if basis is EvidenceBasisType.DERIVED_PHYSICS and (
            not refs or any(fact["kind"] != "DERIVED_PHYSICS" for fact in resolved)
        ):
            raise _error(malfunction, scenario, prompt_version, batch, split_path, split_depth,
                         code=ScenarioEvidenceErrorCode.DERIVED_PHYSICS_KIND_MISMATCH,
                         hop=hop_name, claim=claim, basis_type=basis.value, invalid_refs=refs,
                         reason="DERIVED_PHYSICS must cite registered bounded derivation")
        if basis is EvidenceBasisType.APPROVED_RULE and (
            not refs or any(fact["kind"] != "APPROVED_RULE" for fact in resolved)
        ):
            raise _error(malfunction, scenario, prompt_version, batch, split_path, split_depth,
                         code=ScenarioEvidenceErrorCode.APPROVED_RULE_KIND_MISMATCH,
                         hop=hop_name, claim=claim, basis_type=basis.value, invalid_refs=refs,
                         reason="APPROVED_RULE must cite approved rule ref")
        if basis is EvidenceBasisType.ASSUMPTION:
            assumption_hops += 1
            if causal and index < 4:
                raise _error(malfunction, scenario, prompt_version, batch, split_path, split_depth,
                             code=ScenarioEvidenceErrorCode.ASSUMPTION_IN_POSITIVE_CHAIN,
                             hop=hop_name, claim=claim, basis_type=basis.value, invalid_refs=[],
                             reason="causal=true cannot depend on assumption")

    changes = item.get("risk_dimension_changes")
    if not isinstance(changes, list):
        raise _error(malfunction, scenario, prompt_version, batch, split_path, split_depth,
                     code=ScenarioEvidenceErrorCode.INVALID_RISK_DIMENSION_SHAPE,
                     hop="risk_dimension_changes", claim="", basis_type="", invalid_refs=[],
                     reason="risk_dimension_changes must be array")
    dimensions: list[str] = []
    for change in changes:
        if not isinstance(change, dict):
            raise _error(malfunction, scenario, prompt_version, batch, split_path, split_depth,
                         code=ScenarioEvidenceErrorCode.INVALID_RISK_DIMENSION_SHAPE,
                         hop="risk_dimension_changes", claim="", basis_type="", invalid_refs=[],
                         reason="dimension change must be object")
        dimension = change.get("dimension")
        refs = change.get("evidence_refs")
        reason = str(change.get("reason", "")).strip()
        if dimension not in RISK_DIMENSION_VALUES:
            raise _error(malfunction, scenario, prompt_version, batch, split_path, split_depth,
                         code=ScenarioEvidenceErrorCode.UNKNOWN_RISK_DIMENSION,
                         hop="risk_dimension_changes", claim=reason, basis_type="",
                         invalid_refs=[], reason=f"unknown dimension={dimension!r}")
        if dimension in dimensions:
            raise _error(malfunction, scenario, prompt_version, batch, split_path, split_depth,
                         code=ScenarioEvidenceErrorCode.DUPLICATE_RISK_DIMENSION,
                         hop="risk_dimension_changes", claim=reason, basis_type="",
                         invalid_refs=[], reason=f"duplicate dimension={dimension!r}")
        if not isinstance(refs, list) or not refs or any(not isinstance(ref, str) for ref in refs):
            raise _error(malfunction, scenario, prompt_version, batch, split_path, split_depth,
                         code=ScenarioEvidenceErrorCode.INVALID_RISK_DIMENSION_EVIDENCE,
                         hop="risk_dimension_changes", claim=reason, basis_type="",
                         invalid_refs=[], reason="dimension evidence refs required")
        invalid_refs = [ref for ref in refs if registry.resolve(ref) is None]
        if invalid_refs:
            raise _error(malfunction, scenario, prompt_version, batch, split_path, split_depth,
                         code=ScenarioEvidenceErrorCode.UNRESOLVED_EVIDENCE_REF,
                         hop="risk_dimension_changes", claim=reason, basis_type="",
                         invalid_refs=invalid_refs, reason="unresolved evidence refs")
        if not reason:
            raise _error(malfunction, scenario, prompt_version, batch, split_path, split_depth,
                         code=ScenarioEvidenceErrorCode.MISSING_RISK_DIMENSION_REASON,
                         hop="risk_dimension_changes", claim=reason, basis_type="",
                         invalid_refs=[], reason="dimension reason must be non-empty")
        dimensions.append(dimension)
    if not causal and dimensions:
        raise _error(malfunction, scenario, prompt_version, batch, split_path, split_depth,
                     code=ScenarioEvidenceErrorCode.CAUSAL_FALSE_WITH_DIMENSIONS,
                     hop="risk_dimension_changes", claim="", basis_type="", invalid_refs=[],
                     reason="causal=false requires no dimension changes")
    return dimensions, {"assumption_hop_count": assumption_hops}


def compile_causal_assessment(
    *,
    malfunction: MalfunctionCandidate,
    scenario: ScenarioCandidate,
    item: dict[str, Any],
    registry: FactRegistry,
) -> ScenarioCausalAssessment:
    """Compile an already schema/evidence-validated Provider payload to the typed contract."""

    chain = item["causal_chain"]
    node_specs = (
        (
            "malfunction", CausalNodeType.MALFUNCTION,
            malfunction.description, ("MF.description",),
        ),
        (
            "behavior", CausalNodeType.SYSTEM_BEHAVIOR_CHANGE,
            malfunction.functional_effect or chain["m_to_b"]["claim"],
            tuple(chain["m_to_b"].get("evidence_refs", [])),
        ),
        (
            "consequence", CausalNodeType.OPERATIONAL_CONSEQUENCE,
            str(chain.get("b_to_i", {}).get("claim", "")),
            tuple(chain.get("b_to_i", {}).get("evidence_refs", [])),
        ),
        (
            "hazard", CausalNodeType.HAZARD,
            str(item.get("hazardous_event", "")).strip()
            or str(chain.get("i_to_h", {}).get("claim", "")),
            tuple(chain.get("i_to_h", {}).get("evidence_refs", [])),
        ),
    )
    hop_specs = (
        ("m_to_b", "malfunction", "behavior"),
        ("b_to_i", "behavior", "consequence"),
        ("i_to_h", "consequence", "hazard"),
    )
    present_hops = [name for name, _, _ in hop_specs if isinstance(chain.get(name), dict)]
    required_node_names = {"malfunction"}
    for name, source, target in hop_specs:
        if name in present_hops:
            required_node_names.update((source, target))
    node_id = {
        name: f"{scenario.scenario_id}:{malfunction.malfunction_id}:{name}"
        for name, *_ in node_specs
    }
    nodes = tuple(
        CausalNode(node_id[name], node_type, description, provenance)
        for name, node_type, description, provenance in node_specs
        if name in required_node_names and description.strip()
    )
    available_nodes = {item.node_id for item in nodes}
    edges: list[CausalEdge] = []
    bindings: list[EvidenceBinding] = []
    unsupported: list[str] = []
    all_refs: list[str] = []
    for name, source, target in hop_specs:
        hop = chain.get(name)
        if not isinstance(hop, dict):
            continue
        source_id, target_id = node_id[source], node_id[target]
        if source_id not in available_nodes or target_id not in available_nodes:
            continue
        refs = tuple(str(value) for value in hop.get("evidence_refs", []))
        basis = EvidenceKind(hop["basis_type"])
        edge_id = name.upper()
        edges.append(CausalEdge(
            edge_id=edge_id,
            source=source_id,
            target=target_id,
            relation=CausalRelation.CAUSES,
            description=str(hop["claim"]),
            evidence_refs=refs,
        ))
        source_refs: list[SourceRef] = []
        resolved_records = []
        for evidence_ref in refs:
            record = registry.resolve_record(evidence_ref)
            if record is not None:
                resolved_records.append(record)
                for source_ref in record.source_refs:
                    if source_ref not in source_refs:
                        source_refs.append(source_ref)
        binding_status = (
            ReviewStatus.FINALIZED
            if (
                basis is EvidenceKind.ASSUMPTION
                or (
                    bool(refs)
                    and len(resolved_records) == len(refs)
                    and all(
                        (record.approval_status is ReviewStatus.FINALIZED
                         or (scenario.analysis_instance.get("physical_hypothesis") and
                             _conditional_evidence(record, malfunction.malfunction_id, scenario.scenario_id)))
                        and bool(record.source_refs)
                        for record in resolved_records
                    )
                )
            )
            else ReviewStatus.PENDING
        )
        bindings.append(EvidenceBinding(
            edge_id=edge_id,
            evidence_refs=refs,
            basis_type=basis,
            source_refs=tuple(source_refs),
            status=binding_status,
        ))
        all_refs.extend(refs)
        if basis is EvidenceKind.ASSUMPTION:
            unsupported.append(edge_id)
    breakpoint = CausalBreakpoint(item["breakpoint"])
    if breakpoint is not CausalBreakpoint.NONE:
        breakpoint_edge = breakpoint.value
        if breakpoint_edge not in unsupported:
            unsupported.append(breakpoint_edge)
    ordered_nodes = tuple(
        node_id[name] for name, *_ in node_specs if node_id[name] in available_nodes
    )
    changes = tuple(RiskDimensionChange(
        dimension=str(value["dimension"]),
        evidence_refs=tuple(str(item) for item in value["evidence_refs"]),
        reason=str(value["reason"]),
    ) for value in item["risk_dimension_changes"])
    change_records = [
        registry.resolve_record(evidence_ref)
        for change in changes
        for evidence_ref in change.evidence_refs
    ]
    review_status = (
        ReviewStatus.FINALIZED
        if (
            malfunction.status is ReviewStatus.FINALIZED
            and all(item.status is ReviewStatus.FINALIZED for item in bindings)
            and all(
                record is not None
                and (record.approval_status is ReviewStatus.FINALIZED
                     or (scenario.analysis_instance.get("physical_hypothesis") and
                         _conditional_evidence(record, malfunction.malfunction_id, scenario.scenario_id)))
                and bool(record.source_refs)
                for record in change_records
            )
        )
        else ReviewStatus.PENDING
    )
    return ScenarioCausalAssessment(
        scenario_id=scenario.scenario_id,
        causal_graph=CausalGraph(nodes=nodes, edges=tuple(edges)),
        causal_chain=ordered_nodes,
        breakpoint=breakpoint,
        evidence_bindings=tuple(bindings),
        unsupported_links=tuple(unsupported),
        risk_dimension_changes=changes,
        hazardous_event=str(item.get("hazardous_event", "")).strip(),
        potential_harm=str(item.get("potential_harm", "")).strip(),
        provenance=tuple(dict.fromkeys(all_refs)),
        review_status=review_status,
    )


def _error(malfunction, scenario, prompt_version, batch, split_path, split_depth,
           *, code, hop, claim, basis_type, invalid_refs, reason):
    message = (
        "Scenario evidence contract violation: "
        f"code={code.value} "
        f"malfunction_id={malfunction.malfunction_id} scenario_id={scenario.scenario_id} "
        f"semantic_fingerprint={scenario.semantic_fingerprint} hop={hop} claim={claim!r} "
        f"basis_type={basis_type!r} invalid_evidence_refs={invalid_refs!r} "
        f"scenario_contract_version={SCENARIO_CONTRACT_VERSION} "
        f"assessment_contract_version={SCENARIO_ASSESSMENT_CONTRACT_VERSION} "
        f"prompt_version={prompt_version} batch={batch} split_path={split_path} "
        f"split_depth={split_depth} reason={reason}"
    )
    return ScenarioEvidenceContractError(
        message, code=code, hop=hop, reason=reason,
        malfunction_id=malfunction.malfunction_id, scenario_id=scenario.scenario_id,
        semantic_fingerprint=scenario.semantic_fingerprint,
        invalid_evidence_refs=invalid_refs, basis_type=basis_type, claim=claim,
    )
