from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


SCENARIO_SYNTHESIS_CONTRACT_VERSION = "scenario-synthesis-v4"


class CandidateOrigin(str, Enum):
    DIRECT_PROJECT_BINDING = "DIRECT_PROJECT_BINDING"
    APPROVED_ALIAS = "APPROVED_ALIAS"
    METHOD_TEMPLATE = "METHOD_TEMPLATE"
    ODD_CONSTRAINED_CATALOG = "ODD_CONSTRAINED_CATALOG"
    HAZARD_CAUSAL_SEMANTIC_CANDIDATE = "HAZARD_CAUSAL_SEMANTIC_CANDIDATE"
    DETERMINISTIC_DERIVATION = "DETERMINISTIC_DERIVATION"
    LLM_SELECTED_FROM_APPROVED_CANDIDATES = "LLM_SELECTED_FROM_APPROVED_CANDIDATES"
    BINDING_REFINEMENT = "BINDING_REFINEMENT"


class SynthesisValidationStatus(str, Enum):
    VALIDATED = "VALIDATED"
    PENDING = "PENDING"
    REJECTED = "REJECTED"


class ScenarioSynthesisStatus(str, Enum):
    METHOD_VALID = "METHOD_VALID"
    PENDING_SCENARIO_SYNTHESIS = "PENDING_SCENARIO_SYNTHESIS"
    METHOD_INVALID = "METHOD_INVALID"
    SOURCE_CONFLICT = "SOURCE_CONFLICT"


class CoverageLabel(str, Enum):
    TYPICAL = "typical"
    BOUNDARY = "boundary"
    EXTREME = "extreme"


class ScenarioBindingAuthority(str, Enum):
    EXACT_PROJECT_FACT = "EXACT_PROJECT_FACT"
    EXACT_METHOD_MAPPING = "EXACT_METHOD_MAPPING"
    APPROVED_ALIAS = "APPROVED_ALIAS"
    RANGE_CONTAINMENT = "RANGE_CONTAINMENT"
    METHOD_TEMPLATE_INFERENCE = "METHOD_TEMPLATE_INFERENCE"
    ANALYTICAL_SELECTION = "ANALYTICAL_SELECTION"
    DERIVED_COMPOUND = "DERIVED_COMPOUND"


class ScenarioDimensionApplicability(str, Enum):
    REQUIRED = "REQUIRED"
    OPTIONAL = "OPTIONAL"
    NOT_APPLICABLE = "NOT_APPLICABLE"


class SemanticCompatibility(str, Enum):
    """Tri-state semantic relation between a query and a Method atom."""

    SUPPORTED = "SUPPORTED"
    UNKNOWN = "UNKNOWN"
    CONTRADICTED = "CONTRADICTED"


class PhysicalValueAuthority(str, Enum):
    PROJECT_FACT = "PROJECT_FACT"
    METHOD_DEFINED = "METHOD_DEFINED"
    SCENARIO_DEFINED = "SCENARIO_DEFINED"
    DERIVED = "DERIVED"
    ENGINEERING_ANALYSIS_ASSUMPTION = "ENGINEERING_ANALYSIS_ASSUMPTION"
    HUMAN_CONFIRMATION = "HUMAN_CONFIRMATION"
    UNAVAILABLE = "UNAVAILABLE"


def _serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_serialize(item) for item in value]
    return value


@dataclass(frozen=True)
class ScenarioBindingDecision:
    dimension: str
    authority: ScenarioBindingAuthority
    refinable: bool
    source_refs: tuple[str, ...]
    basis: str
    parent_atom_id: str = ""
    project_speed_envelope_kph: tuple[float | None, float | None] | None = None
    parent_speed_range_kph: tuple[float | None, float | None] | None = None

    def __post_init__(self) -> None:
        if not self.dimension or not self.basis.strip() or not self.source_refs:
            raise ValueError("Scenario binding decision requires dimension and evidence")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


@dataclass(frozen=True)
class ScenarioDimensionApplicabilityDecision:
    dimension: str
    status: ScenarioDimensionApplicability
    reason: str
    trigger_evidence: tuple[str, ...]
    source_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.dimension or not self.reason.strip() or not self.source_refs:
            raise ValueError("Scenario dimension applicability requires evidence")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


@dataclass(frozen=True)
class ScenarioCoveragePlan:
    active_mechanism: str
    fixed_dimensions: tuple[str, ...]
    primary_variation_dimensions: tuple[str, ...]
    secondary_variation_dimensions: tuple[str, ...]
    prohibited_trivial_only_dimensions: tuple[str, ...]
    desired_variant_count: int
    variant_intents: tuple[dict[str, Any], ...]
    source_refs: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.active_mechanism.strip() or not 1 <= self.desired_variant_count <= 3:
            raise ValueError("Scenario coverage plan requires a mechanism and 1-3 variants")
        labels = [str(item.get("coverage_label", "")) for item in self.variant_intents]
        if len(labels) != self.desired_variant_count or len(labels) != len(set(labels)):
            raise ValueError("Scenario coverage plan intents must match desired variant count")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


@dataclass(frozen=True)
class ScenarioDimensionCandidate:
    atom_id: str
    canonical_atom_id: str
    dimensions: tuple[str, ...]
    label: str
    source_asset: str
    source_rule: str
    source_tag: str
    candidate_origin: CandidateOrigin
    supporting_context_refs: tuple[str, ...]
    selection_reason: str
    method_semantics: dict[str, Any] = field(default_factory=dict)
    validation_status: SynthesisValidationStatus = SynthesisValidationStatus.PENDING
    speed_range_kph: tuple[float | None, float | None] | None = None
    binding_authority: str = "ANALYTICAL_SELECTION"
    template_relationship: str = "NONE"
    semantic_compatibility: SemanticCompatibility = SemanticCompatibility.UNKNOWN
    semantic_family: str = ""
    ranking_scores: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.atom_id or not self.canonical_atom_id or not self.dimensions:
            raise ValueError("Scenario atom candidate requires identity and dimensions")
        if not self.source_asset or not self.source_rule:
            raise ValueError("Scenario atom candidate requires source provenance")
        if not self.supporting_context_refs or not self.selection_reason.strip():
            raise ValueError("Scenario atom candidate requires selection evidence")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


@dataclass(frozen=True)
class ScenarioAtomCandidateSet:
    dimension: str
    catalog_size: int
    candidates: tuple[ScenarioDimensionCandidate, ...]
    hard_filtered_pool_size: int
    applicability: ScenarioDimensionApplicabilityDecision
    binding_decision: ScenarioBindingDecision
    locked_atom_ids: tuple[str, ...] = ()
    resolution_status_before: str = "PENDING"
    generation_status: str = "CANDIDATES_AVAILABLE"
    reason: str = ""
    shortlist_truncated: bool = False
    shortlist_budget: int = 0
    shortlist_policy: str = "NOT_APPLICABLE"
    shortlist_diagnostics: dict[str, int] = field(default_factory=dict)
    hard_filter_diagnostics: dict[str, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        ids = [item.atom_id for item in self.candidates]
        if not self.dimension or len(ids) != len(set(ids)):
            raise ValueError("Scenario candidate set requires a dimension and unique atoms")
        if any(self.dimension not in item.dimensions for item in self.candidates):
            raise ValueError("Scenario candidate belongs to the wrong dimension")
        if any(atom_id not in ids for atom_id in self.locked_atom_ids):
            raise ValueError("Locked Scenario atom must remain inside the candidate set")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


@dataclass(frozen=True)
class ScenarioSynthesisInput:
    malfunction_id: str
    parent_scenario_id: str
    hazardous_event_id: str
    function_id: str
    malfunction: dict[str, Any]
    parent_scenario: dict[str, Any]
    causal_assessment: dict[str, Any]
    project_context: dict[str, Any]
    method_contract_hash: str
    dimension_candidate_sets: tuple[ScenarioAtomCandidateSet, ...]
    semantic_group_id: str
    structured_semantic_query: dict[str, Any]
    coverage_plan: ScenarioCoveragePlan
    fm_scenario_template: dict[str, Any] = field(default_factory=dict)
    contextual_speed: dict[str, Any] = field(default_factory=dict)
    situation_selection_audit: dict[str, Any] = field(default_factory=dict)
    contract_version: str = SCENARIO_SYNTHESIS_CONTRACT_VERSION

    def __post_init__(self) -> None:
        if not all((self.malfunction_id, self.parent_scenario_id,
                    self.hazardous_event_id, self.semantic_group_id)):
            raise ValueError("ScenarioSynthesisInput requires stable identities")
        dimensions = [item.dimension for item in self.dimension_candidate_sets]
        if len(dimensions) != len(set(dimensions)):
            raise ValueError("ScenarioSynthesisInput dimensions must be unique")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


@dataclass(frozen=True)
class ScenarioSynthesisAssessment:
    semantic_group_id: str
    coverage_label: CoverageLabel
    selected_atoms: dict[str, tuple[str, ...]]
    semantic_rationale: str
    context_refs: tuple[str, ...]
    validation_status: SynthesisValidationStatus
    validation_reasons: tuple[str, ...] = ()
    selection_authority: str = "BOUNDED_PROVIDER_SELECTION"

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


@dataclass(frozen=True)
class AnalyticalScenarioInstantiation:
    scenario_id: str
    parent_scenario_id: str
    malfunction_id: str
    hazardous_event_id: str
    selected_atoms: tuple[ScenarioDimensionCandidate, ...]
    dimension_bindings: dict[str, dict[str, Any]]
    deterministic_validations: tuple[dict[str, Any], ...]
    provider_evidence: dict[str, Any]
    project_facts_used: tuple[str, ...]
    method_facts_used: tuple[str, ...]
    synthesis_version: str = SCENARIO_SYNTHESIS_CONTRACT_VERSION

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


@dataclass(frozen=True)
class ScenarioSynthesisResult:
    synthesis_input: ScenarioSynthesisInput
    assessments: tuple[ScenarioSynthesisAssessment, ...]
    instantiations: tuple[AnalyticalScenarioInstantiation, ...]
    status: ScenarioSynthesisStatus
    pending_reason: str = ""
    provider_trace: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


@dataclass(frozen=True)
class AnalyticalPhysicalInput:
    field: str
    authority: PhysicalValueAuthority
    value: Any = None
    unit: str = ""
    allowed_range: dict[str, Any] = field(default_factory=dict)
    reason: str = ""
    selection_basis: str = ""
    source_atom_ids: tuple[str, ...] = ()
    review_status: str = "PENDING"
    source_refs: tuple[dict[str, Any], ...] = ()
    applicable_scope: dict[str, Any] = field(default_factory=dict)
    project_rule_id: str = ""
    project_policy_version: str = ""

    def __post_init__(self) -> None:
        if not self.field or not self.reason.strip():
            raise ValueError("Analytical physical input requires field and reason")
        if self.authority is PhysicalValueAuthority.UNAVAILABLE and self.value is not None:
            raise ValueError("UNAVAILABLE physical input cannot carry a value")

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


__all__ = [
    "SCENARIO_SYNTHESIS_CONTRACT_VERSION", "CandidateOrigin",
    "SynthesisValidationStatus", "ScenarioSynthesisStatus", "CoverageLabel",
    "ScenarioBindingAuthority", "ScenarioDimensionApplicability",
    "PhysicalValueAuthority", "ScenarioBindingDecision",
    "ScenarioDimensionApplicabilityDecision", "ScenarioCoveragePlan",
    "ScenarioDimensionCandidate",
    "ScenarioAtomCandidateSet", "ScenarioSynthesisInput",
    "ScenarioSynthesisAssessment",
    "AnalyticalScenarioInstantiation", "ScenarioSynthesisResult",
    "AnalyticalPhysicalInput",
]
