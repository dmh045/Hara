from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field, is_dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any

from .method import RoleBinding, TemplateRole

if TYPE_CHECKING:
    from .risk_calculation import StructuredRiskMethod
    from .situation_selection import MalfunctionSituationSelectionMethod


METHOD_CONTRACT_VERSION = "method-contract-v3"
TEMPLATE_COMPILER_VERSION = "full-template-compiler-v2"


class CompileStatus(str, Enum):
    READY = "READY"
    READY_WITH_WARNINGS = "READY_WITH_WARNINGS"
    NOT_READY = "NOT_READY"


class CompilerDiagnosticSeverity(str, Enum):
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"


class CompilerDiagnosticCode(str, Enum):
    ROLE_MISSING = "ROLE_MISSING"
    ROLE_AMBIGUOUS = "ROLE_AMBIGUOUS"
    RULE_PARSE_FAILED = "RULE_PARSE_FAILED"
    RULE_CONFLICT = "RULE_CONFLICT"
    RULE_RANGE_GAP = "RULE_RANGE_GAP"
    RULE_RANGE_OVERLAP = "RULE_RANGE_OVERLAP"
    UNRESOLVED_RULE_VARIABLE = "UNRESOLVED_RULE_VARIABLE"
    SEMANTIC_VARIABLE_UNRESOLVED = "SEMANTIC_VARIABLE_UNRESOLVED"
    AMBIGUOUS_RESULT = "AMBIGUOUS_RESULT"
    EXAMPLE_NOT_EXECUTABLE = "EXAMPLE_NOT_EXECUTABLE"
    ASSUMPTION_UNRESOLVED = "ASSUMPTION_UNRESOLVED"
    EXPOSURE_METHOD_SELECTION_UNRESOLVED = "EXPOSURE_METHOD_SELECTION_UNRESOLVED"
    ASIL_MATRIX_INCOMPLETE = "ASIL_MATRIX_INCOMPLETE"
    ASIL_NA_SEMANTICS_UNRESOLVED = "ASIL_NA_SEMANTICS_UNRESOLVED"
    REPORT_FIELD_MISMATCH = "REPORT_FIELD_MISMATCH"
    WORKFLOW_COORDINATE_MISMATCH = "WORKFLOW_COORDINATE_MISMATCH"
    SOURCE_REF_MISSING = "SOURCE_REF_MISSING"
    MALFORMED_TEMPLATE_TEXT = "MALFORMED_TEMPLATE_TEXT"
    METHOD_SOURCE_INVALID = "METHOD_SOURCE_INVALID"
    METHOD_SOURCE_CONFLICT = "METHOD_SOURCE_CONFLICT"
    BASELINE_SCENARIO_BINDING_INCOMPLETE = "BASELINE_SCENARIO_BINDING_INCOMPLETE"
    FTTI_METHOD_UNCOMPILED = "FTTI_METHOD_UNCOMPILED"
    SEVERITY_FALLBACK_UNCOMPILED = "SEVERITY_FALLBACK_UNCOMPILED"


class RuleType(str, Enum):
    NORMATIVE_RULE = "NORMATIVE_RULE"
    CRITERION = "CRITERION"
    DEFINITION = "DEFINITION"
    EXAMPLE = "EXAMPLE"
    ASSUMPTION = "ASSUMPTION"
    WORKFLOW_INSTRUCTION = "WORKFLOW_INSTRUCTION"
    REFERENCE = "REFERENCE"
    OUTPUT_MAPPING = "OUTPUT_MAPPING"


class NormativeStrength(str, Enum):
    NORMATIVE = "NORMATIVE"
    CRITERION = "CRITERION"
    EXAMPLE = "EXAMPLE"
    ASSUMPTION = "ASSUMPTION"
    INSTRUCTION = "INSTRUCTION"
    REFERENCE = "REFERENCE"


class ParseStatus(str, Enum):
    COMPILED = "COMPILED"
    NEEDS_REVIEW = "NEEDS_REVIEW"
    NON_EXECUTABLE = "NON_EXECUTABLE"
    FAILED = "FAILED"


class PredicateOperator(str, Enum):
    EQ = "EQ"
    NE = "NE"
    LT = "LT"
    LE = "LE"
    GT = "GT"
    GE = "GE"
    BETWEEN = "BETWEEN"
    IN = "IN"
    NOT_IN = "NOT_IN"
    EXISTS = "EXISTS"


class FactType(str, Enum):
    VEHICLE_SPEED = "VEHICLE_SPEED"
    RELATIVE_SPEED = "RELATIVE_SPEED"
    IMPACT_SPEED = "IMPACT_SPEED"
    DELTA_V = "DELTA_V"
    TTC = "TTC"
    SPEED_UNSPECIFIED = "SPEED_UNSPECIFIED"
    COLLISION_TYPE = "COLLISION_TYPE"
    ROAD_USER_TYPE = "ROAD_USER_TYPE"
    DURATION_PERCENT = "DURATION_PERCENT"
    OCCURRENCE_FREQUENCY = "OCCURRENCE_FREQUENCY"
    SITUATION_CLASSIFICATION = "SITUATION_CLASSIFICATION"
    DRIVER_STATE = "DRIVER_STATE"
    DRIVER_IN_VEHICLE = "DRIVER_IN_VEHICLE"
    DIRECT_CONTROL_AVAILABLE = "DIRECT_CONTROL_AVAILABLE"
    INTERVENTION_AVAILABLE = "INTERVENTION_AVAILABLE"
    REMOTE_INTERVENTION_AVAILABLE = "REMOTE_INTERVENTION_AVAILABLE"
    FUNCTION_TYPE = "FUNCTION_TYPE"
    AVOIDABILITY_PERCENT = "AVOIDABILITY_PERCENT"
    FUNCTION = "FUNCTION"
    OUTPUT = "OUTPUT"
    GUIDEWORD = "GUIDEWORD"
    MALFUNCTION = "MALFUNCTION"
    HAZARD = "HAZARD"
    SCENARIO = "SCENARIO"
    SCENARIO_DETAIL = "SCENARIO_DETAIL"
    HAZARDOUS_EVENT = "HAZARDOUS_EVENT"
    SEVERITY = "SEVERITY"
    EXPOSURE = "EXPOSURE"
    CONTROLLABILITY = "CONTROLLABILITY"
    ASIL = "ASIL"
    SAFETY_GOAL = "SAFETY_GOAL"
    SAFE_STATE = "SAFE_STATE"


class FactOrigin(str, Enum):
    PROJECT_FACT = "PROJECT_FACT"
    SCENARIO_FACT = "SCENARIO_FACT"
    DERIVED_FACT = "DERIVED_FACT"
    HUMAN_EVIDENCE = "HUMAN_EVIDENCE"


@dataclass(frozen=True)
class SourceRef:
    workbook: str
    template_hash: str
    sheet: str
    range: str
    raw_text: str
    source_hash: str

    @classmethod
    def create(
        cls, *, workbook: str, template_hash: str, sheet: str, range: str, raw_text: str
    ) -> "SourceRef":
        payload = f"{template_hash}\n{sheet}\n{range}\n{raw_text}"
        return cls(
            workbook=workbook,
            template_hash=template_hash,
            sheet=sheet,
            range=range,
            raw_text=raw_text,
            source_hash=hashlib.sha256(payload.encode("utf-8")).hexdigest(),
        )

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SourceRef":
        return cls(**data)


@dataclass(frozen=True)
class Predicate:
    field: FactType
    operator: PredicateOperator
    value: Any = None
    unit: str = ""
    source: SourceRef | None = None


@dataclass(frozen=True)
class RangePredicate:
    field: FactType
    lower: float | None
    upper: float | None
    lower_inclusive: bool | None
    upper_inclusive: bool | None
    unit: str = ""
    source: SourceRef | None = None


@dataclass(frozen=True)
class CategoricalPredicate:
    field: FactType
    values: tuple[str, ...]
    matching_semantics: str = "EXACT_CANONICAL"
    source: SourceRef | None = None


RulePredicate = Predicate | RangePredicate | CategoricalPredicate


@dataclass(frozen=True)
class CompilerDiagnostic:
    severity: CompilerDiagnosticSeverity
    code: CompilerDiagnosticCode
    message: str
    role: TemplateRole | None = None
    source_refs: tuple[SourceRef, ...] = ()
    details: dict[str, Any] = field(default_factory=dict)
    blocking: bool = False


@dataclass(frozen=True)
class CompiledRule:
    rule_id: str
    rule_type: RuleType
    role: TemplateRole
    predicates: tuple[RulePredicate, ...]
    result: str | None
    alternatives: tuple[str, ...]
    priority: int
    normative_strength: NormativeStrength
    source_refs: tuple[SourceRef, ...]
    raw_text: str
    parser: str
    parse_status: ParseStatus
    diagnostics: tuple[str, ...] = ()
    resolution_requirement: str = ""
    assessment_method: str = ""
    executable: bool = True


@dataclass(frozen=True)
class WorkflowStep:
    activity: str
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    description: str
    source_ref: SourceRef


@dataclass(frozen=True)
class WorkflowContract:
    steps: tuple[WorkflowStep, ...]
    source_binding: RoleBinding


@dataclass(frozen=True)
class Guideword:
    guideword_id: str
    name: str
    description: str
    order: int
    source_ref: SourceRef


@dataclass(frozen=True)
class GuidewordContract:
    guidewords: tuple[Guideword, ...]
    source_binding: RoleBinding


@dataclass(frozen=True)
class ScenarioDimension:
    dimension_id: str
    canonical_name: str
    display_name: str
    values: tuple[str, ...]
    unit: str
    source_ref: SourceRef
    semantics: str = "METHOD_SCENARIO_ONTOLOGY"


class ScenarioConstraintDisposition(str, Enum):
    PHYSICALLY_IMPOSSIBLE = "PHYSICALLY_IMPOSSIBLE"
    SEMANTICALLY_INCOMPATIBLE = "SEMANTICALLY_INCOMPATIBLE"
    RARE_BUT_FEASIBLE = "RARE_BUT_FEASIBLE"
    EXPLICITLY_ALLOWED = "EXPLICITLY_ALLOWED"


@dataclass(frozen=True)
class FMTemplateMatch:
    """Exact, source-defined selectors for an FM scenario template."""

    keywords: tuple[str, ...]
    component_categories: tuple[str, ...]
    failure_types: tuple[str, ...]
    matching_semantics: str = "KEYWORD_OR_COMPONENT_CATEGORY_OR_FAILURE_TYPE"


@dataclass(frozen=True)
class FailureModeTaxonomyValue:
    """One source-governed selector value and its explicit aliases."""

    canonical_id: str
    aliases: tuple[str, ...]
    source_ref: SourceRef
    description: str = ""


@dataclass(frozen=True)
class FailureModeSelectorTaxonomy:
    """Typed vocabulary used to validate FM template selectors.

    This is deliberately a vocabulary contract, not a classifier: a value is
    resolved only when it is an exact canonical ID or a source-declared alias.
    """

    component_categories: tuple[FailureModeTaxonomyValue, ...]
    failure_types: tuple[FailureModeTaxonomyValue, ...]
    component_source_ref: SourceRef
    failure_type_source_ref: SourceRef
    source_role: str = "TAXONOMY"


@dataclass(frozen=True)
class FMTemplateSelectorMapping:
    """A reviewed vocabulary bridge from one template selector to one taxonomy ID."""

    mapping_id: str
    selector_type: str
    source_template_value: str
    canonical_target: str
    mapping_semantics: str
    runtime_status: str
    template_source_ref: SourceRef
    taxonomy_source_ref: SourceRef | None
    candidate_targets: tuple[str, ...] = ()


@dataclass(frozen=True)
class FMTemplateSelectorAdapter:
    """Typed reconciliation of confirmed template and FM taxonomy vocabularies."""

    mappings: tuple[FMTemplateSelectorMapping, ...]
    source_ref: SourceRef
    source_role: str = "FM_TEMPLATE_SELECTOR_ADAPTER"


@dataclass(frozen=True)
class FMTemplateScenario:
    label: str
    obj_type: str
    obj_position: str
    obj_distance_m: float | None
    obj_v_kph: float | None
    collision_type: str
    source_ref: SourceRef


@dataclass(frozen=True)
class FMScenarioTemplate:
    template_id: str
    match: FMTemplateMatch
    required_scenarios: tuple[FMTemplateScenario, ...]
    source_ref: SourceRef
    source_role: str = "SCENARIO_TEMPLATE_CONSTRAINT"
    original_precedence: int = 0


@dataclass(frozen=True)
class FMScenarioTemplateCatalog:
    templates: tuple[FMScenarioTemplate, ...]
    odd_geometry_m: tuple[tuple[str, float], ...]
    source_ref: SourceRef
    source_role: str = "SCENARIO_TEMPLATE_CONSTRAINT"


@dataclass(frozen=True)
class DomainTriggeringStateMapping:
    rule_id: str
    collision_type: str
    target: str
    v_other_min_kmh: float | None
    v_other_max_kmh: float | None
    triggering_state_key: str
    source_ref: SourceRef
    source_role: str = "CONFIRMED_DOMAIN_MAPPING"


@dataclass(frozen=True)
class DomainKinematicDefaults:
    v_other_kmh: tuple[float, ...]
    gap_m: tuple[tuple[str, float], ...]
    reaction_delay_s: float | None
    source_ref: SourceRef
    source_role: str = "DOMAIN_DEFAULT"


@dataclass(frozen=True)
class ConfirmedFallbackDimension:
    source_dimension: str
    terms: tuple[str, ...]
    target_dimension: str
    source_ref: SourceRef
    source_role: str = "CONFIRMED_FALLBACK_TERM"
    target_status: str = "TARGET_DIMENSION_PENDING"


@dataclass(frozen=True)
class ScenarioDomainKnowledge:
    triggering_state_mappings: tuple[DomainTriggeringStateMapping, ...]
    kinematic_defaults: DomainKinematicDefaults | None
    fallback_dimensions: tuple[ConfirmedFallbackDimension, ...]
    numeric_sections: tuple[str, ...]
    source_ref: SourceRef


@dataclass(frozen=True)
class ScenarioMethodContract:
    """Typed, non-causal Scenario knowledge compiled from selected method assets."""

    fm_template_catalog: FMScenarioTemplateCatalog | None = None
    failure_mode_selector_taxonomy: FailureModeSelectorTaxonomy | None = None
    fm_template_selector_adapter: FMTemplateSelectorAdapter | None = None
    domain_knowledge: ScenarioDomainKnowledge | None = None
    example_catalogs: tuple[SourceRef, ...] = ()


@dataclass(frozen=True)
class ScenarioConstraintPredicate:
    dimension: str
    values: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.dimension or not self.values:
            raise ValueError("ScenarioConstraintPredicate requires dimension and values")


@dataclass(frozen=True)
class ScenarioConstraintRule:
    rule_id: str
    predicates: tuple[ScenarioConstraintPredicate, ...]
    disposition: ScenarioConstraintDisposition
    reason: str
    normative_strength: NormativeStrength
    source_ref: SourceRef
    executable: bool = True

    def __post_init__(self) -> None:
        if not self.rule_id or not self.predicates or not self.reason.strip():
            raise ValueError("ScenarioConstraintRule requires identity, predicates and reason")


@dataclass(frozen=True)
class ScenarioModel:
    dimensions: tuple[ScenarioDimension, ...]
    structural_constraints: tuple[str, ...]
    source_binding: RoleBinding
    source_type: str = "METHOD_SCENARIO_ONTOLOGY"
    constraint_rules: tuple[ScenarioConstraintRule, ...] = ()
    scenario_method: ScenarioMethodContract = field(default_factory=ScenarioMethodContract)


@dataclass(frozen=True)
class ScaleLevel:
    level: str
    description: str
    criterion: str
    evidence: str
    source_ref: SourceRef


@dataclass(frozen=True)
class SeverityScale:
    levels: tuple[ScaleLevel, ...]
    source_binding: RoleBinding


@dataclass(frozen=True)
class SeverityContract:
    scale: SeverityScale
    rules: tuple[CompiledRule, ...]
    diagnostics: tuple[CompilerDiagnostic, ...]


@dataclass(frozen=True)
class ExposureEntry:
    entry_id: str
    source_role: TemplateRole
    description: str
    duration_rating: str
    frequency_rating: str
    normative_strength: NormativeStrength
    source_ref: SourceRef
    metadata: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class ExposureContract:
    scale: tuple[ScaleLevel, ...]
    duration_rules: tuple[CompiledRule, ...]
    frequency_rules: tuple[CompiledRule, ...]
    situation_mappings: tuple[ExposureEntry, ...]
    examples: tuple[CompiledRule, ...]
    diagnostics: tuple[CompilerDiagnostic, ...]


@dataclass(frozen=True)
class ControllabilityContract:
    scale: tuple[ScaleLevel, ...]
    criteria: tuple[CompiledRule, ...]
    examples: tuple[CompiledRule, ...]
    references: tuple[CompiledRule, ...]
    assumptions: tuple[MethodAssumption, ...]
    diagnostics: tuple[CompilerDiagnostic, ...]


@dataclass(frozen=True)
class ASILMapping:
    severity: str
    exposure: str
    controllability: str
    result: str
    source_ref: SourceRef


@dataclass(frozen=True)
class ASILMatrix:
    severity_levels: tuple[str, ...]
    exposure_levels: tuple[str, ...]
    controllability_levels: tuple[str, ...]
    mappings: tuple[ASILMapping, ...]
    na_semantics: str
    source_binding: RoleBinding
    diagnostics: tuple[CompilerDiagnostic, ...]


@dataclass(frozen=True)
class MethodAssumption:
    parameter: str
    value: str
    unit: str
    scope: str
    normative_strength: NormativeStrength
    source_ref: SourceRef


@dataclass(frozen=True)
class DerivationMethod:
    inputs_required: tuple[str, ...]
    derivation_pattern: str
    qm_handling: str
    aggregation_instructions: str
    semantic_derivation_required: bool
    assumptions: tuple[MethodAssumption, ...]
    instructions: tuple[CompiledRule, ...]
    source_binding: RoleBinding


@dataclass(frozen=True)
class ReportFieldMapping:
    canonical_field: str
    sheet: str
    column: str
    column_index: int
    header_rows: tuple[int, ...]
    source_ref: SourceRef


@dataclass(frozen=True)
class ReportContract:
    hara_fields: tuple[ReportFieldMapping, ...]
    safety_goal_fields: tuple[ReportFieldMapping, ...]
    source_bindings: tuple[RoleBinding, ...]
    diagnostics: tuple[CompilerDiagnostic, ...]


@dataclass(frozen=True)
class RequiredFactSpec:
    fact_type: FactType
    required_for: tuple[str, ...]
    unit: str
    constraints: tuple[str, ...]
    condition: str
    origin: FactOrigin
    source_rule_ids: tuple[str, ...]
    source_refs: tuple[SourceRef, ...]


def _serialize(value: Any) -> Any:
    if isinstance(value, Enum):
        return value.value
    if is_dataclass(value):
        return {key: _serialize(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_serialize(item) for item in value]
    return value


@dataclass(frozen=True)
class MethodContract:
    metadata: dict[str, Any]
    workflow: WorkflowContract
    guidewords: GuidewordContract
    scenario_model: ScenarioModel
    severity: SeverityContract
    exposure: ExposureContract
    controllability: ControllabilityContract
    asil: ASILMatrix
    safety_goal_method: DerivationMethod
    safe_state_method: DerivationMethod
    report_contract: ReportContract
    required_fact_specs: tuple[RequiredFactSpec, ...]
    diagnostics: tuple[CompilerDiagnostic, ...]
    sources: tuple[SourceRef, ...]
    compile_status: CompileStatus
    engineering_rules_compiled: bool
    structured_risk_method: StructuredRiskMethod | None = None
    situation_selection: MalfunctionSituationSelectionMethod | None = None
    contract_version: str = METHOD_CONTRACT_VERSION
    compiler_version: str = TEMPLATE_COMPILER_VERSION

    @property
    def blocking_diagnostics(self) -> tuple[CompilerDiagnostic, ...]:
        return tuple(item for item in self.diagnostics if item.blocking)

    @property
    def warnings(self) -> tuple[CompilerDiagnostic, ...]:
        return tuple(
            item for item in self.diagnostics
            if item.severity is CompilerDiagnosticSeverity.WARNING
        )

    def all_rules(self) -> tuple[CompiledRule, ...]:
        return (
            self.severity.rules
            + self.exposure.duration_rules
            + self.exposure.frequency_rules
            + self.exposure.examples
            + self.controllability.criteria
            + self.controllability.examples
            + self.controllability.references
            + self.safety_goal_method.instructions
            + self.safe_state_method.instructions
        )

    def to_dict(self) -> dict[str, Any]:
        result = _serialize(self)
        if self.structured_risk_method is None:
            result.pop("structured_risk_method", None)
        if self.situation_selection is None:
            result.pop("situation_selection", None)
        scenario_method = result.get("scenario_model", {}).get("scenario_method")
        if scenario_method == {
            "fm_template_catalog": None,
            "failure_mode_selector_taxonomy": None,
            "fm_template_selector_adapter": None,
            "domain_knowledge": None,
            "example_catalogs": [],
        }:
            # Preserve the existing template-route contract/checkpoint shape when
            # no YAML Scenario method source has been compiled.
            result["scenario_model"].pop("scenario_method", None)
        result["blocking_diagnostics"] = _serialize(self.blocking_diagnostics)
        result["warnings"] = _serialize(self.warnings)
        return result

    def audit_snapshot(self) -> dict[str, Any]:
        """Return a small, mutation-sensitive regression manifest."""

        full = self.to_dict()
        sections = (
            "workflow", "guidewords", "scenario_model", "severity", "exposure",
            "controllability", "asil", "safety_goal_method", "safe_state_method",
            "report_contract", "required_fact_specs",
        )
        if self.structured_risk_method is not None:
            sections += ("structured_risk_method",)
        if self.situation_selection is not None:
            sections += ("situation_selection",)

        def digest(value: Any) -> str:
            payload = json.dumps(
                value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
            )
            return hashlib.sha256(payload.encode("utf-8")).hexdigest()

        source_hashes = sorted({item.source_hash for item in self.sources})
        return {
            "snapshot_schema": "method-contract-audit-v2",
            "metadata": self.metadata,
            "contract_version": self.contract_version,
            "compiler_version": self.compiler_version,
            "compile_status": self.compile_status.value,
            "engineering_rules_compiled": self.engineering_rules_compiled,
            "counts": {
                "workflow_steps": len(self.workflow.steps),
                "guidewords": len(self.guidewords.guidewords),
                "scenario_dimensions": len(self.scenario_model.dimensions),
                "severity_rules": len(self.severity.rules),
                "exposure_duration_rules": len(self.exposure.duration_rules),
                "exposure_frequency_rules": len(self.exposure.frequency_rules),
                "exposure_examples": len(self.exposure.examples),
                "exposure_situation_mappings": len(self.exposure.situation_mappings),
                "controllability_criteria": len(self.controllability.criteria),
                "controllability_examples": len(self.controllability.examples),
                "asil_mappings": len(self.asil.mappings),
                "hara_report_fields": len(self.report_contract.hara_fields),
                "safety_goal_report_fields": len(
                    self.report_contract.safety_goal_fields
                ),
                "required_fact_specs": len(self.required_fact_specs),
                "diagnostics": len(self.diagnostics),
                "unique_sources": len(source_hashes),
            },
            "section_hashes": {
                name: digest(full[name]) for name in sections
            },
            "source_hash_digest": digest(source_hashes),
            "diagnostics": [
                {
                    "severity": item.severity.value,
                    "code": item.code.value,
                    "blocking": item.blocking,
                    "role": item.role.value if item.role else None,
                }
                for item in self.diagnostics
            ],
        }

    def write_json(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary = target.with_suffix(target.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        temporary.replace(target)
        return target


def unique_sources(items: list[SourceRef]) -> tuple[SourceRef, ...]:
    unique: dict[tuple[str, str, str], SourceRef] = {}
    for item in items:
        unique[(item.sheet, item.range, item.source_hash)] = item
    return tuple(unique.values())
