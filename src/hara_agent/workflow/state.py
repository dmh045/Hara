from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from hara_agent.models import (
    EvidenceValue,
    ReviewStatus,
    RiskAssessment,
    SafetyGoal,
    ScenarioCandidate,
    SourceRef,
)
from hara_agent.services.semantic.scenario_contract import SCENARIO_CONTRACT_VERSION
from hara_agent.services.semantic.scenario_evidence import SCENARIO_ASSESSMENT_CONTRACT_VERSION


class WorkflowStage(str, Enum):
    INITIALIZE = "initialize"
    EXTRACT = "extract"
    FUNCTIONS = "functions"
    HAZOP = "hazop"
    MALFUNCTIONS = "malfunctions"
    SCENARIOS = "scenarios"
    SCORING = "scoring"
    SAFETY_GOALS = "safety_goals"
    QUALITY_GATE = "quality_gate"
    RENDER = "render"
    COMPLETE = "complete"
    BLOCKED = "blocked"


@dataclass
class HARAState:
    run_id: str
    stage: WorkflowStage = WorkflowStage.INITIALIZE
    method_contract: dict[str, Any] = field(default_factory=dict)
    item_definition: dict[str, Any] = field(default_factory=dict)
    functions: list[dict[str, Any]] = field(default_factory=list)
    guideword_assessments: list[dict[str, Any]] = field(default_factory=list)
    malfunctions: list[dict[str, Any]] = field(default_factory=list)
    scenarios: list[ScenarioCandidate] = field(default_factory=list)
    risk_results: list[RiskAssessment] = field(default_factory=list)
    safety_goals: list[SafetyGoal] = field(default_factory=list)
    pending_reviews: list[dict[str, Any]] = field(default_factory=list)
    audit_trail: list[dict[str, Any]] = field(default_factory=list)
    errors: list[dict[str, Any]] = field(default_factory=list)
    scenario_contract_version: str = SCENARIO_CONTRACT_VERSION
    scenario_assessment_contract_version: str = SCENARIO_ASSESSMENT_CONTRACT_VERSION

    @property
    def can_publish(self) -> bool:
        from hara_agent.services.validation import ReleaseGateValidator

        return ReleaseGateValidator().evaluate(self).ready_for_release

    def record(self, event: str, **details: Any) -> None:
        self.audit_trail.append({"event": event, **details})

    def to_dict(self) -> dict[str, Any]:
        result = self._serialize(asdict(self))
        result["can_publish"] = self.can_publish
        return result

    @classmethod
    def read_committed(cls, value: dict[str, Any]) -> "HARAState":
        """Decode stored state without scheduling or invalidating any stage.

        Consumers of a committed snapshot validate their stage dependencies;
        workflow resume additionally applies generation-version invalidation.
        """
        state = cls(
            run_id=str(value["run_id"]),
            stage=WorkflowStage(value.get("stage", WorkflowStage.INITIALIZE.value)),
            method_contract=dict(value.get("method_contract", {})),
            item_definition=dict(value.get("item_definition", {})),
            functions=list(value.get("functions", [])),
            guideword_assessments=list(value.get("guideword_assessments", [])),
            malfunctions=list(value.get("malfunctions", [])),
            pending_reviews=list(value.get("pending_reviews", [])),
            audit_trail=list(value.get("audit_trail", [])),
            errors=list(value.get("errors", [])),
            scenario_contract_version=str(
                value.get("scenario_contract_version", SCENARIO_CONTRACT_VERSION)
            ),
            scenario_assessment_contract_version=str(
                value.get("scenario_assessment_contract_version", "")
            ),
        )
        state.scenarios = [cls._scenario_from_dict(item) for item in value.get("scenarios", [])]
        state.risk_results = [cls._risk_from_dict(item) for item in value.get("risk_results", [])]
        state.safety_goals = [cls._goal_from_dict(item) for item in value.get("safety_goals", [])]
        return state

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "HARAState":
        state = cls.read_committed(value)
        loaded_contract = str(value.get("scenario_contract_version", ""))
        loaded_assessment_contract = str(value.get("scenario_assessment_contract_version", ""))
        scenario_outputs_exist = bool(
            state.scenarios
            or state.item_definition.get("scenario_assessments")
            or state.risk_results
            or state.safety_goals
        )
        contract_mismatch = (
            loaded_contract != SCENARIO_CONTRACT_VERSION
            or loaded_assessment_contract != SCENARIO_ASSESSMENT_CONTRACT_VERSION
        )
        if contract_mismatch and scenario_outputs_exist:
            state.scenarios = []
            state.risk_results = []
            state.safety_goals = []
            state.item_definition.pop("scenario_assessments", None)
            typed = state.item_definition.get("typed")
            if isinstance(typed, dict):
                scenario_fact_ids = {
                    str(item.get("fact_id", ""))
                    for item in typed.get("risk_facts", [])
                    if (
                        isinstance(item, dict)
                        and str(item.get("produced_by", "")).startswith(
                            "scenario-risk-facts-"
                        )
                    )
                }
                typed["risk_facts"] = [
                    item for item in typed.get("risk_facts", [])
                    if not (
                        isinstance(item, dict)
                        and str(item.get("fact_id", "")) in scenario_fact_ids
                    )
                ]
                typed["method_risk_fact_bindings"] = [
                    item for item in typed.get("method_risk_fact_bindings", [])
                    if not (
                        isinstance(item, dict)
                        and str(item.get("source_fact_id", "")) in scenario_fact_ids
                    )
                ]
            state.pending_reviews = [
                item for item in state.pending_reviews
                if not (
                    item.get("field") in {
                        "scenarios", "scenario_risk_facts", "severity", "exposure",
                        "controllability", "asil", "safety_goal",
                    }
                    or "assessment_id" in item
                    or "safety_goal_id" in item
                )
            ]
            state.stage = WorkflowStage.MALFUNCTIONS
            state.audit_trail.append({
                "event": "scenario_checkpoint_invalidated",
                "loaded_scenario_contract_version": loaded_contract or "unversioned",
                "required_scenario_contract_version": SCENARIO_CONTRACT_VERSION,
                "loaded_scenario_assessment_contract_version": (
                    loaded_assessment_contract or "unversioned"
                ),
                "required_scenario_assessment_contract_version": (
                    SCENARIO_ASSESSMENT_CONTRACT_VERSION
                ),
                "preserved_through_stage": WorkflowStage.MALFUNCTIONS.value,
            })
        state.scenario_contract_version = SCENARIO_CONTRACT_VERSION
        state.scenario_assessment_contract_version = SCENARIO_ASSESSMENT_CONTRACT_VERSION
        return state

    @staticmethod
    def _sources(values: list[dict[str, Any]]) -> list[SourceRef]:
        return [SourceRef(**item) for item in values]

    @classmethod
    def _evidence(cls, value: dict[str, Any]) -> EvidenceValue:
        return EvidenceValue(
            value=value.get("value"),
            status=ReviewStatus(value.get("status", ReviewStatus.PENDING.value)),
            sources=cls._sources(value.get("sources", [])),
            confidence=value.get("confidence"),
            rule_version=str(value.get("rule_version", "")),
            review_reason=str(value.get("review_reason", "")),
        )

    @classmethod
    def _scenario_from_dict(cls, value: dict[str, Any]) -> ScenarioCandidate:
        return ScenarioCandidate(
            scenario_id=str(value["scenario_id"]),
            operating_scenario=str(value.get("operating_scenario", "")),
            situational_description=str(value.get("situational_description", "")),
            situational_detailing=str(value.get("situational_detailing", "")),
            operating_mode=str(value.get("operating_mode", "")),
            facts=dict(value.get("facts", {})),
            context_resolution=dict(value.get("context_resolution", {})),
            fact_provenance=dict(value.get("fact_provenance", {})),
            status=ReviewStatus(value.get("status", ReviewStatus.PENDING.value)),
            sources=cls._sources(value.get("sources", [])),
            rule_version=str(value.get("rule_version", "")),
            review_reason=str(value.get("review_reason", "")),
            source_scenario_id=str(value.get("source_scenario_id", "")),
            atomic_variant=str(value.get("atomic_variant", "")),
            semantic_fingerprint=str(value.get("semantic_fingerprint", "")),
            scenario_contract_version=str(value.get("scenario_contract_version", "")),
            exposure_context=list(value.get("exposure_context", [])),
            analysis_instance=dict(value.get("analysis_instance", {})),
        )

    @classmethod
    def _risk_from_dict(cls, value: dict[str, Any]) -> RiskAssessment:
        return RiskAssessment(
            assessment_id=str(value["assessment_id"]),
            scenario_id=str(value["scenario_id"]),
            severity=cls._evidence(value["severity"]),
            exposure=cls._evidence(value["exposure"]),
            controllability=cls._evidence(value["controllability"]),
            asil=cls._evidence(value["asil"]),
            malfunction_id=str(value.get("malfunction_id", "")),
            hazardous_event=str(value.get("hazardous_event", "")),
            potential_harm=str(value.get("potential_harm", "")),
            exposure_tf=str(value.get("exposure_tf", "")),
            safety_goal_id=str(value.get("safety_goal_id", "")),
            branch_evaluations=list(value.get("branch_evaluations", [])),
            branch_summary=dict(value.get("branch_summary", {})),
        )

    @staticmethod
    def _goal_from_dict(value: dict[str, Any]) -> SafetyGoal:
        return SafetyGoal(
            sg_id=str(value["sg_id"]),
            text=str(value["text"]),
            safe_state=str(value["safe_state"]),
            max_asil=str(value["max_asil"]),
            associated_scenario_ids=list(value.get("associated_scenario_ids", [])),
            status=ReviewStatus(value.get("status", ReviewStatus.PENDING.value)),
        )

    @classmethod
    def _serialize(cls, value: Any) -> Any:
        if isinstance(value, Enum):
            return value.value
        if isinstance(value, dict):
            return {key: cls._serialize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [cls._serialize(item) for item in value]
        if isinstance(value, tuple):
            return [cls._serialize(item) for item in value]
        return value
