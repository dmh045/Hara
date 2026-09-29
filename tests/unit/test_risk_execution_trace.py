from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import json
from copy import deepcopy

from hara_agent.contracts import (
    CausalBreakpoint, CausalEdge, CausalGraph, CausalNode, CausalNodeType,
    CausalRelation, EvidenceBinding, ExposureDimensionCoverage,
    ExposureDimensionCoverageDecision, ExposureDimensionCoverageStatus,
    ExposureDimensionRequirementStatus, ScenarioCausalAssessment,
)
from hara_agent.method_sources import YamlBaselineCompiler
from hara_agent.models import (
    EvidenceKind, ReviewStatus, ScenarioCandidate, ScenarioFeasibilityAssessment,
    evaluate_risk_eligibility_payload,
)
from hara_agent.services.analysis import (
    ExposureDimensionCoverageService, RiskExecutionTraceService,
    MethodContractASILService, MethodRuleScoringService, StructuredRiskScoringService,
)
from hara_agent.template import TemplateRoleCompiler
from hara_agent.workflow import HARAState, ReviewArtifactReader, ReviewArtifactWriter, WorkflowStage
from hara_agent.workflow.nodes import score_structured_scenarios


ROOT = Path(__file__).resolve().parents[2]


def _method():
    report = TemplateRoleCompiler().compile_method(
        ROOT / "references/HARA_Template_AI_20260327.xlsx"
    ).report_contract
    return YamlBaselineCompiler().compile(
        ROOT / "method_assets/fusa_baseline_v1/manifest.yaml", report_contract=report,
    )


def _assessment(*, finalized: bool = True, causal: bool = True) -> dict:
    return {
        "malfunction_id": "MF-1", "scenario_id": "SC-1", "hazardous_event": "hazard",
        "rationale": "test", "confidence": 0.9, "breakpoint": "NONE",
        "status": "FINALIZED" if finalized else "PENDING",
        "physically_feasible": causal, "functionally_relevant": causal,
        "causally_relevant": causal, "final_retain": causal,
        "causal_assessment": _causal_assessment(finalized=finalized).to_dict(),
    }


def _causal_assessment(*, finalized: bool) -> ScenarioCausalAssessment:
    nodes = (
        CausalNode("M", CausalNodeType.MALFUNCTION, "malfunction"),
        CausalNode("B", CausalNodeType.SYSTEM_BEHAVIOR_CHANGE, "behavior"),
        CausalNode("I", CausalNodeType.OPERATIONAL_CONSEQUENCE, "consequence"),
        CausalNode("H", CausalNodeType.HAZARD, "hazard"),
    )
    edges = tuple(CausalEdge(
        edge_id, source, target, CausalRelation.CAUSES, edge_id, (f"TEST.{edge_id}",),
    ) for edge_id, source, target in (
        ("M_TO_B", "M", "B"), ("B_TO_I", "B", "I"), ("I_TO_H", "I", "H"),
    ))
    return ScenarioCausalAssessment(
        "SC-1", CausalGraph(nodes, edges), ("M", "B", "I", "H"),
        CausalBreakpoint.NONE,
        tuple(EvidenceBinding(
            edge.edge_id, edge.evidence_refs, EvidenceKind.DIRECT_FACT,
            status=ReviewStatus.FINALIZED,
        ) for edge in edges),
        (), (), "hazard", review_status=(
            ReviewStatus.FINALIZED if finalized else ReviewStatus.PENDING
        ),
    )


def _facts() -> dict:
    return {
        "scenario_id": "SC-1", "malfunction_id": "MF-1", "delta_v_kph": 20.0,
        "collision_type": "FRONTAL", "road_user_type": "VEHICLE",
        "component_category": "sensor_camera", "scenario_atom_ids": ["SO010", "PH005"],
        "relative_distance_m": 10.0, "relative_speed_kph": 9.0, "ttc_s": 4.0,
        "_fact_provenance": {"relative_speed_kph": {
            "provenance": "PROJECT_INPUT",
            "source_refs": [{"location": "tests/test_risk_execution_trace.py"}],
        }},
        "driver_in_vehicle": True, "remote_intervention_available": False,
        "other_road_user_avoidance_possible": False,
    }


def _resolved_coverage(method) -> ExposureDimensionCoverageDecision:
    dimensions = []
    for item in method.scenario_model.dimensions:
        status = (
            ExposureDimensionRequirementStatus.REQUIRED
            if item.canonical_name in {"WHERE", "EGO_ACTION"}
            else ExposureDimensionRequirementStatus.NOT_APPLICABLE
        )
        dimensions.append(ExposureDimensionCoverage(
            dimension=item.canonical_name,
            status=status,
            rule_id="TEST-COVERAGE",
            source_ref="tests/test_risk_execution_trace.py#TEST-COVERAGE",
        ))
    return ExposureDimensionCoverageDecision(
        assessment_key="MF-1::SC-1",
        method_contract_hash=str(method.metadata["template_hash"]),
        coverage_status=ExposureDimensionCoverageStatus.RESOLVED,
        dimensions=tuple(dimensions),
        coverage_rule_ids=("TEST-COVERAGE",),
        granularity="TEST",
    )


def _scoring_facts(method) -> dict:
    facts = _facts()
    facts["method_scenario_dimensions"] = {
        "WHERE": {"resolution_status": "RESOLVED", "atom_id": "SO010"},
        "EGO_ACTION": {"resolution_status": "RESOLVED", "atom_id": "PH005"},
    }
    facts["_exposure_dimension_coverage_decision"] = _resolved_coverage(method)
    return facts


def test_complete_structured_execution_trace_is_observational():
    method = _method()
    facts = _scoring_facts(method)
    scored = StructuredRiskScoringService(method).score(facts, "hazard")
    risk = SimpleNamespace(
        malfunction_id="MF-1", scenario_id="SC-1",
        severity=SimpleNamespace(value=scored["severity"]["severity_score"]),
        exposure=SimpleNamespace(value=scored["exposure"]["exposure_score"]),
        controllability=SimpleNamespace(value=scored["controllability"]["controllability_score"]),
        asil=SimpleNamespace(value="A", status=SimpleNamespace(value="FINALIZED"), sources=[]),
    )
    trace = RiskExecutionTraceService(method).project(
        run_id="trace", assessments=[_assessment()], candidates=[], committed=True,
        scored={("MF-1", "SC-1"): (
            scored["severity"], scored["exposure"], scored["controllability"], facts,
        )}, risks=[risk],
    )
    row = trace["assessments"][0]
    assert trace["risk_stage_status"] == "COMPLETED"
    assert row["risk_eligibility"]["status"] == "ELIGIBLE"
    assert row["severity"]["input_semantic"] == "RELATIVE_SPEED"
    assert row["severity"]["semantic_resolution"] == "CONFIRMED_RELATIVE_VELOCITY"
    assert row["severity"]["ego_speed_constraint"]["USED_FOR_SEVERITY"] is False
    assert row["exposure"]["atom_ids"] == ["SO010", "PH005"]
    assert row["controllability"]["derived_ttc"]["formula_identity"]
    assert row["asil"]["matrix_key"]


def test_missing_relative_speed_and_atoms_remain_pending_inputs():
    method = _method()
    facts = _scoring_facts(method)
    facts.pop("relative_speed_kph")
    facts["scenario_atom_ids"] = []
    scored = StructuredRiskScoringService(method).score(facts, "hazard")
    trace = RiskExecutionTraceService(method).project(
        run_id="trace", assessments=[_assessment()], candidates=[], committed=True,
        scored={("MF-1", "SC-1"): (
            scored["severity"], scored["exposure"], scored["controllability"], facts,
        )},
    )
    row = trace["assessments"][0]
    assert row["severity"]["status"] == "PENDING_INPUT"
    assert row["severity"]["pending_reason"] == "MISSING_RELATIVE_SPEED"
    assert row["exposure"]["status"] == "PENDING_INPUT"
    assert row["exposure"]["pending_reason"] == "EXPOSURE_SCENARIO_ATOM_SET_EMPTY"
    assert row["exposure"]["readiness_status"] == "PENDING_ATOM_BINDING"


def test_current_project_unknown_policy_trace_routes_to_ttc():
    method = _method()
    facts = _scoring_facts(method)
    facts.pop("driver_in_vehicle")
    scored = StructuredRiskScoringService(method).score(facts, "hazard")
    trace = RiskExecutionTraceService(method).project(
        run_id="trace", assessments=[_assessment()], candidates=[], committed=True,
        scored={("MF-1", "SC-1"): (
            scored["severity"], scored["exposure"], scored["controllability"], facts,
        )},
    )
    controllability = trace["assessments"][0]["controllability"]
    assert controllability["decision_status"] == "TTC_AFTER_UNKNOWN_OVERRIDE"
    assert controllability["unknown_policy_action"] == "SKIP_TO_TTC"
    assert controllability["rule_match_states"][0]["state"] == "UNKNOWN"
    assert "driver_in_vehicle" not in facts


def test_pending_coverage_trace_records_native_fusa_execution():
    method = _method()
    facts = _scoring_facts(method)
    facts["scenario_atom_ids"] = ["FA001"]
    facts["method_scenario_dimensions"] = {}
    facts["_exposure_dimension_coverage_decision"] = (
        ExposureDimensionCoverageService(method).decide(
            assessment_key="MF-1::SC-1", function=None, operating_mode="Active",
        )
    )
    scored = StructuredRiskScoringService(method).score(facts, "hazard")
    trace = RiskExecutionTraceService(method).project(
        run_id="trace", assessments=[_assessment()], candidates=[], committed=True,
        scored={("MF-1", "SC-1"): (
            scored["severity"], scored["exposure"], scored["controllability"], facts,
        )},
    )
    exposure = trace["assessments"][0]["exposure"]

    assert exposure["status"] == "FINALIZED"
    assert exposure["coverage_status"] == "PENDING_METHOD_SEMANTICS"
    assert exposure["coverage_gate_applied"] is False
    assert exposure["executor_invoked"] is True
    assert exposure["pending_reason"] == ""
    assert exposure["missing_method_semantics"] == ""
    assert exposure["atom_available"] is True
    assert exposure["atom_used_for_scoring"] is True
    assert exposure["atom_bindings"][0]["used"] is True


def test_production_scoring_keeps_coverage_diagnostic_without_blocking_fusa():
    method = _method()
    state = HARAState(run_id="coverage-runtime", stage=WorkflowStage.SCORING)
    state.functions = [{
        "function_id": "FUN-1", "name": "braking control", "output": "braking request",
    }]
    state.malfunctions = [{
        "malfunction_id": "MF-1", "function_id": "FUN-1", "guideword": "Less",
        "description": "braking request insufficient", "component_category": "sensor_camera",
    }]
    state.scenarios = [ScenarioCandidate(
        "SC-1", "Active", "object ahead", "closing vehicle", _facts(),
    )]
    state.item_definition["scenario_assessments"] = [_assessment()]

    score_structured_scenarios(
        state, MethodRuleScoringService(method), MethodContractASILService(method),
    )

    risk = state.risk_results[0]
    assert risk.exposure.value == "E4"
    assert risk.exposure.status is ReviewStatus.FINALIZED
    assert risk.asil.value == ""
    assert risk.asil.status is ReviewStatus.PENDING
    calculation = next(
        item for item in state.audit_trail
        if item["event"] == "structured_risk_scoring_completed"
    )["risk_calculation_inputs"][0]
    assert calculation["exposure_input"]["status"] == "PENDING_METHOD_SEMANTICS"


def test_governed_speed_enters_production_risk_context_and_trace(tmp_path):
    method = _method()
    source = {
        "provenance": "PROJECT_INPUT", "approval": "FINALIZED",
        "source_refs": [{
            "source_type": "item_document", "source_id": "ItemDef.docx",
            "location": "环境条件:控车范围", "excerpt": "0–7 kph",
        }],
    }
    facts = {
        "ego_speed_constraint": {"min_kph": 0.0, "max_kph": 7.0, "unit": "km/h"},
        "object_speed_kph": 0.0, "relative_distance_m": 7.0,
        "ego_longitudinal_direction": "FORWARD",
        "object_longitudinal_direction": "STATIONARY",
        "object_position": "front", "road_user_type": "VEHICLE",
        "collision_type": "FRONTAL", "driver_in_vehicle": True,
        "remote_intervention_available": False,
        "other_road_user_avoidance_possible": False,
        "scenario_atom_ids": ["SO010", "PH005"],
        "method_scenario_dimensions": {
            "WHERE": {"resolution_status": "RESOLVED", "atom_id": "SO010"},
            "EGO_ACTION": {"resolution_status": "RESOLVED", "atom_id": "PH005"},
        },
    }
    state = HARAState(run_id="sec-speed-test", stage=WorkflowStage.SCORING)
    state.functions = [{"function_id": "FUN-1", "name": "parking control"}]
    state.malfunctions = [{
        "malfunction_id": "MF-1", "function_id": "FUN-1",
        "guideword": "Less", "description": "control loss",
        "component_category": "sensor_camera",
    }]
    state.scenarios = [ScenarioCandidate(
        "SC-1", "parking", "approach stationary vehicle", "front contact",
        facts=facts, fact_provenance={
            key: dict(source) for key in facts
            if key not in {"scenario_atom_ids", "method_scenario_dimensions"}
        },
    )]
    state.item_definition["scenario_assessments"] = [_assessment()]
    writer = ReviewArtifactWriter(state.run_id, tmp_path)

    score_structured_scenarios(
        state, MethodRuleScoringService(method), MethodContractASILService(method),
        review_artifact_writer=writer,
    )

    assert state.scenarios[0].facts["ego_speed_kph"] == 7.0
    selected = state.scenarios[0].fact_provenance["ego_speed_kph"]
    assert selected["project_rule_id"] == "AVP_EGO_SPEED_CLOSED_UPPER_BOUND_V1"
    assert selected["analysis_value_semantic"] == "CURRENT_OPERATION_EGO_SPEED"
    calculation = next(
        item for item in state.audit_trail
        if item["event"] == "structured_risk_scoring_completed"
    )["risk_calculation_inputs"][0]
    context = calculation["hazardous_event_risk_context"]
    assert context["ego_speed_kph"]["value"] == 7.0
    assert context["relative_speed_kph"]["value"] == 7.0
    assert context["ttc_s"]["value"] == 3.6
    assert state.risk_results[0].severity.value.startswith("S")
    assert state.risk_results[0].exposure.value.startswith("E")
    trace = json.loads((tmp_path / state.run_id / "risk_execution_trace.json").read_text())
    assert trace["scenario_eligibility_summary"]["risk_scoring_invoked"] == 1
    assert trace["assessments"][0]["controllability"]["derived_ttc"]["closing_speed_kph"] == 7.0

    conflicted = deepcopy(state)
    conflicted.stage = WorkflowStage.SCORING
    conflicted.scenarios[0].facts["relative_speed_kph"] = 12.0
    conflicted.scenarios[0].fact_provenance["relative_speed_kph"] = dict(source)
    score_structured_scenarios(
        conflicted, MethodRuleScoringService(method),
        MethodContractASILService(method),
    )
    audit = conflicted.audit_trail[-1]["risk_calculation_inputs"][0]
    assert audit["physics_conflicts"][0]["field"] == "relative_speed_kph"
    assert audit["hazardous_event_risk_context"]["relative_speed_kph"]["status"] == "UNAVAILABLE"
    assert conflicted.risk_results[0].exposure.value.startswith("E")


def test_persisted_review_without_canonical_commit_is_not_ineligible():
    review_root = Path("runtime/review")
    writer = ReviewArtifactWriter("risk-trace-interrupted-v2", review_root)
    writer.record_scenario_candidate({"scenario_id": "SC-1", "facts": {}})
    writer.record_scenario_feasibility(_assessment())
    trace = RiskExecutionTraceService(_method()).project_review_run(
        ReviewArtifactReader("risk-trace-interrupted-v2", review_root)
    )
    row = trace["assessments"][0]
    assert trace["risk_stage_status"] == "NOT_REACHED"
    assert row["feasibility"]["persisted_review"] is True
    assert row["feasibility"]["committed_to_state"] is False
    assert row["risk_eligibility"]["status"] == "NOT_COMMITTED"
    assert row["risk_scoring_invoked"] is False


def test_completed_feasibility_with_causal_gap_is_ineligible():
    trace = RiskExecutionTraceService(_method()).project(
        run_id="trace", assessments=[_assessment(causal=False)], candidates=[], committed=True,
    )
    assert trace["assessments"][0]["risk_eligibility"]["status"] == "INELIGIBLE_CAUSAL"


def test_final_retain_is_a_derived_projection_not_an_independent_input():
    payload = _assessment()
    payload["final_retain"] = False
    decision = evaluate_risk_eligibility_payload(payload)
    assert decision.eligible is True
    assert decision.final_retain_value is True
    assert decision.final_retain_role == "DERIVED_FROM_RISK_ELIGIBILITY"


def test_review_projection_uses_canonical_causal_hazard_node_as_stable_identity():
    review_root = Path("runtime/review")
    writer = ReviewArtifactWriter("hazardous-event-id-projection-v1", review_root)
    writer.record_scenario_feasibility(_assessment())

    record = ReviewArtifactReader(
        "hazardous-event-id-projection-v1", review_root,
    ).read_all()["scenario_feasibility"][-1]
    assert record["hazardous_event_id"] == "HE::MF-1::SC-1::H"

    trace = RiskExecutionTraceService(_method()).project_review_run(
        ReviewArtifactReader("hazardous-event-id-projection-v1", review_root)
    )
    assert trace["assessments"][0]["hazardous_event_id"] == "HE::MF-1::SC-1::H"


def test_review_writer_uses_assessment_eligibility_projection():
    assessment = ScenarioFeasibilityAssessment(
        malfunction_id="MF-1", scenario_id="SC-1", physically_feasible=True,
        functionally_relevant=True, causally_relevant=True, hazardous_event="hazard",
        risk_dimensions_changed=[], rationale="test", confidence=0.9, breakpoint="NONE",
        causal_assessment=_causal_assessment(finalized=True),
        status=ReviewStatus.FINALIZED,
    )
    review_root = Path("runtime/review")
    writer = ReviewArtifactWriter("derived-retain-projection-v1", review_root)
    writer.record_scenario_feasibility(assessment)
    record = ReviewArtifactReader(
        "derived-retain-projection-v1", review_root,
    ).read_all()["scenario_feasibility"][-1]
    assert record["final_retain"] is True
    assert record["final_retain_role"] == "DERIVED_FROM_RISK_ELIGIBILITY"
    assert record["risk_eligibility_status"] == "ELIGIBLE"


def test_missing_legacy_causal_contract_fails_closed():
    payload = _assessment()
    payload.pop("causal_assessment")
    decision = evaluate_risk_eligibility_payload(payload)
    assert decision.status.value == "PENDING_FEASIBILITY"
    assert "PENDING_LEGACY_ELIGIBILITY" in decision.reason_codes
