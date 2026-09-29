from __future__ import annotations

from pathlib import Path

from hara_agent.contracts import (
    CausalBreakpoint, CausalEdge, CausalGraph, CausalNode, CausalNodeType,
    CausalRelation, EvidenceBinding, RiskDimensionChange,
    ScenarioCausalAssessment,
)
from hara_agent.models import EvidenceKind
from hara_agent.models import ReviewStatus, ScenarioCandidate
from hara_agent.services.analysis import (
    MethodContractASILService,
    MethodRuleScoringService,
    MethodSafetyGoalService,
)
from hara_agent.services.reporting import HARAExcelRenderer
from hara_agent.template import TemplateRoleCompiler
from hara_agent.workflow import HARAState, WorkflowStage
from hara_agent.workflow.nodes import (
    aggregate_safety_goals, pass_quality_gate, score_structured_scenarios,
)


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "references" / "HARA_Template_AI_20260327.xlsx"


def _method():
    return TemplateRoleCompiler().compile_method(TEMPLATE, use_manifest=False)


def _source(location: str) -> dict:
    return {
        "approval": "FINALIZED",
        "provenance": "PROJECT_INPUT",
        "source_refs": [{
            "source_type": "project_input",
            "source_id": "item.docx",
            "location": location,
            "excerpt": location,
        }],
    }


def _canonical_facts() -> tuple[dict, dict]:
    facts = {
        "collision_type": "VEHICLE_TO_ROAD_USER",
        "road_user_type": "PEDESTRIAN",
        "speed_unspecified_kph": 20.0,
        "exposure_method": "F",
        "occurrence_frequency": "MONTHLY_OR_MORE",
        "avoidability_percent": 99.5,
    }
    provenance = {key: _source(key) for key in facts if key != "exposure_method"}
    provenance["exposure_method"] = _source("exposure_method")
    return facts, provenance


def _causal_assessment() -> dict:
    node_ids = ("M", "B", "I", "H", "HARM")
    node_types = (
        CausalNodeType.MALFUNCTION,
        CausalNodeType.SYSTEM_BEHAVIOR_CHANGE,
        CausalNodeType.OPERATIONAL_CONSEQUENCE,
        CausalNodeType.HAZARD,
        CausalNodeType.HARM,
    )
    edge_ids = ("M_TO_B", "B_TO_I", "I_TO_H", "H_TO_HARM")
    edges = tuple(CausalEdge(
        edge_id, source, target, CausalRelation.CAUSES,
        f"{source} causes {target}", (f"TEST.{edge_id}",),
    ) for edge_id, source, target in zip(edge_ids, node_ids, node_ids[1:]))
    assessment = ScenarioCausalAssessment(
        "SCN-1",
        CausalGraph(
            tuple(CausalNode(node_id, node_type, node_type.value)
                  for node_id, node_type in zip(node_ids, node_types)),
            edges,
        ),
        node_ids,
        CausalBreakpoint.NONE,
        tuple(EvidenceBinding(
            edge.edge_id, edge.evidence_refs, EvidenceKind.DIRECT_FACT,
            status=ReviewStatus.FINALIZED,
        ) for edge in edges),
        (),
        (RiskDimensionChange("severity", ("TEST.I_TO_H",), "test change"),),
        "vehicle approaches pedestrian",
        "pedestrian injury",
        review_status=ReviewStatus.FINALIZED,
    )
    return assessment.to_dict()


def test_compiler_parses_natural_language_controllability_thresholds():
    method = _method()
    by_result = {item.result: item for item in method.controllability.criteria}

    c1 = by_result["C1"].predicates[0]
    c3 = by_result["C3"].predicates[0]
    assert c1.lower == 99.0 and c1.lower_inclusive is False
    assert c3.upper == 90.0 and c3.upper_inclusive is False


def test_method_rule_service_executes_exact_facts_and_preserves_method_ambiguity():
    method = _method()
    facts, provenance = _canonical_facts()
    facts["_fact_provenance"] = provenance

    scored = MethodRuleScoringService(method).score(facts, "vehicle approaches pedestrian")

    assert scored["severity"]["severity_score"] == "S2"
    assert scored["severity"]["engineering_status"] == "PENDING"
    assert "Template ambiguity" in scored["severity"]["engineering_basis"]
    assert scored["exposure"]["exposure_score"] == "E3"
    assert scored["exposure"]["engineering_status"] == "FINALIZED"
    assert scored["controllability"]["controllability_score"] == "C1"
    assert scored["controllability"]["engineering_status"] == "FINALIZED"
    assert scored["severity"]["engineering_source_type"] == "method_contract"


def test_method_rule_service_fails_closed_on_missing_fact_and_open_boundary():
    method = _method()
    facts, provenance = _canonical_facts()
    facts.pop("speed_unspecified_kph")
    provenance.pop("speed_unspecified_kph")
    facts["avoidability_percent"] = 99.0
    facts["_fact_provenance"] = provenance

    scored = MethodRuleScoringService(method).score(facts, "hazard")

    assert scored["severity"]["severity_score"] == ""
    assert "SPEED_UNSPECIFIED" in scored["severity"]["missing_fact_types"]
    assert scored["controllability"]["controllability_score"] == ""
    assert "boundary" in scored["controllability"]["engineering_basis"]


def test_exposure_method_requires_its_own_grounded_provenance():
    method = _method()
    facts, provenance = _canonical_facts()
    provenance.pop("exposure_method")
    facts["_fact_provenance"] = provenance

    scored = MethodRuleScoringService(method).score(facts, "hazard")

    assert scored["exposure"]["exposure_score"] == "E3"
    assert scored["exposure"]["engineering_status"] == "PENDING"
    assert "exposure_method" in scored["exposure"]["engineering_basis"]


def test_unresolved_exposure_and_controllability_use_engineering_paths_not_item_guesses():
    method = _method()
    facts = {
        "collision_type": "FRONTAL",
        "speed_unspecified_kph": 20.0,
        "_fact_provenance": {
            "collision_type": _source("collision_type"),
            "speed_unspecified_kph": _source("speed_unspecified_kph"),
        },
    }

    scored = MethodRuleScoringService(method).score(facts, "hazard")

    assert scored["exposure"]["engineering_status"] == "PENDING"
    assert "no T/F project fact is requested" in scored["exposure"]["engineering_basis"]
    assert scored["exposure"]["reference_mapping_count"] == 376
    assert "exposure_situation_mapping" in scored["exposure"]["calculation_path"]
    assert scored["controllability"]["engineering_status"] == "PENDING"
    assert "not extracted or guessed" in scored["controllability"]["engineering_basis"]
    assert "avoidability_executor" in scored["controllability"]["calculation_path"]


def test_synthetic_causal_hara_regression_preserves_scoring_goal_gate_and_draft():
    method = _method()
    facts, provenance = _canonical_facts()
    facts["avoidability_percent"] = 80.0
    facts["relative_distance"] = "10 m"
    facts["relative_speed_kph"] = 18.0
    facts["closing_speed_kph"] = 18.0
    provenance["relative_distance"] = _source("relative_distance")
    provenance["relative_speed_kph"] = _source("relative_speed_kph")
    provenance["closing_speed_kph"] = _source("closing_speed_kph")
    state = HARAState(run_id="method-scoring", stage=WorkflowStage.SCORING)
    state.functions = [{"function_id": "FUN-1", "name": "braking control"}]
    state.scenarios = [ScenarioCandidate(
        scenario_id="SCN-1",
        operating_scenario="Parking",
        situational_description="Parking",
        situational_detailing="vehicle approaches pedestrian",
        facts=facts,
        fact_provenance=provenance,
    )]
    state.malfunctions = [{
        "malfunction_id": "MF-1",
        "function_id": "FUN-1",
        "guideword": "loss",
        "description": "braking request lost",
    }]
    state.item_definition["scenario_assessments"] = [{
        "malfunction_id": "MF-1",
        "scenario_id": "SCN-1",
        "physically_feasible": True,
        "functionally_relevant": True,
        "causally_relevant": True,
        "breakpoint": "NONE",
        "risk_dimensions_changed": ["severity"],
        "hazardous_event": "vehicle approaches pedestrian",
        "potential_harm": "pedestrian injury",
        "status": "FINALIZED",
        "causal_assessment": _causal_assessment(),
    }]

    result = score_structured_scenarios(
        state,
        MethodRuleScoringService(method),
        MethodContractASILService(method),
    )

    risk = result.risk_results[0]
    assert risk.severity.value == "S2"
    assert risk.severity.status is ReviewStatus.PENDING
    assert risk.exposure.status is ReviewStatus.FINALIZED
    assert risk.controllability.value == "C3"
    assert risk.controllability.status is ReviewStatus.FINALIZED
    assert risk.asil.value == "B"
    assert risk.asil.status is ReviewStatus.PENDING
    assert risk.asil.sources[0].source_type == "method_contract"
    calculation_event = next(
        item for item in result.audit_trail
        if item["event"] == "structured_risk_scoring_completed"
    )
    calculation = calculation_event["risk_calculation_inputs"][0]
    assert calculation["controllability_input"]["ttc_s"] == 2.0
    assert "ttc_s" in calculation["controllability_input"]["inputs_used"]
    assert calculation["ftti_input"]["asil"] == "B"
    assert calculation["ftti_input"]["status"] == "PENDING_METHOD_SEMANTICS"

    aggregate_safety_goals(result, MethodSafetyGoalService(method))
    pass_quality_gate(result)
    output_path = ROOT / "tmp" / "synthetic-causal-hara.xlsx"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = HARAExcelRenderer(
        method.report_contract,
        template_hash=str(method.metadata["template_hash"]),
    ).render(result, TEMPLATE, output_path, draft=True)

    assert len(result.functions) == 1
    assert len(result.malfunctions) == 1
    assert len(result.scenarios) == 1
    assert len(result.risk_results) == 1
    assert len(result.safety_goals) == 1
    assert result.stage is WorkflowStage.QUALITY_GATE
    assert output.exists()
    output.unlink()


def test_method_safety_goal_service_aggregates_exact_intent_and_stays_pending():
    service = MethodSafetyGoalService(_method())
    first = service.register_intent(
        function_name="制动控制",
        malfunction="制动请求丢失",
        guideword="loss",
        scenario_id="SCN-1",
        scenario_description="停车场泊车",
        hazard_event="车辆接近行人",
        asil="A",
    )
    second = service.register_intent(
        function_name="制动控制",
        malfunction="制动请求丢失",
        guideword="loss",
        scenario_id="SCN-2",
        scenario_description="停车场泊车",
        hazard_event="车辆接近墙体",
        asil="C",
    )

    assert first["sg_id"] == second["sg_id"]
    entry = service.to_dict()[first["sg_id"]]
    assert entry["max_asil"] == "C"
    assert entry["derivation_status"] == "NEEDS_REVIEW"
    assert service.is_approved is False
