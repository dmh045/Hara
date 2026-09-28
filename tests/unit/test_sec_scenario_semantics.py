from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pytest

from hara_agent.contracts import (
    CausalBreakpoint, CausalEdge, CausalGraph, CausalNode, CausalNodeType,
    CausalRelation, EvidenceBinding, FactType, ScenarioCausalAssessment,
)
from hara_agent.method_sources import YamlBaselineCompiler
from hara_agent.models import (
    EvidenceKind, FactProvenance, ItemDefinitionFacts, MalfunctionCandidate, ReviewStatus,
    RiskFact, ScenarioCandidate, SourceRef,
)
from hara_agent.services.analysis.driver_configuration_service import (
    DriverConfigurationBrancher,
)
from hara_agent.services.analysis.hazardous_event_risk_context_service import (
    HazardousEventRiskContextService,
)
from hara_agent.services.analysis.method_risk_fact_service import (
    MethodRiskFactBindingService,
)
from hara_agent.services.analysis import MethodContractASILService, MethodRuleScoringService
from hara_agent.services.analysis.scenario_method_service import ScenarioMethodService
from hara_agent.services.analysis.scenario_selection_quality import (
    ScenarioBindingPolicy, ScenarioDimensionApplicabilityService,
    ScenarioSemanticQueryBuilder,
)
from hara_agent.services.analysis.scenario_synthesis_service import (
    ConstrainedScenarioSynthesisService,
)
from hara_agent.template import TemplateRoleCompiler
from hara_agent.workflow import HARAState, WorkflowStage
from hara_agent.workflow.nodes import score_structured_scenarios
from hara_agent.workflow.scenario_synthesis import ScenarioSynthesisRunner


ROOT = Path(__file__).resolve().parents[2]
SOURCE = SourceRef(
    "item_definition", "ItemDef.docx", "table[14].row[13]",
    "位姿状态 | 在驾驶位/不在驾驶位",
)


@pytest.fixture(scope="module")
def method():
    report = TemplateRoleCompiler().compile_method(
        ROOT / "references/HARA_Template_AI_20260327.xlsx"
    ).report_contract
    return YamlBaselineCompiler().compile(
        ROOT / "method_assets/fusa_baseline_v1/manifest.yaml",
        report_contract=report,
    )


def _parent(*, detail: str = "车辆在泊出时起步") -> ScenarioCandidate:
    return ScenarioCandidate(
        scenario_id="SCN-PARENT", operating_scenario="parking garage",
        situational_description=detail, situational_detailing=detail,
        operating_mode="active",
        facts={"ego_speed_constraint": {"min_kph": 0.0, "max_kph": 7.0}},
        status=ReviewStatus.FINALIZED,
        analysis_instance={"malfunction_id": "MF-STATE", "validation_status": "VALIDATED"},
    )


def _state_malfunction() -> dict:
    return {
        "malfunction_id": "MF-STATE", "function_id": "F-STATE",
        "guideword": "No/Loss",
        "description": "泊出取消和接管失效",
        "functional_effect": "控制应生效但未生效，车辆未产生预期加速",
        "vehicle_level_hazard": "车辆在泊出时未按预期运动",
        "component_category": "control", "failure_type": "loss",
    }


def _assessment() -> dict:
    return {
        "hazardous_event_id": "HE-STATE", "hazardous_event": "泊出时与后方车辆交互",
        "causal_assessment": {"status": "VALIDATED", "causal_chain": [
            "取消失效", "车辆未产生预期加速", "与后方车辆交互",
        ]},
    }


def _risk_fact(position: str, value: str) -> dict:
    return {
        "fact_id": f"RF-{position}", "parameter": "DRIVER_IN_VEHICLE",
        "value": value, "approval": "FINALIZED",
        "context": {"allowed_driver_position": position},
        "source_refs": [SOURCE.__dict__.copy()],
    }


def test_state_failure_keeps_identity_and_uses_evidenced_physical_action(method):
    synthesis = ConstrainedScenarioSynthesisService(method).build_input(
        malfunction=_state_malfunction(), parent=_parent(),
        assessment=_assessment(), project_context={
            "odd_locations": ["parking garage"],
            "speed_min_kph": 0.0, "speed_max_kph": 7.0,
        },
    )
    query = synthesis.structured_semantic_query
    assert query["state_failure_action_categories"] == ["ACTION_ABORT"]
    assert "ACTION_ABORT" not in query["action_categories"]
    assert "ACTION_PARK" in query["operating_action_categories"]
    assert "ACTION_ACCELERATE" in query["control_consequence_action_categories"]
    assert "ACTION_PARK" in query["action_categories"]
    assert query["explicit_category_evidence"]["ACTION_PARK"]
    action = next(
        item for item in synthesis.dimension_candidate_sets
        if item.dimension == "EGO_ACTION"
    )
    assert action.generation_status != "METHOD_GAP"


def test_state_failure_without_physical_operation_keeps_specific_method_gap(method):
    synthesis = ConstrainedScenarioSynthesisService(method).build_input(
        malfunction={
            **_state_malfunction(),
            "functional_effect": "状态交接请求未执行",
            "vehicle_level_hazard": "状态停留在原模式",
        },
        parent=_parent(detail="parking garage"),
        assessment={
            "hazardous_event_id": "HE-STATE",
            "hazardous_event": "接管状态不切换",
            "causal_assessment": {"status": "VALIDATED", "causal_chain": ["交接失效"]},
        },
        project_context={"odd_locations": ["parking garage"]},
    )
    assert synthesis.structured_semantic_query["action_categories"] == ["ACTION_ABORT"]
    action = next(
        item for item in synthesis.dimension_candidate_sets
        if item.dimension == "EGO_ACTION"
    )
    assert action.generation_status == "METHOD_GAP"


def test_static_object_exception_does_not_cover_unknown_or_mixed_object():
    builder = ScenarioSemanticQueryBuilder()
    applicability = ScenarioDimensionApplicabilityService()
    decision = ScenarioBindingPolicy({}).decide(
        _parent(), "OBJECT", (0.0, 7.0),
    )

    def assess(object_type: str, hazard: str = ""):
        parent = replace(_parent(), facts={"object_type": object_type})
        query = builder.build(
            malfunction={"description": "loss", "functional_effect": "loss",
                         "vehicle_level_hazard": hazard},
            parent=parent, assessment={"hazardous_event": hazard},
            project_context={}, fm_template={},
        )
        return query, applicability.assess("OBJECT", query, decision)

    static_query, static = assess("static_obstacle")
    assert static.status.value == "NOT_APPLICABLE"
    assert static_query["object_categories"] == ["OBJECT_STATIC"]
    unknown_query, unknown = assess("stroller")
    assert unknown.status.value == "REQUIRED"
    assert unknown_query["unmapped_object_sources"] == ["PARENT.facts.object_type"]
    mixed_query, mixed = assess("static_obstacle", "行人与静态障碍物相邻")
    assert mixed.status.value == "REQUIRED"
    assert "OBJECT_PEDESTRIAN" in mixed_query["object_categories"]


def test_driver_seat_branches_are_stable_source_linked_and_not_vehicle_false():
    brancher = DriverConfigurationBrancher(policy_id="AVP_SEC_2026_09_V1")
    parent = _parent()
    facts = (
        _risk_fact("in_driver_seat", "true"),
        _risk_fact("outside_driver_seat", "false"),
    )
    children = brancher.expand(parent, facts)
    assert len(children) == 2
    assert len({item.scenario_id for item in children}) == 2
    assert [item.facts["driver_position"] for item in children] == [
        "in_driver_seat", "outside_driver_seat",
    ]
    assert children[0].facts["driver_in_vehicle"] is True
    assert "driver_in_vehicle" not in children[1].facts
    assert children[0].fact_provenance["driver_position"]["source_refs"][0]["location"] == SOURCE.location
    assert children[1].analysis_instance["driver_configuration_branch"][
        "driver_in_vehicle_resolution"
    ] == "UNKNOWN_OUTSIDE_SEAT_IS_NOT_OUTSIDE_VEHICLE"
    assert [item.scenario_id for item in brancher.expand(parent, facts)] == [
        item.scenario_id for item in children
    ]
    assert brancher.expand(children[0], facts) == (children[0],)


def test_single_driver_position_reuses_existing_scenario_identity():
    brancher = DriverConfigurationBrancher(policy_id="AVP_SEC_2026_09_V1")
    parent = replace(_parent(), facts={"driver_position": "in_driver_seat"})
    children = brancher.expand(parent, (
        _risk_fact("in_driver_seat", "true"),
        _risk_fact("outside_driver_seat", "false"),
    ))
    assert len(children) == 1
    assert children[0].scenario_id == parent.scenario_id
    assert children[0].facts["driver_in_vehicle"] is True


def test_branch_inherits_scoped_analysis_fact_into_real_risk_context(method):
    parent = replace(
        _parent(),
        facts={"object_speed_kph": 0.0},
        fact_provenance={"object_speed_kph": {
            "provenance": "SCENARIO_DEFINED", "origin": "SCENARIO_DEFINED",
            "approval": "FINALIZED", "validation_status": "VALIDATED",
            "source_refs": [SOURCE.__dict__.copy()],
            "applicable_scope": {
                "malfunction_id": "MF-STATE", "scenario_id": "SCN-PARENT",
            },
        }},
    )
    child, = DriverConfigurationBrancher(policy_id="AVP_SEC_2026_09_V1").expand(
        parent, (_risk_fact("in_driver_seat", "true"),)
    )
    assert child.fact_provenance["object_speed_kph"]["applicable_scope"][
        "scenario_id"
    ] == child.scenario_id
    assert parent.fact_provenance["object_speed_kph"]["applicable_scope"][
        "scenario_id"
    ] == parent.scenario_id
    context = HazardousEventRiskContextService(method).build(
        malfunction_id="MF-STATE", scenario_id=child.scenario_id,
        hazard_node_id="H", scenario={
            **child.facts, "_fact_provenance": child.fact_provenance,
        },
    )
    assert context.object_speed_kph.status.value == "AVAILABLE"
    assert context.driver_in_vehicle.status.value == "AVAILABLE"


def test_outside_seat_removes_inherited_false_without_independent_vehicle_source():
    brancher = DriverConfigurationBrancher(policy_id="AVP_SEC_2026_09_V1")
    parent = replace(
        _parent(),
        facts={"driver_position": "outside_driver_seat", "driver_in_vehicle": False},
        fact_provenance={"driver_in_vehicle": {
            "source_refs": [SOURCE.__dict__.copy()],
        }},
    )
    child, = brancher.expand(parent, (_risk_fact("outside_driver_seat", "false"),))
    assert child.scenario_id == parent.scenario_id
    assert "driver_in_vehicle" not in child.facts

    explicit_source = SourceRef(
        "item_definition", "ItemDef.docx", "table[14].row[14]", "驾驶员在车外",
    )
    explicit_parent = replace(parent, fact_provenance={"driver_in_vehicle": {
        "source_refs": [explicit_source.__dict__.copy()],
    }})
    explicit_child, = brancher.expand(
        explicit_parent, (_risk_fact("outside_driver_seat", "false"),)
    )
    assert explicit_child.facts["driver_in_vehicle"] is False


def test_explicit_outside_vehicle_condition_excludes_in_seat_branch():
    source = SourceRef(
        "item_definition", "ItemDef.docx", "table[14].row[14]", "驾驶员在车外",
    )
    parent = replace(
        _parent(), facts={"driver_in_vehicle": False},
        fact_provenance={"driver_in_vehicle": {
            "provenance": "PROJECT_INPUT", "approval": "FINALIZED",
            "source_refs": [source.__dict__.copy()],
        }},
    )
    children = DriverConfigurationBrancher(policy_id="AVP_SEC_2026_09_V1").expand(
        parent, (
            _risk_fact("in_driver_seat", "true"),
            _risk_fact("outside_driver_seat", "false"),
        ),
    )
    assert len(children) == 1
    assert children[0].facts["driver_position"] == "outside_driver_seat"
    assert children[0].facts["driver_in_vehicle"] is False


def test_driver_outside_seat_binding_cannot_prove_outside_vehicle(method):
    service = MethodRiskFactBindingService(method)
    in_seat = RiskFact(
        "RF-IN", "DRIVER_IN_VEHICLE", "true", context={
            "allowed_driver_position": "in_driver_seat",
        }, source_refs=[SOURCE], provenance=FactProvenance.PROJECT_INPUT,
        approval=ReviewStatus.FINALIZED,
    )
    outside_seat = replace(in_seat, fact_id="RF-OUT", value="false", context={
        "allowed_driver_position": "outside_driver_seat",
    })
    outside_vehicle = replace(
        in_seat, fact_id="RF-VEHICLE-OUT", value="false",
        context={"driver_position": "outside_vehicle"},
    )
    assert service._normalize_value(FactType.DRIVER_IN_VEHICLE, in_seat) is True
    assert service._normalize_value(FactType.DRIVER_IN_VEHICLE, outside_seat) is None
    assert service._normalize_value(FactType.DRIVER_IN_VEHICLE, outside_vehicle) is False
    explicit = replace(outside_seat, source_refs=[SourceRef(
        "item_definition", "ItemDef.docx", "table[14].row[14]", "驾驶员在车外",
    )])
    assert service._normalize_value(FactType.DRIVER_IN_VEHICLE, explicit) is False
    project = ItemDefinitionFacts(
        system_description="AVP", item_boundary="AVP control boundary",
        sources=[SOURCE], risk_facts=[in_seat, outside_seat],
    )
    inside_binding = service.bind(project, {
        "allowed_driver_position": "in_driver_seat",
    })
    outside_binding = service.bind(project, {
        "allowed_driver_position": "outside_driver_seat",
    })
    assert inside_binding.values["driver_in_vehicle"] is True
    assert "driver_in_vehicle" not in outside_binding.values
    assert outside_binding.audit["invalid_fact_types"] == ["DRIVER_IN_VEHICLE"]
    conflicting = service.bind(project, {
        "allowed_driver_position": "in_driver_seat",
        "driver_configuration_source_conflict": True,
    })
    assert "driver_in_vehicle" not in conflicting.values
    assert conflicting.audit["conflicting_fact_types"] == ["DRIVER_IN_VEHICLE"]


def _scoring_state_for_driver_fact(project_fact: RiskFact) -> HARAState:
    child = DriverConfigurationBrancher(policy_id="AVP_SEC_2026_09_V1").expand(
        _parent(), (_risk_fact("in_driver_seat", "true"),),
    )[0]
    child.facts["scenario_atom_ids"] = ["SO010", "PH005"]
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
    causal = ScenarioCausalAssessment(
        child.scenario_id, CausalGraph(nodes, edges), ("M", "B", "I", "H"),
        CausalBreakpoint.NONE,
        tuple(EvidenceBinding(
            edge.edge_id, edge.evidence_refs, EvidenceKind.DIRECT_FACT,
            status=ReviewStatus.FINALIZED,
        ) for edge in edges),
        (), (), "driver position hazard", review_status=ReviewStatus.FINALIZED,
    )
    state = HARAState(run_id="driver-bool-scoring", stage=WorkflowStage.SCORING)
    state.functions = [{"function_id": "F-STATE", "name": "parking control"}]
    state.malfunctions = [{
        "malfunction_id": "MF-STATE", "function_id": "F-STATE",
        "guideword": "Less", "description": "control loss",
        "component_category": "sensor_camera",
    }]
    state.scenarios = [child]
    state.item_definition["typed"] = asdict(ItemDefinitionFacts(
        system_description="AVP", item_boundary="AVP control boundary",
        sources=[SOURCE], risk_facts=[project_fact],
    ))
    state.item_definition["scenario_assessments"] = [{
        "malfunction_id": "MF-STATE", "scenario_id": child.scenario_id,
        "hazardous_event": "driver position hazard", "breakpoint": "NONE",
        "risk_dimensions_changed": [], "physically_feasible": True,
        "functionally_relevant": True, "causally_relevant": True,
        "final_retain": True, "status": "FINALIZED",
        "causal_assessment": causal.to_dict(),
    }]
    return state


def test_driver_bool_binding_reaches_scoring_and_conflict_fails_closed(method):
    approved = RiskFact(
        "RF-IN", "DRIVER_IN_VEHICLE", "true",
        context={"allowed_driver_position": "in_driver_seat"},
        source_refs=[SOURCE], provenance=FactProvenance.PROJECT_INPUT,
        approval=ReviewStatus.FINALIZED,
    )
    binding = MethodRiskFactBindingService(method)
    state = _scoring_state_for_driver_fact(approved)
    score_structured_scenarios(
        state, MethodRuleScoringService(method), MethodContractASILService(method),
        risk_fact_binding=binding,
    )
    assert len(state.risk_results) == 1
    calculation = next(
        item for item in state.audit_trail
        if item["event"] == "structured_risk_scoring_completed"
    )["risk_calculation_inputs"][0]
    driver = calculation["hazardous_event_risk_context"]["driver_in_vehicle"]
    assert driver["status"] == "AVAILABLE"
    assert driver["value"] is True
    assert driver["source_provenance"] == "METHOD_RISK_FACT_BINDING"

    contradictory = _scoring_state_for_driver_fact(replace(approved, value="false"))
    score_structured_scenarios(
        contradictory, MethodRuleScoringService(method),
        MethodContractASILService(method), risk_fact_binding=binding,
    )
    assert contradictory.risk_results[0].exposure.value.startswith("E")
    assert contradictory.risk_results[0].exposure.status is ReviewStatus.FINALIZED
    assert contradictory.risk_results[0].controllability.status is ReviewStatus.PENDING
    event = next(
        item for item in contradictory.audit_trail
        if item["event"] == "structured_risk_scoring_completed"
    )
    assert event["risk_fact_binding_audits"][0]["source_conflicts"] == [{
        "field": "driver_in_vehicle",
        "scenario_value": True,
        "project_value": False,
        "reason": "FACT_SOURCE_CONFLICT",
    }]
    assert event["risk_calculation_inputs"][0]["hazardous_event_risk_context"][
        "driver_in_vehicle"
    ]["status"] == "UNAVAILABLE"


def test_driver_branch_requires_new_causal_check_without_noninterference_proof(method):
    runner = ScenarioSynthesisRunner(method=method, client=None)
    child = DriverConfigurationBrancher(policy_id="AVP_SEC_2026_09_V1").expand(
        _parent(), (_risk_fact("in_driver_seat", "true"),),
    )[0]
    parent_assessment = _assessment()
    parent_assessment["dependency_metadata"] = {
        "complete": True,
        "causal_evidence_fields": ["MF.functional_effect"],
        "physical_feasibility_fields": [],
        "scenario_identity_fields": [],
        "child_subset_refinement": "Method atom subset refinement only.",
    }
    synthesis_input = ConstrainedScenarioSynthesisService(method).build_input(
        malfunction=_state_malfunction(), parent=_parent(),
        assessment=parent_assessment,
        project_context={"odd_locations": ["parking garage"]},
    )
    delta = runner._causal_delta(
        synthesis_input=synthesis_input,
        parent_assessment=parent_assessment, child=child,
    )
    assert delta["status"] == "CAUSAL_REVALIDATION_REQUIRED"
    assert delta["reason"] == "DRIVER_CONFIGURATION_NONINTERFERENCE_UNPROVEN"
    assert delta["changed_fields"] == [
        "driver_position", "allowed_driver_position", "driver_in_vehicle",
    ]


def test_normal_analyze_instantiation_expands_only_applicable_driver_set(method):
    malfunction = MalfunctionCandidate(
        **_state_malfunction(), causal_chain=["M", "B"],
        status=ReviewStatus.FINALIZED,
    )
    facts = ItemDefinitionFacts(
        system_description="AVP", item_boundary="AVP control boundary",
        sources=[SOURCE],
        risk_facts=[
            RiskFact(
                "RF-IN", "DRIVER_IN_VEHICLE", "true",
                context={"allowed_driver_position": "in_driver_seat"},
                source_refs=[SOURCE], provenance=FactProvenance.PROJECT_INPUT,
                approval=ReviewStatus.FINALIZED,
            ),
            RiskFact(
                "RF-OUT", "DRIVER_IN_VEHICLE", "false",
                context={"allowed_driver_position": "outside_driver_seat"},
                source_refs=[SOURCE], provenance=FactProvenance.PROJECT_INPUT,
                approval=ReviewStatus.FINALIZED,
            ),
        ],
    )
    service = ScenarioMethodService(method)
    base, _ = service.instantiate_analytical_candidates(malfunction, [_parent()])
    branched, audit = service.instantiate_analytical_candidates(
        malfunction, [_parent()], project_facts=facts,
    )
    assert len(branched) == len(base) * 2
    assert audit["driver_configuration"]["post_branch_count"] == len(branched)
    assert {item.analysis_instance["driver_configuration_branch"]["driver_position"]
            for item in branched} == {"in_driver_seat", "outside_driver_seat"}
    assert all(item.analysis_instance["malfunction_id"] == malfunction.malfunction_id
               for item in branched)

    one_position = replace(
        _parent(), facts={"driver_position": "in_driver_seat"},
    )
    constrained, _ = service.instantiate_analytical_candidates(
        malfunction, [one_position], project_facts=facts,
    )
    assert len(constrained) == len(base)
    assert all(item.facts["driver_in_vehicle"] is True for item in constrained)


def test_normal_analyze_static_template_keeps_object_without_extra_e_atom(method):
    malfunction = MalfunctionCandidate(
        malfunction_id="MF-UPA", function_id="F-UPA", guideword="No/Loss",
        description="超声波障碍物漏检", functional_effect="障碍物未被识别",
        vehicle_level_hazard="前方障碍物碰撞", causal_chain=["M", "B"],
        component_category="sensor_ultrasonic", failure_type="loss",
        sources=[SOURCE], status=ReviewStatus.FINALIZED,
    )
    parent = replace(_parent(), facts={
        "scenario_atom_ids": ["SO010", "PH005"],
        "ego_speed_constraint": {"min_kph": 0.0, "max_kph": 7.0},
    })
    instances, audit = ScenarioMethodService(method).instantiate_analytical_candidates(
        malfunction, [parent],
    )
    assert any(item.facts.get("object_type") == "static_obstacle" for item in instances), audit
    static = next(item for item in instances if item.facts.get("object_type") == "static_obstacle")
    assert audit["selection_mode"] == "STRONG_TEMPLATE_ANALYTICAL_INSTANCES"
    assert static.facts["object_position"] == "front"
    assert static.facts["object_speed_kph"] == 0
    assert static.facts["scenario_atom_ids"] == parent.facts["scenario_atom_ids"]
    assert static.analysis_instance["source_option_values"]["obj_type"] == "static_obstacle"
