"""C response credit follows the actual scoped path and continuous trajectory."""
from copy import deepcopy
import json

import pytest

from test_autonomous_physics import method, raw_plan, make_candidate, instantiate, score
from test_c_reference_completion import crossing_candidate
from test_scoped_timing_and_avoidance import timed_plan
from hara_agent.contracts.physical_scenario_plan import PhysicalScenarioPlan
from hara_agent.services.analysis import StructuredRiskScoringService, ExposureDimensionCoverageService
from hara_agent.services.analysis.scenario_physics import moving_target_avoidance, rectilinear_moving_contact
from hara_agent.services.analysis.risk_execution_trace_service import RiskExecutionTraceService
from hara_agent.services.reporting.engineering_text_mapper import EngineeringReportTextMapper


@pytest.mark.parametrize("text", [
    "停放车辆造成视线遮挡。",
    "道路无遮挡，但停放车辆遮挡行人视线。",
    "The road is unobstructed, but parked vehicles occlude the pedestrian.",
    "车库立柱限制视野，行人直到近距离才被发现。",
    "Recognition is delayed by parked vehicles.",
    "不是实测上界，邻车遮挡视线。",
    "道路无遮挡而停放车辆遮挡行人。",
    "标线不构成遮挡且停放车辆遮挡行人。",
    "The road has no occlusion and parked vehicles occlude the pedestrian.",
    "The driver does not see the pedestrian because a parked vehicle occludes the view.",
    "The driver cannot see the pedestrian because a parked vehicle occludes the view.",
    "并非没有遮挡，车辆挡住了行人。",
    "不假定实体遮挡物，但邻车遮挡行人视线。",
    "未验证共同计算平台、电源或传感器遮挡导致的共同失效；邻车遮挡行人视线。",
    "未验证独立性，停放车辆遮挡行人视线。",
    "未验证当前已发生的传感器遮挡导致的共同失效能否被响应消除。",
    "未验证当前遮挡导致的共同失效后果。",
    "不据此声明实体阻挡、遮挡或限制通行，但停放车辆遮挡行人视线。",
])
def test_reference_exclusion_is_local_to_the_actual_condition(method, text):
    child = crossing_candidate(method)
    child.facts["physical_analysis"]["hypothesis"]["conditions"].append(text)
    before = deepcopy(child)
    result = score(method, child)
    c = result["controllability"]
    assert c["controllability_score"] == ""
    assert c["reference_scope_exclusions"]
    assert any(row.get("reason_code") in {"CONFLICT_VISIBILITY_OCCLUDED", "CONFLICT_RECOGNITION_DELAYED"}
        for row in c["reference_scope_exclusions"])
    trace = RiskExecutionTraceService(method)._controllability_trace(child.facts, c)
    assert "遮挡或延迟识别" in EngineeringReportTextMapper.controllability_branch({"controllability": trace})
    assert child == before and result["severity"]["severity_score"]


@pytest.mark.parametrize("text", [
    "本假设不覆盖遮挡场景；道路无遮挡。",
    "标线不构成实体遮挡，驾驶员持续监控。",
    "No visual occlusion; delayed recognition is not modeled here.",
    "Unobstructed sightline. This transfer excludes occlusion.",
    "while the marking itself does not physically block or occlude the pedestrian",
    "也不据此推定行人被阻挡或视线被遮挡",
    "不构成实体障碍或遮挡",
    "未建模行人加速、折返或拥挤遮挡",
    "The marking is not treated as a physical wall, curb, or proven occluder.",
    "没有降低可见性，驾驶员持续监控。",
    "There is no limited visibility; the driver monitors the conflict.",
    "本条仅保留可能遮挡的未覆盖条件，不据此声称实际视野受限。",
    "不假定实体遮挡物，也不改变路面附着。",
    "车位标线及斜向车位几何仅作规划与几何参照，不据此声明实体阻挡、遮挡或限制通行。",
])
def test_no_conflict_reference_keeps_its_transparent_monitoring_assumption(method, text):
    child = crossing_candidate(method)
    child.facts["physical_analysis"]["hypothesis"]["conditions"].append(text)
    child.facts["physical_analysis"]["hypothesis"]["uncovered_conditions"].append("遮挡与延迟识别未覆盖")
    c = score(method, child)["controllability"]
    assert c["controllability_score"] == "C2"
    assert c["decision_status"] == "CONDITIONAL_REFERENCE_MATCH"
    assert c["engineering_status"] == "PENDING"


def test_automatic_brake_execution_does_not_supply_the_manual_path(method):
    raw = raw_plan(initial=0, acceleration=.5)
    raw["hypotheses"][0]["remaining_controls"] = ["braking", "automatic_braking_command"]
    child = instantiate(method, raw, make_candidate(0))[0]
    original = deepcopy(child)
    blocked = score(method, child)
    assert child.facts["emergency_braking_available"] is True
    assert not blocked["controllability"]["controllability_score"]
    assert blocked["controllability"]["intervention_applicability"]["manual"]["status"] == "MANUAL_BRAKING_PATH_UNRESOLVED"
    assert child == original
    child.facts["physical_analysis"]["hypothesis"]["remaining_controls"].append("driver_braking")
    accepted = score(method, child)
    assert accepted["controllability"]["controllability_score"]
    for dimension in ("severity", "exposure"):
        assert {k: v for k, v in accepted[dimension].items() if k != "branch_evaluations"} == {
            k: v for k, v in blocked[dimension].items() if k != "branch_evaluations"}
        for left, right in zip(accepted["branch_evaluations"], blocked["branch_evaluations"], strict=True):
            assert left["scores"][dimension] == right["scores"][dimension]
    # Human-approved scoped control declarations use the same authority path.
    child.fact_provenance["physical_analysis"].update(provenance="HUMAN_CONFIRMATION", approval="FINALIZED")
    assert score(method, child)["controllability"]["controllability_score"]


def test_crossing_manual_path_gap_is_reported_as_a_control_input(method):
    child = crossing_candidate(method)
    child.facts["physical_analysis"]["hypothesis"]["remaining_controls"] = ["braking", "automatic_braking_command"]
    c = score(method, child)["controllability"]
    assert c["decision_status"] == "REFERENCE_APPLICABILITY_UNRESOLVED"
    assert not c["controllability_score"]
    trace = RiskExecutionTraceService(method)._controllability_trace(child.facts, c)
    text = EngineeringReportTextMapper.controllability_branch({"controllability": trace})
    assert "驾驶员人工制动路径" in text and "无适用 C 规则" not in text


def rpa_world():
    return dict(fault_motion_model="RECTILINEAR_MOVING_TARGET_CONTACT", ego_speed_kph=5,
        longitudinal_acceleration_mps2=0, model_valid_until_s=4, object_speed_kph=10,
        object_offset_x_m=-6, object_offset_y_m=6, contact_half_length_m=2.6,
        contact_half_width_m=1.9, object_extent="FINITE_CONTACT_ENVELOPE",
        ego_longitudinal_direction="REVERSE", object_longitudinal_direction="RIGHT", object_position="REAR")


def test_reported_rpa_escape_requires_a_finite_stop_and_reorientation():
    facts = rpa_world()
    proposal = {"avoidance": dict(recognition_delay_s=1, escape_direction="LEFT", escape_speed_mps=1,
                                 available_clearance_m=4.1)}
    before = deepcopy((facts, proposal))
    assert rectilinear_moving_contact(facts)["time_s"] == pytest.approx(2.448)
    assert moving_target_avoidance(facts, proposal)["status"] == "AVOIDANCE_TRANSITION_UNRESOLVED"
    assert before == (facts, proposal)
    proposal["avoidance"]["transition"] = dict(deceleration_mps2=2.5, acceleration_mps2=1, reorientation_time_s=.2)
    failed = moving_target_avoidance(facts, proposal)
    assert failed["status"] == "ESCAPE_DOES_NOT_ESTABLISH_AVOIDANCE"
    assert failed["escape"]["time_s"] == pytest.approx(2.448)
    # An explicitly stronger conditional response is a useful positive case,
    # not a default rider capability or a change to the unmitigated world.
    proposal["avoidance"]["transition"] = dict(deceleration_mps2=10, acceleration_mps2=4, reorientation_time_s=.1)
    accepted = moving_target_avoidance(facts, proposal)
    assert accepted["status"] == "CONDITIONAL_ESCAPE_VALIDATED"
    assert accepted["original"] == failed["original"]
    phases = accepted["target_response_segments"]
    for left, right in zip(phases, phases[1:]):
        dt = left["end_s"] - left["begin_s"]
        assert left["end_s"] == right["begin_s"]
        assert left["position_m"] + left["velocity_mps"]*dt + left["acceleration_mps2"]*dt*dt/2 == pytest.approx(right["position_m"])
        assert left["velocity_mps"] + left["acceleration_mps2"]*dt == pytest.approx(right["velocity_mps"])
    assert facts == before[0]


def test_direct_scoring_cannot_bypass_replay_evasion_check(method):
    from test_bounded_interaction_coverage import moving_plan
    raw = moving_plan(initial=0, target_direction="FORWARD", x=-8, position="REAR", collision="REAR_END")
    raw["hypotheses"][0]["remaining_controls"] = []
    parent = make_candidate(0)
    parent.facts["other_road_user_avoidance_possible"] = True
    child = instantiate(method, raw, parent)[0]
    view = {**child.facts, "malfunction_id": "MF", "scenario_id": child.scenario_id,
        "_fact_provenance": child.fact_provenance}
    view["_exposure_dimension_coverage_decision"] = ExposureDimensionCoverageService(method).decide(
        assessment_key="test", function=None, operating_mode=child.operating_mode)
    original = deepcopy(view)
    result = StructuredRiskScoringService(method).score(view, "same contact")
    assert result["controllability"]["decision_status"] != "OVERRIDE_MATCH"
    assert view == original
    # The no-model source/table-only contract remains readable and executable.
    incomplete = {key: value for key, value in view.items() if key != "fault_motion_model"}
    assert StructuredRiskScoringService(method).score(incomplete, "analytical projection missing model")["controllability"]["decision_status"] != "OVERRIDE_MATCH"
    legacy = {key: value for key, value in view.items() if key not in {
        "fault_motion_model", "collision_branches", "physical_analysis", "control_analysis"}}
    assert StructuredRiskScoringService(method).score(legacy, "source table fact")["controllability"]["decision_status"] == "OVERRIDE_MATCH"


def scoped_target_braking_case(method, analysis_key):
    from test_bounded_interaction_coverage import moving_plan
    raw = moving_plan(initial=0, target_direction="FORWARD", x=-8, position="REAR", collision="REAR_END")
    h = raw["hypotheses"][0]
    h["remaining_controls"] = []
    h["intervention_assumptions"] = {"other_road_user_avoidance_possible": {
        "value": True, "rationale": "Conditional target braking", "basis_refs": ["MF.description"],
        "conditions": ["The target stops within this unchanged finite interaction"], "avoidance": {
            "recognition_delay_s": .1, "escape_direction": "FORWARD", "escape_speed_mps": 0,
            "available_clearance_m": 1, "transition": {"deceleration_mps2": 1}}}}
    parent = make_candidate(0)
    parent.facts["other_road_user_avoidance_possible"] = True
    child = instantiate(method, raw, parent)[0]
    # General confirmed capability does not approve a response in every world.
    child.fact_provenance["other_road_user_avoidance_possible"] = {
        "provenance": "HUMAN_CONFIRMATION", "approval": "FINALIZED",
        "source_refs": [{"source_type": "human_confirmation", "source_id": "independent-capability",
            "location": "general target response capability"}]}
    if analysis_key == "control_analysis":
        child.facts[analysis_key] = child.facts.pop("physical_analysis")
        child.fact_provenance[analysis_key] = child.fact_provenance.pop("physical_analysis")
    return child


def target_braking_consumers(method, child):
    from hara_agent.services.analysis.hazardous_event_risk_context_service import HazardousEventRiskContextService
    view = {**child.facts, "malfunction_id": "MF", "scenario_id": child.scenario_id,
        "_fact_provenance": child.fact_provenance}
    context = HazardousEventRiskContextService(method).build(malfunction_id="MF", scenario_id=child.scenario_id,
        hazard_node_id="target-braking-scope", scenario=view)
    view["_exposure_dimension_coverage_decision"] = ExposureDimensionCoverageService(method).decide(
        assessment_key="target-braking-scope", function=None, operating_mode=child.operating_mode)
    return context, StructuredRiskScoringService(method).score(view, "same finite contact")


@pytest.mark.parametrize("analysis_key", ["physical_analysis", "control_analysis"])
def test_same_scope_target_response_survives_old_checkpoint_read(method, analysis_key):
    from dataclasses import asdict
    from hara_agent.workflow import HARAState
    child = scoped_target_braking_case(method, analysis_key)
    restored = HARAState._scenario_from_dict(json.loads(json.dumps(asdict(child))))
    before = deepcopy(restored)
    context, direct = target_braking_consumers(method, restored)
    assert context.other_road_user_avoidance_possible.value is True
    assert direct["controllability"]["controllability_score"] == "C1"
    assert direct["controllability"]["intervention_applicability"]["other_road_user"]["status"] == "CONDITIONAL_ESCAPE_VALIDATED"
    assert score(method, restored)["controllability"]["controllability_score"] == "C1"
    assert restored == before


@pytest.mark.parametrize("analysis_key", ["physical_analysis", "control_analysis"])
@pytest.mark.parametrize("scope_field,other_identity", [("scenario_id", "another-scenario"), ("malfunction_id", "another-malfunction")])
def test_confirmed_target_capability_cannot_borrow_another_scope_response(method, analysis_key, scope_field, other_identity):
    child = scoped_target_braking_case(method, analysis_key)
    child.fact_provenance[analysis_key]["applicable_scope"][scope_field] = other_identity
    before = deepcopy(child)
    context, direct = target_braking_consumers(method, child)
    assert context.other_road_user_avoidance_possible.value is None
    c = direct["controllability"]
    assert c["decision_status"] != "OVERRIDE_MATCH"
    assert c["intervention_applicability"]["other_road_user"]["status"] == "AVOIDANCE_PROPOSAL_SOURCE_UNRESOLVED"
    assert score(method, child)["controllability"]["decision_status"] != "OVERRIDE_MATCH"
    assert child == before


def test_optional_transition_preserves_old_plan_checkpoint_key_and_zero_request_resume(method, tmp_path):
    from test_geometry_completion import Client, run
    from hara_agent.services.semantic.physical_scenario_agent import PhysicalScenarioAgent
    from hara_agent.workflow import HARAState
    raw = timed_plan()
    client = Client([raw])
    agent = PhysicalScenarioAgent(client, response_checkpoint_dir=tmp_path)
    agent.RESPONSE_VALIDATION_VERSION = "physical-response-validation-v33-remote-trajectory-applicability"
    parent = make_candidate(3.6)
    state, children = run(method, agent, candidates={"MF": [parent]})
    old_keys = set(state.item_definition["physical_analysis_plans"])
    assert len(client.requests) == 1
    restored = HARAState.read_committed(json.loads(json.dumps(state.to_dict())))
    agent.RESPONSE_VALIDATION_VERSION = PhysicalScenarioAgent.RESPONSE_VALIDATION_VERSION
    _, resumed = run(method, agent, state=restored, candidates={"MF": [parent]})
    assert resumed == children and set(restored.item_definition["physical_analysis_plans"]) == old_keys
    assert len(client.requests) == 1
    assert PhysicalScenarioPlan.from_dict(raw, evidence_ids={"MF.description"}).to_dict() == raw


@pytest.mark.parametrize("transition", [
    {"acceleration_mps2": -1}, {"deceleration_mps2": True},
    {"reorientation_time_s": float("nan")}, {"acceleration_mps2": "1 m/s2"},
    {"instant_reversal": True}, {},
])
def test_transition_contract_rejects_unbounded_or_untyped_physics(transition):
    raw = timed_plan()
    raw["hypotheses"][0]["intervention_assumptions"] = {"other_road_user_avoidance_possible": {
        "value": True, "rationale": "Conditional response candidate", "basis_refs": ["MF.description"],
        "conditions": ["Unchanged finite contact envelope"], "avoidance": {
            "recognition_delay_s": .1, "escape_direction": "FORWARD", "escape_speed_mps": 1,
            "available_clearance_m": 5, "transition": transition}}}
    before = deepcopy(raw)
    with pytest.raises(ValueError):
        PhysicalScenarioPlan.from_dict(raw, evidence_ids={"MF.description"})
    assert raw == before


def automatic_plan():
    raw = timed_plan()
    h = raw["hypotheses"][0]
    h["remaining_controls"] = ["automatic_braking_command", "braking"]
    h["safe_state_proposal"].update(required_controls=h["remaining_controls"][:],
        conditions=["The automatic command and execution path are conditionally independent."])
    return raw


@pytest.mark.parametrize("declaration", ["remaining", "failed", "unknown"])
def test_existing_motion_manual_assessment_is_control_only_and_resumes_without_requests(method, tmp_path, declaration):
    from test_geometry_completion import Client, run
    from hara_agent.contracts.physical_scenario_plan import CONTROL_ASSESSMENT_REQUIRED, CONTROL_ASSESSMENT_OPTIONAL
    from hara_agent.services.analysis.physical_hypothesis_instantiation_service import PhysicalHypothesisInstantiationService
    from hara_agent.services.semantic.physical_scenario_agent import PhysicalScenarioAgent
    from hara_agent.workflow import HARAState
    raw = automatic_plan()
    parent = make_candidate(3.6)
    parent.facts.update(driver_position="in_driver_seat", driver_in_vehicle=True, operating_mode="parking_vehicle_control")
    child = instantiate(method, raw, parent)[0]
    before = deepcopy(child)
    producer = PhysicalHypothesisInstantiationService(method)
    assert producer.task_scope(child, "MF") == "CONTROL_INPUTS_ONLY"
    h = raw["hypotheses"][0]
    assessment = {key: deepcopy(value) for key, value in h.items()
        if key in CONTROL_ASSESSMENT_REQUIRED | CONTROL_ASSESSMENT_OPTIONAL}
    if declaration == "remaining":
        assessment["remaining_controls"].append("driver_braking")
    elif declaration == "failed":
        assessment["failed_controls"].append("driver_braking")
    else:
        assessment["unresolved_controls"] = {"driver_braking": "The source supports the automatic path only; pedal independence is unestablished."}
    reply = dict(hypotheses=[], uncovered_conditions=raw["uncovered_conditions"], control_assessment=assessment)
    client = Client([reply])
    agent = PhysicalScenarioAgent(client, response_checkpoint_dir=tmp_path)
    state, generated = run(method, agent, candidates={"MF": [child]})
    revised = generated["MF"][0]
    assert len(client.requests) == 1
    assert json.loads(client.requests[0].user_prompt)["context"]["task_scope"] == "CONTROL_INPUTS_ONLY"
    for key in ("motion_model_outcome", "ego_impact_speed_kph", "relative_speed_kph", "ttc_s", "collision_branches"):
        assert revised.facts[key] == before.facts[key]
    assert child == before
    assert producer.task_scope(revised, "MF") == ""
    restored = HARAState.read_committed(json.loads(json.dumps(state.to_dict())))
    _, resumed = run(method, agent, state=restored, candidates={"MF": [child]})
    assert resumed == generated and len(client.requests) == 1
    c = score(method, revised)["controllability"]
    if declaration == "unknown":
        assert not c["controllability_score"]
    elif declaration == "failed":
        assert c["controllability_score"] == "C3"
        assert c["engineering_rule_id"] == "IAV_UNAVAILABLE_BRAKING_REFERENCE"
        assert revised.facts["emergency_braking_available"] is True
        from hara_agent.services.analysis.risk_calculation_input_service import RiskCalculationInputService
        timing = RiskCalculationInputService().ftti("MF", revised.scenario_id, "",
            {**revised.facts, "_fact_provenance": revised.fact_provenance}, method=method)
        assert timing.hazard_time_limit_s > 0 and timing.timing_trace["required_controls"] == ["automatic_braking_command", "braking"]
        from hara_agent.models import EvidenceValue, ReviewStatus
        trace = RiskExecutionTraceService(method)._controllability_trace(revised.facts, c)
        evidence = EvidenceValue(value=c["controllability_score"], status=ReviewStatus.PENDING)
        text = EngineeringReportTextMapper().controllability_rationale(evidence, trace)
        assert "人工制动路径已明确不可用" in text and c["controllability_score"] in text
        assert "自动制动路径" in text and "FTTI" in text
        assert "当前故障条件明确不能使用制动避险" not in text
    else:
        assert c["controllability_score"]


def test_cached_completed_motion_gets_only_one_manual_completeness_correction(method, monkeypatch, tmp_path):
    from test_geometry_completion import Client, run
    from hara_agent.services.analysis.physical_hypothesis_instantiation_service import PhysicalHypothesisInstantiationService
    from hara_agent.services.semantic.physical_scenario_agent import PhysicalScenarioAgent
    from hara_agent.workflow import HARAState
    raw = automatic_plan()
    repaired = deepcopy(raw)
    repaired["hypotheses"][0]["unresolved_controls"] = {"driver_braking": "A compatible pedal path was assessed but fault independence is unresolved."}
    client = Client([raw, repaired])
    agent = PhysicalScenarioAgent(client, response_checkpoint_dir=tmp_path)
    agent.RESPONSE_VALIDATION_VERSION = "physical-response-validation-v34-c-response-applicability"
    parent = make_candidate(3.6)
    parent.facts.update(driver_position="in_driver_seat", driver_in_vehicle=True)
    original = PhysicalHypothesisInstantiationService.manual_path_gap
    monkeypatch.setattr(PhysicalHypothesisInstantiationService, "manual_path_gap", lambda *args: None)
    state, before = run(method, agent, candidates={"MF": [parent]})
    assert len(client.requests) == 1
    old_keys = set(state.item_definition["physical_analysis_plans"])
    entry = next(iter(state.item_definition["physical_analysis_plans"].values()))
    entry.update(semantic_correction_completed=True, initial_plan=deepcopy(raw), correction_diagnostics=[])
    monkeypatch.setattr(PhysicalHypothesisInstantiationService, "manual_path_gap", original)
    agent.RESPONSE_VALIDATION_VERSION = PhysicalScenarioAgent.RESPONSE_VALIDATION_VERSION
    restored = HARAState.read_committed(json.loads(json.dumps(state.to_dict())))
    _, after = run(method, agent, state=restored, candidates={"MF": [parent]})
    assert len(client.requests) == 2 and set(restored.item_definition["physical_analysis_plans"]) == old_keys
    current = next(iter(restored.item_definition["physical_analysis_plans"].values()))
    assert current["validation_history"]
    assert {d["reason"] for d in current["correction_diagnostics"]} == {"CONTROL_MANUAL_PATH_UNASSESSED"}
    assert after["MF"][0].facts["motion_model_outcome"] == before["MF"][0].facts["motion_model_outcome"]
    resumed = HARAState.read_committed(json.loads(json.dumps(restored.to_dict())))
    _, repeated = run(method, agent, state=resumed, candidates={"MF": [parent]})
    assert repeated == after and len(client.requests) == 2
