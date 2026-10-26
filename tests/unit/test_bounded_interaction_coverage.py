"""Moving worlds through the ordinary proposal/physics/causal/risk consumers."""
from copy import deepcopy
from dataclasses import replace
import json

import pytest

from test_autonomous_physics import raw_plan, make_candidate, instantiate, score, mf
from test_geometry_completion import Client, run
from test_p7_closure import method, source
from hara_agent.services.analysis.scenario_physics import branch_physics_view, rectilinear_moving_contact
from hara_agent.services.semantic.physical_scenario_agent import PhysicalScenarioAgent


def moving_plan(*, target_direction="REVERSE", x=8, y=0, initial=3.6, target_speed=3.6,
                acceleration=0, position="FRONT", collision="FRONTAL", horizon=10):
    raw = raw_plan(initial=initial)
    h = raw["hypotheses"][0]
    h["model"] = "RECTILINEAR_MOVING_TARGET_CONTACT"
    h["geometry"].update(object_extent="FINITE_CONTACT_ENVELOPE", object_position=position,
        collision_type=collision, object_longitudinal_direction=target_direction)
    h["conditions"] = ["constant target velocity, fixed body axes; envelope is an analytical assumption"]
    h["uncovered_conditions"] = ["target turning and reaction; unmeasured vehicle dimensions"]
    h["remaining_controls"] = ["driver_braking"]
    values = {"ego_speed_kph": initial, "object_speed_kph": target_speed,
        "longitudinal_acceleration_mps2": acceleration, "object_offset_x_m": x, "object_offset_y_m": y,
        "contact_half_length_m": 1, "contact_half_width_m": .5, "model_valid_until_s": horizon}
    from hara_agent.contracts.physical_scenario_plan import PARAMETER_UNITS
    h["parameters"] = [dict(field=k, value=v, kind="POINT", unit=PARAMETER_UNITS[k],
        rationale="explicit offline representative, not a vehicle specification") for k, v in values.items()]
    return raw


@pytest.mark.parametrize("kwargs,time,relative", [
    ({}, 3.5, 7.2),
    ({"target_direction": "FORWARD", "initial": 7.2}, 7, 3.6),
    ({"target_direction": "FORWARD", "initial": 0, "x": -8,
      "position": "REAR", "collision": "REAR_END"}, 7, 3.6),
    ({"target_direction": "RIGHT", "x": 4, "y": 4, "position": "LEFT", "collision": "SIDE"}, 3.5, 3.6*2**.5),
    ({"target_direction": "LEFT", "x": 4, "y": -4, "position": "RIGHT", "collision": "SIDE"}, 3.5, 3.6*2**.5),
])
def test_relative_contact_produces_scoped_risk_without_static_target_conversion(method, kwargs, time, relative):
    raw = moving_plan(**kwargs)
    child = instantiate(method, raw, make_candidate(kwargs.get("initial", 3.6)))[0]
    assert child.facts["object_speed_kph"] == kwargs.get("target_speed", 3.6)
    assert child.facts["first_contact_time_s"] == pytest.approx(time)
    assert child.facts["relative_speed_kph"] == pytest.approx(relative)
    assert child.fact_provenance["relative_speed_kph"]["approval"] == "PENDING"
    risk = score(method, child)
    assert risk["severity"]["severity_score"]
    if kwargs.get("collision") == "SIDE":
        assert "ttc_s" not in child.facts
    else:
        assert child.facts["ttc_s"] == pytest.approx(time)
        if kwargs.get("position") == "REAR":
            assert not risk["controllability"]["controllability_score"]
            assert risk["controllability"]["pending_reason"] == "REAR_APPROACH_BRAKING_APPLICABILITY_UNRESOLVED"
        else:
            assert risk["controllability"]["controllability_score"]


def test_crossing_pedestrian_does_not_require_vehicle_side_impact_label(method):
    raw = moving_plan(target_direction="RIGHT", x=4, y=4, position="FRONT", collision="VEHICLE_TO_ROAD_USER")
    raw['hypotheses'][0]['geometry']['road_user_type'] = 'PEDESTRIAN'
    parent = make_candidate(3.6)
    parent.facts['road_user_type'] = 'PEDESTRIAN'
    child = instantiate(method, raw, parent)[0]
    assert child.facts['motion_model_outcome']['status'] == 'CONTACT'
    assert child.facts['first_contact_time_s'] == 3.5
    assert child.facts['collision_type'] == 'VEHICLE_TO_ROAD_USER'
    assert 'ttc_s' not in child.facts


def test_explicit_post_stop_world_reaches_contact_and_invalidates_old_phase(method):
    raw = moving_plan(initial=7, target_speed=7, target_direction='FORWARD',
        x=-6, acceleration=-7, position='REAR', collision='REAR_END', horizon=4)
    h = raw['hypotheses'][0]
    next(p for p in h['parameters'] if p['field'] == 'contact_half_length_m')['value'] = 4.3
    parent = make_candidate(7)
    before = deepcopy(parent)
    original = instantiate(method, raw, parent)[0]
    assert original.facts['motion_model_outcome']['limiting_condition'] == 'EGO_STOP'
    assert not original.facts.get('collision_branches')
    h['post_stop_motion'] = 'HOLD_POSITION'
    h['conditions'].append('Conditional braking force maintains the stopping position on this road for four seconds; no restart or rollaway.')
    client = Client([raw])
    agent = PhysicalScenarioAgent(client)
    state, results = run(method, agent, candidates={'MF': [parent]})
    child = results['MF'][0]
    stop = (7 / 3.6) / 7
    distance = (7 / 3.6)**2 / (2 * 7)
    assert child.facts['first_contact_time_s'] == pytest.approx((6 + distance - 4.3) / (7 / 3.6))
    assert child.facts['first_contact_time_s'] > stop
    assert child.facts['ego_impact_speed_kph'] == 0
    assert child.facts['relative_speed_kph'] == 7
    assert child.facts['motion_model_outcome']['contact_phase'] == 'HELD_AFTER_STOP'
    assert child.fact_provenance['post_stop_motion']['approval'] == 'PENDING'
    assert score(method, child)['severity']['severity_score']
    assert parent == before and child.semantic_fingerprint != original.semantic_fingerprint
    assert child.scenario_id != original.scenario_id
    view = {**child.facts, 'scenario_id': child.scenario_id, 'malfunction_id': 'MF',
        '_fact_provenance': child.fact_provenance}
    removed = branch_physics_view(view, {'post_stop_motion': None})
    assert removed['motion_model_outcome']['limiting_condition'] == 'EGO_STOP'
    assert 'ttc_s' not in removed and 'relative_speed_kph' not in removed
    run(method, agent, state=state, candidates={'MF': [deepcopy(parent)]})
    assert len(client.requests) == 1


@pytest.mark.parametrize('reverse,lateral_offset,horizon,contact', [
    (False, 0, 10, True), (True, 0, 10, True),
    (False, 2, 10, False), (False, 0, 2, False),
])
def test_post_stop_contact_respects_body_direction_envelope_and_horizon(method, reverse, lateral_offset, horizon, contact):
    raw = moving_plan(initial=3.6, target_speed=3.6,
        target_direction='REVERSE' if reverse else 'FORWARD',
        x=8 if reverse else -8, y=lateral_offset, acceleration=-2,
        position='FRONT' if reverse else 'REAR', collision='REAR_END', horizon=horizon)
    h=raw['hypotheses'][0]
    h['post_stop_motion']='HOLD_POSITION'
    h['geometry']['ego_longitudinal_direction']='REVERSE' if reverse else 'FORWARD'
    parent=make_candidate(3.6)
    parent.facts['ego_longitudinal_direction']=h['geometry']['ego_longitudinal_direction']
    child=instantiate(method,raw,parent)[0]
    outcome=child.facts['motion_model_outcome']
    assert (outcome['status']=='CONTACT') is contact
    if contact:
        assert outcome['time_s']==7.25
        assert outcome['ego_impact_speed_kph']==0 and outcome['relative_speed_kph']==3.6
    else:
        assert outcome['evaluated_until_s']==horizon and outcome['limiting_condition']=='MODEL_HORIZON'
        assert not child.facts.get('collision_branches') and 'ttc_s' not in child.facts


@pytest.mark.parametrize('change', ['unknown_mode','wrong_model','no_conditions'])
def test_post_stop_assumption_is_model_scoped(change):
    from hara_agent.contracts.physical_scenario_plan import PhysicalScenarioPlan
    raw=moving_plan();h=raw['hypotheses'][0]
    h['post_stop_motion']='HOLD_POSITION'
    if change=='unknown_mode':h['post_stop_motion']='REVERSE'
    if change=='wrong_model':h['model']='CONSTANT_ACCELERATION_TO_IMPACT_DISTANCE'
    if change=='no_conditions':h['conditions']=[]
    with pytest.raises(ValueError,match='post_stop_motion'):
        PhysicalScenarioPlan.from_dict(raw,evidence_ids={'MF.description'})


@pytest.mark.parametrize('kwargs,contact_time', [
    ({'target_direction':'RIGHT','x':.25,'y':5,'position':'LEFT','collision':'SIDE'},4.5),
    ({'target_direction':'REVERSE','x':8,'position':'FRONT','collision':'FRONTAL'},6.75),
])
def test_post_stop_contact_does_not_inherit_a_pre_stop_braking_time_grade(method,kwargs,contact_time):
    raw=moving_plan(initial=3.6,acceleration=-2,**kwargs)
    raw['hypotheses'][0]['post_stop_motion']='HOLD_POSITION'
    child=instantiate(method,raw,make_candidate(3.6))[0]
    assert child.facts['motion_model_outcome']['contact_phase']=='HELD_AFTER_STOP'
    assert child.facts['first_contact_time_s']==contact_time
    risk=score(method,child)
    assert risk['severity']['severity_score']
    assert not risk['controllability']['controllability_score']
    assert risk['controllability']['pending_reason']=='POST_STOP_BRAKING_APPLICABILITY_UNRESOLVED'
    if kwargs['collision']=='SIDE':assert 'ttc_s' not in child.facts


def test_driver_scoped_hypotheses_are_not_cross_producted(method):
    from hara_agent.contracts.physical_scenario_plan import PhysicalScenarioPlan
    from hara_agent.services.analysis.physical_hypothesis_instantiation_service import PhysicalHypothesisInstantiationService
    raw = moving_plan()
    inside = raw['hypotheses'][0]
    inside['driver_positions'] = ['in_driver_seat']
    outside = deepcopy(inside)
    outside.update(hypothesis_id='outside', driver_positions=['outside_driver_seat'],
        remaining_controls=[], unresolved_controls={'braking': 'This outside-only condition does not credit direct intervention'})
    raw['hypotheses'].append(outside)
    plan = PhysicalScenarioPlan.from_dict(raw, evidence_ids={'MF.description'})
    for position, expected in [('in_driver_seat', True), ('outside_driver_seat', None)]:
        parent = make_candidate(3.6)
        parent.facts.update(driver_position=position, driver_in_vehicle=position=='in_driver_seat')
        rows, errors = PhysicalHypothesisInstantiationService(method).instantiate(parent, mf(), plan,
            plan_key='scoped', response_model='offline')
        assert not errors and len(rows) == 1
        assert rows[0].facts.get('emergency_braking_available') is expected
        assert rows[0].facts['physical_analysis']['hypothesis']['driver_positions'] == [position]


def test_numeric_target_position_conflict_uses_one_existing_family_correction(method):
    bad = moving_plan(x=-8, position='FRONT')
    good = moving_plan(x=8, position='FRONT')
    client = Client([bad, good])
    agent = PhysicalScenarioAgent(client)
    parent = make_candidate(3.6)
    state, result = run(method, agent, candidates={'MF': [parent]})
    assert len(client.requests) == 2
    plan = next(iter(state.item_definition['physical_analysis_plans'].values()))
    assert plan['initial_plan'] == bad
    assert plan['correction_diagnostics'][0]['reason'] == 'TARGET_OFFSET_POSITION_CONFLICT'
    assert result['MF'][0].facts['motion_model_outcome']['status'] == 'CONTACT'
    run(method, agent, state=state, candidates={'MF': [deepcopy(parent)]})
    assert len(client.requests) == 2


@pytest.mark.parametrize("kwargs,status", [
    ({"target_direction": "RIGHT", "x": 4, "y": 8, "position": "LEFT", "collision": "SIDE"}, "NO_CONTACT_WITHIN_MODEL"),
    ({"target_direction": "FORWARD", "target_speed": 7.2}, "NO_CONTACT_WITHIN_MODEL"),
    ({"horizon": 1}, "NO_CONTACT_WITHIN_MODEL"),
    ({"acceleration": -2, "target_direction": "FORWARD"}, "NO_CONTACT_WITHIN_MODEL"),
    ({"x": .5}, "INITIAL_CONTACT_REQUIRES_SEPARATE_PHASE"),
    ({"x": -8}, "MODEL_GEOMETRY_NOT_APPLICABLE"),
])
def test_noncontact_does_not_invent_impact_or_ttc(method, kwargs, status):
    child = instantiate(method, moving_plan(**kwargs), make_candidate(3.6))[0]
    assert child.facts["motion_model_outcome"]["status"] == status
    assert not child.facts.get("collision_branches")
    assert "ttc_s" not in child.facts
    assert "ego_impact_speed_kph" not in child.facts


def test_moving_target_response_shared_between_drivers_and_recovered(method, tmp_path):
    parent = make_candidate(3.6)
    parent.facts.update(object_speed_kph=3.6, object_id="object1")
    parent.fact_provenance["object_speed_kph"] = source()
    other = deepcopy(parent)
    other.scenario_id = "outside"
    other.facts["driver_in_vehicle"] = False
    client = Client([moving_plan()])
    agent = PhysicalScenarioAgent(client, response_checkpoint_dir=tmp_path)
    state, result = run(method, agent, candidates={"MF": [parent, other]})
    assert len(client.requests) == 1 and len(result["MF"]) == 2
    assert all(c.facts["motion_model_outcome"]["status"] == "CONTACT" for c in result["MF"])
    original = [c.scenario_id for c in result["MF"]]
    _, recovered = run(method, agent, state=state, candidates={"MF": [parent, other]})
    assert len(client.requests) == 1
    assert [c.scenario_id for c in recovered["MF"]] == original
    child = result["MF"][0]
    view = {**child.facts, "scenario_id": child.scenario_id, "malfunction_id": "MF", "_fact_provenance": child.fact_provenance}
    changed = branch_physics_view(view, {"object_offset_y_m": 10,
        "_fact_provenance": {"object_offset_y_m": child.fact_provenance["object_offset_y_m"]}})
    assert "ttc_s" not in changed and "relative_speed_kph" not in changed
    assert changed["motion_model_outcome"]["status"] == "NO_CONTACT_WITHIN_MODEL"


def test_instantiated_driver_pair_keeps_one_physical_family(method):
    from hara_agent.services.analysis.scenario_equivalence import driver_families
    inside = make_candidate(3.6)
    inside.facts.update(driver_position='in_driver_seat', driver_in_vehicle=True)
    outside = deepcopy(inside)
    outside.scenario_id = 'outside'
    outside.facts.update(driver_position='outside_driver_seat', driver_in_vehicle=False)
    children = [instantiate(method, moving_plan(), p)[0] for p in (inside, outside)]
    assert children[0].facts['impact_time_id'] != children[1].facts['impact_time_id']
    before = deepcopy(children)
    assert len(driver_families(children)) == 1
    assert children == before
    children[1].facts['object_id'] = 'different-actual-target'
    assert len(driver_families(children)) == 2


def test_total_parameter_tasks_bounded_but_original_range_retained(method):
    raw = raw_plan()
    raw["hypotheses"][0]["parameters"][-1].update(kind="RANGE", lower=0, upper=10)
    children = instantiate(method, raw)
    assert len(children) == 3
    sensitivity = children[0].facts["physical_analysis"]["sensitivity"]
    assert sensitivity["deferred_points"]
    assert not sensitivity["deferred_points_evaluated"]
    assert not sensitivity["range_invariance_proven"]


def test_selected_item_condition_reaches_physics_identity_causal_input_and_report(method):
    from hara_agent.contracts.physical_scenario_plan import PhysicalScenarioPlan
    from hara_agent.services.analysis.physical_hypothesis_instantiation_service import PhysicalHypothesisInstantiationService
    from hara_agent.services.analysis.scenario_equivalence import scenario_conditions, signature
    from hara_agent.services.reporting.engineering_text_mapper import EngineeringReportTextMapper
    raw = moving_plan(target_direction="RIGHT", x=4, y=4, position="LEFT", collision="SIDE")
    condition = {"source_field": "odd_road_types", "value": "垂直车位",
        "physical_effect": "出库纵向轨迹与左侧通道目标横穿轨迹相交；接触取决于同时到达，非标签判断"}
    raw["hypotheses"][0]["operating_condition"] = condition
    service = PhysicalHypothesisInstantiationService(method)
    project = {"odd_road_types": ["水平车位", "垂直车位"], "sources": []}
    children, errors = service.instantiate(make_candidate(3.6), mf(),
        PhysicalScenarioPlan.from_dict(raw, evidence_ids={"MF.description"}),
        plan_key="source-conditioned", response_model="offline", project_context=project)
    assert not errors and len(children) == 1
    child = children[0]
    assert child.facts["selected_source_condition"] == condition
    assert child.facts["source_operating_conditions"]["selected_condition"] == condition
    assert child.fact_provenance["selected_source_condition"]["approval"] == "PENDING"
    assert "垂直车位" in EngineeringReportTextMapper.physical_inputs(child, {})
    # Production main-sheet text must expose the new interaction, not look
    # identical to the old stationary-object scenario.
    description, detail = EngineeringReportTextMapper().scenario(child)
    assert '垂直车位' in description and '横穿' in description and '3.6 km/h' in description
    assert condition['physical_effect'] in detail
    other = instantiate(method, moving_plan(target_direction="RIGHT", x=4, y=4, position="LEFT", collision="SIDE"), make_candidate(3.6))[0]
    assert signature(scenario_conditions(child)) != signature(scenario_conditions(other))
    raw["hypotheses"][0]["operating_condition"]["value"] = "高速公路"
    children, errors = service.instantiate(make_candidate(3.6), mf(),
        PhysicalScenarioPlan.from_dict(raw, evidence_ids={"MF.description"}),
        plan_key="outside-scope", response_model="offline", project_context=project)
    assert not children and errors[0]["reason_code"] == "OPERATING_CONDITION_OUTSIDE_ITEM_SCOPE"


def test_environment_alternatives_survive_primary_bundle_selection(method):
    from test_p6_scenario_pipeline import parent, project, malfunction, function
    from dataclasses import asdict
    from hara_agent.services.analysis import ConstrainedScenarioSynthesisService
    service = ConstrainedScenarioSynthesisService(method)
    inputs = service.build_input(malfunction=asdict(malfunction()), parent=parent(), assessment={},
        project_context=asdict(project()), function=asdict(function()))
    bundles = service.logical_selection_bundles(inputs)
    expanded = [service.expand_logical_atom_ids(inputs, b) for b in bundles]
    primary = inputs.coverage_plan.primary_variation_dimensions
    groups = {}
    for row in expanded:
        key = tuple(tuple(row.get(d, ())) for d in primary)
        groups.setdefault(key, set()).add(tuple(row.get("ROAD", ())))
    assert any(len(roads) > 1 for roads in groups.values())
    assignments = service.logical_selection_assignments(inputs)
    assert 0 < len(assignments) <= 24
    assert all(len(a) == inputs.coverage_plan.desired_variant_count <= 3 for a in assignments)
    assert all(a[0] == bundles[0] for a in assignments)  # typical is retained


def test_crossing_causal_claim_uses_simultaneous_contact_not_path_intersection(method):
    from hara_agent.services.analysis.scenario_physics import causal_physics_conflicts
    child = instantiate(method, moving_plan(target_direction="RIGHT", x=4, y=8,
        position="LEFT", collision="SIDE"), make_candidate(3.6))[0]
    payload = {"hazardous_event": "与左侧目标接触", "physical_interaction": {
        "target": "SCENARIO_OBJECT", "object_id": child.facts["object_id"], "object_position": "LEFT",
        "assessment_reference": "MODEL_TRAJECTORY", "relation": "CONTACT"}}
    assert "NO_CONTACT_TRAJECTORY_CONTRADICTS_CONTACT_CLAIM" in causal_physics_conflicts(child, payload)


def test_source_focus_and_plan_cache_are_independent_of_input_order(method):
    from hara_agent.workflow import HARAState
    from hara_agent.workflow.physical_analysis import construct_physical_scenarios
    from hara_agent.models import ItemDefinitionFacts, SourceRef
    first, second = make_candidate(), make_candidate()
    second.scenario_id = 'alternate'
    second.facts['road_surface_conditions'] = 'separate declared condition'
    project = ItemDefinitionFacts('parking', 'test', odd_road_types=['水平车位', '垂直车位'],
        odd_weather_conditions=['小雨'], sources=[SourceRef('item_definition', 'offline', 'scope', '水平/垂直车位，小雨')])
    client = Client([raw_plan(assessed=True), raw_plan(assessed=True)])
    agent = PhysicalScenarioAgent(client)
    state = HARAState('order-independent')
    for values in ([first, second], [second, first]):
        construct_physical_scenarios(state, {'MF': values}, [mf()], method=method, agent=agent, project_facts=project)
    assert len(client.requests) == 2
    assert len(state.item_definition['physical_analysis_plans']) == 2
    assert len(state.item_definition['physical_analysis_metrics']['source_condition_uncovered_ids']) == 2
