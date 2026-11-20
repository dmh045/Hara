"""A turn changes velocity direction, not independently its x/y energy."""
from copy import deepcopy
import json
import math

import pytest

from test_autonomous_physics import method, raw_plan, make_candidate, instantiate, score, mf
from hara_agent.services.analysis.scenario_physics import constant_speed_curved_contact, branch_physics_view
from hara_agent.services.analysis.asil_service import MethodContractASILService
from hara_agent.services.semantic.physical_scenario_agent import PhysicalScenarioAgent
from hara_agent.infrastructure.llm import LLMResponse
from hara_agent.services.reporting.engineering_text_mapper import EngineeringReportTextMapper
from hara_agent.workflow import HARAState, CheckpointRepository
from hara_agent.workflow.physical_analysis import construct_physical_scenarios


def curved_plan(speed=7, radius=5, clearance=1.75, side='LEFT'):
    raw = raw_plan('CONSTANT_LATERAL_ACCELERATION_TO_CONTACT', initial=speed)
    h = raw['hypotheses'][0]
    h.update(model='CONSTANT_SPEED_CURVATURE_TO_CONTACT', remaining_controls=['driver_braking'])
    h['geometry']['ego_lateral_direction'] = side
    h['parameters'] = [dict(field=k, unit=u, kind='POINT', value=v, rationale='Explicit offline kinematic assumption')
        for k,u,v in [('ego_speed_kph','km/h',speed),('turn_radius_m','m',radius),
                      ('lateral_clearance_m','m',clearance),('model_valid_until_s','s',5)]]
    h['conditions'] = ['Constant speed, no sideslip, assumed radius and effective first-contact clearance']
    h['uncovered_conditions'] = ['Not a grip or steering capability bound; no instability or drive acceleration']
    return raw


def test_curved_response_correction_reaches_timing_without_longitudinal_ttc(method, tmp_path):
    from test_geometry_completion import Client, run
    from test_scoped_timing_and_avoidance import timed_plan
    from hara_agent.services.analysis.risk_calculation_input_service import RiskCalculationInputService
    full = curved_plan()
    h = full['hypotheses'][0]
    reference = timed_plan()['hypotheses'][0]
    h['safe_state_proposal'] = deepcopy(reference['safe_state_proposal'])
    h['safe_state_proposal']['conditions'] = ['Braking follows the same fixed-curvature path; no sideslip or steering escape.']
    h['hazard_timing'] = deepcopy(reference['hazard_timing'])
    missing = deepcopy(full)
    missing['hypotheses'][0]['safe_state_proposal'].pop('transition')
    client = Client([missing, full])
    agent = PhysicalScenarioAgent(client, response_checkpoint_dir=tmp_path)
    state, generated = run(method, agent, candidates={'MF': [make_candidate(7)]})
    child = generated['MF'][0]
    timing = RiskCalculationInputService().ftti('MF', child.scenario_id, '',
        {**child.facts, '_fact_provenance': child.fact_provenance}, method=method)
    arc = 5 * math.acos(1 - 1.75/5)
    expected = arc/(7/3.6) - (7/3.6)/(2*3)
    assert timing.hazard_time_limit_s == pytest.approx(expected)
    assert timing.timing_trace['available_path_distance_m'] == pytest.approx(arc)
    assert timing.status.value == 'CONDITIONAL' and 'ttc_s' not in child.facts
    assert len(client.requests) == 2 and agent.RESPONSE_TIMING_POLICY in client.requests[-1].system_prompt
    entry = next(iter(state.item_definition['physical_analysis_plans'].values()))
    for field in ('geometry', 'parameters'):
        assert entry['plan']['hypotheses'][0][field] == missing['hypotheses'][0][field]
    _, resumed = run(method, agent, state=state, candidates={'MF': [make_candidate(7)]})
    assert resumed == generated and len(client.requests) == 2


@pytest.mark.parametrize('speed', [3.6, 7, 14.9999999, 15, 20])
def test_pure_turn_preserves_speed_norm_and_scoped_c_limit(method, speed):
    child = instantiate(method, curved_plan(speed), make_candidate(speed))[0]
    child.facts.update(component_category='computing', failure_type='wrong_value',
                       _function_output='steering angle request', operating_mode='parking_vehicle_control',
                       scenario_atom_ids=['CN_parking_multistory','FB001','FA040'])
    expected_t = 5 * math.acos(1 - 1.75/5) / (speed/3.6)
    assert child.facts['first_contact_time_s'] == pytest.approx(expected_t)
    assert child.facts['ego_impact_speed_kph'] == child.facts['trajectory_max_speed_kph'] == speed
    assert 'ttc_s' not in child.facts
    assert '自车侧向位移方向：向左' in EngineeringReportTextMapper.physical_inputs(child, {})
    result = score(method, child)
    assert result['controllability']['controllability_score'] == ('C1' if speed < 15 else '')
    assert result['severity']['engineering_status'] == 'PENDING'
    for branch in result['branch_evaluations']:
        if branch['C']:
            assert branch['ASIL'] == MethodContractASILService(method).determine(branch['S'],branch['E'],branch['C'])


@pytest.mark.parametrize('change', [
    {'ego_lateral_direction':'RIGHT'}, {'lateral_clearance_m':6}, {'model_valid_until_s':.1},
])
def test_changed_geometry_invalidates_old_contact_without_reusing_ttc(method, change):
    sc = instantiate(method, curved_plan(), make_candidate(7))[0]
    view = {**sc.facts, 'scenario_id':sc.scenario_id, 'malfunction_id':'MF','_fact_provenance':sc.fact_provenance}
    branch = {**change, '_fact_provenance':{key:deepcopy(sc.fact_provenance[key]) for key in change}}
    new = branch_physics_view(view, branch)
    assert new['motion_model_outcome']['status'] == 'NO_CONTACT_WITHIN_MODEL'
    for key in ('ttc_s','first_contact_time_s','relative_speed_kph','ego_impact_speed_kph'):
        assert key not in new
    assert sc.facts['motion_model_outcome']['status'] == 'CONTACT'


def test_turn_direction_and_reverse_are_independent_of_target_location(method):
    raw = curved_plan()
    raw['hypotheses'][0]['geometry']['ego_longitudinal_direction'] = 'REVERSE'
    sc = instantiate(method, raw, make_candidate(7))[0]
    outcome = sc.facts['motion_model_outcome']
    assert outcome['contact_x_m'] < 0 and outcome['contact_y_m'] > 0
    # The old 6 m/s² condition cannot simply become a turn assumption: for
    # illustration R=v²/a would not reach the 1.75 m boundary in a quarter turn.
    alternate = {**sc.facts, 'turn_radius_m':(7/3.6)**2/6}
    assert constant_speed_curved_contact(alternate)['status'] == 'NO_CONTACT_WITHIN_MODEL'


def test_radius_range_stays_bounded_and_symbol_does_not_inherit_old_point(method):
    raw = curved_plan()
    parameter = next(p for p in raw['hypotheses'][0]['parameters'] if p['field']=='turn_radius_m')
    parameter.update(kind='RANGE',lower=1,upper=10)
    children = instantiate(method,raw,make_candidate(7))
    assert len(children)==3
    assert {c.facts['motion_model_outcome']['status'] for c in children} == {'CONTACT','NO_CONTACT_WITHIN_MODEL'}
    assert all(not c.facts['physical_analysis']['sensitivity']['range_invariance_proven']
        for c in children)
    parameter.clear()
    parameter.update(field='turn_radius_m',unit='m',kind='SYMBOLIC',symbol='R',rationale='Radius unresolved')
    parent = deepcopy(children[0])
    # New proposal uses the same object, but explicitly keeps R symbolic.
    symbolic = instantiate(method,raw,parent)[0]
    assert 'turn_radius_m' not in symbolic.facts and 'relative_speed_kph' not in symbolic.facts
    relations = symbolic.facts['physical_analysis']['sensitivity']['parameter_relations']
    assert relations[0]['symbolic_parameters']=={'turn_radius_m':'R'}


def test_real_workflow_reuses_plan_not_changed_geometry_or_old_causal(method,tmp_path):
    from test_scoped_timing_and_avoidance import timed_plan
    complete = curved_plan(); h = complete['hypotheses'][0]
    h['remaining_controls'] = ['automatic_braking_command', 'braking']
    h['unresolved_controls'] = {'driver_braking': 'This fixture assesses automatic braking only; the manual path is explicitly unresolved.'}
    for field in ('safe_state_proposal', 'hazard_timing'):
        h[field] = deepcopy(timed_plan()['hypotheses'][0][field])
    h['safe_state_proposal']['required_controls'] = h['remaining_controls'][:]
    h['safe_state_proposal'].pop('driver_positions')
    class Frozen:
        calls = 0
        def complete_json(self, request):
            self.calls += 1
            assert '不能用它代替纯转向' in request.system_prompt
            catalog = json.loads(request.user_prompt)['model_catalog']
            assert 'CONSTANT_SPEED_CURVATURE_TO_CONTACT' in catalog
            return LLMResponse(deepcopy(complete), 'OFFLINE_SYNTHETIC')
    a = make_candidate(7)
    b = deepcopy(a); b.scenario_id += '-outside'
    a.facts.update(driver_position='in_driver_seat', driver_in_vehicle=True)
    b.facts.update(driver_position='outside_driver_seat', driver_in_vehicle=False)
    client = Frozen(); agent = PhysicalScenarioAgent(client,response_checkpoint_dir=tmp_path/'responses')
    state=HARAState('curved-reuse');repo=CheckpointRepository(tmp_path/'checkpoints')
    result = construct_physical_scenarios(state, {'MF':[a,b]}, [mf()], method=method,
        agent=agent,project_facts=None,checkpoint=repo.save)
    assert client.calls == 1 and len(result['MF']) == 2
    assert all(c.scenario_id not in {a.scenario_id,b.scenario_id} for c in result['MF'])
    assert not state.item_definition.get('scenario_assessments')
    resumed = construct_physical_scenarios(repo.load(state.run_id), {'MF':[a,b]}, [mf()],
        method=method,agent=agent,project_facts=None,checkpoint=repo.save)
    assert resumed == result and client.calls == 1
    a.facts['object_position']='RIGHT'  # A new condition is a different task.
    construct_physical_scenarios(state, {'MF':[a]}, [mf()], method=method,
        agent=agent,project_facts=None,checkpoint=repo.save)
    assert client.calls > 1


def curve_facts(*, reverse=False):
    return dict(ego_speed_kph=5, turn_radius_m=5, lateral_clearance_m=1.75,
        model_valid_until_s=5, ego_lateral_direction='LEFT', object_position='LEFT',
        ego_longitudinal_direction='REVERSE' if reverse else 'FORWARD',
        object_extent='EXTENDED_ALONG_TRAVEL_PATH', object_speed_kph=0, collision_type='SIDE')


@pytest.mark.parametrize('reverse', [False, True])
def test_extended_curve_is_conditional_on_longitudinal_target_coverage(reverse):
    facts = curve_facts(reverse=reverse)
    before = deepcopy(facts)
    result = constant_speed_curved_contact(facts)
    assert result['status'] == 'CONTACT' and result['impact_speed_kph'] == 5
    assert result['time_s'] == pytest.approx(3.107562804250348)
    coverage = result['target_longitudinal_coverage']
    assert coverage['contact_x_m'] == pytest.approx((-1 if reverse else 1)*3.799671038392666)
    assert coverage['coordinate_frame'] == 'BODY_FORWARD_POSITIVE'
    assert coverage['finite_span_checked'] is False and coverage['contact_within_interval'] is None
    assert coverage['effective_contact_interval_m'] is None
    assert '单个有限对象' in coverage['limitation'] and '未由当前输入核验' in coverage['limitation']
    assert facts == before and 'contact_half_length_m' not in facts  # no default length


@pytest.mark.parametrize('reverse', [False, True])
def test_existing_effective_contact_envelope_checks_curved_contact_x(reverse):
    facts = curve_facts(reverse=reverse)
    original = constant_speed_curved_contact(facts)
    contact_x = original['contact_x_m']
    declared = {**facts, 'object_offset_x_m':contact_x, 'contact_half_length_m':.5}
    result = constant_speed_curved_contact(declared)
    assert result['status'] == 'CONTACT'
    for key in ('time_s', 'impact_speed_kph', 'heading_change_rad', 'contact_x_m', 'contact_y_m'):
        assert result[key] == original[key]
    coverage = result['target_longitudinal_coverage']
    assert coverage['effective_contact_interval_m'] == [contact_x-.5, contact_x+.5]
    assert coverage['finite_span_checked'] and coverage['contact_within_interval']
    assert '不是实际车长' in coverage['limitation']
    # A declared interval endpoint is inclusive, with no invented tolerance.
    assert constant_speed_curved_contact({**declared, 'object_offset_x_m':contact_x+.5})['status'] == 'CONTACT'
    outside = constant_speed_curved_contact({**declared, 'object_offset_x_m':0})
    assert outside['status'] == 'MODEL_GEOMETRY_NOT_APPLICABLE'
    assert outside['reason'] == 'CONTACT_OUTSIDE_TARGET_LONGITUDINAL_EXTENT'
    assert outside['target_longitudinal_coverage']['contact_within_interval'] is False
    assert outside['candidate_extended_boundary_contact']['contact_x_m'] == contact_x
    assert 'time_s' not in outside  # neither finite-vehicle contact nor no hazard is claimed


@pytest.mark.parametrize('span', [
    {'object_offset_x_m':4}, {'contact_half_length_m':1},
    {'object_offset_x_m':4,'contact_half_length_m':0},
    {'object_offset_x_m':float('nan'),'contact_half_length_m':1},
    {'object_offset_x_m':4,'contact_half_length_m':True},
])
def test_partial_or_invalid_curved_span_never_becomes_an_extended_target_default(span):
    result = constant_speed_curved_contact({**curve_facts(), **span})
    assert result['status'] == 'MODEL_INPUTS_UNRESOLVED'
    assert result['reason'] == 'TARGET_LONGITUDINAL_EXTENT_UNRESOLVED'
    assert 'time_s' not in result


def test_curved_span_lineage_and_branch_recalculation_remove_out_of_scope_contact(method):
    from hara_agent.contracts.physical_scenario_plan import PARAMETER_UNITS
    from hara_agent.services.analysis.risk_calculation_input_service import RiskCalculationInputService
    from test_scoped_timing_and_avoidance import timed_plan
    raw = curved_plan(speed=5)
    h = raw['hypotheses'][0]
    for key, value in [('object_offset_x_m',4),('contact_half_length_m',1)]:
        h['parameters'].append(dict(field=key,unit=PARAMETER_UNITS[key],kind='POINT',value=value,
            rationale='Declared effective combined contact-envelope interval, not actual vehicle length'))
    for field in ('safe_state_proposal', 'hazard_timing'):
        h[field] = deepcopy(timed_plan()['hypotheses'][0][field])
    child = instantiate(method,raw,make_candidate(5))[0]
    assert child.facts['motion_model_outcome']['target_longitudinal_coverage']['contact_within_interval']
    lineage = child.fact_provenance['first_contact_time_s']['inputs']
    assert 'SCN.object_offset_x_m' in lineage and 'SCN.contact_half_length_m' in lineage
    assert child.fact_provenance['object_offset_x_m']['approval'] == 'PENDING'
    before = deepcopy(child)
    changed = branch_physics_view({**child.facts, 'scenario_id':child.scenario_id, 'malfunction_id':'MF',
        '_fact_provenance':child.fact_provenance}, {'object_offset_x_m':0,
        '_fact_provenance':{'object_offset_x_m':child.fact_provenance['object_offset_x_m']}})
    assert changed['motion_model_outcome']['reason'] == 'CONTACT_OUTSIDE_TARGET_LONGITUDINAL_EXTENT'
    assert 'first_contact_time_s' not in changed and 'ego_impact_speed_kph' not in changed
    assert 'relative_speed_kph' not in changed
    assert RiskCalculationInputService().ftti('MF',child.scenario_id,'',changed,method=method).hazard_time_limit_s is None
    assert child == before
    # Explicit symbolic span inputs do not silently revert to an unbounded side.
    for parameter in h['parameters']:
        if parameter['field'] in {'object_offset_x_m','contact_half_length_m'}:
            parameter.pop('value'); parameter.update(kind='SYMBOLIC',symbol=parameter['field'])
    symbolic = instantiate(method,raw,make_candidate(5))[0]
    assert symbolic.facts['motion_model_outcome']['reason'] == 'TARGET_LONGITUDINAL_EXTENT_UNRESOLVED'
    assert 'first_contact_time_s' not in symbolic.facts


def test_curve_catalog_change_isolates_requests_but_old_plans_still_read(method, tmp_path, monkeypatch):
    from hara_agent.contracts.physical_scenario_plan import MODEL_CATALOG, PhysicalScenarioPlan
    from test_geometry_completion import Client
    raw = curved_plan()
    old_plan = json.loads(json.dumps(raw))
    current = deepcopy(MODEL_CATALOG['CONSTANT_SPEED_CURVATURE_TO_CONTACT'])
    previous = deepcopy(current)
    previous.pop('optional')
    previous['scope'] = 'Previously compiled constant-curvature extended-side scope without finite-span verification.'
    context = {'plan_key':'curve-catalog-isolation', 'evidence':{'MF.description':'source fault'}}
    client = Client([raw,raw])
    agent = PhysicalScenarioAgent(client,response_checkpoint_dir=tmp_path/'responses')
    monkeypatch.setitem(MODEL_CATALOG,'CONSTANT_SPEED_CURVATURE_TO_CONTACT',previous)
    first = agent.propose(context)[0]
    assert agent.propose(context)[0] == first and len(client.requests) == 1
    monkeypatch.setitem(MODEL_CATALOG,'CONSTANT_SPEED_CURVATURE_TO_CONTACT',current)
    second = agent.propose(context)[0]
    assert first == second and len(client.requests) == 2
    assert client.requests[0].user_prompt != client.requests[1].user_prompt
    assert agent.propose(context)[0] == second and len(client.requests) == 2
    assert PhysicalScenarioPlan.from_dict(old_plan,evidence_ids={'MF.description'}).to_dict() == raw
    child = instantiate(method,old_plan,make_candidate(7))[0]
    restored = HARAState.read_committed(json.loads(json.dumps(HARAState('old-curve',scenarios=[child]).to_dict())))
    assert restored.scenarios == [child]
