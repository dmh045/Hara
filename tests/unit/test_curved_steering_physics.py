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
    class Frozen:
        calls = 0
        def complete_json(self, request):
            self.calls += 1
            assert '不能用它代替纯转向' in request.system_prompt
            catalog = json.loads(request.user_prompt)['model_catalog']
            assert 'CONSTANT_SPEED_CURVATURE_TO_CONTACT' in catalog
            return LLMResponse(curved_plan(), 'OFFLINE_SYNTHETIC')
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
