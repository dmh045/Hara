"""Moving response trajectories use the ordinary producer, scorer and report chain."""
from copy import deepcopy
from dataclasses import asdict
import json
import pytest
from test_p7_closure import method
from test_autonomous_physics import instantiate, make_candidate, score, mf
from test_bounded_interaction_coverage import moving_plan
from test_scoped_timing_and_avoidance import timed_plan
from hara_agent.services.analysis.scenario_physics import moving_braking_avoidability_boundary
from hara_agent.services.analysis.risk_calculation_input_service import RiskCalculationInputService


def following():
    return dict(fault_motion_model='RECTILINEAR_MOVING_TARGET_CONTACT', ego_speed_kph=7.2,
        longitudinal_acceleration_mps2=0, object_speed_kph=3.6, object_offset_x_m=5,
        object_offset_y_m=0, contact_half_length_m=1, contact_half_width_m=1,
        model_valid_until_s=10, ego_longitudinal_direction='FORWARD',
        object_longitudinal_direction='FORWARD', object_extent='FINITE_CONTACT_ENVELOPE', object_position='FRONT')


def test_following_response_boundary_includes_relative_closing_during_braking():
    # Closing speed 1 m/s, net b=2. Minimum separation occurs .5 s into
    # braking; the remaining closing distance is 1²/(2*2)=.25 m.
    result = moving_braking_avoidability_boundary(following(), 2)
    assert result['time_s'] == pytest.approx((5-1-.25)/1)
    assert result['minimum_at_s'] == pytest.approx(4.25)
    assert result['response_delay_interval_proven']
    assert result['boundary_bracket_s'][1]-result['boundary_bracket_s'][0] < 1e-12
    reverse = {**following(), 'ego_longitudinal_direction':'REVERSE', 'object_longitudinal_direction':'REVERSE',
               'object_offset_x_m':-5, 'object_position':'REAR'}
    assert moving_braking_avoidability_boundary(reverse, 2)['time_s'] == result['time_s']


def test_crossing_response_checks_held_stop_until_target_has_passed():
    facts={**following(),'ego_speed_kph':14.4,'object_speed_kph':7.2,'object_offset_x_m':6,
        'object_offset_y_m':-4,'object_longitudinal_direction':'LEFT','object_position':'RIGHT'}
    result=moving_braking_avoidability_boundary(facts,4)
    assert result['lateral_overlap_interval_s'] == [1.5,2.5]
    # Stop before x=5 m: 4*t + 4²/(2*4) < 5, i.e. t < .75 s.
    assert result['time_s'] == pytest.approx(.75)
    assert result['stop_time_s'] == pytest.approx(1.75)
    assert result['evaluated_until_s'] == 10
    later={**facts,'object_offset_y_m':-12}  # Original ego passes first; no contact to mitigate.
    assert moving_braking_avoidability_boundary(later,4)['status'] == 'NO_UNMITIGATED_CONTACT_BOUNDARY'


@pytest.mark.parametrize('change', [
    {'object_offset_x_m':-5,'object_speed_kph':10.8,'object_position':'REAR'},
    {'object_offset_x_m':10,'object_speed_kph':7.2,'object_longitudinal_direction':'REVERSE'},
])
def test_stopping_does_not_make_rear_approach_or_oncoming_target_disappear(change):
    result=moving_braking_avoidability_boundary({**following(),**change},2)
    assert result['status']=='NO_IMMEDIATE_SAFE_BRAKING_WINDOW'
    assert 'time_s' not in result
    assert result['immediate_response']['maximum_separation_m'] >= -1


def test_weaker_response_still_rejects_an_immediate_contact():
    facts={**following(),'ego_speed_kph':36,'longitudinal_acceleration_mps2':-3}
    assert moving_braking_avoidability_boundary(facts,2)['status']=='NO_IMMEDIATE_SAFE_BRAKING_WINDOW'


def test_release_excessive_deceleration_has_a_bounded_rear_contact_window():
    facts={**following(),'ego_speed_kph':36,'longitudinal_acceleration_mps2':-3,
        'object_offset_x_m':-5,'object_position':'REAR','object_speed_kph':18,
        'model_valid_until_s':5.5,'post_stop_motion':'HOLD_POSITION'}
    original=deepcopy(facts)
    result=moving_braking_avoidability_boundary(facts,2)
    # Stop position = 25-5*delay+.75*delay²; rear envelope reaches x=23.5
    # at horizon 5.5. Solve the smaller root; no sampled fit or default grade.
    expected=(5-(25-4*.75*1.5)**.5)/(2*.75)
    assert result['status']=='CONDITIONAL_BOUNDARY'
    assert result['time_s']==pytest.approx(expected)
    assert result['response_clearance_side']=='TARGET_BEHIND'
    assert result['stop_time_s']<5.5 and result['immediate_response']['clearance_m']>0
    assert facts==original
    reverse={**facts,'ego_longitudinal_direction':'REVERSE','object_longitudinal_direction':'REVERSE',
        'object_offset_x_m':5,'object_position':'FRONT'}
    assert moving_braking_avoidability_boundary(reverse,2)['time_s']==pytest.approx(expected)
    # Longer target continuation eventually strikes the stopped ego: no claim
    # outside the declared horizon, and no safe-state by dropping the target.
    assert moving_braking_avoidability_boundary({**facts,'model_valid_until_s':8},2)['status']=='NO_IMMEDIATE_SAFE_BRAKING_WINDOW'


def test_late_crossing_after_natural_stop_does_not_make_ego_reverse():
    # Fault braking stops ego at x=2 m in 2 s; the target arrives at 3 s.
    # Any stronger braking started before 2 s stops short of the envelope.
    facts={**following(), 'longitudinal_acceleration_mps2':-1,
        'object_speed_kph':3.6,'object_longitudinal_direction':'LEFT',
        'object_offset_x_m':3,'object_offset_y_m':-4,'object_position':'RIGHT',
        'post_stop_motion':'HOLD_POSITION'}
    result=moving_braking_avoidability_boundary(facts,2)
    assert result['original_contact_time_s']==3
    assert result['time_s']==pytest.approx(2,abs=1e-6)
    assert result['stop_time_s']<=2


def release_plan():
    raw = moving_plan(initial=7.2, target_speed=7.2, target_direction='FORWARD',
        x=-5, acceleration=-1, position='REAR', collision='REAR_END', horizon=5)
    h = raw['hypotheses'][0]
    h.update(post_stop_motion='HOLD_POSITION', failed_controls=['automatic_braking_command'],
        remaining_controls=['driver_braking', 'braking', 'drive'])
    h['safe_state_proposal'] = dict(description='解除异常制动并保持响应开始时的受控速度',
        required_controls=['driver_braking', 'braking', 'drive'], basis_refs=['MF.description'],
        conditions=['Independent driver override releases erroneous command; healthy actuator and drive balance remain.'],
        transition=dict(model='RELEASE_DECELERATION_TO_CONSTANT_SPEED',
            rationale='Finite conditional override; no jump to pre-fault speed.', basis_refs=['MF.description']))
    h['hazard_timing'] = {**deepcopy(timed_plan()['hypotheses'][0]['hazard_timing']),
        'hazard_criterion':'LOSS_OF_RESPONSE_AVOIDABILITY'}
    return raw


def test_fault_release_uses_onset_speed_and_checks_the_finite_horizon(method):
    from hara_agent.services.reporting.engineering_text_mapper import EngineeringReportTextMapper
    from test_geometry_completion import Client, run
    from hara_agent.services.semantic.physical_scenario_agent import PhysicalScenarioAgent
    raw = release_plan(); original = deepcopy(raw)
    client = Client([raw])
    state, generated = run(method, PhysicalScenarioAgent(client), candidates={'MF':[make_candidate(7.2)]})
    child = generated['MF'][0]
    assert len(client.requests) == 1  # release must not request a fictitious held stop
    view = {**child.facts, '_fact_provenance':child.fact_provenance}
    risk_before = score(method, child)
    timing = RiskCalculationInputService().ftti('MF',child.scenario_id,'',view,method=method)
    # At horizon 5: rear separation = -5 + 5*d - d²/2. Contact at -1 m.
    expected = 5-17**.5
    assert timing.hazard_time_limit_s == pytest.approx(expected)
    assert timing.safe_state_transition_time_s is None  # no fabricated stopping time
    assert timing.timing_trace['brake_onset_speed_mps'] == pytest.approx(2-expected)
    assert timing.timing_trace['response_motion'] == 'CONTINUE_AT_ONSET_SPEED'
    assert timing.timing_trace['engineering_approval'] == 'NOT_GRANTED'
    assert timing.status.value == 'CONDITIONAL'
    text = EngineeringReportTextMapper.ftti_rationale([timing.to_dict()])
    assert '保持响应开始时的速度' in text and '停车保持三段' not in text
    assert risk_before == score(method, child) and raw == original
    outside = {**view, 'driver_in_vehicle':False}
    assert RiskCalculationInputService().ftti('MF',child.scenario_id,'',outside,method=method).hazard_time_limit_s is None
    assert moving_braking_avoidability_boundary(view,0)['status']=='RESPONSE_INPUTS_UNRESOLVED'
    assert moving_braking_avoidability_boundary({**view,'longitudinal_acceleration_mps2':1},0,
        release_deceleration=True)['status'] in {'FAULT_RELEASE_NOT_APPLICABLE','NO_UNMITIGATED_CONTACT_BOUNDARY'}
    failed = deepcopy(raw); failed['hypotheses'][0]['failed_controls'].append('drive')
    from hara_agent.contracts.physical_scenario_plan import PhysicalScenarioPlan
    with pytest.raises(ValueError, match='failed control'):
        PhysicalScenarioPlan.from_dict(failed, evidence_ids={'MF.description'})


def proposal():
    raw=moving_plan(initial=7.2,target_speed=3.6,target_direction='FORWARD',x=5)
    h=raw['hypotheses'][0]; timing=timed_plan()['hypotheses'][0]
    h['safe_state_proposal']=deepcopy(timing['safe_state_proposal'])
    h['safe_state_proposal']['transition'].update(net_deceleration_mps2=2,post_stop_motion='HOLD_POSITION')
    h['hazard_timing']=deepcopy(timing['hazard_timing'])
    return raw


def test_moving_proposal_reaches_numeric_timing_without_changing_unmitigated_rating(method):
    from hara_agent.services.reporting.engineering_text_mapper import EngineeringReportTextMapper
    child=instantiate(method,proposal(),make_candidate(7.2))[0]
    original=deepcopy(child); risk=score(method,child)
    view={**child.facts,'_fact_provenance':child.fact_provenance}
    timing=RiskCalculationInputService().ftti('MF',child.scenario_id,'',view,method=method)
    assert timing.hazard_time_limit_s == pytest.approx(3.75)
    assert timing.status.value=='CONDITIONAL' and timing.timing_trace['engineering_approval']=='NOT_GRANTED'
    assert '移动目标' in EngineeringReportTextMapper.ftti_rationale([timing.to_dict()])
    assert child==original and score(method,child)==risk
    outside=RiskCalculationInputService().ftti('MF',child.scenario_id,'',{**view,'driver_in_vehicle':False},method=method)
    assert outside.hazard_time_limit_s is None
    h=view['physical_analysis']['hypothesis']
    h['safe_state_proposal']['transition'].pop('post_stop_motion')
    assert '不能在停车瞬间截断' in RiskCalculationInputService().ftti('MF',child.scenario_id,'',view,method=method).reason


def test_unsafe_response_is_reported_even_without_an_ftti_clock(method):
    raw = proposal()
    h = raw['hypotheses'][0]
    h.pop('hazard_timing')
    h['geometry']['object_longitudinal_direction'] = 'REVERSE'
    child = instantiate(method, raw, make_candidate(7.2))[0]
    timing = RiskCalculationInputService().ftti('MF', child.scenario_id, '',
        {**child.facts, '_fact_provenance': child.fact_provenance}, method=method)
    assert timing.hazard_time_limit_s is None
    assert timing.timing_trace['status'] == 'NO_IMMEDIATE_SAFE_BRAKING_WINDOW'
    assert '不能把停车直接当安全' in timing.reason


def test_deferred_safe_state_dependency_gets_one_real_workflow_correction_and_resumes(method,tmp_path):
    from hara_agent.infrastructure.llm import LLMResponse
    from hara_agent.services.semantic.physical_scenario_agent import PhysicalScenarioAgent
    from hara_agent.workflow import HARAState
    from hara_agent.workflow.physical_analysis import construct_physical_scenarios
    from hara_agent.services.analysis import MethodSafetyGoalService
    from hara_agent.workflow.nodes.safety_goals import aggregate_safety_goals
    class Client:
        calls=0
        def complete_json(self,request):
            self.calls+=1
            raw=timed_plan()
            if self.calls==1:
                raw['hypotheses'][0]['safe_state_proposal']['required_controls'].append('parking_hold')
            else:
                assert 'SAFE_STATE_REQUIRES_UNRESOLVED_CONTROL' in request.user_prompt
                assert 'parking_hold' in request.user_prompt
            return LLMResponse(raw,'OFFLINE')
    state=HARAState('safe-state-correction',functions=[{'function_id':'F','name':'驱动'}],malfunctions=[asdict(mf())])
    client=Client();agent=PhysicalScenarioAgent(client,response_checkpoint_dir=tmp_path)
    def run(s):return construct_physical_scenarios(s,{'MF':[make_candidate(3.6)]},[mf()],method=method,agent=agent,project_facts=None)
    generated=run(state)['MF']
    assert client.calls==2 and len(generated)==1
    assert generated[0].facts['physical_analysis']['hypothesis']['safe_state_proposal']['required_controls']==['driver_braking']
    entry=next(iter(state.item_definition['physical_analysis_plans'].values()))
    assert entry['initial_plan']['hypotheses'][0].get('safe_state_unresolved_reason')
    assert 'parking_hold' in str(entry['correction_diagnostics'])
    run(HARAState.read_committed(json.loads(json.dumps(state.to_dict()))))
    assert client.calls==2


def test_unreachable_safe_stop_is_returned_to_the_same_bounded_workflow(method):
    from test_geometry_completion import Client, run
    from hara_agent.services.semantic.physical_scenario_agent import PhysicalScenarioAgent
    raw = proposal()
    raw['hypotheses'][0]['geometry']['object_longitudinal_direction'] = 'REVERSE'
    corrected = deepcopy(raw)
    corrected['hypotheses'][0].pop('safe_state_proposal')
    corrected['hypotheses'][0]['safe_state_unresolved_reason'] = 'An oncoming target still reaches the held vehicle; this stop strategy is not sufficient.'
    client = Client([raw, corrected])
    state, result = run(method, PhysicalScenarioAgent(client), candidates={'MF': [make_candidate(7.2)]})
    assert len(client.requests) == 2
    assert 'SAFE_STOP_RESPONSE_UNREACHABLE' in client.requests[1].user_prompt
    child = result['MF'][0]
    assert child.facts['motion_model_outcome']['status'] == 'CONTACT'
    assert child.facts['object_longitudinal_direction'] == 'REVERSE'
    assert 'safe_state_proposal' not in child.facts['physical_analysis']['hypothesis']
    before = len(client.requests)
    run(method, PhysicalScenarioAgent(client), state=state, candidates={'MF': [make_candidate(7.2)]})
    assert len(client.requests) == before


def test_single_motion_envelope_preserves_fields_without_an_extra_provider_request():
    from test_geometry_completion import Client
    from test_autonomous_physics import raw_plan
    from hara_agent.services.semantic.physical_scenario_agent import PhysicalScenarioAgent
    raw = raw_plan()['hypotheses'][0]
    original = deepcopy(raw)
    client = Client([raw])
    plan, audit = PhysicalScenarioAgent(client).propose({'plan_key':'single-motion', 'evidence':{'MF.description':'fault'}})
    assert len(client.requests)==1 and plan.hypotheses[0]==original==raw
    assert audit['control_envelope_normalization']=='WRAPPED_SINGLE_MOTION_HYPOTHESIS'
    assert audit['responses'][0]['response']['data']==original
    invalid = {**raw, 'invented_approval':'APPROVED'}
    with pytest.raises(ValueError):
        PhysicalScenarioAgent(Client([invalid, invalid])).propose({'plan_key':'invalid-single', 'evidence':{'MF.description':'fault'}})


def test_no_contact_does_not_spend_a_correction_proving_an_unneeded_escape(method):
    from test_geometry_completion import Client, run
    from hara_agent.services.semantic.physical_scenario_agent import PhysicalScenarioAgent
    raw=moving_plan(target_direction='LEFT',x=8,y=-20,position='RIGHT',horizon=2)
    raw['hypotheses'][0]['intervention_assumptions']={
        'other_road_user_avoidance_possible':dict(value=True,rationale='conditional path',
            basis_refs=['MF.description'],conditions=['unobstructed corridor'],
            avoidance=dict(recognition_delay_s=.1,escape_direction='RIGHT',escape_speed_mps=1,available_clearance_m=5))}
    parent=make_candidate(3.6)
    parent.facts.pop('other_road_user_avoidance_possible')
    parent.fact_provenance.pop('other_road_user_avoidance_possible',None)
    client=Client([raw]);state,result=run(method,PhysicalScenarioAgent(client),candidates={'MF':[parent]})
    child=result['MF'][0]
    assert len(client.requests)==1
    assert child.facts['physical_analysis']['avoidance_validation']['status']=='NO_CONTACT_TO_AVOID'
    assert child.facts['motion_model_outcome']['status']=='NO_CONTACT_WITHIN_MODEL'
    assert not child.facts.get('collision_branches')
