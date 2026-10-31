"""Shared production physics, scope and report boundaries; no provider or grade injection."""
from copy import deepcopy
from dataclasses import asdict
import pytest
from test_autonomous_physics import method, raw_plan, instantiate, score
from hara_agent.services.analysis.scenario_physics import moving_target_avoidance, rectilinear_moving_contact, braking_avoidability_boundary
from hara_agent.services.analysis.risk_calculation_input_service import RiskCalculationInputService
from hara_agent.services.reporting.engineering_text_mapper import EngineeringReportTextMapper


def timed_plan():
    raw = raw_plan(initial=3.6, acceleration=1)
    h = raw['hypotheses'][0]
    h['remaining_controls'] = ['driver_braking']
    h['safe_state_proposal'] = dict(description='在障碍物前停止并保持静止', required_controls=['driver_braking'],
        basis_refs=['MF.description'], conditions=['独立人工制动仍可用；净减速假设包括故障驱动力'],
        driver_positions=['in_driver_seat'], transition=dict(model='CONSTANT_NET_DECELERATION',
            net_deceleration_mps2=3, rationale='conditional net response', basis_refs=['MF.description']))
    h['hazard_timing'] = dict(time_origin='FAULT_ONSET', fault_to_model_start_s=0,
        hazard_criterion='LOSS_OF_BRAKING_AVOIDABILITY', rationale='initial state at fault onset',basis_refs=['MF.description'])
    return raw


def test_timing_report_does_not_label_every_missing_value_as_a_missing_model():
    timings = [
        {'hazard_time_limit_s': .4, 'reason': '条件性危险时限'},
        {'hazard_time_limit_s': None, 'reason': '本成员没有适用的剩余制动安全状态路径；不能借用其他驾驶员分支。'},
    ]
    original = deepcopy(timings)
    text = EngineeringReportTextMapper.ftti_rationale(timings)
    assert '1个成员尚无数值时间边界' in text and '最小值不代表全组结论' in text
    assert '缺少适用时间模型' not in text
    assert timings == original


@pytest.mark.parametrize('reason_field', ['timing_unresolved_reason', 'safe_state_unresolved_reason'])
def test_scoped_timing_gap_reaches_report_without_inventing_a_deadline(method, reason_field):
    from test_autonomous_physics import make_candidate
    raw = timed_plan()
    h = raw['hypotheses'][0]
    h.pop('safe_state_proposal')
    h[reason_field] = '同一后车轨迹下停车仍被接触，尚未建立可解除异常减速的独立路径。'
    child = instantiate(method, raw, make_candidate(3.6))[0]
    before = deepcopy(child)
    timing = RiskCalculationInputService().ftti('MF', child.scenario_id, '',
        {**child.facts, '_fact_provenance': child.fact_provenance}, method=method)
    assert timing.hazard_time_limit_s is None
    assert h[reason_field] in timing.reason
    assert h[reason_field] in EngineeringReportTextMapper.ftti_rationale([timing.to_dict()])
    assert child == before


def test_zero_margin_is_not_presented_as_a_usable_zero_second_ftti():
    result=braking_avoidability_boundary(7/3.6,1,.3,2)
    assert result['time_s']==0 and result['stopping_distance_m']==pytest.approx(.94521604938)
    lost={'hazard_time_limit_s':0,'timing_trace':result}
    before=deepcopy(lost)
    assert EngineeringReportTextMapper.ftti_value([lost])=='初始已无制动避碰余量'
    assert '必须在该边界前完成' not in EngineeringReportTextMapper.ftti_rationale([lost])
    assert '不是可供设计采用' in EngineeringReportTextMapper.ftti_rationale([lost])
    mixed=[lost,{'hazard_time_limit_s':2}, {'hazard_time_limit_s':None}]
    text=EngineeringReportTextMapper.ftti_value(mixed)
    assert text.startswith('2 s；含初始无制动避碰余量') and '1个成员待确定' in text
    assert '0 s～' not in text and lost==before


def test_group_ftti_displays_minimum_valid_member_without_changing_member_values():
    timings = [{'hazard_time_limit_s':3.1}, {'hazard_time_limit_s':.154286},
               {'hazard_time_limit_s':.158}, {'hazard_time_limit_s':None}]
    original = deepcopy(timings)
    assert EngineeringReportTextMapper.ftti_value(timings) == '0.15 s；1个成员待确定'
    assert EngineeringReportTextMapper.ftti_value(list(reversed(timings))) == '0.15 s；1个成员待确定'
    assert '最小值' in EngineeringReportTextMapper.ftti_rationale(timings)
    assert EngineeringReportTextMapper.ftti_value(timings[:3]) == '0.15 s'
    assert timings == original


def test_constant_representative_reaches_existing_timing_calculator_without_defaulting_unknown_model(method):
    from test_autonomous_physics import make_candidate
    child=instantiate(method,timed_plan(),make_candidate(3.6))[0]
    facts={**deepcopy(child.facts),'_fact_provenance':deepcopy(child.fact_provenance)}
    facts['fault_motion_model']='CONSTANT_SPEED_REPRESENTATIVE'
    facts.pop('longitudinal_acceleration_mps2');facts.pop('travel_to_impact_m')
    before=deepcopy(facts)
    timing=RiskCalculationInputService().ftti('MF',child.scenario_id,'',facts,method=method)
    assert timing.hazard_time_limit_s==pytest.approx(2-1/6)
    assert timing.timing_trace['fault_acceleration_mps2']==0
    assert facts==before
    for change in ({'fault_motion_model':'CONSTANT_ACCELERATION_TO_IMPACT_DISTANCE'},
                   {'longitudinal_acceleration_mps2':1},{'object_position':'REAR'}, {'object_speed_kph':1}):
        assert RiskCalculationInputService().ftti('MF',child.scenario_id,'',{**facts,**change},method=method).hazard_time_limit_s is None


@pytest.mark.parametrize('code,explanation', [
    ('RESPONSE_NOT_STRONGER_THAN_FAULT_DECELERATION', '弱于故障轨迹'),
    ('NO_UNMITIGATED_CONTACT_BOUNDARY', '未干预轨迹未形成接触'),
    ('NO_CONTACT_UNDER_THIS_MOTION', '该运动假设下不会发生接触'),
])
def test_timing_report_explains_physical_boundary_diagnostics_without_internal_codes(code, explanation):
    timings = [{'contact_time_s': .25, 'reason': '未形成该危险判据的数值边界：' + code}]
    original = deepcopy(timings)
    text = EngineeringReportTextMapper.ftti_rationale(timings)
    assert explanation in text and code not in text
    assert '接触时间不能作为 FTTI' in text
    assert timings == original


def test_scoped_optional_atom_exclusion_is_not_a_catalog_or_project_id_rule(method):
    from dataclasses import asdict
    from test_p6_scenario_pipeline import parent, project, malfunction, function
    from hara_agent.services.analysis import ConstrainedScenarioSynthesisService
    from hara_agent.services.analysis.physical_hypothesis_instantiation_service import PhysicalHypothesisInstantiationService
    service = PhysicalHypothesisInstantiationService(method)
    bases, _ = ConstrainedScenarioSynthesisService(method).generate_candidates(
        malfunction=asdict(malfunction()), parent=parent('another-project-world'),
        project_context=asdict(project()), function=asdict(function()))
    base = next(c for c in bases if c.facts['method_scenario_dimensions']['EGO_ACTION']['atom_id'] == 'FA005')
    before = deepcopy(base)
    arguments = dict(parent=base, malfunction=malfunction(), geometry={'road_user_type':'VEHICLE'},
        project_context=asdict(project()), function=asdict(function()), dimensions=('OBJECT','TRAFFIC_PATTERN'),
        explicit={'TRAFFIC_PATTERN':None}, current_facts={'object_speed_kph':4.2,
            'object_position':'FRONT','object_longitudinal_direction':'FORWARD'})
    unresolved, _ = service._object_binding(**arguments)
    assert unresolved['method_scenario_dimensions']['TRAFFIC_PATTERN']['applicability_status'] == 'OPTIONAL'
    explanation = 'Same-lane following; no lane entry or adjacent-lane flow in this conditional world.'
    scoped, _ = service._object_binding(**arguments, exclusions={'TRAFFIC_PATTERN':explanation})
    binding = scoped['method_scenario_dimensions']['TRAFFIC_PATTERN']
    assert binding['applicability_status'] == 'NOT_APPLICABLE'
    assert binding['applicability_nature'] == 'CONDITIONAL'
    assert binding['applicability_reason'] == explanation
    assert binding['dimension_source'] == 'PHYSICAL_HYPOTHESIS'
    assert base == before
    # An explicit required traffic relation must survive a proposed omission.
    base.facts['traffic_relation'] = 'cut_in'
    rejected, reason = service._object_binding(**arguments, exclusions={'TRAFFIC_PATTERN':explanation})
    assert not rejected and 'UNREPRESENTED_TRAFFIC_RELATION:TRAFFIC_CUT_IN' in reason
    base.facts.pop('traffic_relation')
    base.facts['method_scenario_dimensions']['TRAFFIC_PATTERN'] = {
        'atom_id':'CN_cutting_in','resolution_status':'RESOLVED','binding_authority':'EXACT_PROJECT_FACT'}
    rejected, reason = service._object_binding(**arguments, exclusions={'TRAFFIC_PATTERN':explanation})
    assert not rejected and reason == 'AUTHORITATIVE_CONDITION_CANNOT_BE_EXCLUDED:TRAFFIC_PATTERN'


def test_exclusion_cannot_contradict_selected_atom():
    from hara_agent.contracts.physical_scenario_plan import PhysicalScenarioPlan
    raw = timed_plan(); h = raw['hypotheses'][0]
    h['operating_atoms'] = {'EGO_ACTION':'custom-action','EGO_DYNAMICS':'custom-dynamics','TRAFFIC_PATTERN':'custom-traffic'}
    h['operating_atom_exclusions'] = {'TRAFFIC_PATTERN':'does not apply'}
    with pytest.raises(ValueError, match='explicitly empty selection'):
        PhysicalScenarioPlan.from_dict(raw, evidence_ids={'MF.description'})


def test_existing_non_driver_response_path_is_usable_without_a_new_scorer(method):
    from test_autonomous_physics import make_candidate
    raw = timed_plan(); h = raw['hypotheses'][0]
    h['remaining_controls'] = ['automatic_braking_command','braking']
    h['safe_state_proposal']['required_controls'] = h['remaining_controls']
    h['safe_state_proposal']['driver_positions'] = ['in_driver_seat','outside_driver_seat']
    h['safe_state_proposal']['conditions'] = ['Separate conditional command path and actuator remain after drive fault.']
    parent = make_candidate(3.6); parent.facts['driver_in_vehicle'] = False
    child = instantiate(method, raw, parent)[0]
    timing = RiskCalculationInputService().ftti('MF',child.scenario_id,'',
        {**child.facts,'_fact_provenance':child.fact_provenance},method=method)
    assert timing.hazard_time_limit_s == pytest.approx(.9364916731)
    assert timing.status.value == 'CONDITIONAL'
    unknown = {**child.facts, 'driver_in_vehicle':None, '_fact_provenance':child.fact_provenance}
    assert RiskCalculationInputService().ftti('MF',child.scenario_id,'',unknown,method=method).hazard_time_limit_s is None
    h['failed_controls'] = ['drive','braking']
    from hara_agent.contracts.physical_scenario_plan import PhysicalScenarioPlan
    with pytest.raises(ValueError, match='failed control'):
        PhysicalScenarioPlan.from_dict(raw, evidence_ids={'MF.description'})


def test_response_boundary_includes_stopping_travel_not_post_contact_time():
    b = braking_avoidability_boundary(1, 1, 2, 3)
    t = b['time_s']
    assert t == pytest.approx(.9364916731)
    assert t + t*t/2 + (1+t)**2/6 == pytest.approx(2)
    assert b['distance_residual_m'] == pytest.approx(0)
    assert braking_avoidability_boundary(2, 1e-12, 10, 4)['time_s'] == pytest.approx(4.75)
    assert braking_avoidability_boundary(4, 0, 1, 3)['status'] == 'AVOIDABILITY_ALREADY_LOST'
    assert braking_avoidability_boundary(0, 0, 1, 3)['status'] == 'NO_CONTACT_UNDER_THIS_MOTION'


def test_proposal_instantiation_risk_timing_and_projection_preserve_conditional_scope(method):
    from test_autonomous_physics import make_candidate
    child = instantiate(method, timed_plan(), make_candidate(3.6))[0]
    before = deepcopy(child)
    result = score(method, child)
    view = {**child.facts, '_fact_provenance': child.fact_provenance}
    timing = RiskCalculationInputService().ftti('MF', child.scenario_id, '', view, method=method)
    assert timing.hazard_time_limit_s == pytest.approx(.9364916731)
    assert timing.hazard_time_limit_s < child.facts['ttc_s']
    assert timing.status.value == 'CONDITIONAL'
    assert timing.timing_trace['engineering_approval'] == 'NOT_GRANTED'
    assert timing.safe_state_transition_time_s > 0
    assert child == before
    assert result['severity']['severity_score']
    assert EngineeringReportTextMapper.ftti_value([timing.to_dict()]).endswith(' s')
    assert '条件性' in EngineeringReportTextMapper.ftti_rationale([timing.to_dict()])
    outside = {**view, 'driver_in_vehicle': False}
    blocked = RiskCalculationInputService().ftti('MF', child.scenario_id, '', outside, method=method)
    assert blocked.hazard_time_limit_s is None and '驾驶员' in blocked.reason
    multiple = RiskCalculationInputService().ftti('MF', child.scenario_id, '', view,
        [{'branch_id':'a'}, {'branch_id':'b'}], method=method)
    assert multiple.hazard_time_limit_s is None


def crossing():
    return dict(fault_motion_model='RECTILINEAR_MOVING_TARGET_CONTACT', ego_speed_kph=0,
        longitudinal_acceleration_mps2=.34, model_valid_until_s=3, object_speed_kph=3.6,
        object_offset_x_m=.8, object_offset_y_m=-1.2, contact_half_length_m=2.55,
        contact_half_width_m=1.15, object_extent='FINITE_CONTACT_ENVELOPE',
        ego_longitudinal_direction='FORWARD', object_longitudinal_direction='LEFT', object_position='RIGHT')


def test_hara_278_same_geometry_rejects_unclosed_escape_and_preserves_counterfactual():
    facts = crossing()
    assert rectilinear_moving_contact(facts)['time_s'] == pytest.approx(.05)
    assert rectilinear_moving_contact({**facts,'longitudinal_acceleration_mps2':0})['time_s'] == pytest.approx(.05)
    assert moving_target_avoidance(facts, {'value':True})['status'] == 'AVOIDANCE_GEOMETRY_UNRESOLVED'
    proposal = {'avoidance':dict(recognition_delay_s=.1,escape_direction='RIGHT',escape_speed_mps=1,available_clearance_m=4)}
    assert moving_target_avoidance(facts, proposal)['status'] == 'CONTACT_BEFORE_ESCAPE'
    proposal['avoidance']['recognition_delay_s'] = .01
    assert moving_target_avoidance(facts, proposal)['status'] == 'CONDITIONAL_ESCAPE_VALIDATED'
    proposal['avoidance']['available_clearance_m'] = .01
    assert moving_target_avoidance(facts, proposal)['status'] == 'ESCAPE_CORRIDOR_TOO_SHORT'


@pytest.mark.parametrize('model', ['CONSTANT_ACCELERATION_TO_IMPACT_DISTANCE', 'GRADE_ROLLAWAY_TO_IMPACT_DISTANCE'])
def test_static_contact_model_does_not_credit_unmodelled_target_escape(method, model):
    from test_autonomous_physics import make_candidate
    raw = raw_plan(model)
    h = raw['hypotheses'][0]
    h['remaining_controls'] = ['driver_braking']
    h['intervention_assumptions'] = {'other_road_user_avoidance_possible':{
        'value':True, 'rationale':'hypothetical target side step', 'basis_refs':['MF.description'],
        'conditions':['target can see ego, side corridor assumed open'],
        'avoidance':{'recognition_delay_s':.5,'escape_direction':'LEFT','escape_speed_mps':1.2,'available_clearance_m':1}}}
    before = deepcopy(raw)
    parent = make_candidate(0)
    parent.facts.pop('other_road_user_avoidance_possible')
    parent.fact_provenance.pop('other_road_user_avoidance_possible', None)
    child = instantiate(method, raw, parent)[0]
    assert child.facts['motion_model_outcome']['status'] == 'CONTACT'
    assert child.facts['physical_analysis']['avoidance_validation']['status'] == 'UNSUPPORTED_AVOIDANCE_TRAJECTORY'
    assert 'other_road_user_avoidance_possible' not in child.facts
    result = score(method, child)
    assert result['controllability']['controllability_score'] != 'C1'
    assert result['severity']['severity_score'] and raw == before


def test_new_hazard_origin_cannot_reuse_old_or_unsupported_response(method):
    from test_autonomous_physics import make_candidate
    child = instantiate(method, timed_plan(), make_candidate(3.6))[0]
    view={**child.facts,'_fact_provenance':child.fact_provenance}
    service=RiskCalculationInputService()
    assert service.ftti('OTHER',child.scenario_id,'',view,method=method).hazard_time_limit_s is None
    assert service.ftti('MF',child.scenario_id,'',{**view,'object_speed_kph':1},method=method).hazard_time_limit_s is None
    original=deepcopy(view)
    view['physical_analysis']['hypothesis'].pop('hazard_timing')
    assert service.ftti('MF',child.scenario_id,'',view,method=method).hazard_time_limit_s is None
    assert original['physical_analysis']['hypothesis'].get('hazard_timing')


def test_absent_timing_hypothesis_is_distinct_from_wrong_world_scope(method):
    result = RiskCalculationInputService().ftti('MF', 'SCN', '', {'ttc_s': .7}, method=method)
    assert result.hazard_time_limit_s is None and result.contact_time_s == .7
    assert '尚未形成' in result.reason and '不能继承其他成员' not in result.reason


def test_control_only_safe_state_reaches_sg_and_reports_uncovered_driver_members(method):
    from test_report_review_corrections import mixed_pending_group
    from hara_agent.services.analysis import MethodSafetyGoalService
    from hara_agent.workflow.nodes.safety_goals import aggregate_safety_goals
    from hara_agent.services.reporting import HARAReportProjectionService, load_report_schema
    from hara_agent.contracts.physical_scenario_plan import PhysicalScenarioPlan
    state = mixed_pending_group()
    h = timed_plan()['hypotheses'][0]
    assessment = {k: deepcopy(h[k]) for k in ('rationale','basis_refs','conditions','uncovered_conditions',
        'failed_controls','remaining_controls','safe_state_proposal','hazard_timing')}
    parsed = PhysicalScenarioPlan.from_dict({'hypotheses':[], 'uncovered_conditions':[],
        'control_assessment':assessment}, evidence_ids={'MF.description'})
    assert parsed.control_assessment['safe_state_proposal']
    for i, child in enumerate(state.scenarios):
        child.facts['driver_in_vehicle'] = i == 0
        child.facts['control_analysis'] = {'plan_key':'control-only', 'hypothesis':deepcopy(assessment)}
    raw_risks = deepcopy(state.risk_results)
    aggregate_safety_goals(state, MethodSafetyGoalService(method))
    catalog = state.item_definition['safety_goal_catalog']
    goal = state.safety_goals[0]
    assert len(catalog[goal.sg_id]['safe_state_candidates']) == 1
    assert len(catalog[goal.sg_id]['safe_state_gaps']) == 13
    view = HARAReportProjectionService(load_report_schema()).project(state, method)
    assert '适用1个成员' in view.safety_goals[0].safe_state
    assert '13个成员' in view.safety_goals[0].safe_state
    assert '驾驶员不在驾驶位，踏板方案不适用' in view.safety_goals[0].safe_state
    assert any('没有在驾驶位操作踏板的适用条件' in d.analysis_basis for d in view.scenario_details)
    assert all(getattr(a,k)==getattr(b,k) for a,b in zip(raw_risks,state.risk_results)
        for k in ('severity','exposure','controllability','asil'))


def test_new_proposal_cannot_silently_omit_operating_binding():
    from hara_agent.services.semantic.physical_scenario_agent import PhysicalScenarioAgent
    from hara_agent.infrastructure.llm import LLMResponse
    class Client:
        calls=0
        def complete_json(self, request):
            self.calls+=1
            return LLMResponse(raw_plan(), 'synthetic-legacy-shape')
    client=Client()
    with pytest.raises(ValueError, match='operating_atoms'):
        PhysicalScenarioAgent(client).propose({'plan_key':'test-bounded-repair','evidence':{'MF.description':'fault'},
            'operating_atom_options':{'EGO_ACTION':[{'atom_id':'FV010'}],'EGO_DYNAMICS':[{'atom_id':'FA005'}],'TRAFFIC_PATTERN':[]}})
    assert client.calls == 2  # Initial + bounded schema repair, no retry loop.


def test_driver_qualified_holding_reuses_motion_but_binds_each_actor(method):
    from dataclasses import replace
    from test_p6_scenario_pipeline import parent, project, malfunction, function
    from hara_agent.services.analysis import ConstrainedScenarioSynthesisService
    from hara_agent.services.analysis.physical_hypothesis_instantiation_service import PhysicalHypothesisInstantiationService
    from hara_agent.contracts.physical_scenario_plan import PhysicalScenarioPlan
    fault = replace(malfunction(), description='驻车保持失效导致坡道溜车')
    children, _ = ConstrainedScenarioSynthesisService(method).generate_candidates(
        malfunction=asdict(fault), parent=parent(), project_context=asdict(project()), function=asdict(function()))
    base = next(c for c in children if c.facts['method_scenario_dimensions']['EGO_DYNAMICS']['atom_id'] == 'Audi_state_standing')
    service = PhysicalHypothesisInstantiationService(method)
    base.facts['driver_in_vehicle'] = True
    options = service.operating_atom_options(base, fault, asdict(project()), asdict(function()))
    assert {'PH002', 'PH003'} <= {a['atom_id'] for a in options['EGO_ACTION']}
    raw = raw_plan(initial=0)
    h = raw['hypotheses'][0]
    h['condition_changes'] = dict(initial_state=False, target=True, rationale='Separate stationary object on the travel path')
    h['operating_atoms'] = dict(EGO_ACTION='PH002', EGO_DYNAMICS='Audi_state_standing', TRAFFIC_PATTERN=None)
    h['driver_operating_atoms'] = {'outside_driver_seat': {'EGO_ACTION': 'PH003'}}
    plan = PhysicalScenarioPlan.from_dict(raw, evidence_ids={'MF.description'})
    produced = []
    for inside in (True, False):
        base.facts['driver_in_vehicle'] = inside
        generated, errors = service.instantiate(replace(base, scenario_id=base.scenario_id + ('-IN' if inside else '-OUT')), fault, plan, plan_key='same-motion-plan', response_model='OFFLINE',
            project_context=asdict(project()), function=asdict(function()))
        assert not errors
        assert len(generated) == 1
        produced.extend(generated)
    assert [c.facts['method_scenario_dimensions']['EGO_ACTION']['atom_id'] for c in produced] == ['PH002', 'PH003']
    assert produced[0].facts['ttc_s'] == produced[1].facts['ttc_s']
    assert produced[0].scenario_id != produced[1].scenario_id
    assert all(c.status.value == 'PENDING' for c in produced)
    h.pop('driver_operating_atoms')
    bad = PhysicalScenarioPlan.from_dict(raw, evidence_ids={'MF.description'})
    rejected, errors = service.instantiate(base, fault, bad, plan_key='wrong-actor', response_model='OFFLINE',
        project_context=asdict(project()), function=asdict(function()))
    assert not rejected and 'ATOM_DRIVER_CONDITION_CONFLICT' in str(errors)


def test_driver_override_cannot_break_shared_compound_atom(tmp_path):
    from test_geometry_completion import Client
    from hara_agent.services.semantic.physical_scenario_agent import PhysicalScenarioAgent
    raw = raw_plan(initial=5)
    h = raw['hypotheses'][0]
    h['operating_atoms'] = {'EGO_ACTION':'custom-moving', 'EGO_DYNAMICS':'custom-moving', 'TRAFFIC_PATTERN':None}
    h['driver_operating_atoms'] = {'outside_driver_seat': {'EGO_ACTION':'custom-hold'}}
    corrected = deepcopy(raw)
    corrected['hypotheses'][0].pop('driver_operating_atoms')
    context = {'plan_key':'compound-shared-driver', 'evidence':{'MF.description':'drive fault'},
        'operating_atom_options':{
            'EGO_ACTION':[{'atom_id':'custom-moving','fills':['EGO_ACTION','EGO_DYNAMICS']},
                          {'atom_id':'custom-hold','fills':['EGO_ACTION']}],
            'EGO_DYNAMICS':[{'atom_id':'custom-moving','fills':['EGO_ACTION','EGO_DYNAMICS']}],
            'TRAFFIC_PATTERN':[]}}
    client = Client([raw, corrected])
    agent = PhysicalScenarioAgent(client, response_checkpoint_dir=tmp_path)
    plan, audit = agent.propose(context)
    assert len(client.requests) == 2 and audit['schema_repairs'] == 1
    assert 'COMPOUND_ATOM_CONFLICT' in str(audit['validation_errors'])
    assert 'required assignments' in str(audit['validation_errors'])
    assert '"EGO_ACTION": "custom-moving", "EGO_DYNAMICS": "custom-moving"' in str(audit['validation_errors'])
    assert audit['responses'][0]['response']['data'] == raw
    assert plan.hypotheses[0]['parameters'] == raw['hypotheses'][0]['parameters']
    assert not plan.hypotheses[0].get('driver_operating_atoms')
    agent.propose(context)
    assert len(client.requests) == 2


def test_safe_state_field_location_is_explicit_without_moving_raw_values():
    from hara_agent.contracts.physical_scenario_plan import PhysicalScenarioPlan
    raw = timed_plan()
    raw['hypotheses'][0]['safe_state_proposal']['post_stop_motion'] = 'HOLD_POSITION'
    before = deepcopy(raw)
    with pytest.raises(ValueError, match='safe_state_proposal .*post_stop_motion belongs inside transition'):
        PhysicalScenarioPlan.from_dict(raw, evidence_ids={'MF.description'})
    assert raw == before


@pytest.mark.parametrize('local_time', ['inherit', 'explicit', 'unresolved', 'different_model'])
def test_driver_specific_response_preserves_same_world_fault_clock(method, local_time):
    from hara_agent.contracts.physical_scenario_plan import controls_for_driver
    from test_autonomous_physics import make_candidate
    raw = timed_plan(); h = raw['hypotheses'][0]
    h['safe_state_proposal'].pop('driver_positions')
    h['hazard_timing']['fault_to_model_start_s'] = .25
    local = {k: deepcopy(h[k]) for k in ('rationale', 'basis_refs', 'conditions',
        'uncovered_conditions', 'failed_controls', 'remaining_controls', 'safe_state_proposal')}
    local['remaining_controls'] = ['automatic_braking_command', 'braking']
    local['safe_state_proposal']['required_controls'] = ['automatic_braking_command', 'braking']
    if local_time == 'explicit':
        local['hazard_timing'] = {**h['hazard_timing'], 'fault_to_model_start_s': .5}
    elif local_time == 'unresolved':
        local['timing_unresolved_reason'] = 'The local response uses a different unbound assessment phase.'
    elif local_time == 'different_model':
        local['safe_state_proposal']['transition']['model'] = 'RELEASE_FAULT_DECELERATION'
    h['driver_control_assessments'] = {'outside_driver_seat': local}
    original = deepcopy(raw)
    effective = controls_for_driver(h, 'outside_driver_seat')
    assert 'driver_braking' not in effective['remaining_controls']
    assert raw == original
    if local_time in {'unresolved', 'different_model'}:
        assert 'hazard_timing' not in effective
        return
    parent = make_candidate(3.6)
    parent.facts['driver_in_vehicle'] = False
    child = instantiate(method, raw, parent)[0]
    timing = RiskCalculationInputService().ftti('MF', child.scenario_id, '',
        {**child.facts, '_fact_provenance':child.fact_provenance}, method=method)
    assert timing.hazard_time_limit_s == pytest.approx(15 ** .5 / 2 - 1 + (.5 if local_time == 'explicit' else .25))
    assert timing.timing_trace['required_controls'] == ['automatic_braking_command', 'braking']


@pytest.mark.parametrize('mutation,hint', [
    ('nested', 'not in a nested control_assessment'),
    ('missing', 'do not invent an available control'),
    ('both', 'Do not choose false merely because evidence is absent'),
])
def test_correction_feedback_names_the_actual_control_contract_issue(mutation, hint):
    from hara_agent.contracts.physical_scenario_plan import PhysicalScenarioPlan
    raw = timed_plan(); h = raw['hypotheses'][0]
    if mutation == 'nested':
        h['control_assessment'] = {'unresolved_controls': {'braking':'unknown'}}
    elif mutation == 'missing':
        h.pop('remaining_controls')
    else:
        h['unresolved_controls'] = {'remote_intervention_available':'unknown'}
        h['intervention_assumptions'] = {'remote_intervention_available': {
            'value':False, 'rationale':'conditional no credited path', 'basis_refs':['MF.description'], 'conditions':['offline']}}
    before = deepcopy(raw)
    with pytest.raises(ValueError, match=hint):
        PhysicalScenarioPlan.from_dict(raw, evidence_ids={'MF.description'})
    assert raw == before


@pytest.mark.parametrize('explicit_model_gap', [False, True])
@pytest.mark.parametrize('missing', ['response', 'fault_clock'])
def test_safe_stop_response_is_completed_once_in_production_workflow(method, tmp_path, explicit_model_gap, missing):
    import json
    from test_autonomous_physics import make_candidate, mf
    from hara_agent.infrastructure.llm import LLMResponse
    from hara_agent.services.semantic.physical_scenario_agent import PhysicalScenarioAgent
    from hara_agent.workflow import HARAState
    from hara_agent.workflow.physical_analysis import construct_physical_scenarios
    class Client:
        calls = 0
        def complete_json(self, request):
            self.calls += 1
            raw = timed_plan()
            if self.calls == 1:
                if missing == 'response':
                    raw['hypotheses'][0]['safe_state_proposal'].pop('transition')
                else:
                    raw['hypotheses'][0].pop('hazard_timing')
                if explicit_model_gap:
                    raw['hypotheses'][0]['timing_unresolved_reason'] = (
                        'This representative requires a time-varying brake response outside the constant-net-response model.'
                        if missing == 'response' else
                        'The initial model phase is known, but its offset from the intermittent fault onset is not related.')
            else:
                assert ('SAFE_STOP_RESPONSE_NOT_MODELED' if missing == 'response'
                        else 'SAFE_STOP_FAULT_CLOCK_NOT_BOUND') in request.user_prompt
            return LLMResponse(raw, 'OFFLINE')
    client = Client()
    state = HARAState('response-completion')
    child = make_candidate(3.6)
    agent = PhysicalScenarioAgent(client, response_checkpoint_dir=tmp_path)
    results = construct_physical_scenarios(state, {'MF': [child]}, [mf()], method=method, agent=agent, project_facts=None)
    assert len(results['MF']) == 1
    assert client.calls == (1 if explicit_model_gap else 2)
    restored = HARAState.read_committed(json.loads(json.dumps(state.to_dict())))
    construct_physical_scenarios(restored, {'MF': [child]}, [mf()], method=method, agent=agent, project_facts=None)
    assert client.calls == (1 if explicit_model_gap else 2)
    facts = results['MF'][0].facts
    assert facts['ttc_s'] == pytest.approx(1.236068)
    if not explicit_model_gap:
        assert facts['physical_analysis']['hypothesis']['safe_state_proposal']['transition']['net_deceleration_mps2'] == 3
    timing = RiskCalculationInputService().ftti('MF', results['MF'][0].scenario_id, '',
        {**facts, '_fact_provenance': results['MF'][0].fact_provenance}, method=method)
    if explicit_model_gap:
        assert timing.hazard_time_limit_s is None
    else:
        # 1*t + t²/2 + (1+t)²/(2*3) = 2 m, independent of the executor.
        assert timing.hazard_time_limit_s == pytest.approx(15 ** .5 / 2 - 1)


def test_current_safe_state_gap_does_not_reactivate_historical_proposal():
    from types import SimpleNamespace
    from hara_agent.models import ReviewStatus
    from hara_agent.services.reporting import HARAReportProjectionService
    goal = SimpleNamespace(associated_scenario_ids=['member'], safe_state='历史停车草案', status=ReviewStatus.PENDING)
    catalog = {'safe_state_candidates': [], 'safe_state_gaps': [{'scenario_id': 'member', 'reason': '即刻制动也不能在障碍物前停止'}],
               'prior_safe_state': {'description': '历史停车草案'}}
    text = HARAReportProjectionService._safe_state_display(goal, catalog)
    assert '即刻制动' in text and '历史停车草案' not in text
    assert catalog['prior_safe_state']['description'] == goal.safe_state


def test_qm_without_goal_is_not_reported_as_missing_safe_state(method):
    from test_report_review_corrections import mixed_pending_group
    from hara_agent.services.reporting import HARAReportProjectionService, load_report_schema
    from hara_agent.workflow.nodes.safety_goals import aggregate_safety_goals
    from hara_agent.services.analysis import MethodSafetyGoalService
    state = mixed_pending_group()
    from hara_agent.models import EvidenceValue, ReviewStatus
    state.risk_results = state.risk_results[:1]
    risk = state.risk_results[0]
    risk.asil = EvidenceValue('QM', ReviewStatus.FINALIZED, rule_version='synthetic-report-fixture')
    risk.safety_goal_id = ''
    aggregate_safety_goals(state, MethodSafetyGoalService(method))
    view = HARAReportProjectionService(load_report_schema()).project(state, method)
    assert all('无需分配SG' in r.safe_state and not r.sg_id for r in view.rows)
