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
    assert '车外' in view.safety_goals[0].safe_state
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
    assert audit['responses'][0]['response']['data'] == raw
    assert plan.hypotheses[0]['parameters'] == raw['hypotheses'][0]['parameters']
    assert not plan.hypotheses[0].get('driver_operating_atoms')
    agent.propose(context)
    assert len(client.requests) == 2


@pytest.mark.parametrize('explicit_model_gap', [False, True])
def test_safe_stop_response_is_completed_once_in_production_workflow(method, tmp_path, explicit_model_gap):
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
                raw['hypotheses'][0]['safe_state_proposal'].pop('transition')
                if explicit_model_gap:
                    raw['hypotheses'][0]['timing_unresolved_reason'] = 'This representative requires a time-varying brake response outside the constant-net-response model.'
            else:
                assert 'SAFE_STOP_RESPONSE_NOT_MODELED' in request.user_prompt
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
