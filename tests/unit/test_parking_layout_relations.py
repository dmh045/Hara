"""Parking axes are conditional geometry, not source permission or labels."""
from copy import deepcopy
from dataclasses import asdict
import json

import pytest

from hara_agent.contracts.physical_scenario_plan import PhysicalScenarioPlan
from hara_agent.services.analysis.physical_hypothesis_instantiation_service import PhysicalHypothesisInstantiationService
from hara_agent.services.semantic.physical_scenario_agent import PhysicalScenarioAgent
from hara_agent.services.semantic.scenario_batching import _scenario_json
from hara_agent.services.semantic.scenario_evidence import build_fact_registry, CausalEvidenceSelector
from hara_agent.workflow import HARAState
from hara_agent.workflow.physical_analysis import construct_physical_scenarios
from hara_agent.infrastructure.llm import LLMResponse
from test_autonomous_physics import make_candidate, mf
from test_bounded_interaction_coverage import moving_plan
from test_reference_interaction_physics import parameters
from test_p7_closure import method


def layout_plan(kind='PARALLEL'):
    raw = moving_plan(initial=7.2, target_speed=3.6, target_direction='FORWARD', x=5)
    h = raw['hypotheses'][0]
    angle = 60 if kind == 'ANGLED' else 90 if kind == 'PERPENDICULAR' else 0
    h['layout_relation'] = {'kind': kind, 'basis': 'ANALYSIS_ASSUMPTION',
        'axes_deg': {'bay': 0, 'aisle': angle},
        'paths': {'ego': 'BAY_AXIS', 'object': 'AISLE_AXIS'},
        'rationale': 'The bay exit meets aisle traffic; assumed axes are not measured product limits.'}
    if kind != 'PARALLEL':
        h['geometry'].update(object_position='RIGHT', object_longitudinal_direction='LEFT')
        parameters(raw, object_offset_x_m=8, object_offset_y_m=-4, object_heading_angle_deg=angle)
    return raw


def instantiate(method, raw):
    plan = PhysicalScenarioPlan.from_dict(deepcopy(raw), evidence_ids={'MF.description'})
    return PhysicalHypothesisInstantiationService(method).instantiate(make_candidate(7.2), mf(), plan,
        plan_key='layout', response_model='OFFLINE')


def test_two_layouts_change_real_relative_motion_and_reach_bounded_causal_payload(method):
    parallel, errors = instantiate(method, layout_plan())
    angled, errors2 = instantiate(method, layout_plan('ANGLED'))
    assert not errors and not errors2
    p, a = parallel[0], angled[0]
    assert p.facts['first_contact_time_s'] == pytest.approx(4)
    assert a.facts['first_contact_time_s'] == pytest.approx(14/3)
    assert p.facts['relative_speed_kph'] != a.facts['relative_speed_kph']
    assert p.semantic_fingerprint != a.semantic_fingerprint
    for child in (p, a):
        relation = child.facts['layout_relation']
        assert relation['geometry_scope'] == 'TWO_AXIS_RELATIVE_MOTION'
        assert relation['coordinate_frame'] == 'BODY_FORWARD_POSITIVE'
        assert relation['assessment_reference'] == 'MODEL_INITIAL_STATE'
        assert child.fact_provenance['layout_relation']['approval'] == 'PENDING'
        assert child.fact_provenance['layout_relation']['provenance'] == 'LLM_INFERENCE'
        selection = CausalEvidenceSelector().select(build_fact_registry(mf(), child))
        assert 'SCN.layout_relation' in selection.selected_refs and selection.selected_evidence_count <= 10
        payload = json.loads(_scenario_json(mf(), child))
        assert payload['facts']['layout_relation'] == relation
        assert payload['conditional_physical_analysis']['hypothesis']['layout_relation']['kind'] == relation['kind']
        assert 'selected_source_condition' not in child.facts


@pytest.mark.parametrize('kind,angle', [('PARALLEL', 60), ('ANGLED', 0), ('ANGLED', 90), ('PERPENDICULAR', 45)])
def test_layout_kind_cannot_disagree_with_its_axes(kind, angle):
    raw = layout_plan(kind)
    raw['hypotheses'][0]['layout_relation']['axes_deg']['aisle'] = angle
    with pytest.raises(ValueError, match='LAYOUT_AXES_KIND_CONFLICT'):
        PhysicalScenarioPlan.from_dict(raw, evidence_ids={'MF.description'})


def test_incompatible_heading_is_not_silently_rotated_to_fit_layout(method):
    raw = layout_plan('ANGLED')
    parameters(raw, object_heading_angle_deg=30)
    before = deepcopy(raw)
    children, errors = instantiate(method, raw)
    assert not children and 'LAYOUT_PATH_AXIS_CONFLICT' in errors[0]['reason']
    assert raw == before


@pytest.mark.parametrize('actor', ['ego', 'object'])
def test_explicit_symbolic_heading_cannot_be_backfilled_from_cardinal_direction(method, actor):
    raw = layout_plan('ANGLED')
    field = actor+'_heading_angle_deg'
    parameters(raw, **{field: 0})
    parameter = next(p for p in raw['hypotheses'][0]['parameters'] if p['field']==field)
    parameter.pop('value'); parameter.update(kind='SYMBOLIC', symbol=actor+'_theta')
    h = raw['hypotheses'][0]
    values = {p['field']: p['value'] for p in h['parameters'] if p['kind']=='POINT'}
    relation = PhysicalHypothesisInstantiationService._layout_relation(h, values, {})
    assert relation['path_checks'][actor]['status'] == 'SYMBOLIC_OR_UNRESOLVED'
    assert relation['geometry_scope'] == 'LOCAL_INITIAL_RELATION_ONLY'
    assert field not in values


def test_undeclared_heading_uses_existing_cardinal_direction(method):
    children, errors = instantiate(method, layout_plan())
    assert not errors and children[0].facts['layout_relation']['path_checks']['object']['motion_heading_deg'] == 0


@pytest.mark.parametrize('label', ['longitudinal gradient 10%', 'transverse slope', 'Längsgefälle', 'Schräglage'])
def test_grade_words_are_not_parking_layout_restrictions(label):
    h = layout_plan()['hypotheses'][0]
    values = {p['field']: p['value'] for p in h['parameters']}
    relation = PhysicalHypothesisInstantiationService._layout_relation(h, values, {'odd_road_types':[label]})
    assert relation['project_scope_kinds'] == []


@pytest.mark.parametrize('label', ['perpendicular parking spaces', '垂直车位', 'Querparken'])
def test_explicit_source_layout_is_not_expanded_by_a_new_hypothesis(label):
    h = layout_plan()['hypotheses'][0]
    values = {p['field']: p['value'] for p in h['parameters']}
    with pytest.raises(ValueError, match='LAYOUT_OUTSIDE_PROJECT_SCOPE'):
        PhysicalHypothesisInstantiationService._layout_relation(h, values, {'odd_road_types':[label]})


def test_source_layout_kind_does_not_approve_generated_axes(method):
    raw = layout_plan('ANGLED'); h = raw['hypotheses'][0]
    h['operating_condition'] = {'source_field':'odd_road_types','value':'斜列车位',
        'physical_effect':'The permitted parking layout constrains this conditional moving interaction.'}
    h['layout_relation']['basis'] = 'SELECTED_SOURCE_CONDITION'
    plan = PhysicalScenarioPlan.from_dict(raw,evidence_ids={'MF.description'})
    children, errors = PhysicalHypothesisInstantiationService(method).instantiate(make_candidate(7.2),mf(),plan,
        plan_key='source-layout',response_model='OFFLINE',project_context={'odd_road_types':['斜列车位']})
    assert not errors
    assert children[0].facts['layout_relation']['project_scope_kinds'] == ['ANGLED']
    assert children[0].fact_provenance['layout_relation']['approval'] == 'PENDING'
    assert children[0].fact_provenance['layout_relation']['provenance'] == 'LLM_INFERENCE'


def test_response_correction_cannot_change_the_unreported_layout_world():
    raw = layout_plan(); corrected = layout_plan('ANGLED')
    with pytest.raises(ValueError,match='CORRECTION_CHANGED_UNREPORTED_WORLD.*layout_relation'):
        PhysicalScenarioPlan.validate_correction_scope(raw,corrected,[])
    corrected = deepcopy(raw)
    corrected['hypotheses'][0]['layout_relation']['rationale'] = 'Same world explained more clearly.'
    PhysicalScenarioPlan.validate_correction_scope(raw,corrected,[])


def test_label_only_and_stationary_target_motion_claims_do_not_establish_coverage(method):
    raw = layout_plan()
    raw['hypotheses'][0]['layout_relation']['paths'] = {'ego':'UNSPECIFIED','object':'UNSPECIFIED'}
    with pytest.raises(ValueError, match='LAYOUT_WITHOUT_MOTION_RELATION'):
        PhysicalScenarioPlan.from_dict(raw, evidence_ids={'MF.description'})
    raw = layout_plan(); parameters(raw, object_speed_kph=0)
    children, errors = instantiate(method, raw)
    assert not children and 'LAYOUT_OBJECT_PATH_REQUIRES_MOTION' in errors[0]['reason']
    raw['hypotheses'][0]['layout_relation']['paths']['object'] = 'UNSPECIFIED'
    children, errors = instantiate(method, raw)
    assert not errors and children[0].facts['layout_relation']['geometry_scope'] == 'LOCAL_INITIAL_RELATION_ONLY'


def test_curved_path_is_checked_only_at_its_initial_tangent(method):
    from test_curved_steering_physics import curved_plan
    raw = curved_plan()
    h = raw['hypotheses'][0]
    h['layout_relation'] = layout_plan('ANGLED')['hypotheses'][0]['layout_relation']
    h['layout_relation']['paths']['object'] = 'UNSPECIFIED'
    plan = PhysicalScenarioPlan.from_dict(raw, evidence_ids={'MF.description'})
    children, errors = PhysicalHypothesisInstantiationService(method).instantiate(make_candidate(7),mf(),plan,
        plan_key='curved-layout',response_model='OFFLINE')
    assert not errors
    assert children[0].facts['layout_relation']['trajectory_scope'] == 'INITIAL_TANGENT_ONLY'
    assert children[0].facts['layout_relation']['geometry_scope'] == 'LOCAL_INITIAL_RELATION_ONLY'


def test_existing_legacy_world_needs_no_extra_request_on_resume(method):
    raw = layout_plan();raw['hypotheses'][0].pop('layout_relation')
    children, errors = instantiate(method, raw)
    assert not errors and 'layout_relation' not in children[0].facts
    class Forbidden:
        def complete_json(self, request):raise AssertionError('An assessed legacy world must not request a new plan')
    state = HARAState('legacy-layout',scenarios=children)
    restored = HARAState.read_committed(json.loads(json.dumps(state.to_dict())))
    result = construct_physical_scenarios(restored,{'MF':restored.scenarios},[mf()],method=method,
        agent=PhysicalScenarioAgent(Forbidden()),project_facts=None)
    assert result['MF'] == children


def test_control_only_contract_and_identity_do_not_depend_on_motion_prompt_version(monkeypatch):
    from test_autonomous_physics import raw_plan
    h = raw_plan(assessed=True)['hypotheses'][0]
    raw = {'hypotheses':[], 'uncovered_conditions':[], 'control_assessment': {
        key:deepcopy(h[key]) for key in ('rationale','basis_refs','conditions','uncovered_conditions','failed_controls','remaining_controls','unresolved_controls')}}
    class Capture:
        def __init__(self):self.requests=[]
        def complete_json(self, request):self.requests.append(request);return LLMResponse(deepcopy(raw),'OFFLINE')
    context = {'plan_key':'control','task_scope':'CONTROL_INPUTS_ONLY','evidence':{'MF.description':'fault'}}
    before = Capture();PhysicalScenarioAgent(before).propose(context)
    monkeypatch.setattr(PhysicalScenarioAgent,'PROMPT_VERSION','another-motion-prompt')
    after = Capture();PhysicalScenarioAgent(after).propose(context)
    assert asdict(before.requests[0]) == asdict(after.requests[0])
    assert 'layout_relation' not in json.loads(after.requests[0].user_prompt)['output_contract']


def test_motion_policy_compares_real_interactions_in_one_existing_task_and_reuses_it(method):
    from test_geometry_completion import Client, run
    from hara_agent.services.analysis.scenario_equivalence import signature
    raw = {'hypotheses':[], 'uncovered_conditions':['Frozen context-only response; not an assertion of coverage.']}
    original = make_candidate(7.2); duplicate = deepcopy(original); duplicate.scenario_id = 'same-world-other-source-id'
    client = Client([raw]); agent = PhysicalScenarioAgent(client)
    state, _ = run(method,agent,candidates={'MF':[original,duplicate]})
    entry = next(iter(state.item_definition['physical_analysis_plans'].values()))
    context = entry['context']; policy = context['interaction_coverage_policy']
    assert 'inferred MF objects are not immutable source constraints' in policy
    assert "do not borrow another function's perpendicular restriction" in policy
    assert 'prioritize a compatible moving bay/aisle interaction' in policy
    assert 'parallel/angled geometry' in policy and 'specific fault/model incompatibility' in policy
    assert 'at most 3 total instances' in policy
    assert len(client.requests) == 1
    old = {k:v for k,v in context.items() if k!='plan_key'}
    old['interaction_coverage_policy'] = 'Previous interaction selection policy'
    assert signature(old) != context['plan_key']
    run(method,agent,state=state,candidates={'MF':[original,duplicate]})
    assert len(client.requests) == 1


def test_odd_action_candidate_does_not_claim_its_interactions_are_already_generated(method):
    from test_geometry_completion import Client, run
    parent = make_candidate(7.2)
    parent.facts.pop('road_user_type'); parent.fact_provenance.pop('road_user_type',None)
    raw = layout_plan('ANGLED')
    raw['hypotheses'][0]['rationale'] = 'A moving aisle vehicle changes the bay-exit overlap; fixed nearby objects are a different conditional world.'
    client = Client([raw]); state, result = run(method,PhysicalScenarioAgent(client),candidates={'MF':[parent]})
    assert len(client.requests) == 1 and len(result['MF']) == 1
    request = json.loads(client.requests[0].user_prompt)
    strategy = next(iter(state.item_definition['physical_analysis_plans'].values()))['context']['analysis_strategy']
    assert '还没有分开对象运动或布局交互' in strategy
    assert '默认只输出一个完整假设' not in strategy
    assert '只有当前输入明确包含' not in strategy
    assert '具体不适用依据' in request['output_shape']['hypotheses'][0]['rationale']
    child=result['MF'][0]
    assert child.facts['layout_relation']['geometry_scope']=='TWO_AXIS_RELATIVE_MOTION'
    assert child.facts['first_contact_time_s']==pytest.approx(14/3)
    assert child.fact_provenance['layout_relation']['approval']=='PENDING'


def test_control_manual_unknown_retains_actuator_and_isolates_previous_prompt_cache(tmp_path):
    from dataclasses import replace
    from hara_agent.infrastructure.llm.response_checkpoint import ResponseCheckpoint
    raw={'hypotheses':[], 'uncovered_conditions':[], 'control_assessment':{
        'rationale':'The actuator may remain; no basis establishes the manual pedal path.',
        'basis_refs':['MF.description'],'conditions':['Conditional actuator retention; manual path unresolved.'],
        'uncovered_conditions':['Manual pedal path and common cause are unverified.'],
        'failed_controls':['drive'],'remaining_controls':['braking'],
        'unresolved_controls':{'driver_braking':'No source or compatible assumption establishes the pedal command path.'}}}
    class Client:
        def __init__(self):self.requests=[]
        def complete_json(self,request):self.requests.append(request);return LLMResponse(deepcopy(raw),'OFFLINE')
    context={'plan_key':'manual-channel','task_scope':'CONTROL_INPUTS_ONLY','driver_configurations':['in_driver_seat'],
        'intervention_inputs':['emergency_braking_available'],'evidence':{'MF.description':'drive fault'}}
    capture=Client();PhysicalScenarioAgent(capture).propose(context)
    request=capture.requests[0]
    assert 'unresolved_controls.driver_braking' in request.system_prompt
    assert '不能据此认定踏板已可用' in request.system_prompt
    store=ResponseCheckpoint(tmp_path);client=Client()
    store.complete(client,replace(request,prompt_version='post-fault-control-inputs-v39-manual-channel-scope'))
    agent=PhysicalScenarioAgent(client,response_checkpoint_dir=tmp_path)
    plan,_=agent.propose(context)
    assert len(client.requests)==2  # Revised manual semantics cannot hit the legacy response.
    assert plan.control_assessment['remaining_controls']==['braking']
    assert 'driver_braking' in plan.control_assessment['unresolved_controls']
    agent.propose(context)
    assert len(client.requests)==2  # The revised task itself remains recoverable.


@pytest.mark.parametrize('manual_state,expected', [('conditional', True), ('architecture_unknown', None), ('failed', False)])
def test_conditional_control_contract_keeps_manual_authority_separate_from_verification(method, manual_state, expected):
    assessment = {'rationale':'The present fault is limited to the drive command; assess the existing pedal path in the same representative.',
        'basis_refs':['MF.description'], 'conditions':['Existing pedal and braking actuator; conditional isolation of the failed drive command.'],
        'uncovered_conditions':['Common controller failure and actual isolation remain unverified.'],
        'failed_controls':['automatic_drive_command'], 'remaining_controls':['braking']}
    if manual_state == 'conditional':
        assessment['remaining_controls'].append('driver_braking')
    elif manual_state == 'architecture_unknown':
        assessment['unresolved_controls']={'driver_braking':'The existing pedal interface routes through the failed shared command path; no compatible retained pedal path is established in this representative.'}
    else:
        assessment['failed_controls'].append('driver_braking')
    raw={'hypotheses':[], 'uncovered_conditions':assessment['uncovered_conditions'], 'control_assessment':assessment}
    class Client:
        def complete_json(self, request):
            self.request=request
            return LLMResponse(deepcopy(raw),'OFFLINE')
    client=Client(); context={'plan_key':'conditional-control-definition','task_scope':'CONTROL_INPUTS_ONLY',
        'driver_configurations':['in_driver_seat'],'intervention_inputs':['emergency_braking_available'],
        'evidence':{'MF.description':'single drive command fault; an existing pedal interface'}}
    before=deepcopy(context)
    plan,_=PhysicalScenarioAgent(client).propose(context)
    contract=json.loads(client.request.user_prompt)['output_contract']['control_arrays']
    assert '本代表条件性剩余能力' in contract['meaning'] and '不是已验证能力' in contract['meaning']
    assert '人工/自动采用同一故障范围标准' in contract['meaning']
    assert '仅未实测或未证明独立性不是未知理由' in client.request.system_prompt
    assert client.request.prompt_version == 'post-fault-control-inputs-v40-conditional-control-scope'
    assert context == before and plan.control_assessment == assessment
    from test_autonomous_physics import instantiate as instantiate_motion, raw_plan
    from hara_agent.services.analysis.risk_calculation_input_service import RiskCalculationInputService
    parent=instantiate_motion(method,raw_plan(initial=7.2),make_candidate(7.2))[0]
    original=deepcopy(parent)
    children,errors=PhysicalHypothesisInstantiationService(method).instantiate(parent,mf(),plan,
        plan_key=context['plan_key'],response_model='OFFLINE')
    assert not errors and len(children)==1 and parent==original
    child=children[0]
    # Retained braking execution is not a declaration of manual C credit.
    assert child.facts['emergency_braking_available'] is True
    applicability=RiskCalculationInputService.manual_braking_applicability(
        {**child.facts,'_fact_provenance':child.fact_provenance},malfunction_id='MF',scenario_id=child.scenario_id)
    assert applicability['status'] == ('DECLARED_MANUAL_BRAKING_PATH' if expected is True else
        'MANUAL_BRAKING_PATH_UNAVAILABLE' if expected is False else 'MANUAL_BRAKING_PATH_UNRESOLVED')
    assert child.fact_provenance['emergency_braking_available']['approval']=='PENDING'
    assert child.fact_provenance['emergency_braking_available']['calculation_scope']=='CONDITIONAL'
