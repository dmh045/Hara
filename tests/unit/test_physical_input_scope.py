"""Targeted F01–F03 regressions, through the production plan consumer."""
from copy import deepcopy
import json
import pytest
from test_autonomous_physics import raw_plan, make_candidate, mf, instantiate
from test_p7_closure import method, source
from hara_agent.contracts.physical_scenario_plan import PhysicalScenarioPlan
from hara_agent.services.analysis.physical_hypothesis_instantiation_service import PhysicalHypothesisInstantiationService


def produce(method, raw, parent):
    return PhysicalHypothesisInstantiationService(method).instantiate(parent, mf(),
        PhysicalScenarioPlan.from_dict(raw, evidence_ids={'MF.description'}), plan_key='scope-test', response_model='OFFLINE')


def symbol(raw, field):
    p = next(p for p in raw['hypotheses'][0]['parameters'] if p['field'] == field)
    p.pop('value'); p.update(kind='SYMBOLIC', symbol='unknown_parameter')


@pytest.mark.parametrize('position', ['in_driver_seat', 'outside_driver_seat'])
def test_explicit_driver_scope_conditions_unknown_parent_without_source_approval(method, position):
    parent = make_candidate()
    for field in ('driver_position', 'driver_in_vehicle'):
        parent.facts.pop(field, None); parent.fact_provenance.pop(field, None)
    before = deepcopy(parent)
    raw = raw_plan(); raw['hypotheses'][0]['driver_positions'] = [position]
    children, errors = produce(method, raw, parent)
    assert not errors and len(children) == 1 and parent == before
    child = children[0]
    assert child.facts['driver_position'] == position
    assert child.fact_provenance['driver_position']['approval'] == 'PENDING'
    assert child.fact_provenance['driver_position']['provenance'] == 'LLM_INFERENCE'
    assert child.fact_provenance['driver_position']['source_refs'][0]['source_type'] == 'analysis_hypothesis'
    assert child.status.value == 'PENDING' and child.scenario_id != parent.scenario_id
    if position == 'in_driver_seat':
        assert child.facts['driver_in_vehicle'] is True
    else:
        assert 'driver_in_vehicle' not in child.facts
    from hara_agent.services.reporting.engineering_text_mapper import EngineeringReportTextMapper
    mapper = EngineeringReportTextMapper()
    assert '分析假设' in mapper.scenario(child)[0]
    assert '分析假设' in mapper.driver_branch(child, {})


def test_driver_scope_respects_known_position_and_boolean_handoff(method):
    parent = make_candidate()
    raw = raw_plan(); raw['hypotheses'][0]['driver_positions'] = ['in_driver_seat']
    children, errors = produce(method, raw, parent)
    assert not errors and len(children) == 1
    assert 'driver_configuration_assumption' not in children[0].facts['physical_analysis']
    parent.facts.update(driver_position='outside_driver_seat', driver_in_vehicle=False)
    before = deepcopy(parent)
    children, errors = produce(method, raw, parent)
    assert not children and errors[0]['reason'] == 'HYPOTHESIS_DRIVER_SCOPE_NOT_APPLICABLE'
    assert parent == before


@pytest.mark.parametrize('positions', [None, ['in_driver_seat', 'outside_driver_seat']])
def test_unrestricted_driver_scope_does_not_invent_a_position_or_duplicate_motion(method, positions):
    parent = make_candidate()
    parent.facts.pop('driver_in_vehicle'); parent.fact_provenance.pop('driver_in_vehicle', None)
    raw = raw_plan()
    if positions:
        raw['hypotheses'][0]['driver_positions'] = positions
    children, errors = produce(method, raw, parent)
    assert not errors and len(children) == 1
    assert 'driver_position' not in children[0].facts and 'driver_in_vehicle' not in children[0].facts


@pytest.mark.parametrize('control_only', [False, True])
@pytest.mark.parametrize('failed,remaining,expected', [
    (['automatic_steering_command'], ['steering'], True),
    (['steering'], ['braking'], False),
    (['drive'], ['braking'], None),
])
def test_declared_steering_capability_reaches_both_production_consumers(method, control_only, failed, remaining, expected):
    parent = make_candidate()
    before = deepcopy(parent)
    raw = raw_plan()
    h = raw['hypotheses'][0]
    h.update(failed_controls=failed, remaining_controls=remaining)
    if control_only:
        raw = {'hypotheses': [], 'uncovered_conditions': [], 'control_assessment': {
            k: h[k] for k in ('rationale','basis_refs','conditions','uncovered_conditions','failed_controls','remaining_controls')}}
    children, errors = produce(method, raw, parent)
    assert not errors and len(children) == 1 and parent == before
    child = children[0]
    assert child.facts.get('steering_control_available') is expected
    if expected is not None:
        assert child.fact_provenance['steering_control_available']['approval'] == 'PENDING'
        assert child.fact_provenance['steering_control_available']['source_refs'][0]['source_type'] == 'analysis_hypothesis'
        from hara_agent.services.analysis.hazardous_event_risk_context_service import HazardousEventRiskContextService
        service = HazardousEventRiskContextService(method)
        view = {**child.facts, '_fact_provenance':child.fact_provenance}
        context = service.build(malfunction_id='MF',
            scenario_id=child.scenario_id, hazard_node_id='impact',
            scenario=view)
        assert service.scoring_facts(context, view)['steering_control_available'] is expected


@pytest.mark.parametrize('control_only', [False, True])
def test_retained_steering_does_not_overwrite_an_existing_conflicting_condition(method, control_only):
    parent = make_candidate()
    parent.facts['steering_control_available'] = False
    parent.fact_provenance['steering_control_available'] = source()
    before = deepcopy(parent)
    raw = raw_plan(); raw['hypotheses'][0]['remaining_controls'] = ['steering']
    if control_only:
        raw = {'hypotheses': [], 'uncovered_conditions': [], 'control_assessment': {
            k: raw['hypotheses'][0][k] for k in ('rationale','basis_refs','conditions','uncovered_conditions','failed_controls','remaining_controls')}}
    children, errors = produce(method, raw, parent)
    assert not children and errors[0]['reason_code'] == 'CONFLICT_WITH_SELECTED_CONDITION'
    assert parent == before


@pytest.mark.parametrize('field', ['longitudinal_acceleration_mps2', 'relative_distance_m'])
def test_explicit_symbol_cannot_use_parent_number_or_alias(method, field):
    parent = make_candidate()
    parent.facts.update(longitudinal_acceleration_mps2=2, relative_distance_m=2, travel_to_impact_m=2)
    parent.fact_provenance.update({k: source() for k in ('longitudinal_acceleration_mps2', 'relative_distance_m', 'travel_to_impact_m')})
    parent.fact_provenance['longitudinal_acceleration_mps2']['analysis_value_semantic'] = 'PRE_FAULT_OPERATING_ACCELERATION'
    before = deepcopy(parent)
    raw = raw_plan(); symbol(raw, field)
    child = instantiate(method, raw, parent)[0]
    assert field not in child.facts
    assert 'ego_impact_speed_kph' not in child.facts and 'ttc_s' not in child.facts
    assert parent == before
    assert child.facts['physical_analysis']['model_input_view'][field]['action'] == 'SYMBOLIC'


@pytest.mark.parametrize('phase,obj,expected', [('POST_FAULT_MOTION_PARAMETER','EGO',True),
    ('PRE_FAULT_OPERATING_ACCELERATION','EGO',False), ('POST_FAULT_MOTION_PARAMETER','OTHER',False)])
def test_omitted_parameter_inherits_only_applicable_role_and_object(method, phase, obj, expected):
    parent = make_candidate()
    parent.facts['longitudinal_acceleration_mps2'] = 2
    parent.fact_provenance['longitudinal_acceleration_mps2'] = {**source(), 'analysis_value_semantic':phase, 'object_id':obj}
    raw = raw_plan(); raw['hypotheses'][0]['parameters'].pop()
    inputs, _ = PhysicalHypothesisInstantiationService(method)._model_inputs(parent, raw['hypotheses'][0], 'MF')
    assert ('longitudinal_acceleration_mps2' in inputs) is expected
    children, errors = produce(method, raw, parent)
    if expected:
        assert children[0].facts['longitudinal_acceleration_mps2'] == 2
        assert 'ego_impact_speed_kph' in children[0].facts and not errors
    else:
        assert not children
        assert errors[0]['reason'] == 'REQUIRED_MODEL_INPUT_OMITTED'
        assert errors[0]['fields'] == ['longitudinal_acceleration_mps2']


def test_method_standing_initial_cannot_silently_change(method):
    parent = make_candidate()
    parent.facts['ego_speed_constraint'] = {'min_kph':0,'max_kph':7}
    parent.facts['method_scenario_dimensions'] = {'EGO_DYNAMICS':{'atom_id':'Audi_state_standing', 'speed_intersection_kph':[0,0]}}
    parent.fact_provenance['ego_speed_kph']['provenance'] = 'METHOD_CONTRACT'
    children, errors = produce(method, raw_plan(initial=7), parent)
    assert not children and errors[0]['reason_code'] == 'INITIAL_STATE_ATOM_CONFLICT'


@pytest.mark.parametrize('constraint,allowed', [
    ({'min_kph': None, 'max_kph': None}, True),
    ({'speed_min_kph': None, 'speed_max_kph': None}, True),
    ({'min_kph': None, 'max_kph': 8}, True),
    ({'min_kph': 6, 'max_kph': None}, True),
    ({'min_kph': None, 'max_kph': 6}, False),
    ({'min_kph': 8, 'max_kph': None}, False),
    ({'min_kph': None, 'max_kph': 7, 'max_inclusive': False}, False),
    ({'min_kph': 7, 'max_kph': None, 'min_inclusive': False}, False),
])
def test_nullable_source_speed_bounds_preserve_known_limits(method, constraint, allowed):
    parent = make_candidate(7)
    parent.facts['ego_speed_constraint'] = constraint
    before = deepcopy(parent)
    children, errors = produce(method, raw_plan(initial=7), parent)
    assert parent == before
    if allowed:
        assert not errors and children[0].facts['ego_speed_kph'] == 7
        assert children[0].facts['physical_analysis']['calculation_nature'] == 'CONDITIONAL'
    else:
        assert not children and errors[0]['reason'] == 'INITIAL_SPEED_OUTSIDE_OPERATING_CONSTRAINT'


def test_rollaway_input_view_excludes_moving_parent_without_defaulting_zero(method):
    parent = make_candidate(7)
    before = deepcopy(parent)
    service = PhysicalHypothesisInstantiationService(method)
    status = service.input_status(parent, 'MF')['models']
    grade = status['GRADE_ROLLAWAY_TO_IMPACT_DISTANCE']
    assert 'ego_speed_kph' not in grade['available']
    assert 'ego_speed_kph' in grade['needs_hypothesis']
    assert 'separate stopped condition' in grade['initial_state_requirement']
    assert status['CONSTANT_ACCELERATION_TO_IMPACT_DISTANCE']['available']['ego_speed_kph']['value'] == 7
    raw = grade_raw(); raw['hypotheses'][0]['parameters'].pop(0)
    children, errors = produce(method, raw, parent)
    assert not children and errors[0]['reason'] == 'REQUIRED_MODEL_INPUT_OMITTED'
    assert parent == before
    stationary = service.input_status(make_candidate(0), 'MF')['models']['GRADE_ROLLAWAY_TO_IMPACT_DISTANCE']
    assert stationary['available']['ego_speed_kph']['value'] == 0


def test_method_moving_target_cannot_become_stationary_same_world(method):
    parent = make_candidate(); parent.facts.update(object_id='object1',object_speed_kph=10)
    parent.fact_provenance['object_speed_kph'] = {**source(), 'provenance':'METHOD_CONTRACT'}
    children, errors = produce(method, raw_plan(), parent)
    assert not children and errors[0]['field'] == 'object_speed_kph'
    assert errors[0]['old_value'] == 10 and errors[0]['proposed_value'] == 0


def test_explicit_new_target_preserves_original_and_gets_new_identity(method):
    parent = make_candidate(); parent.facts.update(object_id='object1',object_speed_kph=10)
    parent.fact_provenance['object_speed_kph'] = source()
    raw = raw_plan(); h=raw['hypotheses'][0]
    h['geometry']['object_id']='different-stationary-object'
    h['condition_changes']={'initial_state':False,'target':True,'rationale':'separate stationary roadside object'}
    child=instantiate(method,raw,parent)[0]
    assert parent.facts['object_speed_kph']==10 and child.facts['object_speed_kph']==0
    assert child.scenario_id!=parent.scenario_id and child.status.value=='PENDING'
    assert child.facts['physical_object_exposure_unresolved']


def grade_raw(heading='UPHILL', position='REAR', resistance=0):
    raw=raw_plan('GRADE_ROLLAWAY_TO_IMPACT_DISTANCE'); h=raw['hypotheses'][0]
    h['grade_frame'].update(heading_relative_to_downhill=heading,downhill_body_direction='REAR' if heading=='UPHILL' else 'FRONT')
    h['geometry'].update(ego_longitudinal_direction='REVERSE' if heading=='UPHILL' else 'FORWARD', object_position=position)
    h['parameters'][-1]['value']=resistance
    return raw


@pytest.mark.parametrize('location', ['condition_changes', 'parameters'])
def test_misplaced_grade_frame_reports_its_actual_contract_path(location):
    valid = grade_raw()
    raw = deepcopy(valid)
    h = raw['hypotheses'][0]
    frame = h.pop('grade_frame')
    if location == 'condition_changes':
        h.setdefault(location, {'initial_state': False, 'target': False,
                                'rationale': 'same world'})['grade_frame'] = frame
    else:
        h[location].append({'field': 'grade_frame', 'unit': 'struct', 'kind': 'SYMBOLIC',
                            'symbol': json.dumps(frame), 'rationale': 'orientation'})
    before = deepcopy(raw)
    with pytest.raises(ValueError, match=r'hypotheses\[\]\.grade_frame'):
        PhysicalScenarioPlan.from_dict(raw, evidence_ids={'MF.description'})
    assert raw == before  # Diagnostics never move or rewrite a model response.
    assert PhysicalScenarioPlan.from_dict(valid, evidence_ids={'MF.description'}).to_dict() == valid


@pytest.mark.parametrize('heading,position,contact,sign', [('UPHILL','REAR',True,-1),('UPHILL','FRONT',False,-1),('DOWNHILL','FRONT',True,1)])
def test_body_grade_direction_controls_contact(method,heading,position,contact,sign):
    child=instantiate(method,grade_raw(heading,position))[0]
    assert (child.facts.get('motion_model_outcome',{}).get('status')=='CONTACT') is contact
    assert child.facts['gravity_body_acceleration_mps2']*sign>0
    assert ('ttc_s' in child.facts) is contact


def test_holding_resistance_does_not_launch_reverse_motion(method):
    child=instantiate(method,grade_raw(resistance=5))[0]
    assert child.facts['longitudinal_acceleration_mps2']==0
    assert 'ttc_s' not in child.facts and 'ego_impact_speed_kph' not in child.facts


def test_no_contact_outcome_reaches_causal_request_and_scopes_contact_check(method):
    from hara_agent.services.semantic.scenario_batching import _scenario_json
    from hara_agent.services.analysis.scenario_physics import causal_physics_conflicts
    child = instantiate(method, raw_plan(initial=7, acceleration=-8), make_candidate(7))[0]
    before = deepcopy(child)
    assert child.facts['motion_model_outcome']['status'] == 'STOPPED_BEFORE_CONTACT'
    request = json.loads(_scenario_json(mf(), child))
    view = request['physical_interaction']
    assert view['conditions']['motion_model_outcome'] == child.facts['motion_model_outcome']
    assert view['evidence_refs']['motion_model_outcome'] == 'DERIVED.motion_model_outcome'
    assert view['derived_physics_evidence']['DERIVED.motion_model_outcome']['value']['status'] == 'STOPPED_BEFORE_CONTACT'
    claim = {'target': 'SCENARIO_OBJECT', 'object_position': child.facts['object_position'],
             'relation': 'CONTACT', 'assessment_reference': 'MODEL_TRAJECTORY'}
    assert 'NO_CONTACT_TRAJECTORY_CONTRADICTS_CONTACT_CLAIM' in causal_physics_conflicts(
        child, {'physical_interaction': claim})
    # Initial approach and a different target/phase are not disproved by this trajectory.
    for difference in ({'relation': 'APPROACH'}, {'target': 'OTHER'},
                       {'assessment_reference': 'OTHER'}, {'object_id': 'another-object'}):
        assert not causal_physics_conflicts(child, {'physical_interaction': {**claim, **difference}})
    contact = instantiate(method, raw_plan(initial=7, acceleration=2), make_candidate(7))[0]
    assert not causal_physics_conflicts(contact, {'physical_interaction': claim})
    assert child == before


def test_unknown_slope_reference_needs_explicit_orientation(method):
    raw=grade_raw(); raw['hypotheses'][0].pop('grade_frame')
    parent=make_candidate();parent.facts['source_slope_direction']='上坡'
    children,errors=produce(method,raw,parent)
    assert not children and errors[0]['reason_code']=='GRADE_ORIENTATION_REQUIRED'
    children,errors=produce(method,grade_raw(),parent)
    assert not errors and children[0].facts['grade_orientation_basis']=='ANALYSIS_ASSUMPTION'


def test_known_slope_reference_rejects_opposite_heading(method):
    parent=make_candidate();parent.facts.update(source_slope_direction='上坡',slope_reference='EGO_FORWARD')
    raw=grade_raw('DOWNHILL','FRONT');raw['hypotheses'][0]['grade_frame']['slope_reference']='EGO_FORWARD'
    children,errors=produce(method,raw,parent)
    assert not children and errors[0]['reason_code']=='GRADE_HEADING_CONFLICT'


def test_same_world_enum_spelling_keeps_source_and_parent_unchanged(method):
    parent=make_candidate()
    parent.facts.update(object_position='front',road_user_type='vehicle',collision_type='frontal',
                        object_longitudinal_direction='stationary',object_speed_kph=0)
    for field in ('object_position','road_user_type','collision_type','object_longitudinal_direction','object_speed_kph'):
        parent.fact_provenance[field]=deepcopy(parent.fact_provenance['ego_speed_kph'])
    before=deepcopy(parent)
    children,errors=produce(method,grade_raw('DOWNHILL','FRONT'),parent)
    assert not errors and len(children)==1
    assert parent==before
    child=children[0]
    assert child.facts['object_position']=='FRONT' and child.facts['road_user_type']=='VEHICLE'
    assert child.fact_provenance['object_position']['approval']==parent.fact_provenance['object_position']['approval']
    assert child.fact_provenance['object_position']['source_refs']==parent.fact_provenance['object_position']['source_refs']
    assert child.facts['motion_model_outcome']['status']=='CONTACT'


@pytest.mark.parametrize('field,old', [('object_position','rear'),('object_id','Object1')])
def test_enum_normalization_does_not_merge_changed_geometry_or_object_identity(method,field,old):
    parent=make_candidate();parent.facts[field]=old
    parent.fact_provenance[field]=deepcopy(parent.fact_provenance['ego_speed_kph'])
    children,errors=produce(method,grade_raw('DOWNHILL','FRONT'),parent)
    assert not children and errors[0]['field']==field


def test_bounded_semantic_correction_retains_initial_response_and_resumes(method,tmp_path):
    from hara_agent.workflow import HARAState
    from hara_agent.workflow.physical_analysis import construct_physical_scenarios
    from hara_agent.services.semantic.physical_scenario_agent import PhysicalScenarioAgent
    from hara_agent.infrastructure.llm.protocol import LLMResponse
    class Client:
        calls=0
        def complete_json(self,request):
            self.calls+=1
            context=json.loads(request.user_prompt)['context']
            raw=grade_raw()
            raw['hypotheses'][0]['unresolved_controls'] = {'braking': 'Synthetic test: independence of the braking path is not resolved'}
            if not context.get('correction_diagnostics'):
                raw['hypotheses'][0].pop('grade_frame')
            return LLMResponse(raw,'OFFLINE')
    state=HARAState('bounded-correction'); client=Client(); agent=PhysicalScenarioAgent(client,response_checkpoint_dir=tmp_path)
    def run():
        return construct_physical_scenarios(state,{'MF':[make_candidate()]},[mf()],method=method,agent=agent,project_facts=None)
    assert len(run()['MF'])==1 and client.calls==2
    entry=next(iter(state.item_definition['physical_analysis_plans'].values()))
    assert len(entry['responses'])==2 and entry['initial_plan'] and entry['correction_diagnostics']
    assert entry['semantic_correction_completed'] and not entry['instantiations']['SC']['failures']
    run();assert client.calls==2


def test_retained_grade_failure_gets_specific_feedback_once_on_version_change(method, tmp_path, monkeypatch):
    from hara_agent.workflow import HARAState
    from hara_agent.workflow.physical_analysis import construct_physical_scenarios
    from hara_agent.services.semantic.physical_scenario_agent import PhysicalScenarioAgent
    from hara_agent.infrastructure.llm.protocol import LLMResponse
    from hara_agent.services.analysis.physical_hypothesis_instantiation_service import (
        PhysicalHypothesisInstantiationService, PhysicalInputConflict,
    )
    mapping = PhysicalHypothesisInstantiationService._grade_mapping
    def old_feedback(parent, hypothesis):
        try:
            return mapping(parent, hypothesis)
        except PhysicalInputConflict as error:
            error.details['resolution'] = 'Preserve the applicable input, or explicitly propose a supported new initial condition/target; do not silently overwrite.'
            raise
    monkeypatch.setattr(PhysicalHypothesisInstantiationService, '_grade_mapping', staticmethod(old_feedback))
    good = grade_raw('UPHILL', 'REAR')
    good['hypotheses'][0]['unresolved_controls'] = {'braking': 'Actuator independence is not established in this fixture.'}
    bad = deepcopy(good)
    bad['hypotheses'][0]['grade_frame']['heading_relative_to_downhill'] = 'DOWNHILL'
    class Client:
        calls = 0
        corrected = False
        def complete_json(self, request):
            self.calls += 1
            if self.corrected:
                assert 'heading_relative_to_downhill=DOWNHILL' in request.user_prompt
                assert 'downhill_body_direction=FRONT' in request.user_prompt
                assert 'ego_longitudinal_direction=FORWARD' in request.user_prompt
            return LLMResponse(deepcopy(good if self.corrected else bad), 'OFFLINE')
    client = Client(); agent = PhysicalScenarioAgent(client, response_checkpoint_dir=tmp_path)
    agent.RESPONSE_VALIDATION_VERSION = 'old-grade-feedback'
    state = HARAState('retained-grade')
    def run(current):
        return construct_physical_scenarios(current, {'MF': [make_candidate()]}, [mf()],
            method=method, agent=agent, project_facts=None)['MF']
    rejected = run(state)
    assert client.calls == 2 and rejected[0].scenario_id == 'SC'
    assert not rejected[0].facts.get('motion_model_outcome')
    entry = next(iter(state.item_definition['physical_analysis_plans'].values()))
    assert not entry['instantiations']['SC']['generated_ids']
    assert entry['instantiations']['SC']['failures'][0]['reason_code'] == 'GRAVITY_DIRECTION_CONFLICT'
    initial = deepcopy(entry['initial_plan'])
    monkeypatch.setattr(PhysicalHypothesisInstantiationService, '_grade_mapping', staticmethod(mapping))
    client.corrected = True
    agent.RESPONSE_VALIDATION_VERSION = PhysicalScenarioAgent.RESPONSE_VALIDATION_VERSION
    children = run(state)
    assert len(children) == 1 and client.calls == 3
    assert children[0].facts['ego_longitudinal_direction'] == 'REVERSE'
    assert entry['initial_plan'] == initial and entry['validation_history']
    restored = HARAState.read_committed(json.loads(json.dumps(state.to_dict())))
    assert run(restored) == children and client.calls == 3


@pytest.mark.parametrize('alignment,heading,position', [('SAME','UPHILL','REAR'),('OPPOSITE','DOWNHILL','FRONT')])
def test_road_reference_heading_maps_into_body_coordinates(method,alignment,heading,position):
    parent=make_candidate();parent.facts.update(source_slope_direction='上坡',slope_reference='ROAD_REFERENCE')
    raw=grade_raw(heading,position)
    raw['hypotheses'][0]['grade_frame'].update(slope_reference='ROAD_REFERENCE',ego_forward_relative_to_reference=alignment)
    child=instantiate(method,raw,parent)[0]
    assert child.facts['motion_model_outcome']['status']=='CONTACT'


def test_road_reference_conflict_feedback_requires_coordinate_repair(method):
    parent = make_candidate()
    parent.facts.update(source_slope_direction='上坡', slope_reference='ROAD_REFERENCE')
    raw = grade_raw('DOWNHILL', 'FRONT')
    h = raw['hypotheses'][0]
    h['grade_frame'].update(slope_reference='ROAD_REFERENCE', ego_forward_relative_to_reference='SAME')
    h['condition_changes'] = {'initial_state': True, 'target': False, 'rationale': 'Independent stopped initial condition'}
    service = PhysicalHypothesisInstantiationService(method)
    mapping = service.input_status(parent, 'MF')['grade_reference_mapping']
    assert mapping['selected_source_slope_direction'] == '上坡'
    assert mapping['road_reference_heading_by_alignment'] == {'SAME': 'UPHILL', 'OPPOSITE': 'DOWNHILL'}
    children, errors = service.instantiate(parent, mf(), PhysicalScenarioPlan.from_dict(raw, evidence_ids={'MF.description'}),
        plan_key='frame-conflict', response_model='offline')
    assert not children and errors[0]['reason_code'] == 'GRADE_HEADING_CONFLICT'
    assert errors[0]['relation'] == 'GRADE_REFERENCE_AND_BODY_ALIGNMENT'
    assert 'OPPOSITE' in errors[0]['resolution'] and 'condition_changes.initial_state' in errors[0]['resolution']
    # A different rationale alone did not bypass the guard. A coherent body
    # mapping does calculate, retaining the source uphill label and approval.
    h['grade_frame']['ego_forward_relative_to_reference'] = 'OPPOSITE'
    child = instantiate(method, raw, parent)[0]
    assert child.facts['motion_model_outcome']['status'] == 'CONTACT'
    assert parent.facts['source_slope_direction'] == '上坡'
    assert child.fact_provenance['relative_speed_kph']['approval'] == 'PENDING'


def test_budget_stop_during_semantic_correction_keeps_original_and_resumes(method,tmp_path):
    from hara_agent.workflow import HARAState, CheckpointRepository
    from hara_agent.workflow.physical_analysis import construct_physical_scenarios
    from hara_agent.services.semantic.physical_scenario_agent import PhysicalScenarioAgent
    from hara_agent.infrastructure.llm.protocol import LLMResponse
    from hara_agent.infrastructure.llm.provider_budget import ProviderAttemptBudgetExceeded
    class Client:
        calls=0
        stopped=False
        def complete_json(self,request):
            self.calls+=1
            context=json.loads(request.user_prompt)['context']
            raw=grade_raw()
            if context.get('correction_diagnostics'):
                if not self.stopped:
                    self.stopped=True
                    raise ProviderAttemptBudgetExceeded(limit=1,attempts=1,ledger_path=tmp_path/'dummy')
            else:
                raw['hypotheses'][0].pop('grade_frame')
            return LLMResponse(raw,'OFFLINE')
    client=Client();repository=CheckpointRepository(tmp_path/'state');state=HARAState('resume-semantic')
    agent=PhysicalScenarioAgent(client,response_checkpoint_dir=tmp_path/'responses')
    def run():
        return construct_physical_scenarios(state,{'MF':[make_candidate()]},[mf()],method=method,agent=agent,project_facts=None,checkpoint=repository.save)
    with pytest.raises(ProviderAttemptBudgetExceeded):run()
    state=repository.load(state.run_id)
    entry=next(iter(state.item_definition['physical_analysis_plans'].values()))
    assert entry['status']=='CORRECTION_REQUESTED' and len(entry['responses'])==1
    assert entry['initial_plan'] and entry['correction_diagnostics']
    run();assert client.calls==3
    assert entry['status']=='VALIDATED' and len(entry['responses'])==2
    run();assert client.calls==3


def test_input_metadata_object_change_changes_plan_identity():
    from hara_agent.services.analysis.scenario_equivalence import scenario_conditions, signature
    parent=make_candidate();a=signature(scenario_conditions(parent))
    parent.fact_provenance['ego_speed_kph']['object_id']='different-vehicle'
    assert signature(scenario_conditions(parent))!=a


@pytest.mark.parametrize('project_scope', [None, 'different_item_definition', 'current_avp_item_definition'])
@pytest.mark.parametrize('controls_only', [False, True])
def test_production_physical_request_scopes_project_driver_convention(method, project_scope, controls_only):
    from hara_agent.workflow import HARAState
    from hara_agent.workflow.physical_analysis import construct_physical_scenarios
    from hara_agent.services.semantic.physical_scenario_agent import PhysicalScenarioAgent, expand_physical_prompt_view
    parent = instantiate(method)[0] if controls_only else make_candidate()
    if controls_only:
        parent.facts.pop('driver_in_vehicle', None)
        parent.fact_provenance.pop('driver_in_vehicle', None)
    if project_scope is not None:
        parent.facts['project_scope'] = project_scope
    before = deepcopy(method.metadata)
    class RequestCaptured(Exception):
        pass
    class Client:
        def complete_json(self, request):
            payload = json.loads(request.user_prompt)
            view = expand_physical_prompt_view(payload['context'], payload['source_texts'])
            policy = view['method_identity']['project_analysis_policy']
            assert ('driver_location_convention' in policy) == (project_scope == before['project_analysis_policy']['project_scope'])
            assert 'decisions' not in policy
            assert (view.get('task_scope') == 'CONTROL_INPUTS_ONLY') == controls_only
            assert 'outside_driver_seat仅表示不在驾驶位，不等于人员在车外' in request.system_prompt
            assert view['method_identity']['method_source_hash'] == before['method_source_hash']
            raise RequestCaptured
    with pytest.raises(RequestCaptured):
        construct_physical_scenarios(HARAState('scope-check', item_definition={
            'item_sha256': before['project_analysis_policy']['bound_item_sha256'][0]}), {'MF': [parent]}, [mf()],
            method=method, agent=PhysicalScenarioAgent(Client()), project_facts=None)
    assert method.metadata == before


def test_grade_change_invalidates_old_contact(method):
    from hara_agent.services.analysis.scenario_physics import branch_physics_view
    child=instantiate(method,grade_raw())[0]
    view={**child.facts,'scenario_id':child.scenario_id,'malfunction_id':'MF','_fact_provenance':child.fact_provenance}
    changed=branch_physics_view(view,{'ego_heading_relative_to_downhill':'DOWNHILL',
        '_fact_provenance':{'ego_heading_relative_to_downhill':child.fact_provenance['ego_heading_relative_to_downhill']}})
    assert 'ego_impact_speed_kph' not in changed and 'ttc_s' not in changed
