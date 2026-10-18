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
    child = instantiate(method, raw, parent)[0]
    assert ('ego_impact_speed_kph' in child.facts) is expected
    assert ('longitudinal_acceleration_mps2' in child.facts) is expected


def test_method_standing_initial_cannot_silently_change(method):
    parent = make_candidate()
    parent.facts['ego_speed_constraint'] = {'min_kph':0,'max_kph':7}
    parent.facts['method_scenario_dimensions'] = {'EGO_DYNAMICS':{'atom_id':'Audi_state_standing', 'speed_intersection_kph':[0,0]}}
    parent.fact_provenance['ego_speed_kph']['provenance'] = 'METHOD_CONTRACT'
    children, errors = produce(method, raw_plan(initial=7), parent)
    assert not children and errors[0]['reason_code'] == 'INITIAL_STATE_ATOM_CONFLICT'


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


@pytest.mark.parametrize('alignment,heading,position', [('SAME','UPHILL','REAR'),('OPPOSITE','DOWNHILL','FRONT')])
def test_road_reference_heading_maps_into_body_coordinates(method,alignment,heading,position):
    parent=make_candidate();parent.facts.update(source_slope_direction='上坡',slope_reference='ROAD_REFERENCE')
    raw=grade_raw(heading,position)
    raw['hypotheses'][0]['grade_frame'].update(slope_reference='ROAD_REFERENCE',ego_forward_relative_to_reference=alignment)
    child=instantiate(method,raw,parent)[0]
    assert child.facts['motion_model_outcome']['status']=='CONTACT'


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


def test_grade_change_invalidates_old_contact(method):
    from hara_agent.services.analysis.scenario_physics import branch_physics_view
    child=instantiate(method,grade_raw())[0]
    view={**child.facts,'scenario_id':child.scenario_id,'malfunction_id':'MF','_fact_provenance':child.fact_provenance}
    changed=branch_physics_view(view,{'ego_heading_relative_to_downhill':'DOWNHILL',
        '_fact_provenance':{'ego_heading_relative_to_downhill':child.fact_provenance['ego_heading_relative_to_downhill']}})
    assert 'ego_impact_speed_kph' not in changed and 'ttc_s' not in changed
