"""A side impact label is independent of a finite body's closing-motion axis."""
from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from test_autonomous_physics import method,make_candidate,instantiate,mf,score
from test_bounded_interaction_coverage import moving_plan
from hara_agent.services.analysis import scenario_equivalence as equivalence
from hara_agent.services.analysis.physical_hypothesis_instantiation_service import PhysicalHypothesisInstantiationService as Producer
from hara_agent.services.analysis.scenario_physics import rectilinear_moving_contact
from hara_agent.services.semantic.scenario_batching import _scenario_json
from hara_agent.services.semantic.scenario_agent import ScenarioFeasibilityAgent
from hara_agent.services.semantic.scenario_evidence import SCENARIO_ASSESSMENT_CONTRACT_VERSION
from hara_agent.workflow.nodes import scenarios as scenario_node


def longitudinal_world(method,*,stale=False,surface='SIDE'):
    # Numeric geometry from the actual bounded first-control result. The
    # fixture's source declarations are ordinary synthetic test hypotheses.
    raw=moving_plan(initial=7,acceleration=1.5,target_direction='STATIONARY',target_speed=0,
        x=6.15,y=1.75,position='LEFT',collision=surface)
    for p in raw['hypotheses'][0]['parameters']:
        if p['field']=='contact_half_length_m':p['value']=4.4
        if p['field']=='contact_half_width_m':p['value']=1.825
    child=instantiate(method,raw,make_candidate(7))[0]
    if stale:
        child.facts['motion_model_outcome']['longitudinal_contact']=False
        child.facts.pop('ttc_s',None);child.fact_provenance.pop('ttc_s',None)
    return child


def claim(child,*,old=False):
    return dict(scenario_id=child.scenario_id,physically_feasible=True,functionally_relevant=True,
        causally_relevant=True,breakpoint='NONE',hazardous_event='The fault brings the same target into contact.',
        causal_chain={
            'm_to_b':dict(claim='defined excessive drive',basis_type='DIRECT_FACT',evidence_refs=['MF.description']),
            'b_to_i':dict(claim='longitudinal_contact=false' if old else 'longitudinal_contact=true; the lateral envelopes already overlap',
                basis_type='DERIVED_PHYSICS',evidence_refs=['DERIVED.motion_model_outcome']),
            'i_to_h':dict(claim='contact along the same unchanged trajectory',basis_type='DERIVED_PHYSICS',evidence_refs=['DERIVED.relative_speed_kph'])},
        risk_dimension_changes=[],confidence=.8,status='PENDING')


def test_surface_does_not_erase_actual_longitudinal_contact_or_manual_c(method):
    old=longitudinal_world(method,stale=True);before=deepcopy(old)
    current=equivalence.scenario_with_current_physics(old)
    assert old==before and current is not old
    assert current.facts['collision_type']=='SIDE'
    assert current.facts['motion_model_outcome']=={**old.facts['motion_model_outcome'],'longitudinal_contact':True}
    assert current.facts['ttc_s']==pytest.approx(.7071301238960774)
    assert current.facts['ttc_s']==current.facts['first_contact_time_s']
    for field in ('ego_speed_kph','object_speed_kph','relative_speed_kph','ego_impact_speed_kph',
                  'object_offset_x_m','object_offset_y_m','driver_in_vehicle'):
        assert current.facts[field]==old.facts[field]
    assert score(method,old)['controllability']['controllability_score']=='C3'
    assert old==before


def test_prompt_and_exact_keys_use_one_current_view_without_changing_physical_request(method,monkeypatch):
    old=longitudinal_world(method,stale=True);same=longitudinal_world(method,surface='REAR_END')
    original=deepcopy((old,same));physical=Producer.physical_conditions(old)
    agent=SimpleNamespace(client=SimpleNamespace(),prompt_version='frozen-causal',assessment_contract_version=SCENARIO_ASSESSMENT_CONTRACT_VERSION)
    with monkeypatch.context() as patch:
        # A saved pre-repair cache used the source inputs verbatim. Keep its
        # actual native key construction; remove only the newly added view.
        patch.setattr(equivalence,'scenario_with_current_physics',lambda value:value)
        patch.setattr(scenario_node,'scenario_with_current_physics',lambda value:value)
        prior_keys={c.scenario_id:equivalence.exact_task_key(agent,mf(),c) for c in (old,same)}
        prior_batch={c.scenario_id:scenario_node._scenario_batch_cache_key(agent,mf(),[c]) for c in (old,same)}
    assert equivalence.exact_task_key(agent,mf(),old)!=prior_keys[old.scenario_id]
    assert scenario_node._scenario_batch_cache_key(agent,mf(),[old])!=prior_batch[old.scenario_id]
    assert equivalence.exact_task_key(agent,mf(),same)==prior_keys[same.scenario_id]
    assert scenario_node._scenario_batch_cache_key(agent,mf(),[same])==prior_batch[same.scenario_id]
    current=equivalence.scenario_with_current_physics(old)
    assert equivalence.exact_task_key(agent,mf(),current)==equivalence.exact_task_key(agent,mf(),old)
    payload=json.loads(_scenario_json(mf(),old,causal_evidence_budget=40))
    assert payload['physical_interaction']['conditions']['motion_model_outcome']['longitudinal_contact'] is True
    assert payload['facts']['motion_model_outcome']['longitudinal_contact'] is True
    assert '"longitudinal_contact": false' not in payload['causal_evidence_view']
    assert Producer.physical_conditions(old)==physical and (old,same)==original


def test_old_causal_checkpoint_rechecks_only_changed_axis_and_resumes_without_request(method,monkeypatch):
    old=longitudinal_world(method,stale=True);same=longitudinal_world(method,surface='REAR_END')
    class Agent:
        client=SimpleNamespace()
        prompt_version=ScenarioFeasibilityAgent.PROMPT_VERSION
        assessment_contract_version=SCENARIO_ASSESSMENT_CONTRACT_VERSION
        calls=[]
        def assess(self,malfunction,candidates,project_registry=None):
            self.calls.extend(c.scenario_id for c in candidates)
            return [ScenarioFeasibilityAgent._parse(malfunction,claim(c),scenario=c) for c in candidates],{'llm_calls':1}
    agent=Agent()
    with monkeypatch.context() as patch:
        patch.setattr(equivalence,'scenario_with_current_physics',lambda value:value)
        saved_keys={c.scenario_id:equivalence.exact_task_key(agent,mf(),c) for c in (old,same)}
    prior={'assessments':[claim(old,old=True),claim(same)],'audit':{'causal_reuse':{'exact_mapping':[
        {'task_key':saved_keys[c.scenario_id],'scenario_ids':[c.scenario_id]} for c in (old,same)]}}}
    original=deepcopy(prior)
    rows,audit=scenario_node._assess_instantiated_scenarios(agent,mf(),[old,same],{},None,prior)
    assert agent.calls==[old.scenario_id]
    assert audit['causal_reuse']['exact_task_checkpoint_hits']==1 and prior==original
    assert next(r for r in rows if r.scenario_id==old.scenario_id).causal_chain['b_to_i']['claim'].startswith('longitudinal_contact=true')
    restored={'assessments':[r.to_dict() for r in rows],'audit':audit}
    rows2,audit2=scenario_node._assess_instantiated_scenarios(agent,mf(),[old,same],{},None,restored)
    assert agent.calls==[old.scenario_id] and audit2['causal_reuse']['exact_task_checkpoint_hits']==2
    assert [r.to_dict() for r in rows2]==[r.to_dict() for r in rows]


@pytest.mark.parametrize('authority',['PROJECT_INPUT','HUMAN_CONFIRMATION','FOREIGN_DERIVED_SCOPE'])
def test_direct_or_foreign_cached_evidence_is_not_overwritten(method,authority):
    child=longitudinal_world(method,stale=True);meta=child.fact_provenance['motion_model_outcome']
    if authority=='FOREIGN_DERIVED_SCOPE':
        meta['applicable_scope']={'malfunction_id':'MF','scenario_id':'OTHER'}
    else:meta.update(provenance=authority,approval='FINALIZED')
    before=deepcopy(child)
    with pytest.raises(ValueError,match='conflicts with deterministic physics'):
        equivalence.scenario_with_current_physics(child)
    assert child==before


@pytest.mark.parametrize('authority',['PROJECT_INPUT','HUMAN_CONFIRMATION','LLM_INFERENCE'])
def test_equal_direct_or_assumed_contact_time_keeps_its_authority_during_axis_refresh(method,authority):
    child=longitudinal_world(method,stale=True)
    meta=deepcopy(child.fact_provenance['first_contact_time_s'])
    meta.update(provenance=authority,approval='FINALIZED' if authority!='LLM_INFERENCE' else 'PENDING',
        source_binding_kind='preserved explicit time input')
    child.fact_provenance['first_contact_time_s']=deepcopy(meta)
    current=equivalence.scenario_with_current_physics(child)
    assert current.facts['ttc_s']==child.facts['first_contact_time_s']
    assert current.fact_provenance['first_contact_time_s']==meta
    assert child.fact_provenance['first_contact_time_s']==meta


def test_real_crossing_heading_remains_two_dimensional(method):
    raw=moving_plan(initial=3.6,target_direction='RIGHT',target_speed=3.6,x=4,y=4,position='LEFT',collision='SIDE')
    child=instantiate(method,raw,make_candidate(3.6))[0]
    assert rectilinear_moving_contact(child.facts)['longitudinal_contact'] is False
    assert 'ttc_s' not in child.facts
    assert equivalence.scenario_with_current_physics(child) is child


def test_added_curve_coverage_diagnostic_preserves_existing_task_identity(method):
    from test_curved_steering_physics import curved_plan
    from hara_agent.services.analysis.hazardous_event_risk_context_service import HazardousEventRiskContextService
    child=instantiate(method,curved_plan(),make_candidate(7))[0]
    child.facts['motion_model_outcome'].pop('target_longitudinal_coverage',None)
    old=deepcopy(child)
    assert equivalence.scenario_with_current_physics(child) is child
    assert child==old
    # Other new business/diagnostic fields have no compatibility exception.
    current={**old.facts['motion_model_outcome'],'contact_axis':'LONGITUDINAL'}
    assert HazardousEventRiskContextService._retained_physics_value_changed(old.facts['motion_model_outcome'],current,field='motion_model_outcome')
    current=deepcopy(old.facts['motion_model_outcome'])
    current['time_s']+=.001
    assert HazardousEventRiskContextService._retained_physics_value_changed(old.facts['motion_model_outcome'],current,field='motion_model_outcome')


@pytest.mark.parametrize('kind',['legacy_table','missing_retained_outcome','other_motion_model'])
def test_current_contact_refresh_keeps_other_input_boundaries(method,kind):
    child=longitudinal_world(method,stale=True)
    if kind=='legacy_table':child.facts.pop('fault_motion_model')
    elif kind=='missing_retained_outcome':child.facts.pop('motion_model_outcome')
    else:child.facts['fault_motion_model']='CONSTANT_ACCELERATION_TO_IMPACT_DISTANCE'
    # A direct table input is consumed by its established source/branch
    # validators, not converted into a new physics derivation by this view.
    child.fact_provenance['relative_speed_kph']={'provenance':'PROJECT_INPUT','approval':'FINALIZED',
        'source_refs':[{'location':'retained direct scoring input'}]}
    before=deepcopy(child)
    assert equivalence.scenario_with_current_physics(child) is child
    assert child==before


def test_independent_context_audit_keeps_equal_curve_source_when_only_diagnostic_was_added(method):
    from dataclasses import asdict
    from test_curved_steering_physics import curved_plan
    from hara_agent.services.analysis.hazardous_event_risk_context_service import HazardousEventRiskContextService
    child=instantiate(method,curved_plan(),make_candidate(7))[0]
    child.facts['motion_model_outcome'].pop('target_longitudinal_coverage',None)
    child.fact_provenance['motion_model_outcome'].update(provenance='PROJECT_INPUT',approval='FINALIZED')
    before=deepcopy(child)
    contexts=HazardousEventRiskContextService(method)
    view=contexts._scenario_with_derived_physics(asdict(child))
    assert view['motion_model_outcome']==child.facts['motion_model_outcome']
    assert view['_fact_provenance']['motion_model_outcome']==child.fact_provenance['motion_model_outcome']
    result=contexts.audit({'scenario_candidate':[asdict(child)],'scenario_feasibility':[{
        'scenario_id':child.scenario_id,'malfunction_id':'MF','status':'FINALIZED',
        'physically_feasible':True,'functionally_relevant':True,'causally_relevant':True}]})
    assert result['summary']['causal_relevant_hazardous_events']==1
    assert child==before
