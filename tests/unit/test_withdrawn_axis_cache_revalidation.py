"""Current derived axis/TTC persistence must not erase a proven unsafe response."""
from copy import deepcopy
from pathlib import Path
import gzip,json
import pytest
from hara_agent.services.analysis.deterministic_risk_executor import FTTIMethodExecutor
from hara_agent.services.analysis.risk_calculation_input_service import RiskCalculationInputService
from hara_agent.services.analysis.scenario_equivalence import scenario_with_current_physics
from test_reuse_reporting_acceptance import active_method
from test_report_current_physics_view import historical_surface_child

CASES=json.loads(gzip.decompress((Path(__file__).resolve().parents[1]/'fixtures/withdrawn_axis_cache.json.gz').read_bytes()))['rows']

def evaluate(method,scenario):
    return RiskCalculationInputService().ftti(scenario.get('malfunction_id','MF-F2-002'),scenario['scenario_id'],'',
        {**scenario['facts'],'_fact_provenance':scenario['fact_provenance']},method=method)

@pytest.mark.parametrize('case',[c for c in CASES if c['expected_physical_status']],ids=lambda c:c['scenario_id'])
def test_real_withdrawn_negative_survives_persisted_current_axis_cache(active_method,case):
    before=deepcopy(case);old=evaluate(active_method,{**case['original'],'malfunction_id':case['malfunction_id']});current=evaluate(active_method,{**case['current'],'malfunction_id':case['malfunction_id']})
    assert old.timing_trace['status']==current.timing_trace['status']=='AVOIDABILITY_ALREADY_LOST'
    assert old.status.value==current.status.value=='NOT_APPLICABLE'
    assert old.hazard_time_limit_s is current.hazard_time_limit_s is None
    assert old.safe_state_transition_time_s==current.safe_state_transition_time_s
    assert old.timing_trace['time_s']==current.timing_trace['time_s']==0
    assert old.timing_trace['stopping_distance_m']==current.timing_trace['stopping_distance_m']
    replay=current.timing_trace['rejected_response_revalidation']
    assert replay['snapshot_revalidation']=={'basis':'RECOMPUTED_CURRENT_DERIVED_LONGITUDINAL_VIEW',
        'changed_fields':['motion_model_outcome.longitudinal_contact','ttc_s'],'frozen_trajectory_recomputed_equal':True,'response_evaluations_equal':True}
    assert replay['proposal_remains_withdrawn'] is True
    assert 'safe_state_proposal' not in case['current']['facts']['physical_analysis']['hypothesis']
    assert case==before

@pytest.mark.parametrize('change',['object','distance','horizon','contact_time','controls','failed','source_scope','source_kind','source_approval','derivation_inputs','source_input_scope'])
def test_axis_cache_revalidation_does_not_accept_world_control_or_authority_changes(active_method,change):
    child=deepcopy(CASES[0]['current']);facts=child['facts'];meta=child['fact_provenance'];h=facts['physical_analysis']['hypothesis']
    if change=='object':facts['object_id']='another-object'
    elif change=='distance':facts['object_offset_x_m']+=.01
    elif change=='horizon':facts['model_valid_until_s']+=1
    elif change=='contact_time':facts['motion_model_outcome']['time_s']+=.01
    elif change=='controls':h['remaining_controls'].remove('braking')
    elif change=='failed':h['failed_controls'].append('braking')
    elif change=='source_scope':meta['physical_analysis']['applicable_scope']['scenario_id']='another-scenario'
    elif change=='source_kind':meta['motion_model_outcome']['provenance']='PROJECT_INPUT'
    elif change=='source_approval':meta['motion_model_outcome']['approval']='REJECTED'
    elif change=='derivation_inputs':meta['motion_model_outcome']['inputs']=['SCN.unrelated']
    else:meta['object_offset_x_m']['applicable_scope']['scenario_id']='another-scenario'
    before=deepcopy(child);result=evaluate(active_method,child)
    assert result.status.value=='PENDING_INPUT' and result.hazard_time_limit_s is None
    assert result.timing_trace.get('status') not in {'AVOIDABILITY_ALREADY_LOST','NO_IMMEDIATE_SAFE_BRAKING_WINDOW'}
    if change in {'controls','failed'}:
        replay=result.timing_trace['rejected_response_revalidation']
        assert replay['current_evaluation']['timing_trace']['pending_reason']=='CONTROL_PATH_UNRESOLVED'
        assert replay['previous_timing_trace']['status']=='AVOIDABILITY_ALREADY_LOST'
    else:assert 'rejected_response_revalidation' not in result.timing_trace
    assert child==before


def test_current_axis_recalculation_positive_does_not_revive_withdrawn_proposal(active_method):
    child=historical_surface_child(active_method);analysis=child.facts['physical_analysis'];proposal=deepcopy(analysis['hypothesis']['safe_state_proposal']);historical_hypothesis=deepcopy(analysis['hypothesis']);analysis['hypothesis'].pop('safe_state_proposal')
    analysis['rejected_response_evaluation']={'proposal':proposal,
        'input_values':FTTIMethodExecutor.response_input_snapshot(child.facts),'method_source_hash':'historical-method',
        'timing_trace':{'status':'NO_IMMEDIATE_SAFE_BRAKING_WINDOW','hypothesis':historical_hypothesis}}
    current=scenario_with_current_physics(child);before=deepcopy(current)
    result=RiskCalculationInputService().ftti('MF',current.scenario_id,'',
        {**current.facts,'_fact_provenance':current.fact_provenance},method=active_method)
    assert result.status.value=='PENDING_INPUT' and result.hazard_time_limit_s is None
    replay=result.timing_trace['rejected_response_revalidation'];assert replay['current_evaluation']['timing_trace']['status']=='CONDITIONAL_BOUNDARY'
    assert replay['previous_timing_trace']['status']=='NO_IMMEDIATE_SAFE_BRAKING_WINDOW' and replay['proposal_remains_withdrawn'] is True
    assert 'safe_state_proposal' not in current.facts['physical_analysis']['hypothesis'] and current==before


def test_real_current_command_gap_stays_pending_after_axis_revalidation(active_method):
    case=next(c for c in CASES if c['expected_physical_status'] is None)
    child={**deepcopy(case['current']),'malfunction_id':case['malfunction_id']};before=deepcopy(child)
    result=evaluate(active_method,child)
    assert result.status.value=='PENDING_INPUT' and result.hazard_time_limit_s is None
    replay=result.timing_trace['rejected_response_revalidation']
    assert replay['previous_timing_trace']['status']=='NO_IMMEDIATE_SAFE_BRAKING_WINDOW'
    assert replay['snapshot_revalidation']['response_evaluations_equal'] is True
    actual=replay['current_evaluation'];assert actual['timing_trace']['pending_reason']=='CONTROL_PATH_UNRESOLVED'
    assert actual['timing_trace']['missing_required_controls']==['automatic_drive_command','automatic_steering_command']
    assert replay['proposal_remains_withdrawn'] is True and child==before


def test_unused_response_prose_and_basis_order_do_not_change_physics_or_source_authority(active_method):
    child=deepcopy(CASES[0]['current']);h=child['facts']['physical_analysis']['hypothesis']
    h['rationale'] += '\nClarifies the same withdrawn response.'
    h['basis_refs']=list(reversed(h['basis_refs']))+['UNUSED_EXPLANATORY_REFERENCE']
    h['remaining_controls']=list(reversed(h['remaining_controls']))
    before=deepcopy(child);result=evaluate(active_method,child)
    assert result.status.value=='NOT_APPLICABLE' and result.timing_trace['status']=='AVOIDABILITY_ALREADY_LOST'
    assert result.timing_trace['rejected_response_revalidation']['snapshot_revalidation']['response_evaluations_equal'] is True
    assert child==before


def test_changed_used_input_source_reference_is_not_just_a_display_edit(active_method):
    child=deepcopy(CASES[0]['current'])
    child['fact_provenance']['object_offset_x_m']['source_refs'][0]['source_id']='DIFFERENT_USED_SOURCE'
    before=deepcopy(child);result=evaluate(active_method,child)
    assert result.status.value=='PENDING_INPUT' and result.hazard_time_limit_s is None
    assert 'rejected_response_revalidation' not in result.timing_trace
    assert child==before
