"""Current calculated display cache and untouched response snapshots have distinct duties."""
from copy import deepcopy
from dataclasses import replace
import pytest
from hara_agent.models import EvidenceValue,ReviewStatus,RiskAssessment
from hara_agent.workflow import HARAState
from hara_agent.workflow.nodes.scoring import _scoring_evidence
from hara_agent.services.analysis.deterministic_risk_executor import FTTIMethodExecutor
from hara_agent.services.analysis.risk_calculation_input_service import RiskCalculationInputService
from hara_agent.services.reporting import HARAReportProjectionService,load_report_schema
from test_autonomous_physics import instantiate,make_candidate,score,mf
from test_bounded_interaction_coverage import moving_plan
from test_response_evidence_reuse import response_plan
from test_reuse_reporting_acceptance import active_method


def historical_surface_child(method, *, rear=False):
    raw=response_plan(moving_plan(target_direction='FORWARD' if rear else 'STATIONARY',
        target_speed=7.2 if rear else 0, x=-8 if rear else 4,
        position='REAR' if rear else 'FRONT',collision='SIDE'))
    child=instantiate(method,raw,make_candidate(3.6))[0]
    assert child.facts['motion_model_outcome']['longitudinal_contact'] is True
    child.facts['motion_model_outcome']['longitudinal_contact']=False
    child.facts.pop('ttc_s',None);child.fact_provenance.pop('ttc_s',None)
    return child


def report_state(method,child):
    scores=score(method,child)
    risk=RiskAssessment('ASSESSMENT',child.scenario_id,
        _scoring_evidence(scores['severity'],'severity_score'),
        _scoring_evidence(scores['exposure'],'exposure_score'),
        _scoring_evidence(scores['controllability'],'controllability_score'),
        EvidenceValue('',ReviewStatus.PENDING),malfunction_id='MF',
        hazardous_event='车辆与同一对象发生接触。',
        branch_evaluations=deepcopy(scores['branch_evaluations']),branch_summary=deepcopy(scores['branch_summary']))
    return HARAState('current-physics-display',functions=[{'function_id':'F','name':'车辆控制','output':'控制请求'}],
        malfunctions=[mf().to_dict()],scenarios=[child],risk_results=[risk])


def source_timing(method,child,branches):
    return RiskCalculationInputService().ftti('MF',child.scenario_id,'',
        {**child.facts,'_fact_provenance':child.fact_provenance},branches,method=method).to_dict()


def test_current_longitudinal_TTC_display_does_not_rewrite_retained_side_world(active_method):
    child=historical_surface_child(active_method);state=report_state(active_method,child);before=deepcopy(state)
    expected=source_timing(active_method,child,state.risk_results[0].branch_evaluations)
    view=HARAReportProjectionService(load_report_schema()).project(state,active_method)
    detail=view.scenario_details[0]
    assert 'TTC：3 s' in detail.physical_inputs and '不作为纵向 TTC' not in detail.physical_inputs
    assert detail.controllability=='C3' and '侧向碰撞' in detail.branch_results
    assert view.projection_metrics['ftti_timing_inputs'][0]==expected
    assert child.facts['collision_type']=='SIDE' and child.facts['motion_model_outcome']['longitudinal_contact'] is False
    assert 'ttc_s' not in child.facts and state==before


def test_current_display_does_not_invalidate_withdrawn_response_snapshot(active_method):
    child=historical_surface_child(active_method,rear=True)
    checked=source_timing(active_method,child,[])
    assert checked['timing_trace']['status']=='NO_IMMEDIATE_SAFE_BRAKING_WINDOW'
    analysis=child.facts['physical_analysis'];proposal=analysis['hypothesis'].pop('safe_state_proposal')
    analysis['rejected_response_evaluation']={
        'proposal':deepcopy(proposal),'input_values':FTTIMethodExecutor.response_input_snapshot(child.facts),
        'timing_trace':deepcopy(checked['timing_trace']),'method_source_hash':active_method.metadata['method_source_hash']}
    state=report_state(active_method,child);before=deepcopy(state)
    expected=source_timing(active_method,child,state.risk_results[0].branch_evaluations)
    assert expected['timing_trace']['status']=='NO_IMMEDIATE_SAFE_BRAKING_WINDOW'
    view=HARAReportProjectionService(load_report_schema()).project(state,active_method)
    timing=view.projection_metrics['ftti_timing_inputs'][0]
    assert timing==expected and timing['hazard_time_limit_s'] is None
    assert timing['timing_trace']['rejected_response_revalidation']['proposal_remains_withdrawn'] is True
    assert '不作为纵向 TTC' not in view.scenario_details[0].physical_inputs
    assert '所选响应不能避碰' in view.rows[0].ftti and state==before


def test_actual_crossing_keeps_first_contact_separate_from_longitudinal_TTC(active_method):
    raw=response_plan(moving_plan(target_direction='RIGHT',x=4,y=4,position='LEFT',collision='SIDE'))
    child=instantiate(active_method,raw,make_candidate(3.6))[0]
    assert child.facts['motion_model_outcome']['longitudinal_contact'] is False
    state=report_state(active_method,child);before=deepcopy(state)
    view=HARAReportProjectionService(load_report_schema()).project(state,active_method)
    assert '不作为纵向 TTC' in view.scenario_details[0].physical_inputs
    assert 'ttc_s' not in child.facts and state==before


def test_project_input_conflict_is_not_hidden_by_report_refresh(active_method):
    child=historical_surface_child(active_method);state=report_state(active_method,child)
    child.fact_provenance['motion_model_outcome']['provenance']='PROJECT_INPUT'
    before=deepcopy(state)
    with pytest.raises(ValueError,match='conflicts with deterministic physics'):
        HARAReportProjectionService(load_report_schema()).project(state,active_method)
    assert state==before


@pytest.mark.parametrize('kind',['curve','lateral_acceleration','angled'])
def test_non_longitudinal_models_do_not_gain_TTC_through_display(active_method,kind):
    if kind=='curve':
        from test_curved_steering_physics import curved_plan
        raw=response_plan(curved_plan(speed=3.6,radius=10,clearance=.5))
    elif kind=='lateral_acceleration':
        from test_autonomous_physics import raw_plan
        raw=response_plan(raw_plan('CONSTANT_LATERAL_ACCELERATION_TO_CONTACT'))
        raw['hypotheses'][0]['remaining_controls']=['driver_braking']
    else:
        raw=response_plan(moving_plan(target_speed=0,target_direction='STATIONARY',x=3,y=3,
            position='LEFT',collision='SIDE'))
        raw['hypotheses'][0]['parameters'].append({'field':'ego_heading_angle_deg','value':45,
            'kind':'POINT','unit':'deg','rationale':'explicit angled ray'})
    child=instantiate(active_method,raw,make_candidate(3.6))[0]
    assert 'ttc_s' not in child.facts and child.facts.get('first_contact_time_s') is not None
    state=report_state(active_method,child);before=deepcopy(state)
    expected=source_timing(active_method,child,state.risk_results[0].branch_evaluations)
    view=HARAReportProjectionService(load_report_schema()).project(state,active_method)
    assert '不作为纵向 TTC' in view.scenario_details[0].physical_inputs
    assert view.projection_metrics['ftti_timing_inputs'][0]==expected and state==before
