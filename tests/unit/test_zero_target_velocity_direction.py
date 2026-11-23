"""A constant zero target velocity keeps its raw, redundant heading."""
from copy import deepcopy
import pytest
from hara_agent.services.analysis import scenario_physics as physics
from test_autonomous_physics import method,instantiate,make_candidate
from test_bounded_interaction_coverage import moving_plan


def world(method):
    return instantiate(method,moving_plan(target_speed=0,target_direction='FORWARD',x=8),make_candidate(3.6))[0]


def test_constant_zero_target_is_proven_without_editing_raw_direction(method):
    child=world(method);before=deepcopy(child)
    assert physics.target_is_stationary_in_model(child,malfunction_id='MF')
    assert child.facts['object_longitudinal_direction']=='FORWARD' and child==before


@pytest.mark.parametrize('difference',['symbolic','foreign_scope','impact_phase','other_object','other_model','invalid_target_phase'])
def test_zero_initial_scalar_does_not_authorize_another_world_or_phase(method,difference):
    child=world(method)
    if difference=='symbolic':child.facts['object_speed_kph']='u0'
    elif difference=='foreign_scope':child.fact_provenance['object_speed_kph']['applicable_scope']['scenario_id']='another'
    elif difference=='impact_phase':child.fact_provenance['object_speed_kph']['analysis_value_semantic']='IMPACT_SPEED'
    elif difference=='other_object':child.fact_provenance['object_speed_kph']['object_id']='another'
    elif difference=='other_model':child.facts['fault_motion_model']='FUTURE_TARGET_ACCELERATION'
    else:child.facts.update(object_reaction_time_s=1,object_braking_deceleration_mps2=1)
    assert not physics.target_is_stationary_in_model(child,malfunction_id='MF')


def test_entire_target_trajectory_is_checked_not_only_initial_speed(method,monkeypatch):
    child=world(method);original=physics._straight_world
    def later_motion(facts):
        ego,target,ev,ov,end,stop=original(facts)
        return ego,[(0,1,0,0,0),(1,2,0,1,0)],ev,ov,end,stop
    monkeypatch.setattr(physics,'_straight_world',later_motion)
    assert not physics.target_is_stationary_in_model(child,malfunction_id='MF')


def test_actual_zero_target_worlds_restore_node_and_direct_scores_without_source_changes(method):
    import gzip,json
    from pathlib import Path
    from hara_agent.workflow import HARAState
    from hara_agent.workflow.risk_rescoring import clean_risk_stage
    from hara_agent.workflow.nodes.scoring import score_structured_scenarios
    from hara_agent.services.analysis.risk_services import RiskScoringServices
    from hara_agent.services.analysis.structured_risk_scoring_service import StructuredRiskScoringService
    from hara_agent.services.analysis.exposure_dimension_coverage_service import ExposureDimensionCoverageService
    raw=json.loads(gzip.decompress((Path(__file__).parents[1]/'fixtures/zero_target_velocity_source.json.gz').read_bytes()))
    state=HARAState.read_committed(raw);before={c.scenario_id:deepcopy(c) for c in state.scenarios}
    prior={r.scenario_id:r for r in state.risk_results};clean_risk_stage(state)
    services=RiskScoringServices.from_method(method);score_structured_scenarios(state,services.scoring,services.asil,services.binding)
    coverage=ExposureDimensionCoverageService(method)
    for candidate in state.scenarios:
        risk=next(r for r in state.risk_results if r.scenario_id==candidate.scenario_id)
        assert not prior[candidate.scenario_id].severity.value and not prior[candidate.scenario_id].controllability.value
        assert risk.severity.value and risk.controllability.value and risk.asil.value
        assert risk.exposure.value==prior[candidate.scenario_id].exposure.value
        assert candidate==before[candidate.scenario_id]
        mf=next(m for m in state.malfunctions if m['malfunction_id']==risk.malfunction_id)
        function=next(f for f in state.functions if f['function_id']==mf['function_id'])
        view={**candidate.facts,'scenario_id':candidate.scenario_id,'malfunction_id':risk.malfunction_id,
            'operating_mode':candidate.operating_mode,'component_category':mf['component_category'],
            'failure_type':mf['failure_type'],'_function_output':function['output'],'_fact_provenance':candidate.fact_provenance,
            '_exposure_dimension_coverage_decision':coverage.decide(assessment_key=risk.assessment_id,
                function=coverage.function_from_projection(function),operating_mode=candidate.operating_mode)}
        direct=StructuredRiskScoringService(method).score(view,risk.hazardous_event)
        assert direct['severity']['severity_score']==risk.severity.value
        assert direct['controllability']['controllability_score']==risk.controllability.value


def test_zero_right_heading_uses_displacement_axis_and_longitudinal_profile(method):
    from hara_agent.services.analysis.scenario_equivalence import scenario_with_current_physics
    from test_autonomous_physics import score
    raw=moving_plan(target_speed=0,target_direction='RIGHT',x=8)
    child=instantiate(method,raw,make_candidate(3.6))[0];before=deepcopy(child)
    outcome=physics.rectilinear_moving_contact(child.facts)
    assert outcome['status']=='CONTACT' and outcome['longitudinal_contact'] is True
    assert child.facts['ttc_s']==outcome['time_s']
    result=score(method,child)['controllability']
    assert result['controllability_score']
    assert result.get('reference_rule',{}).get('rule_id')!='IAV_CROSSING_CONTACT_BRAKING_TIME_TRANSFER'
    assert child==before and child.facts['object_longitudinal_direction']=='RIGHT'
    assert scenario_with_current_physics(child) is child


def test_actual_zero_right_target_preserves_source_and_rebuilds_only_axis_ttc(method):
    import gzip,json
    from pathlib import Path
    from hara_agent.workflow import HARAState
    from hara_agent.services.analysis.scenario_equivalence import scenario_with_current_physics
    from hara_agent.services.analysis.structured_risk_scoring_service import StructuredRiskScoringService
    from hara_agent.services.analysis.exposure_dimension_coverage_service import ExposureDimensionCoverageService
    raw=json.loads(gzip.decompress((Path(__file__).parents[1]/'fixtures/zero_target_right_source.json.gz').read_bytes()))
    child=HARAState._scenario_from_dict(raw);before=deepcopy(child)
    mf=child.analysis_instance['malfunction_id']
    assert physics.target_is_stationary_in_model(child,malfunction_id=mf)
    current=scenario_with_current_physics(child)
    changed={field for field in child.facts.keys()|current.facts.keys()
        if child.facts.get(field)!=current.facts.get(field)}
    assert changed=={'motion_model_outcome','ttc_s'}
    assert current.facts['motion_model_outcome']=={**child.facts['motion_model_outcome'],'longitudinal_contact':True}
    assert current.facts['ttc_s']==child.facts['first_contact_time_s']
    view={**current.facts,'scenario_id':child.scenario_id,'malfunction_id':mf,
        '_fact_provenance':current.fact_provenance,
        '_exposure_dimension_coverage_decision':ExposureDimensionCoverageService(method).decide(
            assessment_key='actual-zero-target',function=None,operating_mode=child.operating_mode)}
    result=StructuredRiskScoringService(method).score(view,'same stationary target contact')
    assert result['severity']['severity_score'] and result['controllability']['controllability_score']
    assert result['controllability'].get('reference_rule',{}).get('rule_id')!='IAV_CROSSING_CONTACT_BRAKING_TIME_TRANSFER'
    assert child==before and child.facts['object_longitudinal_direction']=='RIGHT'


def test_initial_zero_ego_later_lateral_acceleration_is_not_longitudinal(method):
    raw=moving_plan(initial=0,acceleration=1,target_speed=0,target_direction='RIGHT',x=3,y=1)
    raw['hypotheses'][0]['parameters'].append(dict(field='ego_heading_angle_deg',unit='deg',kind='POINT',value=30,rationale='explicit fixed velocity heading'))
    child=instantiate(method,raw,make_candidate(0))[0]
    outcome=physics.rectilinear_moving_contact(child.facts)
    assert outcome['status']=='CONTACT' and outcome['longitudinal_contact'] is False


def test_later_lateral_target_segment_cannot_be_hidden_by_its_initial_zero(method):
    child=world(method);facts={**child.facts,'object_offset_x_m':4.1,'contact_half_length_m':4,'object_offset_y_m':0}
    ego=[(0,5,0,1,0)];target=[(0,1,0,0,0),(1,5,0,1,0)]
    outcome=physics._rectilinear_contact_from_segments(facts,ego,target,(1,0),(0,-1),5,5)
    assert outcome['status']=='CONTACT' and outcome['longitudinal_contact'] is False
