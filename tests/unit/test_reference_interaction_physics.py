"""IAV front/rear/side counterexamples through the single production kernel."""
from dataclasses import asdict
import json
import math

import pytest
from test_p7_closure import method
from test_autonomous_physics import instantiate, make_candidate, mf, score
from test_moving_response_timing import following, proposal
from test_geometry_completion import Client, run
from hara_agent.contracts.physical_scenario_plan import PARAMETER_UNITS, PhysicalScenarioPlan
from hara_agent.services.analysis.scenario_physics import (
    rectilinear_moving_contact, moving_braking_avoidability_boundary, branch_physics_view,
)
from hara_agent.services.analysis.risk_calculation_input_service import RiskCalculationInputService
from hara_agent.services.reporting.engineering_text_mapper import EngineeringReportTextMapper
from hara_agent.services.semantic.physical_scenario_agent import PhysicalScenarioAgent
from hara_agent.workflow import HARAState


def parameters(raw, **values):
    params = raw['hypotheses'][0]['parameters']
    for field, value in values.items():
        current = next((p for p in params if p['field'] == field), None)
        row = dict(field=field, value=value, unit=PARAMETER_UNITS[field], kind='POINT',
                   rationale='Explicit reference-example hypothesis, not measured vehicle capability')
        if current is None:
            params.append(row)
        else:
            current.clear(); current.update(row)
    return raw


def test_front_reference_first_contact_not_evaluation_time_or_response_deadline():
    # Appendix 1-001: workbook uses rounded .155 s; exact contact is .3/(7/3.6).
    f = {**following(), 'ego_speed_kph':7, 'object_speed_kph':0, 'object_offset_x_m':1.3}
    result = rectilinear_moving_contact(f)
    assert result['time_s'] == pytest.approx(.3/(7/3.6))
    assert result['relative_speed_kph'] == pytest.approx(7)
    # Appendix 1-007: .5 s acceleration, then brake. Contact happens BEFORE
    # the entered .598 s; its entered 9.9664 km/h is not first-impact speed.
    f.update(longitudinal_acceleration_mps2=4, object_offset_x_m=2.75)
    result = rectilinear_moving_contact(f, response_delay_s=.5, response_deceleration_mps2=12)
    onset=7/3.6+4*.5
    remaining=1.75-(7/3.6*.5+4*.5**2/2)
    expected_speed=math.sqrt(onset**2-2*12*remaining)
    assert result['time_s'] == pytest.approx(.5+(onset-expected_speed)/12)
    assert result['time_s'] < .598
    assert result['relative_speed_kph'] == pytest.approx(expected_speed*3.6)
    assert result['relative_speed_kph'] > 9.9664
    # Appendix 1-005: same-speed front vehicle; response avoids contact.
    f['object_speed_kph']=7
    assert rectilinear_moving_contact(f, response_delay_s=.5,
        response_deceleration_mps2=12)['status'] == 'NO_CONTACT_WITHIN_MODEL'


def test_rear_reference_other_vehicle_brakes_and_holds_without_reversing():
    f={**following(), 'ego_speed_kph':7, 'longitudinal_acceleration_mps2':-2,
       'object_speed_kph':10, 'object_offset_x_m':-4, 'object_position':'REAR',
       'collision_type':'REAR_END', 'post_stop_motion':'HOLD_POSITION',
       'object_reaction_time_s':.95, 'object_braking_deceleration_mps2':10}
    assert rectilinear_moving_contact(f)['status']=='NO_CONTACT_WITHIN_MODEL'
    original={k:v for k,v in f.items() if not k.startswith('object_reaction') and k!='object_braking_deceleration_mps2'}
    assert rectilinear_moving_contact(original)['status']=='CONTACT'
    # Delayed target response still collides; it is not unconditional evasion.
    f['object_reaction_time_s']=3
    hit=rectilinear_moving_contact(f)
    assert hit['status']=='CONTACT' and hit['object_impact_speed_kph']==10
    assert hit['ego_impact_speed_kph']==0
    assert hit['contact_phase']=='HELD_AFTER_STOP'


def test_target_response_is_on_the_same_clock_as_ego_ftti():
    f={**following(),'object_reaction_time_s':1,'object_braking_deceleration_mps2':1}
    result=moving_braking_avoidability_boundary(f,2)
    # Target ends at x=6.5. Ego must stop before x=5.5; stopping travel=1 m.
    assert result['time_s']==pytest.approx(2.25)
    assert result['target_response_time_origin']=='MODEL_INITIAL_STATE'
    assert result['response_delay_interval_proven']
    for delta, status in [(-1e-5,'NO_CONTACT_WITHIN_MODEL'),(1e-5,'CONTACT')]:
        check=rectilinear_moving_contact(f,response_delay_s=result['time_s']+delta,response_deceleration_mps2=2)
        assert check['status']==status


def test_fixed_heading_moving_response_boundary_uses_the_same_2d_world():
    # Rotate a following relation by 40 degrees. Projected envelope extents
    # define the same one-metre path clearance; no steering dynamics implied.
    x,y=math.cos(math.radians(40)),math.sin(math.radians(40))
    f={**following(),'ego_heading_angle_deg':40,'object_heading_angle_deg':40,
       'object_offset_x_m':5*x,'object_offset_y_m':5*y,
       'contact_half_length_m':x,'contact_half_width_m':y,
       'object_reaction_time_s':1,'object_braking_deceleration_mps2':1}
    result=moving_braking_avoidability_boundary(f,2)
    assert result['time_s']==pytest.approx(2.25)
    assert result['response_delay_interval_proven']
    for delta,status in [(-1e-5,'NO_CONTACT_WITHIN_MODEL'),(1e-5,'CONTACT')]:
        assert rectilinear_moving_contact(f,response_delay_s=result['time_s']+delta,
            response_deceleration_mps2=2)['status']==status


def angle_plan():
    raw=proposal(); h=raw['hypotheses'][0]
    h['geometry'].update(object_position='LEFT',collision_type='SIDE',object_longitudinal_direction='STATIONARY')
    parameters(raw,ego_speed_kph=7,object_speed_kph=0,ego_heading_angle_deg=40,
        object_offset_x_m=0,object_offset_y_m=.6,contact_half_length_m=2,contact_half_width_m=.5)
    h['conditions']=['Fixed velocity heading 40 degrees; stationary side envelope; independent braking remains effective']
    h['uncovered_conditions']=['Changing heading, wheel steering and target evasion are not assumed']
    return raw


def test_side_reference_angle_is_velocity_heading_not_wheel_angle(method):
    child=instantiate(method,angle_plan(),make_candidate(7))[0]
    speed=7/3.6*math.sin(math.radians(40))
    assert child.facts['first_contact_time_s']==pytest.approx(.1/speed)
    assert child.facts['contact_normal_speed_kph']==pytest.approx(4.499513267805774)
    assert child.facts['relative_speed_kph']==pytest.approx(7)  # unchanged active method meaning
    assert 'ttc_s' not in child.facts
    assert score(method,child)['severity']['severity_score']
    assert child.fact_provenance['ego_heading_angle_deg']['approval']=='PENDING'
    text=EngineeringReportTextMapper.physical_inputs(child,{})
    assert '速度方向角' in text and '不是方向盘或车轮转角' in text
    # Mirror the world: signed angle changes identity but preserves time/speed.
    raw=angle_plan();raw['hypotheses'][0]['geometry']['object_position']='RIGHT'
    parameters(raw,ego_heading_angle_deg=-40,object_offset_y_m=-.6)
    mirror=instantiate(method,raw,make_candidate(7))[0]
    assert mirror.scenario_id!=child.scenario_id
    assert mirror.facts['first_contact_time_s']==pytest.approx(child.facts['first_contact_time_s'])
    # Body x separates before lateral entry: do not use gap/v_y alone.
    parameters(raw,contact_half_length_m=.01)
    assert instantiate(method,raw,make_candidate(7))[0].facts['motion_model_outcome']['status']=='NO_CONTACT_WITHIN_MODEL'


def test_new_inputs_invalidate_contact_and_explicit_symbol_never_becomes_zero(method):
    raw=angle_plan();child=instantiate(method,raw,make_candidate(7))[0]
    view={**child.facts,'_fact_provenance':child.fact_provenance}
    changed=branch_physics_view(view,{'ego_heading_angle_deg':0})
    assert 'first_contact_time_s' not in changed and 'contact_normal_speed_kph' not in changed
    p=next(p for p in raw['hypotheses'][0]['parameters'] if p['field']=='ego_heading_angle_deg')
    p.pop('value');p.update(kind='SYMBOLIC',symbol='theta')
    symbolic=instantiate(method,raw,make_candidate(7))[0]
    assert not symbolic.facts.get('collision_branches')
    assert 'first_contact_time_s' not in symbolic.facts
    assert symbolic.facts['physical_analysis']['sensitivity']['parameter_relations']


def test_physical_production_path_with_target_response_scores_projects_and_resumes(method,tmp_path):
    raw=parameters(proposal(),object_reaction_time_s=1,object_braking_deceleration_mps2=1)
    client=Client([raw]); agent=PhysicalScenarioAgent(client,response_checkpoint_dir=tmp_path)
    parent=make_candidate(7.2)
    parent.facts['component_category']='actuator_propulsion'
    parent.facts['scenario_atom_ids']=['CN_parking_multistory']
    state, generated=run(method,agent,candidates={'MF':[parent]})
    child=generated['MF'][0]
    assert len(client.requests)==1
    assert 'object_reaction_time_s' in client.requests[0].system_prompt
    view={**child.facts,'_fact_provenance':child.fact_provenance}
    timing=RiskCalculationInputService().ftti('MF',child.scenario_id,'',view,method=method)
    assert timing.hazard_time_limit_s==pytest.approx(2.25)
    assert timing.status.value=='CONDITIONAL'
    assert timing.timing_trace['engineering_approval']=='NOT_GRANTED'
    assert all(score(method,child)[k][v] for k,v in [('severity','severity_score'),('exposure','exposure_score'),('controllability','controllability_score')])
    restored=HARAState.read_committed(json.loads(json.dumps(state.to_dict())))
    _,again=run(method,agent,state=restored,candidates={'MF':[parent]})
    assert again==generated and len(client.requests)==1


def test_stationary_angle_world_survives_production_scoring_and_branch_recalculation(method):
    from hara_agent.services.semantic.scenario_agent import ScenarioFeasibilityAgent
    from hara_agent.services.analysis import MethodContractASILService, StructuredRiskScoringService
    from hara_agent.workflow.nodes.scoring import score_structured_scenarios
    parent=make_candidate(7)
    parent.facts.update(component_category='actuator_steering',driver_in_vehicle=False,
                        driver_position='outside_driver_seat',scenario_atom_ids=['CN_parking_multistory'])
    child=instantiate(method,angle_plan(),parent)[0]
    assessment=ScenarioFeasibilityAgent._parse(mf(),dict(scenario_id=child.scenario_id,
        physically_feasible=True,functionally_relevant=True,causally_relevant=True,breakpoint='NONE',
        hazardous_event='Offline fixed-angle contact',risk_dimension_changes=[],confidence=.9,
        causal_chain={
            'm_to_b':dict(claim='offline fault',basis_type='DIRECT_FACT',evidence_refs=['MF.description']),
            'b_to_i':dict(claim='trajectory',basis_type='DERIVED_PHYSICS',evidence_refs=['DERIVED.ego_impact_speed_kph']),
            'i_to_h':dict(claim='contact',basis_type='DERIVED_PHYSICS',evidence_refs=['DERIVED.relative_speed_kph'])}),scenario=child).to_dict()
    state=HARAState('offline-angle-scoring',scenarios=[child],malfunctions=[asdict(mf())],
        item_definition={'scenario_assessments':[assessment]})
    score_structured_scenarios(state,StructuredRiskScoringService(method),MethodContractASILService(method))
    risk=state.risk_results[0]
    assert risk.severity.value=='S1'
    assert risk.severity.status.value=='PENDING'
    assert risk.branch_evaluations[0]['relative_speed_kph']==pytest.approx(7)
    assert not risk.branch_evaluations[0]['physical_input_conflicts']
    assert child.facts['object_longitudinal_direction']=='STATIONARY'


@pytest.mark.parametrize('changes',[
    {'object_reaction_time_s':1}, {'object_reaction_time_s':1,'object_braking_deceleration_mps2':0},
    {'ego_heading_angle_deg':181}, {'object_heading_angle_deg':180},
])
def test_invalid_or_conflicting_response_and_heading_are_not_silently_used(changes):
    result=rectilinear_moving_contact({**following(),**changes})
    assert result['status']!='CONTACT'


def test_removed_reference_defaults_are_not_active_authorities(method):
    from hara_agent.method_sources.yaml_loader import load_yaml_mapping
    from pathlib import Path
    root=Path('method_assets/fusa_baseline_v1')
    assert load_yaml_mapping(root/'raw/ftti_formulas.yaml')['formulas']==[]
    assert load_yaml_mapping(root/'raw/ftti_budgets.yaml')['budgets']=={}
    assert method.metadata['project_analysis_policy']['timing_analysis']['hazard_criterion']=='LOSS_OF_BRAKING_AVOIDABILITY'
