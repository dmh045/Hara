"""Report intent and collision wording stay within the actual associated world."""
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import gzip,json
import pytest
from hara_agent.models import EvidenceValue,ReviewStatus
from hara_agent.workflow import HARAState
from hara_agent.services.reporting import HARAReportProjectionService,load_report_schema
from hara_agent.services.reporting.engineering_text_mapper import EngineeringReportTextMapper
from test_reuse_reporting_acceptance import active_method

DATA=json.loads(gzip.decompress((Path(__file__).resolve().parents[1]/'fixtures/f7_report_member_scope.json.gz').read_bytes()))


def test_real_f7_report_uses_associated_pedestrian_intent_and_reverse_contact(active_method):
    state=HARAState.read_committed(deepcopy(DATA['state']));trace=deepcopy(DATA['risk_trace'])
    before,trace_before=deepcopy(state),deepcopy(trace)
    view=HARAReportProjectionService(load_report_schema()).project(state,active_method,risk_trace=trace)
    assert len(view.rows)==2 and len(view.safety_goals)==1
    goal=view.safety_goals[0]
    assert goal.sg_id==state.safety_goals[0].sg_id
    assert '高于目标' in goal.safety_goal and '倒车' in goal.safety_goal
    assert '障碍物' not in goal.safety_goal
    assert goal.original_text==state.item_definition['safety_goal_catalog'][goal.sg_id]['original_safety_goal']
    assert goal.status=='条件性草稿'
    static=next(r for r in view.rows if r.asil=='QM')
    person=next(r for r in view.rows if r.asil=='B')
    assert not static.sg_id and person.sg_id==goal.sg_id
    assert '后向碰撞' in static.hazardous_event and '追尾' not in static.hazardous_event
    assert '倒车碰撞后方车辆' in static.severity_rationale and '追尾' not in static.severity_rationale
    detail=next(d for d in view.scenario_details if d.scenario_id==static.scenario_id)
    assert '倒车碰撞后方车辆' in detail.branch_results
    assert goal.timing_boundary and '0.15 s' in goal.timing_boundary
    assert state==before and trace==trace_before


def member(target,event):
    return dict(scenario_id=target,function='任意路径控制',malfunction='输出与请求不一致',
        vehicle_hazard='车辆非预期运动并与固定障碍物碰撞。',road_user_type=target,
        contact_status='CONTACT',hazardous_event=event)


def test_actual_member_mechanism_is_not_replaced_by_parent_example_object():
    row=member('CYCLIST','车辆非预期偏离训练路线，与骑行者发生接触。');before=deepcopy(row)
    result=EngineeringReportTextMapper().safety_goal_presentation([row],'原安全意图')
    assert result['text']=='防止车辆非预期偏离训练路线。'
    assert result['members'][0]['basis']['hazardous_event']==row['hazardous_event']
    assert '固定障碍物' not in result['text'] and row==before


def test_multiple_actual_objects_share_contact_wording_without_borrowing_unknown_member():
    rows=[member('PEDESTRIAN','车辆与行人发生接触。'),member('VEHICLE','车辆与其他车辆发生碰撞。')]
    before=deepcopy(rows)
    result=EngineeringReportTextMapper().safety_goal_presentation(rows,'原安全意图')
    assert result['text']=='防止车辆与行人、其他车辆发生接触。'
    assert len(result['members'])==2 and rows==before
    rows[1]['hazardous_event']='车辆未与其他车辆发生接触。'
    result=EngineeringReportTextMapper().safety_goal_presentation(rows,'原安全意图')
    assert '原安全意图' in result['text'] and '行人、其他车辆' not in result['text']


@pytest.mark.parametrize('event',['车辆未与行人发生接触。','车辆不会碰撞。','车辆与行人是否发生接触尚待确定。'])
def test_structured_contact_does_not_override_negative_or_unresolved_event(event):
    result=EngineeringReportTextMapper().safety_goal_presentation([member('PEDESTRIAN',event)],'原安全意图')
    assert result['text']=='原安全意图。'


@pytest.mark.parametrize('change',[{}, {'ego_longitudinal_direction':'FORWARD'},
    {'object_speed_kph':5,'object_longitudinal_direction':'FORWARD'},
    {'object_position':'FRONT'}, {'object_longitudinal_direction':None}, {'object_speed_kph':False}])
def test_reverse_contact_label_requires_exact_stationary_rear_vehicle_world(change):
    facts=dict(ego_longitudinal_direction='REVERSE',object_position='REAR',object_speed_kph=0,
        object_longitudinal_direction='STATIONARY',road_user_type='VEHICLE',collision_type='REAR_END')
    facts.update(change);scenario=SimpleNamespace(facts=facts);before=deepcopy(facts)
    raw='车辆与后方车辆发生REAR_END碰撞，来源ID-REAR_END保持。'
    text=EngineeringReportTextMapper.hazardous_event(raw,scenario)
    rationale=EngineeringReportTextMapper().severity_rationale(EvidenceValue('S1',ReviewStatus.PENDING),
        {'physical_inputs':{'road_user_type':'VEHICLE','collision_type':'REAR_END','relative_speed_kph':7}},scenario=scenario)
    if not change:
        assert '后向碰撞' in text and '倒车碰撞后方车辆' in rationale
    else:
        assert '追尾碰撞' in text and '车辆追尾碰撞' in rationale
    assert 'ID-REAR_END' in text and facts==before


def test_unmatched_or_multiple_contact_branches_do_not_borrow_parent_reverse_geometry():
    facts=dict(ego_longitudinal_direction='REVERSE',object_position='REAR',object_speed_kph=0,
        object_longitudinal_direction='STATIONARY',collision_branches=[{'branch_id':'own'}])
    branch={'branch_id':'other','S':'S1','E':'E2','C':'C3','ASIL':'QM',
        'risk_context':{'road_user_type':{'value':'VEHICLE'},'collision_type':{'value':'REAR_END'}}}
    scenario=SimpleNamespace(facts=facts)
    assert '追尾碰撞' in EngineeringReportTextMapper.branch_result(branch,scenario)
    branch['branch_id']='own'
    assert '倒车碰撞后方车辆' in EngineeringReportTextMapper.branch_result(branch,scenario)
    facts['collision_branches'].append({'branch_id':'another'})
    assert '追尾碰撞' in EngineeringReportTextMapper.branch_result(branch,scenario)


def test_correct_shared_object_or_object_neutral_intent_is_retained():
    row=member('VEHICLE','半自动引导提前终止后，车辆与后方车辆发生接触。')
    row['vehicle_hazard']='车辆在泊车中擦碰邻近车辆、障碍物或路缘。'
    mapper=EngineeringReportTextMapper()
    assert mapper.safety_goal_presentation([row],'原安全意图')['text']=='防止车辆在泊车中擦碰邻近车辆、障碍物或路缘。'
    row['vehicle_hazard']='车辆发生非预期急加速。'
    assert mapper.safety_goal_presentation([row],'原安全意图')['text']=='防止车辆非预期急加速。'


def test_parent_object_mismatch_does_not_turn_a_dependent_clause_into_goal():
    row=member('PEDESTRIAN','半自动引导提前终止后，车辆与行人发生接触。')
    result=EngineeringReportTextMapper().safety_goal_presentation([row],'原安全意图')
    assert result['text']=='防止车辆与行人发生接触。'
    row['hazardous_event']='半自动引导提前终止后。'
    assert EngineeringReportTextMapper().safety_goal_presentation([row],'原安全意图')['text']!='防止半自动引导提前终止后。'


def test_parameter_bearing_member_contact_uses_complete_actual_object_intent():
    row=member('PEDESTRIAN','车辆以7 km/h与行人发生接触。')
    result=EngineeringReportTextMapper().safety_goal_presentation([row],'防止接近终点时')
    assert result['text']=='防止车辆与行人发生接触。'
    assert result['members'][0]['basis']['hazardous_event']==row['hazardous_event']
    row['contact_status']='UNRESOLVED'
    result=EngineeringReportTextMapper().safety_goal_presentation([row],'原安全意图')
    assert not result['members'][0]['basis'].get('member_contact_grounded')


def test_actual_contact_alternative_skips_scene_prefix_without_vehicle_predicate():
    row=member('VEHICLE','纵向车位反向泊入接近终点时，自车以8 km/h与边界车辆发生接触。')
    row['vehicle_hazard']='车门与边界物体接触，或车辆泊入后挤压边界区域。'
    assert EngineeringReportTextMapper().safety_goal_presentation([row],'原安全意图')['text']=='防止车辆与其他车辆发生接触。'
