"""Retain an explicit shared actor without manufacturing a driver or hazard."""
from pathlib import Path
from copy import deepcopy
import gzip,json
import pytest
from hara_agent.workflow import HARAState
from hara_agent.services.reporting import HARAReportProjectionService,load_report_schema
from hara_agent.services.reporting.engineering_text_mapper import EngineeringReportTextMapper
from test_reuse_reporting_acceptance import active_method

DATA=json.loads(gzip.decompress((Path(__file__).resolve().parents[1]/'fixtures/sg_shared_subject_038.json.gz').read_bytes()))


def test_real_038_preserves_driver_subject_and_original_goal_identity_sources(active_method):
    state=HARAState.read_committed(deepcopy(DATA));before=deepcopy(state)
    view=HARAReportProjectionService(load_report_schema()).project(state,active_method)
    assert len(view.rows)==len(view.safety_goals)==1
    goal=view.safety_goals[0];risk=state.risk_results[0]
    assert goal.safety_goal=='防止驾驶员无法感知车辆当前报警变化。'
    assert view.rows[0].safety_goal==goal.safety_goal
    assert goal.sg_id==state.safety_goals[0].sg_id==risk.safety_goal_id
    assert goal.original_text==state.item_definition['safety_goal_catalog'][goal.sg_id]['original_safety_goal']
    assert goal.status=='条件性草稿' and not state.can_publish
    assert [getattr(risk,k).value for k in ('severity','exposure','controllability','asil')]==['S2','E2','C3','A']
    assert view.rows[0].ftti=='2.16 s' and state==before
    evidence=view.projection_metrics['safety_goal_display_evidence'][goal.sg_id]
    assert '驾驶员看到过期状态，无法感知车辆当前报警变化。' in str(evidence)


@pytest.mark.parametrize('subject',['驾驶员','乘员','车辆'])
def test_comma_shared_explicit_actor_is_preserved_for_arbitrary_function(subject):
    member={'functional_effect':subject+'收到固定提示，无法识别当前门锁状态。','function':'任意信息功能'}
    before=deepcopy(member)
    result=EngineeringReportTextMapper().safety_goal_presentation([member],'保留原目标')
    assert result['text']=='防止'+subject+'无法识别当前门锁状态。'
    assert result['members'][0]['basis']['functional_effect']==member['functional_effect'] and member==before


@pytest.mark.parametrize('source',['看到过期状态，无法感知当前变化。','操作主体未知，无法感知当前变化。',
    '驾驶员看到旧状态。无法感知当前变化。','驾驶员看到旧状态；无法感知当前变化。',
    '驾驶员身份待确定，无法感知当前变化。'])
def test_unknown_or_separate_sentence_does_not_invent_actor(source):
    member={'functional_effect':source,'driver_position':'in_driver_seat','driver_in_vehicle':True}
    assert EngineeringReportTextMapper().safety_goal_presentation([member],'保留原目标')['text']=='保留原目标。'


@pytest.mark.parametrize('source',['驾驶员看到旧状态，不会无法感知当前变化。',
    '驾驶员看到旧状态，但并非无法感知当前变化。','乘员看到旧状态，并不是不能识别当前状态。',
    '驾驶员看到旧状态，不能确认是否无法感知当前变化。'])
def test_negation_and_uncertainty_do_not_become_affirmative_impairment(source):
    member={'functional_effect':source};before=deepcopy(member)
    result=EngineeringReportTextMapper().safety_goal_presentation([member],'保留原目标')
    assert result['text']=='保留原目标。' and member==before


def test_negated_impairment_clause_does_not_negate_separate_affirmative_contact():
    member={'vehicle_hazard':'车辆与固定障碍物碰撞。','road_user_type':'PEDESTRIAN',
        'contact_status':'CONTACT','hazardous_event':'驾驶员并非无法感知提示，车辆以7 km/h与行人发生接触。'}
    before=deepcopy(member)
    result=EngineeringReportTextMapper().safety_goal_presentation([member],'保留原目标')
    assert result['text']=='防止车辆与行人发生接触。' and member==before
