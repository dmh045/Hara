"""Production-derived numeric report plus unseen and incomplete display cases."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path

import pytest
from hara_agent.models import EvidenceValue, ReviewStatus
from hara_agent.services.reporting import HARAReportProjectionService, load_report_schema
from hara_agent.services.reporting.engineering_text_mapper import EngineeringReportTextMapper
from hara_agent.workflow.state import HARAState
from test_reuse_reporting_acceptance import active_method


@pytest.fixture
def real_sample():
    fixture = json.loads((Path(__file__).parents[1] / 'fixtures/report_numeric_physics_e2e_20261019.json').read_text())
    return HARAState.read_committed(fixture['state']), fixture['trace']


def test_real_numeric_sample_keeps_scenario_and_explains_actual_scores(active_method, real_sample):
    state, trace = real_sample
    before, trace_before = deepcopy(state), deepcopy(trace)
    service = HARAReportProjectionService(load_report_schema())
    view = service.project(state, active_method, risk_trace=trace)
    row, detail = view.rows[0], view.scenario_details[0]
    assert (row.operational_scenario, row.scenario_detail) == service.text.scenario(state.scenarios[0])
    assert '室外停车场' in row.operational_scenario
    assert '泊车控车' in row.operational_scenario and '驾驶员在驾驶位' in row.operational_scenario
    assert '15.6 km/h' in row.severity_rationale and '追尾' in row.severity_rationale
    assert '室外停车场=E4' in row.exposure_rationale and 'E3' in row.exposure_rationale
    assert '1.59 s' in row.controllability_rationale
    assert 'S1 / E3 / C3' in row.asil_rationale
    assert (row.severity,row.exposure,row.controllability,row.asil) == ('S1','E3','C3','A')
    assert '分支 ' not in row.scenario_detail and '个成员' not in row.scenario_detail
    assert row.remark.count('待审试算') == 1
    assert '未覆盖' not in detail.physical_inputs and 'plan_key' not in detail.physical_inputs
    assert '1.5' in detail.physical_inputs and '5 m' in detail.physical_inputs
    assert '未考虑驾驶员接管与制动介入' in detail.physical_inputs
    assert 'LLM 分析假设' in detail.analysis_basis
    assert view.safety_goals[0].safety_goal == '防止车辆非预期急加速。'
    assert view.safety_goals[0].original_text == state.item_definition['safety_goal_catalog'][state.safety_goals[0].sg_id]['original_safety_goal']
    assert '条件性' in view.safety_goals[0].status
    assert all(token not in row.severity_rationale + row.exposure_rationale + row.controllability_rationale
               for token in ('请求', 'C-TTC', 'S-vehicle', 'FUSA_', 'BRANCH-'))
    assert state == before and trace == trace_before


def test_multiple_members_keep_only_shared_raw_conditions_and_all_ratings(active_method, real_sample):
    state, _ = real_sample
    first = state.scenarios[0]
    second = deepcopy(first)
    second.scenario_id += '-different'
    second.facts['driver_position'] = 'outside_driver_seat'
    second.facts['method_scenario_dimensions']['WHERE']['project_value'] = '地下停车场'
    second.facts['source_operating_conditions']['operating_scenario'] = '地下停车场'
    first.facts['source_operating_conditions']['operating_scenario'] = '室外停车场'
    # Same displayed rounding, different raw input: not a shared parameter.
    first.facts['longitudinal_acceleration_mps2'] = 1.504
    second.facts['longitudinal_acceleration_mps2'] = 1.503
    second.facts['relative_distance_m'] = 8
    second.facts['ego_impact_speed_kph'] = 18.125
    second_risk = deepcopy(state.risk_results[0])
    second_risk.scenario_id = second.scenario_id
    second_risk.assessment_id += '-different'
    second_risk.controllability = EvidenceValue(None, ReviewStatus.PENDING, review_reason='MISSING_DISTANCE')
    second_risk.asil = EvidenceValue(None, ReviewStatus.PENDING)
    second_risk.branch_evaluations = []
    state.scenarios.append(second)
    state.risk_results.append(second_risk)
    state.item_definition['scenario_assessments'].append({**deepcopy(state.item_definition['scenario_assessments'][0]), 'scenario_id':second.scenario_id})
    before = deepcopy(state)
    service = HARAReportProjectionService(load_report_schema())
    view = service.project(state,active_method)
    assert len(view.rows) == 1 and len(view.scenario_details) == 2
    row = view.rows[0]
    assert '驾驶员在驾驶位' not in row.operational_scenario
    assert '室外停车场' not in row.operational_scenario and '地下停车场' not in row.operational_scenario
    assert '初始车速' in row.operational_scenario
    assert '参数' in row.remark and '加速度' in row.remark and '驾驶员位置' in row.remark
    assert '最高已知结果不是全组上界' in row.remark
    assert {d.asil for d in view.scenario_details} == {'A','待评定'}
    assert '15.6' not in row.scenario_detail and '18.1' not in row.scenario_detail
    assert state == before


@pytest.mark.parametrize('behavior,expected', [
    ('车辆出现非预期横摆，向相邻车道偏移', '防止车辆非预期横摆。'),
    ('车辆丧失减速能力，无法按请求停车', '防止车辆丧失减速能力。'),
    ('车辆无法保持静止，沿道路坡向移动', '防止车辆无法保持静止。'),
])
def test_unseen_sg_wording_uses_supported_behavior_and_retains_original(behavior,expected):
    mapper=EngineeringReportTextMapper()  # no exact-match catalog
    member={'function':'车辆运动控制','malfunction':'内部执行请求未正确输出','vehicle_hazard':behavior,
            'hazardous_event':behavior,'scenario_id':'new-id'}
    original='保留的原始工程安全目标'
    before=deepcopy(member)
    result=mapper.safety_goal_presentation([member],original)
    assert result['text'] == expected
    assert result['members'][0]['basis']['vehicle_hazard'] == behavior
    assert '制动介入' not in result['text'] and '碰撞' not in result['text']
    assert member == before


def test_negative_sg_claim_never_becomes_unsupported_collision_or_action():
    result=EngineeringReportTextMapper().safety_goal_presentation([
        {'malfunction':'控制消息异常','vehicle_hazard':'未发生非预期运动','hazardous_event':'不会碰撞'}], '现有安全意图')
    assert result['text'] == '现有安全意图。'
    assert result['members'][0]['basis']['original_goal'] == '现有安全意图'


def test_display_uses_the_deciding_branch_physics_not_parent_max_s(active_method,real_sample):
    state,trace=real_sample
    risk=state.risk_results[0]
    low=deepcopy(risk.branch_evaluations[0])
    high=deepcopy(low)
    high.update(branch_id='other-object',S='S2',E='E4',C='C3',ASIL='C',relative_speed_kph=29.456,ttc_s=.854)
    high['risk_context']['relative_speed_kph']['value']=29.456
    high['risk_context']['ttc_s']['value']=.854
    risk.branch_evaluations=[low,high]
    before=deepcopy(state)
    view=HARAReportProjectionService(load_report_schema()).project(state,active_method,risk_trace=trace)
    row=view.rows[0]
    assert (row.severity,row.exposure,row.controllability,row.asil)==('S2','E4','C3','C')
    assert '29.5' in row.severity_rationale and '15.6' not in row.severity_rationale
    assert '0.85' in row.controllability_rationale
    assert 'S2 / E4 / C3' in row.asil_rationale
    assert state==before
