"""Production-derived numeric report plus unseen and incomplete display cases."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

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


@pytest.mark.parametrize('validated', [True, False])
def test_conditional_stationary_start_is_not_reported_as_zero_fault_speed(validated):
    scenario = SimpleNamespace(facts={
        'ego_speed_kph': 0, 'ego_impact_speed_kph': 2.724688,
        'ego_speed_constraint': {'min_kph': 0, 'max_kph': 0},
        'physical_analysis': {'engineering_approval': 'NOT_GRANTED'},
    }, fact_provenance={'ego_speed_kph': {
        'approval': 'PENDING', 'validation_status': 'VALIDATED' if validated else 'PENDING',
        'analysis_value_semantic': 'PRE_FAULT_OPERATING_SPEED',
    }})
    before = deepcopy(scenario)
    text = EngineeringReportTextMapper.speed_text(scenario)
    if validated:
        assert text == '故障前/初始车速：0 km/h；碰撞时本车速度：2.7 km/h'
    else:
        assert text == '适用车速范围：0 km/h'
    assert scenario == before


@pytest.mark.parametrize('reason,text', [
    ('PHYSICAL_MODEL_NOT_APPLICABLE', '当前可用物理模型不覆盖'),
    ('TARGET_NOT_ON_POST_FAULT_TRAVEL_PATH', '目标不在故障后的运动路径'),
])
def test_model_gap_is_concise_and_preserves_original_diagnostics(reason, text):
    scenario = SimpleNamespace(facts={}, fact_provenance={}, analysis_instance={
        'physical_plan_diagnostics': [{'reason':reason, 'uncovered_conditions':['完整未覆盖原文' * 100]}]})
    before = deepcopy(scenario)
    value = EngineeringReportTextMapper.analysis_basis(scenario)
    assert text in value and '未生成碰撞数值' in value
    assert reason not in value and '完整未覆盖原文' not in value
    assert scenario == before


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
    assert '停车场/停车楼/车库=E4' in row.exposure_rationale and 'E3' in row.exposure_rationale
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


@pytest.mark.parametrize('same_condition', [False, True])
def test_group_keeps_source_choice_and_effect_together(real_sample, same_condition):
    state, _ = real_sample
    first = deepcopy(state.scenarios[0])
    second = deepcopy(first)
    first.facts['selected_source_condition'] = {
        'source_field': 'odd_road_types', 'value': 'T字框型', 'physical_effect': '开放端目标位于前方'}
    second.facts['selected_source_condition'] = deepcopy(first.facts['selected_source_condition']) if same_condition else {
        'source_field': 'odd_road_types', 'value': '直线开口型', 'physical_effect': '另一车位边界对应目标位置'}
    before = deepcopy([first, second])
    service = HARAReportProjectionService(load_report_schema())
    text, detail = service._group_scenario([{'scenario': first}, {'scenario': second}])
    assert ('代表运行条件为T字框型' in text) == same_condition
    if not same_condition:
        assert 'T字框型' not in text and '直线开口型' not in text
        assert '开放端目标' not in detail and '另一车位' not in detail
    for scenario in (first, second):
        member_text, member_detail = service.text.scenario(scenario)
        condition = scenario.facts['selected_source_condition']
        assert condition['value'] in member_text and condition['physical_effect'] in member_detail
    assert [first, second] == before


def test_coverage_note_does_not_change_risk_or_approval(active_method, real_sample):
    state, trace = real_sample
    before = deepcopy(state)
    projector = HARAReportProjectionService(load_report_schema())
    base = projector.project(state, active_method, risk_trace=trace)
    note = '原文坡度范围未进入当前候选覆盖；不是对全部场景指定实际坡度。'
    view = projector.project(state, active_method, risk_trace=trace,
        run_summary={'input_coverage_notes':[note]})
    assert note in view.summary.scope['scope_note']
    assert view.rows == base.rows and view.scenario_details == base.scenario_details
    assert view.safety_goals == base.safety_goals and state == before


def test_partial_real_recovery_never_claims_complete_provider_validation(active_method,real_sample):
    state,trace=real_sample
    state.record('real_partial_response_recovery',available_candidates=32,
        validated_response_identities=14,remaining_unvalidated=18)
    before=deepcopy(state)
    view=HARAReportProjectionService(load_report_schema()).project(state,active_method,risk_trace=trace)
    assert view.summary.scope['run_scope']=='PARTIAL REAL RESPONSE RECOVERY'
    note=view.summary.scope['scope_note']
    assert '14 个已通过因果校验' in note and '18 个尚未验证' in note
    assert '补充风险事实模型阶段未执行' in note
    assert view.rows[0].asil=='A' and state==before


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
    assert len(view.rows) == 2 and len(view.scenario_details) == 2
    assert all('初始车速' in row.operational_scenario for row in view.rows)
    assert any('室外停车场' in row.operational_scenario for row in view.rows)
    assert any('地下停车场' in row.operational_scenario for row in view.rows)
    assert any('最高已知结果不是全组上界' in row.remark for row in view.rows)
    assert {d.asil for d in view.scenario_details} == {'A','待评定'}
    assert any('15.6' in row.scenario_detail for row in view.rows)
    assert any('18.1' in row.scenario_detail for row in view.rows)
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


def test_v8_member_intents_do_not_select_other_parent_branch_or_parameters():
    fixture = json.loads((Path(__file__).parents[1] / 'fixtures/report_v8_safety_intents.json').read_text())
    mapper = EngineeringReportTextMapper()
    expected = {
        'SG-METHOD-DBF8EE1CDE': '防止驻车制动请求卡死在释放状态。',
        'SG-METHOD-5725DFEFFD': '防止本应关闭的车辆功能持续运行。',
        'SG-METHOD-77060B1CDE': '防止车辆加速响应滞后。',
    }
    for goal in fixture['goals']:
        before = deepcopy(goal)
        presentation = mapper.safety_goal_presentation(goal['members'], goal['original'])
        assert presentation['text'] == expected[goal['sg_id']]
        assert len(presentation['members']) == len(goal['members'])
        assert goal == before


def test_alternative_intents_keep_both_actual_branches_and_negative_evidence():
    mapper = EngineeringReportTextMapper()
    members = [
        {'vehicle_hazard': '需解锁时无法解锁，或需锁止时无法锁止',
         'hazardous_event': '驻车结束后，执行器无法解锁'},
        {'vehicle_hazard': '需解锁时无法解锁，或需锁止时无法锁止',
         'hazardous_event': '停车场内，执行器无法锁止'},
    ]
    text = mapper.safety_goal_presentation(members, '原文')['text']
    assert '执行器无法解锁' in text and '执行器无法锁止' in text
    members[0]['hazardous_event'] = '未发生无法解锁'
    result = mapper.safety_goal_presentation(members[:1], '原文')
    assert result['text'] == '原文。'


def test_summary_timing_and_harm_projection_keeps_raw_conditional_state(active_method, real_sample):
    from hara_agent.services.analysis.risk_calculation_input_service import RiskCalculationInputService
    state, trace = real_sample
    state.risk_results[0].potential_harm = ''
    before = deepcopy(state)
    view = HARAReportProjectionService(load_report_schema()).project(state, active_method, risk_trace=trace)
    scope = view.summary.scope
    assert scope['numeric_risk_chain_count'] == scope['numeric_asil_count'] == 1
    assert scope['complete_risk_chain_count'] == 0
    assert view.rows[0].potential_harm.startswith('按当前 S 分档：')
    assert view.safety_goals[0].status == '条件性草稿'
    assert view.rows[0].ftti == '待确定'
    assert '不能作为 FTTI' in view.rows[0].ftti_rationale
    assert '1.59' in view.rows[0].ftti_rationale
    assert state == before
    for key in ('ttc_s', 'first_contact_time_s'):
        scenario = {key: .154286, 'object_id': 'target', '_fact_provenance': {
            key: {'assessment_reference': 'MODEL_INITIAL_STATE', 'provenance': 'DERIVED'}}}
        result = RiskCalculationInputService().ftti('MF', 'SCN', 'A', scenario)
        assert result.contact_time_s == .154286
        assert result.contact_time_kind == key
        assert result.hazard_time_limit_s is None
        assert result.safe_state_transition_time_s is None
        assert result.time_origin == 'MODEL_INITIAL_STATE'
        assert result.evidence_refs == (f'DERIVED.{key}',)
        assert result.status.value == 'PENDING_METHOD_SEMANTICS'


def test_unresolved_risk_intents_use_production_aggregation_without_approval(active_method, real_sample):
    from hara_agent.services.analysis.method_safety_goal_service import MethodSafetyGoalService
    from hara_agent.workflow.nodes.safety_goals import aggregate_safety_goals
    state, _ = real_sample
    original_id = state.risk_results[0].safety_goal_id
    original_text = state.item_definition['safety_goal_catalog'][original_id]['original_safety_goal']
    state.risk_results[0].asil = EvidenceValue(None, ReviewStatus.PENDING)
    ratings = deepcopy(state.risk_results[0])
    aggregate_safety_goals(state, MethodSafetyGoalService(active_method))
    assert state.risk_results[0] == ratings
    assert state.risk_results[0].safety_goal_id == original_id
    assert state.item_definition['safety_goal_catalog'][original_id]['original_safety_goal'] == original_text
    assert len(state.safety_goals) == 1
    assert state.safety_goals[0].max_asil == ''
    assert state.safety_goals[0].status is ReviewStatus.PENDING
    view = HARAReportProjectionService(load_report_schema()).project(state, active_method)
    assert view.safety_goals[0].max_asil == '待评定'
    assert view.safety_goals[0].status == '待审核'


def test_source_conditions_keep_pending_separate_from_exclusion(active_method, real_sample):
    state, trace = real_sample
    mid = state.malfunctions[0]['malfunction_id']
    fid = state.malfunctions[0]['function_id']
    generated = []
    for sid, status, reason, location in (
        ('pending', 'PENDING_FEASIBILITY', 'PENDING_CAUSAL_EVIDENCE', '室外停车场'),
        ('excluded', 'INELIGIBLE_CAUSAL', 'CAUSAL_LINK_FALSE', '室内停车场'),
    ):
        trace['assessments'].append({
            'scenario_id': sid, 'malfunction_id': mid,
            'risk_eligibility': {'status': status, 'reason_codes': [reason]},
            'risk_scoring_invoked': False})
        generated.append({'scenario_id': sid, 'facts': {
            'source_operating_conditions': {'operating_scenario': location}}})
    before = deepcopy(state)
    view = HARAReportProjectionService(load_report_schema()).project(
        state, active_method, risk_trace=trace, generated_scenarios=generated)
    rows = {row.scenario_id: row for row in view.generated_children}
    assert rows['pending'].causal_disposition == 'PENDING_CAUSAL'
    assert rows['excluded'].causal_disposition == 'EXCLUDED_FROM_RISK'
    assert rows['pending'].function_id == rows['excluded'].function_id == fid
    assert rows['pending'].source_conditions == '来源场所：室外停车场'
    assert rows['pending'].disposition_reason == 'PENDING_CAUSAL_EVIDENCE'
    assert state == before


def test_partial_qm_remains_a_calculated_result_with_independent_c_and_review_state(active_method, real_sample):
    from hara_agent.services.analysis.risk_execution_trace_service import RiskExecutionTraceService
    state, _ = real_sample
    risk = state.risk_results[0]
    risk.severity = EvidenceValue('S0', ReviewStatus.PENDING, review_reason='CONDITIONAL_TRIAL')
    risk.controllability = EvidenceValue(None, ReviewStatus.PENDING, review_reason='METHOD_GAP')
    risk.asil = EvidenceValue('QM', ReviewStatus.PENDING, review_reason='CONDITIONAL_TRIAL')
    risk.branch_evaluations = []
    risk.branch_summary = {}
    before = deepcopy(risk)
    trace = RiskExecutionTraceService()._asil_trace(risk)
    assert trace['status'] == 'CALCULATED_PENDING_REVIEW'
    assert trace['review_status'] == 'PENDING'
    assert trace['C'] is None and trace['pending_reason'] == ''
    assert HARAReportProjectionService._score_state(risk.asil, trace, scoring_invoked=True) == '待审试算'
    text = EngineeringReportTextMapper().asil_rationale(risk.asil, trace)
    assert 'QM' in text and 'S0' in text and 'C 仍未评定' in text
    assert 'ASIL 暂不评定' not in text
    view = HARAReportProjectionService(load_report_schema()).project(state, active_method)
    assert (view.rows[0].severity, view.rows[0].controllability, view.rows[0].asil) == ('S0', '待评定', 'QM')
    assert '其余维度的全部组合均对应 QM' in view.rows[0].asil_rationale
    assert 'C 本身仍待评定' in view.rows[0].asil_rationale
    assert risk == before


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
    assert '0.85' in row.ftti_rationale and '1.59' in row.ftti_rationale
    timing = view.projection_metrics['ftti_timing_inputs'][0]
    assert timing['contact_time_s'] is None
    assert {b['branch_id'] for b in timing['branch_timings']} == {low['branch_id'], high['branch_id']}
    assert state==before


def test_partial_qm_branch_does_not_describe_missing_c_as_ttc_grade(active_method, real_sample):
    state, _ = real_sample
    risk = state.risk_results[0]
    risk.severity = EvidenceValue('S0', ReviewStatus.PENDING, review_reason='CONDITIONAL_TRIAL')
    risk.controllability = EvidenceValue(None, ReviewStatus.PENDING, review_reason='METHOD_GAP')
    risk.asil = EvidenceValue('QM', ReviewStatus.PENDING, review_reason='CONDITIONAL_TRIAL')
    branch = risk.branch_evaluations[0]
    branch.update(S='S0', C=None, ASIL='QM', ttc_s=5.59)
    branch['scores']['controllability'].update(controllability_score='',
        decision_status='MODEL_DEPENDENCY_UNRESOLVED',
        pending_reason='FAILED_CONTROL_PROFILE_DEPENDENCY_UNRESOLVED')
    before = deepcopy(state)
    row = HARAReportProjectionService(load_report_schema()).project(state, active_method).rows[0]
    assert row.controllability == '待评定' and row.asil == 'QM'
    assert '分档为 待评定' not in row.controllability_rationale
    assert 'C 方法依赖尚未成立' in row.controllability_rationale and 'C 暂未定' in row.controllability_rationale
    assert state == before
def test_event_prose_formats_physical_units_and_only_the_bound_object():
    from types import SimpleNamespace
    from hara_agent.services.reporting.engineering_text_mapper import EngineeringReportTextMapper
    original = '事件ID V1.2，规则R0.154286s；于0.5633344678939333s以7.126km/h接触target1，target10未接触；偏移-0.0001米。'
    scenario = SimpleNamespace(facts={'object_id': 'target1', 'road_user_type': 'VEHICLE'}, analysis_instance={})
    display = EngineeringReportTextMapper.hazardous_event(original, scenario)
    assert '0.56s' in display and '7.1km/h' in display and '其他车辆' in display
    assert 'target10' in display and 'target1，' not in display
    assert 'V1.2' in display and 'R0.154286s' in display and '−0.01 < 值 < 0米' in display
    assert '0.5633344678939333s' in original and scenario.facts['object_id'] == 'target1'


def test_event_prose_translates_contact_labels_without_rewriting_identities():
    from hara_agent.services.reporting.engineering_text_mapper import EngineeringReportTextMapper
    original = '与前方0.3 m处行人发生FRONTAL接触；与后车发生REAR_END碰撞；侧方SIDE 接触；LATERAL碰撞；ID-FRONTAL接触；FRONTAL_RULE；SIDEWAYS'
    display = EngineeringReportTextMapper.hazardous_event(original)
    assert '0.3 m处行人发生正面碰撞' in display
    assert '与后车发生追尾碰撞' in display
    assert display.count('侧向碰撞') == 2
    assert 'ID-FRONTAL接触' in display and 'FRONTAL_RULE' in display and 'SIDEWAYS' in display
    assert 'FRONTAL接触' in original


@pytest.mark.parametrize('reason,expected', [
    ('REAR_APPROACH_BRAKING_APPLICABILITY_UNRESOLVED','后车从本车行进方向的后方接近'),
    ('LATERAL_C_METHOD_NOT_AVAILABLE','侧向接触模型'),
    ('UNSUPPORTED_INTERACTION','当前方法无适用 C 规则'),
    ('POST_STOP_BRAKING_APPLICABILITY_UNRESOLVED','本车停止后被移动目标接触'),
])
def test_pending_c_model_label_uses_the_actual_interaction_reason(reason,expected):
    evidence=EvidenceValue('',ReviewStatus.PENDING)
    trace={'decision_status':'MODEL_NOT_APPLICABLE','pending_reason':reason}
    before=deepcopy(trace)
    text=EngineeringReportTextMapper().controllability_rationale(evidence,trace)
    assert expected in text and 'C 暂未定' in text
    if not reason.startswith('LATERAL_'):assert '侧向' not in text
    assert trace==before and evidence.value=='' and evidence.status is ReviewStatus.PENDING


def test_pending_c_summary_keeps_committed_reason_when_trace_is_compact():
    evidence=EvidenceValue('',ReviewStatus.PENDING,
        review_reason='REAR_APPROACH_BRAKING_APPLICABILITY_UNRESOLVED; conditional')
    trace={'decision_status':'MODEL_NOT_APPLICABLE'}
    display=EngineeringReportTextMapper().controllability_rationale(evidence,trace)
    assert '后车从本车行进方向的后方接近' in display and '侧向' not in display
    assert trace=={'decision_status':'MODEL_NOT_APPLICABLE'}
