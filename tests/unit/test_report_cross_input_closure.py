"""Projection regressions from five inputs, without new semantic responses."""
from copy import deepcopy
from types import SimpleNamespace

from openpyxl import load_workbook

from hara_agent.services.reporting import HARAReportProjectionService, load_report_schema
from hara_agent.services.reporting.canonical_renderer import HARAReportWorkbookRenderer
from hara_agent.services.reporting.engineering_text_mapper import EngineeringReportTextMapper
from test_report_semantic_readability import real_sample
from test_reuse_reporting_acceptance import active_method


def test_unconfirmed_space_intent_is_complete_and_shared_across_member_parameters():
    hazard = '车辆可能朝尚未确认可用的车位开始泊车，形成过早泊车目标状态。'
    events = [
        '车辆基于提前输出的未确认可用车位目标，以约15 km/h前进入位，并与行人发生接触。',
        '在垂直停车位入口的泊车过程中，因可用车位确认前提前输出导致自车启动，与骑行者发生侧向接触。',
        '车辆在车位可用性尚未确认时向被提前报告为可用的斜列停车位低速前向泊车。',
    ]
    members = [dict(scenario_id=str(i), vehicle_hazard=hazard, hazardous_event=event)
               for i, event in enumerate(events)]
    before = deepcopy(members)
    result = EngineeringReportTextMapper().safety_goal_presentation(members, '保留原安全目标')
    assert result['text'] == '防止车辆朝尚未确认可用的车位开始泊车。'
    assert len(result['members']) == len(members) and members == before
    assert all(r['basis']['vehicle_hazard'] == hazard for r in result['members'])
    # An unfamiliar source with only a dependent phrase keeps the original,
    # rather than manufacturing a predicate or a new collision consequence.
    unknown = EngineeringReportTextMapper().safety_goal_presentation([
        {'hazardous_event': '车辆基于提前输出的错误目标，因确认前提前输出。'}], '原安全意图')
    assert unknown['text'] == '原安全意图。'


def test_empty_physical_detail_describes_model_gap_without_changing_source():
    scenario = SimpleNamespace(facts={}, analysis_instance={'physical_plan_diagnostics': [
        {'reason': 'PHYSICAL_MODEL_NOT_APPLICABLE', 'uncovered_conditions': ['车库门交互未覆盖']}]})
    before = deepcopy(scenario)
    _, detail = EngineeringReportTextMapper().scenario(scenario)
    assert '当前可用物理模型不覆盖' in detail
    assert '运行细节未确定' not in detail and scenario == before


def timing(value, driver='outside_driver_seat', controls=('remote_control', 'braking'), **trace):
    return dict(hazard_time_limit_s=value, driver_position=driver, time_origin='FAULT_ONSET',
                physical_model='CONSTANT_ACCELERATION_TO_IMPACT_DISTANCE',
                hazard_criterion='LOSS_OF_BRAKING_AVOIDABILITY',
                timing_trace={'required_controls': list(controls), **trace})


def test_sg_timing_takes_minimum_within_response_scope_and_keeps_missing_members():
    items = [timing(12.53), timing(6), timing(9.18),
             timing(None, pending_reason='CONTROL_PATH_UNRESOLVED'),
             timing(.4, driver='in_driver_seat', controls=('driver_braking', 'braking')),
             timing(.2, response_model_family='MOVING_TARGET_FAULT_RELEASE',
                    response_path_status='REQUIRED_NOT_ESTABLISHED')]
    before = deepcopy(items)
    text = EngineeringReportTextMapper.safety_goal_timing(items, detailed=True)
    assert '6 s' in text and '12.53 s' not in text and '9.18 s' not in text
    assert '1个成员缺少适用控制路径' in text
    assert '0.4 s' in text and '在驾驶位' in text
    assert '0.2 s' in text and '解除异常减速' in text and '通道待建立' in text
    assert len(text.splitlines()) == 3  # three independent response scopes
    assert items == before


def test_many_sg_timing_paths_have_bounded_preview_and_complete_audit():
    items = [dict(timing(i / 10, controls=('braking', f'channel_{i}')), scenario_id=f'SCN-{i}')
             for i in range(1, 21)]
    items[-1]['hazard_time_limit_s'] = None
    items[-1]['timing_trace']['pending_reason'] = 'CONTROL_PATH_UNRESOLVED'
    before = deepcopy(items)
    compact = EngineeringReportTextMapper.safety_goal_timing(items)
    full = EngineeringReportTextMapper.safety_goal_timing(items, detailed=True)
    assert len(compact.splitlines()) == 4 and len(compact) < 350
    assert '0.1 s' in compact and '0.2 s' in compact
    assert '另有18条响应路径（18个成员），其中1个成员无有效边界' in compact
    assert len(full.splitlines()) == 20
    assert '1.9 s' in full and '缺少适用控制路径' in full
    assert '控制依赖' in compact and '时间追溯' in compact and items == before


def test_summary_counts_selected_branch_not_older_aggregate_grade(active_method, real_sample):
    state, trace = real_sample
    risk = state.risk_results[0]
    branch = deepcopy(risk.branch_evaluations[0])
    branch.update(C=None, ASIL=None)
    risk.branch_evaluations = [branch]
    trace['assessments'][0]['branch_evaluations'] = [branch]
    before = deepcopy((state, trace))
    view = HARAReportProjectionService(load_report_schema()).project(state, active_method, risk_trace=trace)
    assert risk.controllability.value == 'C3'
    assert view.scenario_details[0].controllability == '待评定'
    assert view.projection_metrics['score_status_counts']['controllability']['uncalculated'] == 1
    assert view.summary.scope['numeric_risk_chain_count'] == view.summary.scope['numeric_asil_count'] == 0
    assert '可控性: 待评定×1' in view.malfunction_inventory[0].risk_distribution
    assert (state, trace) == before


def test_sg_timing_is_written_by_production_renderer(active_method, real_sample, tmp_path):
    state, trace = real_sample
    before = deepcopy(state)
    schema = load_report_schema()
    view = HARAReportProjectionService(schema).project(state, active_method, risk_trace=trace)
    assert view.safety_goals[0].timing_boundary
    timing_evidence = view.projection_metrics['safety_goal_timing_evidence'][view.safety_goals[0].sg_id]
    assert timing_evidence['scenario_ids'] == state.safety_goals[0].associated_scenario_ids
    assert timing_evidence['full_response_scopes']
    path = tmp_path/'report.xlsx'
    HARAReportWorkbookRenderer().render(view, 'references/HARA_Template_AI_20260327.xlsx', path, schema)
    wb = load_workbook(path, read_only=True, data_only=True)
    assert wb['06_Safety Goal']['H3'].value == '最严条件性时间边界及范围'
    assert wb['06_Safety Goal']['H4'].value == view.safety_goals[0].timing_boundary
    assert wb['06_Safety Goal']['G4'].value == '条件性草稿'
    assert state == before
    wb.close()
    editable = load_workbook(path)
    assert editable['01_Item Definition'].column_dimensions['C'].width == 70
    editable.close()


def test_inventory_localization_keeps_identifiers_and_grade_labels():
    text = EngineeringReportTextMapper.engineering_terms(
        '对象：VEHICLE / CYCLIST；severity: S1×2；exposure: E2×2；RETAINED_FOR_RISK；MF-VEHICLE-01')
    assert text == '对象：其他车辆 / 骑行者；严重度: S1×2；暴露: E2×2；保留用于风险分析；MF-VEHICLE-01'


def test_offline_production_rescore_is_not_presented_as_new_full_causal_run(real_sample, active_method):
    state, trace = real_sample
    state.record('offline_risk_rescoring_completed', source_run_id='original-full-run')
    view = HARAReportProjectionService(load_report_schema()).project(state, active_method, risk_trace=trace)
    assert view.summary.scope['run_scope'] == 'OFFLINE FIXED-POPULATION REPLAY'
    assert 'no new full causal validation' in view.summary.scope['full_project_population']
    assert '0 new Provider calls' in view.summary.scope['scope_note']


def test_item_overview_keeps_function_and_phase_scope_with_renamed_ids(active_method, real_sample):
    state, trace = real_sample
    identity = state.functions[0]['function_id']
    state.functions[0]['name'] = '探索性泊车'
    state.item_definition['typed'] = {
        'odd_road_types': ['平行车位', '垂直车位'],
        'odd_scope_evidence': [{'field': 'odd_road_types', 'values': ['垂直车位'],
                                'function_ids': [identity], 'operating_modes': ['执行阶段']}],
    }
    before = deepcopy(state)
    view = HARAReportProjectionService(load_report_schema()).project(state, active_method, risk_trace=trace)
    assert dict(view.item_facts)['允许道路/车位'] == '平行车位、垂直车位（仅适用探索性泊车，模式：执行阶段）'
    assert state == before
    project = deepcopy(state.item_definition['typed'])
    project['odd_scope_evidence'][0]['function_ids'] = ['renamed-function-835']
    names = {'renamed-function-835': {'name': '换项目后的探索操作'}}
    text = EngineeringReportTextMapper.item_fact_text('odd_road_types', project['odd_road_types'], project, names)
    assert text == '平行车位、垂直车位（仅适用换项目后的探索操作，模式：执行阶段）'
    project['odd_scope_evidence'].append({'field': 'odd_road_types', 'values': ['垂直车位']})
    assert EngineeringReportTextMapper.item_fact_text('odd_road_types', project['odd_road_types'], project, names) == '平行车位、垂直车位'
