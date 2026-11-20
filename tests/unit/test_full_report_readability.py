"""Reviewer presentation regressions independent of one Provider's wording."""
from copy import deepcopy
from types import SimpleNamespace

import pytest
from openpyxl import load_workbook

from hara_agent.models import EvidenceValue, ReviewStatus
from hara_agent.services.reporting import HARAReportProjectionService, load_report_schema
from hara_agent.services.reporting.canonical_renderer import HARAReportWorkbookRenderer
from hara_agent.services.reporting.engineering_text_mapper import EngineeringReportTextMapper
from test_reviewer_report_closure import _p6_report_state, ROOT
from test_reuse_reporting_acceptance import active_method


@pytest.mark.parametrize('label,chinese', [
    ('Normal lateral accel (aY < 3 m/s²)', '正常侧向加速度'),
    ('Increased lateral accel (3 ≤ aY < 5 m/s²)', '较高侧向加速度'),
    ('High lateral accel (5 ≤ aY < 7 m/s²)', '高侧向加速度'),
    ('Very high lateral accel (aY > 7 m/s²)', '很高侧向加速度'),
])
@pytest.mark.parametrize('long_form', [False, True])
def test_lateral_method_labels_are_chinese_without_changing_bands_or_trace(label, chinese, long_form):
    mapper = EngineeringReportTextMapper()
    if long_form:
        label = label.replace('accel ', 'acceleration ')
    threshold = label[label.index('('):]
    trace = {'atom_details': [{'atom_id': 'unseen-lateral-method', 'used': True,
        'e_rank': 'E2', 'dimensions': ['EGO_DYNAMICS']}],
        'scenario_terms': {'EGO_DYNAMICS': {'atom_id': 'unseen-lateral-method', 'atom_label': label}},
        'result': 'E2'}
    before = deepcopy(trace)
    assert mapper.engineering_terms(label) == chinese+' '+threshold
    text = mapper.exposure_rationale(EvidenceValue('E2', ReviewStatus.PENDING), trace, calculated=True)
    assert chinese+' '+threshold+'=E2' in text
    assert 'lateral' not in text.casefold() and trace == before
    identity = 'RAW_Normal_lateral_accel_v3'
    assert identity in mapper.engineering_terms('方法身份 '+identity+' 保留')
    original = 'The operator observes normal lateral acceleration during parking.'
    assert mapper.engineering_terms(original) == original


@pytest.mark.parametrize('interval', [None, [-5, -3]])
def test_target_longitudinal_coverage_keeps_checked_and_assumed_scope_distinct(interval):
    limitation = ('目标有效侧面覆盖接触点；已按所列区间检查。' if interval
        else '目标有效侧面须覆盖接触点；尚未提供数值区间，未完成纵向覆盖核验。')
    facts = {'fault_motion_model': 'CONSTANT_SPEED_CURVATURE_TO_CONTACT',
        'motion_model_outcome': {'status': 'CONTACT', 'target_longitudinal_coverage': {
            'contact_x_m': -3.799671, 'coordinate_frame': 'BODY_FORWARD_POSITIVE',
            'assessment_reference': 'MODEL_INITIAL_STATE', 'finite_span_checked': bool(interval),
            'effective_contact_interval_m': interval,
            'contact_within_interval': True if interval else None, 'limitation': limitation}}}
    before = deepcopy(facts)
    text = EngineeringReportTextMapper.physical_inputs(SimpleNamespace(facts=facts), {})
    assert '侧方接触点纵向坐标：-3.8 m' in text
    assert limitation in text
    assert ('目标有效纵向接触区间' in text) is bool(interval)
    assert facts == before


def test_legacy_curve_without_coverage_trace_does_not_claim_finite_object_validation():
    facts = {'fault_motion_model': 'CONSTANT_SPEED_CURVATURE_TO_CONTACT',
             'motion_model_outcome': {'status': 'CONTACT'}}
    before = deepcopy(facts)
    text = EngineeringReportTextMapper.physical_inputs(SimpleNamespace(facts=facts), {})
    assert '本历史结果未单独核验有限对象的纵向覆盖' in text
    assert '不能等同于任意单辆车辆的接触' in text
    assert '目标有效纵向接触区间' not in text and facts == before


@pytest.mark.parametrize("outcome", ["CONTACT", "NO_CONTACT_WITHIN_MODEL", "GEOMETRY_NOT_APPLICABLE"])
def test_foreign_event_uses_current_world_without_inventing_contact(outcome):
    mapper = EngineeringReportTextMapper()
    scenario = SimpleNamespace(facts={"road_user_type": "CYCLIST", "object_position": "REAR",
        "motion_model_outcome": {"status": outcome}, "first_contact_time_s": 1.234567,
        "physical_analysis": {"hypothesis": {"conditions": ["conditional world"]}}}, analysis_instance={})
    original = "The rear cyclist approaches the unexpected stop, without claiming approved vehicle capability."
    before = deepcopy(scenario)
    text = mapper.review_event(original, scenario, {"description": "车辆非预期减速"})
    assert "骑行者" in text and "后方" in text and "结构化摘要" in text
    assert "完整因果声明及限定" in text
    assert ("在1.23 s" in text) is (outcome == "CONTACT")
    assert ("未确认接触" in text) is (outcome == "NO_CONTACT_WITHIN_MODEL")
    assert scenario == before
    # Unknown projects without enough structured context keep their original.
    assert original.split(",")[0] in mapper.review_event(original, scenario, {})


def test_chinese_control_qualification_is_retained_and_foreign_state_uses_transition():
    mapper = EngineeringReportTextMapper()
    scenario = SimpleNamespace(facts={}, analysis_instance={})
    candidate = {"description": "只在自动制动路径保留时于目标前停车", "required_controls": ["braking"],
        "transition": {"model": "CONSTANT_NET_DECELERATION", "net_deceleration_mps2": 1.2345}}
    assert candidate["description"] in mapper.safe_state_member(candidate, scenario)
    candidate["description"] = "Bring the vehicle to a controlled standstill before the contact envelope."
    text = mapper.safe_state_member(candidate, scenario)
    assert "目标状态：车辆停止" in text and "1.23 m/s²" in text
    assert "响应避碰效果尚待验证" in text and "接触前停止" not in text
    assert "保持静止" not in text  # no holding phase was supplied


def test_mixed_contract_tokens_are_localized_without_rewriting_ids_or_conditions():
    mapper = EngineeringReportTextMapper()
    raw = ('仅在outside_driver_seat及automatic_braking_command独立时，目标向RIGHT横穿；'
           '7.123 kph；NO_CONTACT_WITHIN_MODEL；记录 SCN-RIGHT-123、RIGHT_01、rule-v1.2')
    text = mapper.hazardous_event(raw)
    assert '仅在不在驾驶位及自动制动命令链独立时' in text
    assert '向右横穿' in text and '7.1 km/h' in text and '模型时域内未接触' in text
    assert 'SCN-RIGHT-123、RIGHT_01、rule-v1.2' in text
    assert 'outside_driver_seat' in raw
    assert mapper.engineering_terms('本车向FORWARD方向行驶，目标向REVERSE运动') == '本车向前行驶，目标向后运动'


def test_control_scope_terms_and_timing_reason_keep_qualifications_and_ids():
    mapper = EngineeringReportTextMapper()
    raw = ('driver_configurations为空；未把required路径加入remaining_controls；'
           'remaining braking是假设，不是车辆已确认能力；多故事车库内的passenger car；'
           '记录 ID-driver_configurations-01、remaining_controls_v2。')
    text = mapper.engineering_terms(raw)
    assert '驾驶员配置为空' in text and '加入保留的控制能力' in text
    assert '保留的 制动执行器是假设，不是车辆已确认能力' in text
    assert '多层车库内的乘用车' in text
    assert 'ID-driver_configurations-01、remaining_controls_v2' in text
    timing = [{'reason': 'braking 执行能力失效，未证明独立路径；记录 CTRL-braking-1',
               'contact_time_s': 1.2, 'contact_time_kind': 'first_contact_time_s'}]
    before = deepcopy(timing)
    explanation = mapper.ftti_rationale(timing)
    assert '制动执行器 执行能力失效，未证明独立路径' in explanation
    assert 'CTRL-braking-1' in explanation and '接触时间不能作为 FTTI' in explanation
    assert timing == before
    boundary = mapper.engineering_terms(
        '未建立LOSS_OF_BRAKING_AVOIDABILITY；FIRST_CONTACT_BOUND不是FTTI；'
        'FAULT_ONSET尚未知；不能借PROJECT条件认定CONSTANT_NET_DECELERATION响应有效；'
        '前方发生FRONTAL接触，ID-FAULT_ONSET-1保持不变。')
    assert '未建立失去制动避碰能力边界；首次接触边界不是FTTI' in boundary
    assert '故障发生时刻尚未知' in boundary and '恒定净减速度响应有效' in boundary
    assert '前方发生正面接触，ID-FAULT_ONSET-1保持不变' in boundary
    assert mapper.engineering_terms('缺少 fault_release 响应；暴露 exposure；ID-fault_release-1保留') == (
        '缺少 故障作用解除 响应；暴露 暴露；ID-fault_release-1保留')


@pytest.mark.parametrize('model', list(EngineeringReportTextMapper._PHYSICAL_MODEL_TEXT))
def test_model_names_and_vehicle_alias_are_localized_in_prose_but_not_ids(model):
    mapper = EngineeringReportTextMapper()
    raw = f'静止passenger_car；本工况采用{model}，不证明处置有效；记录 ID-{model}-1、passenger_car_01。'
    text = mapper.hazardous_event(raw)
    assert '静止乘用车' in text and mapper._PHYSICAL_MODEL_TEXT[model] in text
    assert '不证明处置有效' in text
    assert f'ID-{model}-1、passenger_car_01' in text
    timing = [{'reason': raw, 'contact_time_s': None}]
    before = deepcopy(timing)
    rationale = mapper.ftti_rationale(timing)
    assert mapper._PHYSICAL_MODEL_TEXT[model] in rationale and '静止乘用车' in rationale
    assert timing == before


def test_chinese_function_projection_localizes_view_labels_without_mutating_source(active_method):
    from copy import deepcopy
    state = _p6_report_state()
    for function in state.functions:
        function['output'] = '在Rear-、Front-或TopView中显示车位；记录 ID-TopView-01'
    before = deepcopy(state.functions)
    view = HARAReportProjectionService(load_report_schema()).project(state, active_method)
    assert view.rows and view.function_inventory
    assert all('后视、前视或俯视画面' in row.function_output for row in view.rows)
    assert all('ID-TopView-01' in row.function_output for row in view.rows)
    assert state.functions == before


def test_operation_and_configuration_terms_use_one_projection_for_function_and_failure(active_method):
    state = _p6_report_state()
    for function in state.functions:
        function['name'] = '速度 profile 选择'
        function['output'] = '执行泊车 maneuver；保留记录 ID-profile-1'
    for malfunction in state.malfunctions:
        malfunction['description'] = '所选 profile 错误，泊车 maneuver 偏离要求'
        malfunction['vehicle_level_hazard'] = 'maneuver 使车辆侵入相邻车位；ID-maneuver-2'
    before = deepcopy(state)
    view = HARAReportProjectionService(load_report_schema()).project(state, active_method)
    assert view.rows and view.function_inventory and view.malfunction_inventory
    for row in view.rows:
        assert row.function_name == '速度 配置方案 选择'
        assert '泊车 操作' in row.function_output and 'ID-profile-1' in row.function_output
        assert row.malfunction.startswith('所选 配置方案 错误')
        assert row.hazard.startswith('操作 使车辆侵入') and 'ID-maneuver-2' in row.hazard
    assert all(r.malfunction.startswith('所选 配置方案 错误') for r in view.malfunction_inventory)
    assert state == before


def test_localization_reaches_structured_event_fallback_exposure_and_safety_goals(active_method):
    mapper = EngineeringReportTextMapper()
    scenario = SimpleNamespace(facts={'road_user_type': 'PEDESTRIAN', 'object_position': 'FRONT',
        'motion_model_outcome': {'status': 'CONTACT'}, 'first_contact_time_s': 1.234}, analysis_instance={})
    event = mapper.review_event('The display is stale during the parking operation.', scenario,
        {'description': 'TopView延迟期间车辆继续运动'})
    assert '俯视画面延迟' in event and 'TopView' not in event
    assert mapper.hazardous_event('车辆在Parkplatz/Parkhaus/Garage区域运动') == '车辆在停车场/停车楼/车库区域运动。'
    evidence = EvidenceValue('E2', ReviewStatus.PENDING)
    trace = {'atom_details': [{'atom_id': 'unseen', 'used': True, 'e_rank': 'E3', 'dimensions': ['EGO_DYNAMICS']}],
        'scenario_terms': {'EGO_DYNAMICS': {'atom_id': 'unseen', 'atom_label': 'Deceleration normal (< 2.44 m/s²)'}},
        'result': 'E2'}
    before_trace = deepcopy(trace)
    assert '正常减速 (< 2.44 m/s²)=E3' in mapper.exposure_rationale(evidence, trace, calculated=True)
    assert trace == before_trace
    state = _p6_report_state()
    for malfunction in state.malfunctions:
        malfunction['vehicle_level_hazard'] = '车辆在TopView延迟期间继续运动，可能碰撞行人'
    from hara_agent.models import SafetyGoal
    state.safety_goals = [SafetyGoal('SG-display', '原目标TopView', '待确定', '',
        [risk.scenario_id for risk in state.risk_results])]
    for risk in state.risk_results:
        risk.safety_goal_id = 'SG-display'
    before = deepcopy(state)
    view = HARAReportProjectionService(load_report_schema()).project(state, active_method)
    assert view.safety_goals
    assert all('TopView' not in goal.safety_goal and 'TopView' not in goal.hazard for goal in view.safety_goals)
    assert any('俯视画面' in goal.safety_goal for goal in view.safety_goals)
    assert view.safety_goals[0].original_text == '原目标TopView'
    assert state == before


@pytest.mark.parametrize("parent_type,selected_type", [
    ("平行停车位", "垂直停车位"), ("分流车道", "汇入车道"),
])
@pytest.mark.parametrize("location", [None, "室外试验场"])
def test_selected_road_condition_does_not_keep_parent_type_as_location(parent_type, selected_type, location):
    scenario = SimpleNamespace(operating_scenario="parking area", operating_mode="parking_vehicle_control",
        facts={"method_scenario_dimensions": {"WHERE": {"project_value": location or parent_type}},
            "source_operating_conditions": {"operating_scenario": location or parent_type,
                "permitted_road_types": [parent_type, selected_type]},
            "selected_source_condition": {"source_field": "odd_road_types", "value": selected_type,
                "physical_effect": "所选运行条件决定目标相对位置"}}, analysis_instance={})
    before = deepcopy(scenario)
    description, details = EngineeringReportTextMapper().scenario(scenario)
    assert selected_type in description and parent_type not in description
    assert "所选运行条件决定目标相对位置" in details
    if location:
        assert location in description
    assert scenario == before


def test_group_with_different_selected_types_does_not_revive_parent_type_as_location():
    base = SimpleNamespace(operating_scenario="parking area", operating_mode="parking_vehicle_control",
        atomic_variant="typical", context_resolution={}, analysis_instance={}, facts={
            "method_scenario_dimensions": {"WHERE": {"project_value": "旧车位类型"}},
            "source_operating_conditions": {"operating_scenario": "旧车位类型",
                "permitted_road_types": ["旧车位类型", "新类型甲", "新类型乙"]}})
    scenarios = [deepcopy(base), deepcopy(base)]
    for scenario, choice in zip(scenarios, ["新类型甲", "新类型乙"]):
        condition = {"source_field": "odd_road_types", "value": choice, "physical_effect": choice + "的几何影响"}
        scenario.facts['selected_source_condition'] = condition
        scenario.facts['source_operating_conditions']['selected_condition'] = deepcopy(condition)
    before = deepcopy(scenarios)
    text, _ = HARAReportProjectionService(load_report_schema())._group_scenario(
        [{'scenario': scenario} for scenario in scenarios])
    assert all(value not in text for value in ["旧车位类型", "新类型甲", "新类型乙"])
    assert scenarios == before


def test_same_qm_with_missing_c_is_not_described_as_unknown_group_upper_bound(active_method):
    state = _p6_report_state()
    for risk in state.risk_results:
        risk.severity = EvidenceValue("S0", ReviewStatus.PENDING, review_reason="CONDITIONAL_TRIAL")
        risk.controllability = EvidenceValue(None, ReviewStatus.PENDING)
        risk.asil = EvidenceValue("QM", ReviewStatus.PENDING, review_reason="CONDITIONAL_TRIAL")
        risk.branch_evaluations = []
    view = HARAReportProjectionService(load_report_schema()).project(state, active_method)
    assert all(row.asil == "QM" and row.controllability == "待评定" for row in view.rows)
    assert all("不是全组上界" not in row.remark for row in view.rows)


def test_report_groups_by_inventory_and_retains_originals_and_dynamic_cells(active_method, tmp_path):
    state = _p6_report_state()
    original = "Unexpected braking produces a rear interaction under the declared conditional geometry."
    state.risk_results[0].hazardous_event = original
    state.scenarios[0].facts.update(road_user_type="VEHICLE", object_position="REAR",
        motion_model_outcome={"status": "NO_CONTACT_WITHIN_MODEL"},
        control_analysis={"hypothesis": {"conditions": ["Independent driver braking remains outside the drive fault"]}},
        selected_source_condition={"value": "斜向车位", "physical_effect": "The bay markings define the planned entry geometry, not a barrier."})
    # Add an independent function with the SAME displayed name and an earlier
    # scenario hash: the source inventory, not hash order or name, owns ordering.
    function = {**deepcopy(state.functions[0]), "function_id": "other-function"}
    state.functions.append(function)
    mf = {**deepcopy(state.malfunctions[0]), "malfunction_id": "other-fault", "function_id": "other-function"}
    state.malfunctions.append(mf)
    scenario, risk = deepcopy(state.scenarios[0]), deepcopy(state.risk_results[0])
    scenario.scenario_id = risk.scenario_id = "000-earlier-hash"
    risk.assessment_id = "other-risk"
    risk.malfunction_id = "other-fault"
    state.scenarios.append(scenario); state.risk_results.append(risk)
    before = deepcopy(state)
    schema = load_report_schema()
    view = HARAReportProjectionService(schema).project(state, active_method)
    assert view.rows[-1].function_id == "other-function"
    assert view.rows[0].malfunction_id == state.malfunctions[0]["malfunction_id"]
    assert original in view.audit_references[0].original_analysis_text
    assert "not a barrier" in view.audit_references[0].original_analysis_text
    assert "Independent driver braking remains outside the drive fault" in view.audit_references[0].original_analysis_text
    output = tmp_path / "review.xlsx"
    HARAReportWorkbookRenderer().render(view, ROOT / "references/HARA_Template_AI_20260327.xlsx", output, schema)
    with_book = load_workbook(output)
    try:
        sheet = with_book["04_HARA"]
        assert sheet.column_dimensions["I"].width >= schema.field("hazardous_event").width
        last_row = 5 + len(view.rows)
        for area in sheet.merged_cells.ranges:
            if area.max_row < 6:
                continue
            assert area.min_col == area.max_col and 2 <= area.min_col <= 7
            assert area.max_row < last_row  # equal name must not merge two identities
        for index, row in enumerate(view.rows, start=6):
            assert (sheet.cell(index, 17).value or "") == row.asil
            assert (sheet.cell(index, 21).value or "") == row.sg_id
        audit = with_book["99_Audit"]
        assert audit.cell(4, audit.max_column).value == "原始分析说明（保留原语言）"
        assert original in audit.cell(5, audit.max_column).value
        detail = with_book['04A_Scenario Detail']
        for cell in detail[4]:
            if cell.value in schema.presentation['scenario_detail']['column_widths']:
                assert detail.column_dimensions[cell.column_letter].width == schema.presentation['scenario_detail']['column_widths'][cell.value]
    finally:
        with_book.close()
    assert state == before


def test_method_glossary_translates_thresholds_without_changing_trace():
    mapper = EngineeringReportTextMapper(load_report_schema().presentation["language"])
    trace = {"status": "FINALIZED", "result": "E2", "aggregation_rule": "min_when_unequal",
        "atom_bindings": [{"atom_id": "split", "e_rank": "E2", "used": True, "dimension": "ROAD"}],
        "scenario_terms": {"ROAD": {"atom_id": "split", "atom_label": "µ-split (delta-µ > 0.3 left/right)"}}}
    before = deepcopy(trace)
    result = mapper.exposure_rationale(EvidenceValue('E2', ReviewStatus.FINALIZED, rule_version='method'), trace)
    assert "左右分离附着路面（附着系数差>0.3）=E2" in result
    assert "不同暴露等级组合取较低等级，最终 E2" in result
    trace['scenario_terms']['ROAD']['atom_id'] = 'other'
    assert '附着系数差' not in mapper.exposure_rationale(EvidenceValue('E2', ReviewStatus.FINALIZED, rule_version='method'), trace)
    trace['scenario_terms']['ROAD']['atom_id'] = 'split'
    assert trace == before


def test_acceleration_and_reverse_glossary_preserve_all_numeric_boundaries():
    import re
    glossary = load_report_schema().presentation['language']['scenario_condition_translations']
    entries = {key: value for key, value in glossary.items()
               if key.startswith('Longitudinal accel ') or key == 'Reversing (v > 10 kph)'}
    assert len(entries) == 9
    for original, translated in entries.items():
        assert re.findall(r'\d+(?:\.\d+)?|[<>≤≥]', original) == re.findall(r'\d+(?:\.\d+)?|[<>≤≥]', translated)
        assert 'Longitudinal' not in translated and 'Reversing' not in translated
        if '(time)' in original:
            assert '时间口径' in translated
        if '(freq)' in original:
            assert '频次口径' in translated
