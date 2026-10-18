"""Independent-review regressions: incomplete group, summary, existing SG intent."""
from collections import Counter
from copy import deepcopy

from openpyxl import load_workbook

from hara_agent.models import EvidenceValue, ReviewStatus, SafetyGoal
from hara_agent.services.reporting import HARAReportProjectionService, load_report_schema
from hara_agent.services.reporting.canonical_renderer import HARAReportWorkbookRenderer
from hara_agent.services.reporting.engineering_text_mapper import EngineeringReportTextMapper
from test_reviewer_report_closure import _p6_report_state, _pending
from test_reuse_reporting_acceptance import active_method


def mixed_pending_group():
    state = _p6_report_state()
    source, risk = state.scenarios[0], state.risk_results[0]
    causal = state.item_definition['scenario_assessments'][0]
    state.item_definition['scenario_assessments'] = []
    state.scenarios, state.risk_results = [], []
    for i in range(14):
        child, assessment = deepcopy(source), deepcopy(risk)
        child.scenario_id = assessment.scenario_id = f'SCN-{i}'
        assessment.assessment_id = f'RA-{i}'
        assessment.severity = EvidenceValue('S1', ReviewStatus.FINALIZED, rule_version='fixture')
        assessment.exposure = EvidenceValue('E3' if i < 12 else 'E2', ReviewStatus.FINALIZED, rule_version='fixture')
        assessment.controllability = _pending()
        assessment.asil = _pending()
        state.scenarios.append(child)
        state.risk_results.append(assessment)
        state.item_definition['scenario_assessments'].append({**deepcopy(causal), 'scenario_id': child.scenario_id})
    return state


def test_incomplete_group_does_not_display_first_member_noncommon_e(active_method):
    state = mixed_pending_group()
    before = deepcopy(state)
    view = HARAReportProjectionService(load_report_schema()).project(state, active_method)
    row = view.rows[0]
    assert (row.severity, row.exposure, row.controllability, row.asil) == ('S1', '待评定', '待评定', '待评定')
    assert 'E3×12' in row.exposure_rationale and 'E2×2' in row.exposure_rationale
    assert len(view.scenario_details) == 14
    assert Counter(d.exposure for d in view.scenario_details) == {'E3': 12, 'E2': 2}
    assert state == before


def test_incomplete_group_shared_value_and_branch_difference(active_method):
    state = mixed_pending_group()
    for risk in state.risk_results:
        risk.exposure = EvidenceValue('E3', ReviewStatus.FINALIZED, rule_version='fixture')
    project = HARAReportProjectionService(load_report_schema())
    assert project.project(state, active_method).rows[0].exposure == 'E3'
    state.risk_results[0].branch_evaluations = [{'S':'S1', 'E':'E2', 'C':None, 'ASIL':None}]
    view = project.project(state, active_method)
    assert view.rows[0].exposure == '待评定'
    assert '碰撞分支分布：E2×1' in view.rows[0].exposure_rationale


def test_summary_separates_review_groups_and_retained_records(tmp_path, active_method):
    schema = load_report_schema()
    view = HARAReportProjectionService(schema).project(mixed_pending_group(), active_method)
    output = tmp_path / 'report.xlsx'
    HARAReportWorkbookRenderer().render(view, 'references/HARA_Template_AI_20260327.xlsx', output, schema)
    workbook = load_workbook(output)
    summary = {row[1].value: row[2].value for row in workbook['00_Summary'].iter_rows(min_row=5)}
    assert summary['主表审核组数（04_HARA）'] == 1
    assert summary['风险记录数'] == summary['场景明细成员数'] == 14
    assert view.summary.eligible_hazardous_event_count == 14


def test_sg_rewording_requires_all_member_sources_and_keeps_status(active_method):
    schema = load_report_schema()
    mapper = EngineeringReportTextMapper(schema.presentation['language'])
    rules = schema.presentation['language']['safety_goal_rewordings']
    for rule in rules:
        member = {'scenario_id': 'A', 'malfunction_id': 'MF', **rule['source']}
        result = mapper.safety_goal_presentation([member], '原目标')
        assert result['text'] == rule['text']
        assert result['members'][0]['basis'] == rule['source']
        changed = {**member, 'scenario_id': 'B', 'vehicle_hazard': '另一种危险机制', 'malfunction': '另一种失效'}
        result = mapper.safety_goal_presentation([member, changed], '原目标')
        assert '另一种失效' in result['text']  # never narrow all members to the first
        assert result['members'][1]['wording_id'] == 'original_intent'
    # A claim about absent contact must not get rewritten into a collision goal.
    changed = {**rules[1]['source'], 'hazardous_event': '未发生接触'}
    assert mapper.safety_goal_presentation([changed], '原目标')['members'][0]['wording_id'] == 'original_intent'
    state = mixed_pending_group()
    state.malfunctions[0].update(description=rules[0]['source']['malfunction'],
        vehicle_level_hazard=rules[0]['source']['vehicle_hazard'])
    state.risk_results[0].safety_goal_id = 'SG-1'
    state.safety_goals = [SafetyGoal('SG-1', '原目标', '待确定', 'B', ['SCN-0'], ReviewStatus.PENDING)]
    before = deepcopy(state)
    view = HARAReportProjectionService(schema).project(state, active_method)
    assert view.safety_goals[0].safety_goal == rules[0]['text']
    assert view.safety_goals[0].original_text == '原目标'
    assert state == before


def test_harm_translation_preserves_qualifiers_without_inferring_missing_harm():
    mapper = EngineeringReportTextMapper(load_report_schema().presentation['language'])
    text = mapper.potential_harm('Severe injuries, possibly life-threatening, survival probable', severity=_pending())
    assert '可能危及生命' in text and '生存可能性较高' in text
    assert mapper.potential_harm('unknown source phrase', severity=_pending()) == 'unknown source phrase'
    assert mapper.potential_harm('', severity=_pending()) == '待S评定完成后确定'
