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


def test_incomplete_group_keeps_different_actual_exposure_bases(active_method):
    state = mixed_pending_group()
    trace = {'assessments': []}
    for i, risk in enumerate(state.risk_results):
        record = {'malfunction_id': risk.malfunction_id, 'scenario_id': risk.scenario_id,
                  'risk_scoring_invoked': True}
        for field in ('severity', 'exposure', 'controllability', 'asil'):
            evidence = getattr(risk, field)
            record[field] = {'result': evidence.value, 'status': evidence.status.value}
        record['exposure'].update(
            atom_bindings=[{'atom_id': f'place-{i}', 'used': True, 'e_rank': risk.exposure.value, 'dimensions': ['WHERE']}],
            scenario_terms={'WHERE': {'atom_id': f'place-{i}',
                'atom_label': '室外停车场' if i < 12 else '地下停车场', 'project_value': '来源允许的停车场'}},
            aggregation_rule='min_when_unequal')
        trace['assessments'].append(record)
    before = deepcopy((state, trace))
    row = HARAReportProjectionService(load_report_schema()).project(state, active_method, risk_trace=trace).rows[0]
    assert row.exposure == '待评定'
    assert '室外停车场=E3' in row.exposure_rationale
    assert '地下停车场=E2' in row.exposure_rationale
    assert '不同暴露等级组合取较低等级' in row.exposure_rationale
    assert '成员依据示例' in row.exposure_rationale
    assert 'E3×12' in row.exposure_rationale and 'E2×2' in row.exposure_rationale
    assert (state, trace) == before


def test_concrete_severity_gap_is_not_hidden_by_input_production_status():
    mapper = EngineeringReportTextMapper()
    trace = {'pending_reason': 'MISSING_RELATIVE_SPEED',
             'input_production': {'pending_reason': 'AUTONOMOUS_MODEL_UNRESOLVED', 'after': {}}}
    before = deepcopy(trace)
    text = mapper.severity_rationale(_pending(), trace)
    assert '碰撞相对速度' in text and '输入生产尚有未决项' not in text
    assert trace == before
    trace['input_production']['pending_reason'] = 'MOTION_MODEL_OR_ENVELOPE_NOT_SOURCED'
    text = mapper.severity_rationale(_pending(), trace)
    assert '碰撞相对速度' in text and '适用运动模型或速度包络' in text


def test_exposure_rank_labels_the_used_atom_not_the_search_phrase():
    mapper = EngineeringReportTextMapper(load_report_schema().presentation['language'])
    trace = {'result': 'E3', 'aggregation_rule': 'e3_e4_mix',
             'atom_bindings': [{'atom_id': 'FB006', 'used': True, 'E_class': 'E3', 'dimension': ['ROAD']}],
             'scenario_terms': {'ROAD': {'atom_id': 'FB006', 'atom_label': 'Reduced friction (µ < 0.8)',
                                        'project_value': '晴朗'}}}
    before = deepcopy(trace)
    evidence = EvidenceValue('E3', ReviewStatus.FINALIZED, rule_version='fixture')
    text = mapper.exposure_rationale(evidence, trace)
    assert '低附着路面（µ < 0.8）=E3' in text and '晴朗' not in text
    assert trace == before
    # A stale label for a different Atom must not explain this Atom's rank.
    trace['scenario_terms']['ROAD']['atom_id'] = 'different-atom'
    text = mapper.exposure_rationale(evidence, trace)
    assert '道路条件=E3' in text and '低附着' not in text


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
        assert '原目标' in result['text']  # unsupported wording retains its original goal
        assert result['members'][1]['malfunction'] == '另一种失效'
        assert result['members'][1]['wording_id'] == 'grounded_behavior_excerpt'
    # A claim about absent contact must not get rewritten into a collision goal.
    changed = {**rules[1]['source'], 'hazardous_event': '未发生接触'}
    displayed = mapper.safety_goal_presentation([changed], '原目标')
    assert displayed['members'][0]['wording_id'] == 'grounded_behavior_excerpt'
    assert '碰撞' not in displayed['text']
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
