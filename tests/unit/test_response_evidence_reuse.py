"""Calculated response proof and existing clocks are not lost in projection."""
from copy import deepcopy
from pathlib import Path
import gzip
import json

import pytest

from hara_agent.contracts.physical_scenario_plan import controls_for_driver
from hara_agent.services.analysis.risk_calculation_input_service import RiskCalculationInputService
from hara_agent.services.analysis.scenario_physics import braking_avoidability_boundary
from hara_agent.services.reporting.engineering_text_mapper import EngineeringReportTextMapper
from test_autonomous_physics import instantiate, make_candidate, method
from test_bounded_interaction_coverage import moving_plan
from test_scoped_timing_and_avoidance import timed_plan

FIXTURE = json.loads(gzip.decompress((Path(__file__).resolve().parents[1] / 'fixtures/response_evidence_reuse.json.gz').read_bytes()))


def timing(method, child):
    return RiskCalculationInputService().ftti('MF', child.scenario_id, '',
        {**child.facts, '_fact_provenance': child.fact_provenance}, method=method).to_dict()


def response_plan(raw):
    h = raw['hypotheses'][0]
    base = timed_plan()['hypotheses'][0]
    h.update({k: deepcopy(base[k]) for k in ('safe_state_proposal', 'hazard_timing')})
    h['safe_state_proposal']['transition']['post_stop_motion'] = 'HOLD_POSITION'
    return raw


def test_clock_unbound_retains_checked_same_proposal_without_inventing_ftti(method):
    raw = timed_plan(); raw['hypotheses'][0].pop('hazard_timing')
    child = instantiate(method, raw, make_candidate(3.6))[0]
    result = timing(method, child)
    assert result['status'] == 'PENDING_INPUT' and result['hazard_time_limit_s'] is None
    assert result['timing_trace']['pending_reason'] == 'FAULT_CLOCK_UNRESOLVED'
    assert result['timing_trace']['status'] == 'CONDITIONAL_BOUNDARY'
    proposal = {'scenario_id': child.scenario_id, **child.facts['physical_analysis']['hypothesis']['safe_state_proposal']}
    state, text = EngineeringReportTextMapper.safe_state_outcome(proposal, result)
    assert state.endswith('_CHECKED_CLOCK_UNBOUND')
    assert '在已检查范围内避免与目标接触' in text and '尚不能作为FTTI' in text
    changed = {**proposal, 'description': 'another proposal'}
    assert EngineeringReportTextMapper.safe_state_outcome(changed, result)[0].endswith('_UNVERIFIED')


def test_computed_unsafe_same_proposal_is_not_relabelled_unverified(method):
    raw = response_plan(moving_plan(initial=3.6, target_direction='FORWARD', target_speed=7.2,
        x=-8, position='REAR', collision='REAR_END'))
    raw['hypotheses'][0].pop('hazard_timing')
    child = instantiate(method, raw, make_candidate(3.6))[0]
    result = timing(method, child)
    assert result['timing_trace']['status'] == 'NO_IMMEDIATE_SAFE_BRAKING_WINDOW'
    assert result['hazard_time_limit_s'] is None
    proposal = {'scenario_id': child.scenario_id, **child.facts['physical_analysis']['hypothesis']['safe_state_proposal']}
    assert EngineeringReportTextMapper.safe_state_outcome(proposal, result)[0] == 'UNSAFE_RESPONSE'


@pytest.mark.parametrize('case', FIXTURE['stationary_no_contact'], ids=lambda c: c['scenario_id'])
def test_real_stationary_envelope_no_contact_still_checks_selected_response(method, case):
    before = deepcopy(case)
    result = RiskCalculationInputService().ftti(case['malfunction_id'], case['scenario_id'], '',
        {**case['facts'], '_fact_provenance': case['fact_provenance']}, method=method).to_dict()
    trace = result['timing_trace']
    assert result['status'] == 'PENDING_INPUT' and result['hazard_time_limit_s'] is None
    assert trace['status'] == 'NO_UNMITIGATED_CONTACT_BOUNDARY'
    assert trace['original']['status'] == trace['response_contact']['status'] == 'NO_CONTACT_WITHIN_MODEL'
    assert trace['response_contact']['evaluated_until_s'] == case['facts']['model_valid_until_s']
    assert trace['response_model_family'] == 'RECTILINEAR_STATIONARY_TARGET_BRAKING'
    h = (case['facts'].get('control_analysis') or case['facts']['physical_analysis'])['hypothesis']
    proposal = {'scenario_id': case['scenario_id'], **h['safe_state_proposal']}
    assert EngineeringReportTextMapper.safe_state_outcome(proposal, result)[0].endswith('_IMMEDIATE_ONLY')
    assert case == before


def test_stationary_contact_keeps_original_closed_form_numeric_path(method):
    raw = response_plan(moving_plan(initial=3.6, target_direction='STATIONARY', target_speed=0, x=4))
    child = instantiate(method, raw, make_candidate(3.6))[0]
    result = timing(method, child)
    distance = child.facts['first_contact_time_s']
    expected = braking_avoidability_boundary(1, 0, distance, 3)
    assert result['hazard_time_limit_s'] == expected['time_s']
    assert result['safe_state_transition_time_s'] == expected['stopping_time_s']
    assert result['timing_trace']['status'] == expected['status']
    assert result['status'] == 'CONDITIONAL'


def test_stationary_no_contact_needs_an_explicit_post_stop_model(method):
    raw = response_plan(moving_plan(initial=3.6, target_direction='STATIONARY', target_speed=0, x=4, y=2))
    raw['hypotheses'][0]['safe_state_proposal']['transition'].pop('post_stop_motion')
    child = instantiate(method, raw, make_candidate(3.6))[0]
    result = timing(method, child)
    assert result['timing_trace']['pending_reason'] == 'POST_STOP_MOTION_UNRESOLVED'
    assert result['hazard_time_limit_s'] is None
    assert '有限接触包络' in result['reason'] and '移动目标需' not in result['reason']


def test_real_common_fault_clock_can_serve_own_local_response_without_shared_response(method):
    case = FIXTURE['saved_fault_clock']; raw = deepcopy(case['raw_hypothesis']); before = deepcopy(raw)
    effective = controls_for_driver(raw, 'in_driver_seat')
    assert not raw.get('safe_state_proposal')
    assert effective['hazard_timing'] == raw['hazard_timing']
    assert effective['safe_state_proposal'] == raw['driver_control_assessments']['in_driver_seat']['safe_state_proposal']
    facts = deepcopy(case['facts']); facts['physical_analysis']['hypothesis'] = effective
    result = RiskCalculationInputService().ftti(case['malfunction_id'], case['scenario_id'], '',
        {**facts, '_fact_provenance': case['fact_provenance']}, method=method)
    assert result.hazard_time_limit_s == case['expected_time_s']
    assert result.status.value == 'CONDITIONAL'
    assert raw == before


@pytest.mark.parametrize('limitation', ['explicit_gap', 'local_clock', 'no_response', 'other_criterion'])
def test_common_clock_reuse_respects_explicit_local_meaning_and_active_method(method, limitation):
    case = FIXTURE['saved_fault_clock']; raw = deepcopy(case['raw_hypothesis'])
    local = raw['driver_control_assessments']['in_driver_seat']
    if limitation == 'explicit_gap': local['timing_unresolved_reason'] = 'The local clock has not been related to this response'
    elif limitation == 'local_clock': local['hazard_timing'] = {**raw['hazard_timing'], 'fault_to_model_start_s': .5}
    elif limitation == 'no_response': local.pop('safe_state_proposal')
    else: raw['hazard_timing']['hazard_criterion'] = 'FIRST_CONTACT_BOUND'
    effective = controls_for_driver(raw, 'in_driver_seat')
    if limitation in {'explicit_gap', 'no_response'}:
        assert not effective.get('hazard_timing')
    elif limitation == 'local_clock':
        assert effective['hazard_timing'] == local['hazard_timing']
    else:
        assert effective['hazard_timing']['hazard_criterion'] == 'FIRST_CONTACT_BOUND'
        facts = deepcopy(case['facts']); facts['physical_analysis']['hypothesis'] = effective
        result = RiskCalculationInputService().ftti(case['malfunction_id'], case['scenario_id'], '',
            {**facts, '_fact_provenance': case['fact_provenance']}, method=method)
        assert result.hazard_time_limit_s is None
        assert result.timing_trace['pending_reason'] == 'FAULT_CLOCK_UNRESOLVED'


def test_stationary_no_contact_does_not_hide_unsafe_response_after_original_stop(method):
    raw = response_plan(moving_plan(initial=7.2, target_direction='STATIONARY',
        target_speed=0, x=3, acceleration=-2, horizon=6))
    raw['hypotheses'][0]['safe_state_proposal']['transition']['net_deceleration_mps2'] = .5
    child = instantiate(method, raw, make_candidate(7.2))[0]
    before = deepcopy(child)
    result = timing(method, child)
    trace = result['timing_trace']
    assert trace['original'] == {'status': 'NO_CONTACT_WITHIN_MODEL',
        'evaluated_until_s': 1.0, 'limiting_condition': 'EGO_STOP'}
    assert result['status'] == 'NOT_APPLICABLE' and result['hazard_time_limit_s'] is None
    assert trace['status'] == 'NO_IMMEDIATE_SAFE_BRAKING_WINDOW'
    assert trace['response_contact']['time_s'] == pytest.approx(4 - 8 ** .5)
    assert trace['response_contact']['time_s'] > trace['original']['evaluated_until_s']
    assert trace['response_model_family'] == 'RECTILINEAR_STATIONARY_TARGET_BRAKING'
    proposal = {'scenario_id': child.scenario_id,
        **child.facts['physical_analysis']['hypothesis']['safe_state_proposal']}
    assert EngineeringReportTextMapper.safe_state_outcome(proposal, result)[0] == 'UNSAFE_RESPONSE'
    assert EngineeringReportTextMapper.ftti_value([result]) == '所选响应不能避碰'
    assert child == before


@pytest.mark.parametrize('analysis_bound', [True, False])
def test_stationary_no_contact_with_explicit_hold_requires_own_analysis_source(method, analysis_bound):
    raw = response_plan(moving_plan(initial=3.6, target_speed=0,
        target_direction='FORWARD', x=8, horizon=2))
    child = instantiate(method, raw, make_candidate(3.6))[0]
    if not analysis_bound:
        child.fact_provenance.pop('physical_analysis')
    before = deepcopy(child)
    result = timing(method, child)
    assert result['hazard_time_limit_s'] is None and result['status'] == 'PENDING_INPUT'
    proposal = {'scenario_id': child.scenario_id,
        **child.facts['physical_analysis']['hypothesis']['safe_state_proposal']}
    if analysis_bound:
        assert result['timing_trace']['status'] == 'NO_UNMITIGATED_CONTACT_BOUNDARY'
        assert result['timing_trace']['response_contact']['status'] == 'NO_CONTACT_WITHIN_MODEL'
        assert result['timing_trace']['response_contact']['evaluated_until_s'] == 2
        assert EngineeringReportTextMapper.safe_state_outcome(proposal, result)[0].endswith('_IMMEDIATE_ONLY')
    else:
        assert result['timing_trace']['pending_reason'] == 'RESPONSE_INPUTS_UNRESOLVED'
        assert '时间假设未绑定当前失效与场景' in result['reason']
        assert EngineeringReportTextMapper.safe_state_outcome(proposal, result)[0].endswith('_UNVERIFIED')
    assert child == before
