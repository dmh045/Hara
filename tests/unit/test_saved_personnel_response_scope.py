"""Saved real manual responses stay conditional under a validated request scope."""
from copy import deepcopy
from dataclasses import asdict
import gzip
import json
from pathlib import Path

import pytest

from hara_agent.contracts.physical_scenario_plan import PhysicalScenarioPlan, controls_for_driver
from hara_agent.models import ItemDefinitionFacts, MalfunctionCandidate, ReviewStatus, SourceRef
from hara_agent.services.analysis.physical_hypothesis_instantiation_service import PhysicalHypothesisInstantiationService
from hara_agent.services.analysis.risk_calculation_input_service import RiskCalculationInputService
from hara_agent.services.analysis.deterministic_risk_executor import FTTIMethodExecutor
from hara_agent.workflow import HARAState
from hara_agent.workflow.physical_analysis import construct_physical_scenarios
from hara_agent.services.semantic.physical_scenario_agent import PhysicalScenarioAgent
from test_p7_closure import method

FIXTURE = Path(__file__).resolve().parents[1] / 'fixtures/ipa_saved_personnel_response_scope.json.gz'
CASES = json.loads(gzip.decompress(FIXTURE.read_bytes()))['cases']


def instantiate_saved(method, case):
    producer = PhysicalHypothesisInstantiationService(method)
    parent = HARAState._scenario_from_dict(deepcopy(case['parent']))
    raw_mf = deepcopy(case['malfunction'])
    fault = MalfunctionCandidate(**{**raw_mf, 'status': ReviewStatus(raw_mf['status']),
        'sources': [SourceRef(**s) for s in raw_mf['sources']]})
    raw = {'hypotheses': [deepcopy(case['raw_hypothesis'])], 'uncovered_conditions': []}
    plan = PhysicalScenarioPlan.from_dict(raw, evidence_ids=set(case['evidence_ids']))
    context = deepcopy(case['request_context'])
    children, errors = producer.instantiate(parent, fault, plan, plan_key=context['plan_key'],
        response_model='FROZEN_REAL_RESPONSE_OFFLINE', request_context=context,
        project_context=case['project_context'], function=case['function'])
    assert not errors and len(children) == 1
    assert raw == {'hypotheses': [case['raw_hypothesis']], 'uncovered_conditions': []}
    assert context == case['request_context'] and asdict(parent) == case['parent']
    return producer, fault, children[0]


@pytest.mark.parametrize('case', CASES[:], ids=lambda c: c['malfunction']['malfunction_id'])
def test_frozen_real_response_scope_keeps_physics_and_does_not_reask(method, case):
    producer, fault, child = instantiate_saved(method, case)
    for field, value in case['expected_physical_fields'].items():
        assert child.facts.get(field) == value, field
    h = child.facts['physical_analysis']['hypothesis']
    timing = RiskCalculationInputService().ftti(fault.malfunction_id, child.scenario_id, '',
        {**child.facts, '_fact_provenance': child.fact_provenance}, method=method)
    if case['current_shared_proposal_was_lost']:
        assert h['safe_state_proposal'] == case['raw_hypothesis']['safe_state_proposal']
        assert h['hazard_timing'] == case['raw_hypothesis']['hazard_timing']
        assert not h.get('driver_positions')  # Request scope is not relabeled as model output.
        assert timing.hazard_time_limit_s == pytest.approx(case['expected_ftti_s'])
        assert timing.status.value == 'CONDITIONAL'
        assert timing.timing_trace['required_controls'] == sorted(h['safe_state_proposal']['required_controls'])
        assert 'missing_required_controls' not in timing.timing_trace
    else:
        assert not h.get('safe_state_proposal')
        assert h['safe_state_unresolved_reason']
        assert h['timing_unresolved_reason']
        assert timing.hazard_time_limit_s is None
    assert child.fact_provenance['driver_in_vehicle']['approval'] == 'PENDING'
    assert child.fact_provenance['driver_in_vehicle']['selection_basis'] == 'VALIDATED_REQUEST_DRIVER_SCOPE'
    assert case['original_correction_completed']  # This repair never resets its allowance.
    assert producer.task_scope(child, fault.malfunction_id) == ''

    class NoTransport:
        def complete_json(self, request):
            raise AssertionError('A saved assessed world must not spend another Provider request')

    state = HARAState('saved-personnel-response', scenarios=[child])
    restored = HARAState.read_committed(json.loads(json.dumps(state.to_dict())))
    recovered = construct_physical_scenarios(restored, {fault.malfunction_id: restored.scenarios}, [fault],
        method=method, agent=PhysicalScenarioAgent(NoTransport()),
        project_facts=ItemDefinitionFacts.from_dict(case['project_context']))
    assert recovered[fault.malfunction_id] == [child]


@pytest.mark.parametrize('limitation', ['unvalidated', 'outside', 'multiple', 'failed', 'unknown', 'local_gap'])
def test_request_scope_never_overrides_an_explicit_personnel_or_control_limit(limitation):
    h = deepcopy(CASES[0]['raw_hypothesis'])
    scope = 'in_driver_seat'
    if limitation == 'unvalidated': scope = ''
    elif limitation == 'outside': h['safe_state_proposal']['driver_positions'] = ['outside_driver_seat']
    elif limitation == 'multiple': h['driver_positions'] = ['in_driver_seat', 'outside_driver_seat']
    else:
        local = h['driver_control_assessments']['in_driver_seat']
        if limitation == 'local_gap': local['safe_state_unresolved_reason'] = 'This scope was explicitly assessed as unresolved'
        else:
            local['remaining_controls'].remove('driver_braking')
            if limitation == 'failed': local['failed_controls'].append('driver_braking')
            else: local['unresolved_controls'] = {'driver_braking': 'Independence is not established'}
    before = deepcopy(h)
    selected = controls_for_driver(h, 'in_driver_seat', driver_scope=scope)
    assert not selected.get('safe_state_proposal')
    assert not selected.get('hazard_timing')
    assert h == before


def test_real_withdrawn_response_distinguishes_missing_command_from_driver_scope(method):
    case = next(c for c in CASES if not c['current_shared_proposal_was_lost'])
    _, fault, child = instantiate_saved(method, case)
    analysis = child.facts['physical_analysis']
    analysis['rejected_response_evaluation'] = {
        'input_values': FTTIMethodExecutor.response_input_snapshot(child.facts),
        'proposal': deepcopy(case['original_rejected_proposal']),
        'timing_trace': deepcopy(case['original_rejected_trace']),
        'method_source_hash': case['original_method_source_hash']}
    before = deepcopy(child)
    result = RiskCalculationInputService().ftti(fault.malfunction_id, child.scenario_id, '',
        {**child.facts, '_fact_provenance': child.fact_provenance}, method=method)
    assert result.status.value == 'PENDING_INPUT' and result.hazard_time_limit_s is None
    replay = result.timing_trace['rejected_response_revalidation']
    assert replay['proposal_remains_withdrawn'] is True
    assert replay['previous_timing_trace'] == case['original_rejected_trace']
    assert replay['previous_timing_trace']['status'] == 'NO_IMMEDIATE_SAFE_BRAKING_WINDOW'
    current = replay['current_evaluation']
    assert current['timing_trace']['pending_reason'] == 'CONTROL_PATH_UNRESOLVED'
    assert current['timing_trace']['missing_required_controls'] == ['automatic_braking_command']
    assert current['timing_trace']['failed_required_controls'] == []
    assert current['timing_trace']['driver_scope_mismatch'] is False
    assert '所选响应依赖的控制未在本成员中全部建立或不相容' in current['reason']
    assert '不能借用其他驾驶员分支' not in current['reason']
    assert not analysis['hypothesis'].get('safe_state_proposal')
    assert child == before


def test_driver_mismatch_is_reported_only_for_the_actual_driver_guard(method):
    case = next(c for c in CASES if c['current_shared_proposal_was_lost'])
    _, fault, child = instantiate_saved(method, case)
    facts = {**deepcopy(child.facts), 'driver_in_vehicle': False, '_fact_provenance': child.fact_provenance}
    result = RiskCalculationInputService().ftti(fault.malfunction_id, child.scenario_id, '', facts, method=method)
    assert result.status.value == 'PENDING_INPUT'
    assert result.timing_trace['missing_required_controls'] == []
    assert result.timing_trace['failed_required_controls'] == []
    assert result.timing_trace['driver_scope_mismatch'] is True
    assert '不能借用其他驾驶员分支' in result.reason
