"""A remote interface gets C credit only for its scoped, timely response."""
from copy import deepcopy
import pytest

from test_autonomous_physics import method, instantiate, make_candidate, score
from test_scoped_timing_and_avoidance import timed_plan
from hara_agent.contracts.physical_scenario_plan import PhysicalScenarioPlan
from hara_agent.services.analysis.risk_calculation_input_service import RiskCalculationInputService
from hara_agent.services.analysis.hazardous_event_risk_context_service import HazardousEventRiskContextService


def remote_plan(delay=.1):
    raw = timed_plan()
    h = raw['hypotheses'][0]
    h['remaining_controls'] = ['braking', 'remote_control']
    h['driver_positions'] = ['outside_driver_seat']
    h['safe_state_proposal']['required_controls'] = h['remaining_controls'][:]
    h['safe_state_proposal']['driver_positions'] = ['outside_driver_seat']
    h['intervention_assumptions'] = {'remote_intervention_available': {
        'value': True, 'rationale': 'conditional independent remote stop',
        'basis_refs': ['MF.description'], 'conditions': ['operator detects the fault and commands the retained remote path'],
        'response_delay_s': delay}}
    return raw


def remote_child(method, delay=.1):
    parent = make_candidate(3.6)
    parent.facts.update(driver_in_vehicle=False, driver_position='outside_driver_seat',
                        operating_mode='parking_vehicle_control')
    parent.facts.pop('remote_intervention_available')
    parent.fact_provenance.pop('remote_intervention_available', None)
    return instantiate(method, remote_plan(delay), parent)[0]


def view(child):
    return {**child.facts, 'malfunction_id': 'MF', 'scenario_id':child.scenario_id,
            '_fact_provenance':child.fact_provenance}


def test_timely_conditional_remote_response_retains_override_and_source(method):
    child = remote_child(method)
    before = deepcopy(child)
    check = RiskCalculationInputService().remote_intervention_applicability(view(child),
        malfunction_id='MF', scenario_id=child.scenario_id, method=method)
    assert check['status'] == 'CONDITIONAL_REMOTE_RESPONSE_VALIDATED'
    assert check['latest_response_start_s'] == pytest.approx(.9364916731)
    c = score(method, child)['controllability']
    assert c['controllability_score'] == 'C1'
    assert c['engineering_status'] == 'PENDING'
    assert c['decision_status'] == 'OVERRIDE_MATCH'
    assert child == before


@pytest.mark.parametrize('defect', ['no_delay','late','failed_remote','other_driver','other_object','no_window','system_response'])
def test_interface_fact_alone_cannot_credit_an_unproven_response(method, defect):
    child = remote_child(method)
    h = child.facts['physical_analysis']['hypothesis']
    if defect == 'no_delay':
        h['intervention_assumptions']['remote_intervention_available'].pop('response_delay_s')
    elif defect == 'late':
        h['intervention_assumptions']['remote_intervention_available']['response_delay_s'] = 1
    elif defect == 'failed_remote':
        h['remaining_controls'].remove('remote_control')
        h['failed_controls'].append('remote_control')
    elif defect == 'other_driver':
        h['safe_state_proposal']['driver_positions'] = ['in_driver_seat']
    elif defect == 'other_object':
        child.fact_provenance['physical_analysis']['applicable_scope']['scenario_id'] = 'another-world'
    elif defect == 'no_window':
        child.facts['relative_distance_m'] = .01
        child.facts['travel_to_impact_m'] = .01
    else:
        h['safe_state_proposal']['required_controls'] = ['braking', 'automatic_braking_command']
        h['remaining_controls'].append('automatic_braking_command')
    before = deepcopy(child)
    result = score(method, child)
    assert result['controllability']['controllability_score'] != 'C1'
    assert child == before
    assert child.facts['remote_intervention_available'] is True
    # Removing unproven C credit must not change the unmitigated severity/exposure.
    reference = deepcopy(child)
    reference.facts.pop('remote_intervention_available')
    baseline = score(method, reference)
    for dimension, key in [('severity','severity_score'), ('exposure','exposure_score')]:
        assert result[dimension][key] == baseline[dimension][key]


def test_conditioned_remote_false_stays_false_and_does_not_require_delay(method):
    child = remote_child(method)
    child.facts['remote_intervention_available'] = False
    assert RiskCalculationInputService().remote_intervention_applicability(view(child),
        malfunction_id='MF',scenario_id=child.scenario_id,method=method) == {}


@pytest.mark.parametrize('delay', [-1, float('nan'), True, '0.1'])
def test_remote_delay_contract_does_not_coerce_invalid_values(delay):
    with pytest.raises(ValueError):
        PhysicalScenarioPlan.from_dict(remote_plan(delay), evidence_ids={'MF.description'})


def test_existing_plan_without_new_optional_input_still_reads(method):
    raw = remote_plan()
    raw['hypotheses'][0]['intervention_assumptions']['remote_intervention_available'].pop('response_delay_s')
    assert PhysicalScenarioPlan.from_dict(raw, evidence_ids={'MF.description'})
