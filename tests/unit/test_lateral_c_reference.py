"""Scoped reference C is conditional; geometry, identity and normal C stay intact."""
from copy import deepcopy
from dataclasses import replace
import math

import pytest

from test_autonomous_physics import method, raw_plan, make_candidate, instantiate, score
from hara_agent.services.analysis.scenario_physics import branch_physics_view
from hara_agent.services.analysis.risk_execution_trace_service import RiskExecutionTraceService
from hara_agent.services.reporting.engineering_text_mapper import EngineeringReportTextMapper


def steering_candidate(method, *, initial=7, clearance=.5, acceleration=6):
    raw = raw_plan('CONSTANT_LATERAL_ACCELERATION_TO_CONTACT', initial=initial)
    h = raw['hypotheses'][0]
    h['remaining_controls'] = ['braking']
    for p in h['parameters']:
        if p['field'] == 'lateral_clearance_m': p['value'] = clearance
        if p['field'] == 'lateral_acceleration_mps2': p['value'] = acceleration
    sc = instantiate(method, raw, make_candidate(initial))[0]
    sc.facts.update(_function_output='steering angle request', component_category='computing',
                    failure_type='excessive', operating_mode='parking_vehicle_control',
                    scenario_atom_ids=['CN_parking_multistory', 'FB001', 'FA040'])
    return sc


def test_reference_c_uses_generated_trajectory_and_preserves_se(method):
    sc = steering_candidate(method)
    before = deepcopy(sc)
    policy = deepcopy(method.metadata)
    policy['project_analysis_policy']['controllability'].pop('conditional_reference_rules')
    old = score(replace(method, metadata=policy), sc)
    new = score(method, sc)
    assert not old['controllability']['controllability_score']
    c = new['controllability']
    assert c['controllability_score'] == 'C1'
    assert c['decision_status'] == 'CONDITIONAL_REFERENCE_MATCH'
    assert c['engineering_status'] == 'PENDING'
    assert c['reference_rule']['trajectory_max_speed_kph'] == math.hypot(7, math.sqrt(6) * 3.6)
    assert new['severity']['maximum_feasible_severity'] == old['severity']['severity_score']
    assert {k: v for k, v in new['severity'].items() if k not in {'branch_evaluations', 'maximum_feasible_severity'}} == {
        k: v for k, v in old['severity'].items() if k != 'branch_evaluations'}
    assert new['exposure'] == old['exposure']
    for left, right in zip(old['branch_evaluations'], new['branch_evaluations'], strict=True):
        for dimension in ('severity', 'exposure'):
            assert left['scores'][dimension] == right['scores'][dimension]
        assert left['branch_id'] == right['branch_id']
    assert sc == before and 'ttc_s' not in sc.facts
    assert all(b['scores']['controllability']['controllability_score'] == 'C1' for b in new['branch_evaluations'])
    from hara_agent.services.analysis.asil_service import MethodContractASILService
    for branch in new['branch_evaluations']:
        assert branch['ASIL'] == MethodContractASILService(method).determine(branch['S'], branch['E'], branch['C'])
        assert branch['status'] == 'PENDING'
    assert new['branch_summary']['highest_known_asil'] and new['branch_summary']['status'] == 'PENDING'
    trace = RiskExecutionTraceService(method)._controllability_trace(sc.facts, c)
    assert trace['source_status'] == 'CONDITIONAL_REFERENCE'
    assert trace['ttc_band'] == '' and not trace['ttc_branch_eligible']
    assert trace['reference_rule']['source_location'].endswith('D17')
    assert '跨项目适用性待审核' in EngineeringReportTextMapper.controllability_branch({'controllability': trace})


@pytest.mark.parametrize('changes', [
    {'driver_in_vehicle': False}, {'emergency_braking_available': False},
    {'failure_type': 'loss'}, {'failure_type': 'stuck'}, {'failure_type': 'delayed'},
    {'component_category': 'actuator_lateral'}, {'_function_output': 'drive torque'},
    {'operating_mode': 'parking_search'},
])
def test_reference_does_not_generalize_to_other_failures_or_drivers(method, changes):
    sc = steering_candidate(method)
    # No remote/other-party override: inspect just reference scope.
    sc.facts['remote_intervention_available'] = None
    sc.facts['other_road_user_avoidance_possible'] = None
    sc.facts.update(changes)
    if 'emergency_braking_available' in changes:
        sc.facts['physical_analysis']['hypothesis']['remaining_controls'] = []
    assert score(method, sc)['controllability']['decision_status'] != 'CONDITIONAL_REFERENCE_MATCH'


@pytest.mark.parametrize('peak,expected', [(14.9999999, 'C1'), (15., ''), (15.0000001, '')])
def test_speed_limit_uses_unrounded_trajectory_not_normal_speed(method, peak, expected):
    # Longitudinal speed stays below the limit while lateral motion raises the norm.
    acceleration = ((peak ** 2 - 7 ** 2) / 3.6 ** 2) / (2 * .5)
    sc = steering_candidate(method, acceleration=acceleration)
    result = score(method, sc)
    assert result['controllability']['controllability_score'] == expected


def test_branch_geometry_recomputes_reference_speed_bound(method):
    sc = steering_candidate(method)
    view = {**sc.facts, 'scenario_id': sc.scenario_id, 'malfunction_id': 'MF', '_fact_provenance': sc.fact_provenance}
    branch = {'lateral_clearance_m': 5., '_fact_provenance': {
        'lateral_clearance_m': deepcopy(sc.fact_provenance['lateral_clearance_m'])}}
    new = branch_physics_view(view, branch)
    assert new['trajectory_max_speed_kph'] > 15
    assert new['trajectory_max_speed_kph'] != sc.facts['trajectory_max_speed_kph']
