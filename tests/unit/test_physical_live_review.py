"""Frozen-input regressions for the independent live-sample review; no network."""
from copy import deepcopy
import json
import math

import pytest

from test_autonomous_physics import raw_plan, make_candidate, instantiate
from test_physical_input_scope import grade_raw, symbol
from test_p7_closure import method
from hara_agent.infrastructure.llm.protocol import LLMResponse
from hara_agent.services.semantic.physical_scenario_agent import (
    PhysicalScenarioAgent, physical_prompt_view, expand_physical_prompt_view,
)
from hara_agent.services.analysis.scenario_physics import constant_acceleration_contact, causal_physics_conflicts
from hara_agent.services.reporting.engineering_text_mapper import EngineeringReportTextMapper


def test_prompt_projection_is_lossless_and_preserves_different_scopes():
    excerpt = '仅限泊车，不适用于高速公路；坡向不等于车头方向。' * 100
    conditions = {'facts': {'ego_speed_kph': 7.0000001}, 'fact_authority': {
        'a': {'object_id': 'front', 'time_role': 'PRE_FAULT', 'source_refs': [{'source_id': 'one', 'excerpt': excerpt}]},
        'b': {'object_id': 'rear', 'time_role': 'POST_FAULT', 'source_refs': [{'source_id': 'two', 'excerpt': excerpt}]},
        'c': {'source_refs': [{'source_id': 'three', 'excerpt': excerpt + '独有否定条件'}]},
    }}
    context = {'conditions': conditions, 'project_context': {'limit': 7}, 'evidence': {
        'SCENARIO.conditions': deepcopy(conditions), 'PROJECT.operating_context': {'value': {'limit': 7}, 'sources': []}}}
    before = deepcopy(context)
    view, texts = physical_prompt_view(context)
    assert context == before
    assert expand_physical_prompt_view(view, texts) == before
    assert len(texts) == 1 and list(texts.values()) == [excerpt]
    assert view['conditions']['facts']['ego_speed_kph'] == 7.0000001
    assert view['conditions']['fact_authority']['a']['object_id'] == 'front'
    assert view['conditions']['fact_authority']['b']['time_role'] == 'POST_FAULT'
    assert len(json.dumps([view, texts])) < len(json.dumps(context)) / 2
    # Similar text/conditions are not replaced by a reference to a different fact.
    context['evidence']['SCENARIO.conditions']['facts']['ego_speed_kph'] = 7.0000002
    view, texts = physical_prompt_view(context)
    assert '$context_ref' not in view['evidence']['SCENARIO.conditions']
    assert expand_physical_prompt_view(view, texts) == context


def test_agent_sends_compact_view_and_retains_original_basis_ids():
    class Client:
        def complete_json(self, request):
            payload = json.loads(request.user_prompt)
            assert payload['context']['evidence']['SCENARIO.conditions'] == {'$context_ref': 'conditions'}
            assert expand_physical_prompt_view(payload['context'], payload['source_texts']) == context
            return LLMResponse(raw_plan(), 'OFFLINE', usage={})
    context = {'plan_key': 'k', 'conditions': {'facts': {'ego_speed_kph': 0}},
               'evidence': {'MF.description': 'drive fault', 'SCENARIO.conditions': {'facts': {'ego_speed_kph': 0}}}}
    plan, _ = PhysicalScenarioAgent(Client()).propose(context)
    assert plan.hypotheses[0]['basis_refs'] == ['MF.description']


@pytest.mark.parametrize('acceleration', [1.5, -0.1, 0])
def test_symbolic_distance_boundaries_match_existing_numeric_solver(method, acceleration):
    raw = raw_plan(initial=7, acceleration=acceleration)
    symbol(raw, 'relative_distance_m')
    child = instantiate(method, raw, make_candidate(7))[0]
    sensitivity = child.facts['physical_analysis']['sensitivity']
    assert sensitivity['parameter_relations'][0]['symbolic_parameters'] == {'relative_distance_m': 'unknown_parameter'}
    assert 'ego_impact_speed_kph' not in child.facts and 'ttc_s' not in child.facts
    assert sensitivity['method_boundaries']
    for row in sensitivity['method_boundaries']:
        assert row['parameter'] == 'relative_distance_m'
        if row['boundary'] <= 0:
            continue
        result = constant_acceleration_contact(7, acceleration, row['boundary'])
        assert result['status'] == 'CONTACT'
        if 'speed_boundary_kph' in row:
            assert result['impact_speed_kph'] == pytest.approx(row['speed_boundary_kph'], abs=1e-5)
        else:
            assert result['time_s'] == pytest.approx(row['time_boundary_s'], abs=1e-5)
    text = EngineeringReportTextMapper.physical_inputs(child, {})
    assert 'v_impact(d)' in text and 'T(d)' in text


def test_symbolic_lateral_clearance_has_relations_and_severity_boundaries_not_c(method):
    raw = raw_plan('CONSTANT_LATERAL_ACCELERATION_TO_CONTACT', initial=7)
    symbol(raw, 'lateral_clearance_m')
    child = instantiate(method, raw, make_candidate(7))[0]
    sensitivity = child.facts['physical_analysis']['sensitivity']
    assert sensitivity['method_boundaries']
    for row in sensitivity['method_boundaries']:
        assert row['parameter'] == 'lateral_clearance_m' and 'time_boundary_s' not in row
        if row['boundary'] > 0:
            result = constant_acceleration_contact(0, .5, row['boundary'])
            assert math.hypot(7, result['impact_speed_kph']) == pytest.approx(row['speed_boundary_kph'])
    text = EngineeringReportTextMapper.physical_inputs(child, {})
    assert 'w=v_y0' in text and '不作为纵向TTC' in text
    assert 'ttc_s' not in child.facts and 'first_contact_time_s' not in child.facts


@pytest.mark.parametrize('heading,position,opposite', [('UPHILL', 'FRONT', True), ('UPHILL', 'REAR', False), ('DOWNHILL', 'FRONT', False)])
def test_symbolic_resistance_cannot_hide_grade_geometry(method, heading, position, opposite):
    raw = grade_raw(heading, position)
    symbol(raw, 'residual_resistance_mps2')
    child = instantiate(method, raw)[0]
    outcome = child.facts['motion_model_outcome']
    sensitivity = child.facts['physical_analysis']['sensitivity']
    if opposite:
        assert outcome['reason'] == 'TARGET_OUTSIDE_GRAVITY_PATH'
        assert outcome['contact_possible_in_model'] is False
        assert not sensitivity['method_boundaries']
        assert '不构成接近' in EngineeringReportTextMapper.physical_inputs(child, {})
    else:
        assert sensitivity['method_boundaries']
        assert 'a=max(0,g_d−r)' in EngineeringReportTextMapper.physical_inputs(child, {})
    assert child.facts['object_position'] == position
    assert 'collision_branches' not in child.facts and 'ttc_s' not in child.facts
    assert 'residual_resistance_mps2' not in child.facts


def test_symbolic_distance_with_cap_has_piecewise_relations_and_no_uncapped_ttc_threshold(method):
    raw = raw_plan(initial=7, acceleration=1.5)
    symbol(raw, 'relative_distance_m')
    h = raw['hypotheses'][0]
    h['parameters'].append(dict(field='post_fault_speed_cap_kph', unit='km/h', kind='POINT', value=10, rationale='conditional remaining limiter'))
    h['remaining_controls'] = ['speed_limiter']
    child = instantiate(method, raw, make_candidate(7))[0]
    text = EngineeringReportTextMapper.physical_inputs(child, {})
    assert 'd>d_cap' in text
    boundaries = child.facts['physical_analysis']['sensitivity']['method_boundaries']
    assert all('time_boundary_s' not in row and row['speed_boundary_kph'] <= 10 for row in boundaries)


def test_symbolic_uphill_front_does_not_accept_new_contact_claim(method):
    raw = grade_raw('UPHILL', 'FRONT')
    symbol(raw, 'residual_resistance_mps2')
    child = instantiate(method, raw)[0]
    payload = {'hazardous_event': '与前方目标碰撞', 'physical_interaction': {
        'target': 'SCENARIO_OBJECT', 'object_id': child.facts['object_id'], 'object_position': 'FRONT',
        'assessment_reference': 'MODEL_TRAJECTORY', 'relation': 'CONTACT'}}
    assert 'NOT_CLOSING_CONTRADICTS_CAUSAL_CLAIM' in causal_physics_conflicts(child, payload)
    payload['physical_interaction']['relation'] = 'NOT_APPROACHING'
    payload['hazardous_event'] = '不会接触前方目标'
    assert not causal_physics_conflicts(child, payload)
