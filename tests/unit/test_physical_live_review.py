"""Frozen-input regressions for the independent live-sample review; no network."""
from copy import deepcopy
import json
import math
from types import SimpleNamespace

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


def test_repeated_structured_provenance_is_shared_losslessly():
    metadata = {'object_id': 'target', 'stage': 'POST_FAULT', 'inputs': list(range(500)),
        'source_refs': [{'source_id': 'real-item', 'excerpt': 'source qualifier'}]}
    context = {'conditions': {'fact_authority': {'first': metadata, 'second': deepcopy(metadata)}},
        'evidence': {'MF.description': 'fault'}}
    context['conditions']['fact_authority']['third'] = {**metadata, 'stage': 'PRE_FAULT'}
    before = deepcopy(context)
    view, texts = physical_prompt_view(context)
    assert expand_physical_prompt_view(view, texts) == before == context
    assert len(json.dumps([view, texts])) < len(json.dumps(context)) * .6


def test_proposal_context_retains_physical_constraints_without_downstream_grades():
    context = {'plan_key': 'unchanged-audit-key', 'conditions': {'facts': {'ego_speed_kph': 0}},
        'evidence': {'MF.description': 'drive fault'},
        'intervention_inputs': ['emergency_braking_available'],
        'method_identity': {'method_source_hash': 'full-method-hash', 'project_analysis_policy': {
            'controllability': {'result': 'C3', 'rationale': 'downstream-only'},
            'severity_evaluation': {'grading_policy': 'downstream-only'},
            'driver_location_convention': {'outside_driver_seat_is_outside_vehicle': True},
            'engineering_decisions': {'approved_template_id': 'existing-template'}}}}
    before = deepcopy(context)
    class Client:
        def complete_json(self, request):
            payload = json.loads(request.user_prompt)
            projected = expand_physical_prompt_view(payload['context'], payload['source_texts'])
            expected = deepcopy(before)
            policy = expected['method_identity']['project_analysis_policy']
            policy.pop('controllability'); policy.pop('severity_evaluation')
            assert projected == expected
            assert request.prompt_version.endswith('v22-task-context')
            return LLMResponse(raw_plan(), 'OFFLINE', usage={})
    PhysicalScenarioAgent(Client()).propose(context)
    assert context == before


@pytest.mark.parametrize('controls_only', [False, True])
@pytest.mark.parametrize('mode', ['json_object', 'none'])
def test_physical_request_format_contract_and_cached_repair(tmp_path, controls_only, mode):
    from hara_agent.contracts.physical_scenario_plan import INTERVENTION_ASSUMPTION_FIELDS
    control = dict(rationale='Automatic command is lost; no independent path established',
        basis_refs=['MF.description'], conditions=['Current fault only'], uncovered_conditions=['Independent control unknown'],
        failed_controls=['automatic_braking_command'], remaining_controls=[], intervention_assumptions={})
    valid = ({'hypotheses': [], 'uncovered_conditions': [], 'control_assessment': control}
             if controls_only else raw_plan())
    invalid = deepcopy(valid)
    target = invalid['control_assessment'] if controls_only else invalid['hypotheses'][0]
    # Reproduce the real response's confusion of downstream inputs with
    # allowed proposal fields; repair must not silently translate null to false.
    target['intervention_assumptions'] = {'driver_in_vehicle': {'value': None}}
    class Client:
        config = SimpleNamespace(scenario_response_format=mode)
        calls = 0
        def complete_json(self, request):
            self.calls += 1
            assert request.response_format == ({'type': 'json_object'} if mode == 'json_object' else None)
            payload = json.loads(request.user_prompt)
            contract = payload['output_contract']['intervention_assumptions']
            if not controls_only:
                geometry = payload['output_shape']['hypotheses'][0]['geometry']
                assert 'object_longitudinal_direction' in geometry
                assert set(payload['output_contract']['moving_target_geometry']['object_longitudinal_direction']) == {'FORWARD', 'REVERSE', 'LEFT', 'RIGHT'}
                assert payload['output_shape']['hypotheses'][0]['grade_frame']['coordinate_frame'] == 'BODY_FORWARD_POSITIVE'
            assert set(contract['optional_keys']) == INTERVENTION_ASSUMPTION_FIELDS
            assert contract['type'] == 'object'
            if self.calls == 2:
                assert payload['previous_invalid_response'] == invalid
                assert 'driver_in_vehicle' in payload['repair_errors'][0]
            return LLMResponse(deepcopy(invalid if self.calls == 1 else valid), 'OFFLINE')
    context = {'plan_key': 'format-contract', 'evidence': {'MF.description': 'command fault'}}
    if controls_only:
        context['task_scope'] = 'CONTROL_INPUTS_ONLY'
    client = Client()
    agent = PhysicalScenarioAgent(client, response_checkpoint_dir=tmp_path)
    plan, audit = agent.propose(context)
    assert client.calls == 2 and audit['schema_repairs'] == 1
    _, resumed = agent.propose(context)
    assert client.calls == 2 and resumed['response_cache_hits'] == 2
    result = plan.control_assessment if controls_only else plan.hypotheses[0]
    assert result.get('intervention_assumptions', {}) == {}


def test_physical_acceleration_unit_notation_preserves_raw_reply_and_values():
    from hara_agent.contracts.physical_scenario_plan import PhysicalScenarioPlan
    raw = raw_plan('CONSTANT_LATERAL_ACCELERATION_TO_CONTACT', initial=7)
    parameter = next(p for p in raw['hypotheses'][0]['parameters'] if p['field'] == 'lateral_acceleration_mps2')
    parameter['unit'] = 'm/s²'
    before = deepcopy(raw)
    plan = PhysicalScenarioPlan.from_dict(raw, evidence_ids={'MF.description'})
    normalized = next(p for p in plan.hypotheses[0]['parameters'] if p['field'] == parameter['field'])
    assert normalized == {**parameter, 'unit': 'm/s2'}
    assert raw == before
    parameter['unit'] = 'm/s'
    with pytest.raises(ValueError, match='declared unit'):
        PhysicalScenarioPlan.from_dict(raw, evidence_ids={'MF.description'})


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


@pytest.mark.parametrize('envelope', ['unwrapped', 'missing_outer_limits'])
def test_control_envelope_normalization_preserves_reply_and_reuses_response(tmp_path, envelope):
    control = dict(rationale='Conditional independent driver path', basis_refs=['MF.description'],
        conditions=['Manual channel independent of automatic command'],
        uncovered_conditions=['Shared actuator faults not covered'],
        failed_controls=['automatic_braking_command'], remaining_controls=['driver_braking'],
        intervention_assumptions={})
    raw = deepcopy(control) if envelope == 'unwrapped' else {'hypotheses': [], 'control_assessment': deepcopy(control)}
    before = deepcopy(raw)
    class Client:
        calls = 0
        def complete_json(self, request):
            self.calls += 1
            return LLMResponse(raw, 'OFFLINE')
    client = Client()
    agent = PhysicalScenarioAgent(client, response_checkpoint_dir=tmp_path)
    context = {'plan_key': 'control-envelope', 'task_scope': 'CONTROL_INPUTS_ONLY',
               'evidence': {'MF.description': 'Automatic command fault'}}
    records = []
    plan, audit = agent.propose(context, on_response=records.append)
    assert plan.control_assessment == control
    assert list(plan.uncovered_conditions) == control['uncovered_conditions']
    assert raw == before == records[0]['response']['data']
    assert audit['schema_repairs'] == 0 and audit['control_envelope_normalization']
    restored, resumed = agent.propose(context)
    assert restored == plan and resumed['response_cache_hits'] == 1 and client.calls == 1


@pytest.mark.parametrize('defect', ['contradiction', 'missing_limits', 'foreign_score', 'wrong_task'])
def test_control_envelope_normalization_does_not_invent_or_approve_capability(defect):
    raw = dict(rationale='Control scope', basis_refs=['MF.description'],
        conditions=['Independent channel assumed'], uncovered_conditions=['Common cause not covered'],
        failed_controls=['automatic_braking_command'], remaining_controls=['driver_braking'])
    if defect == 'contradiction': raw['unresolved_controls'] = {'braking': 'Unknown even conditionally'}
    if defect == 'missing_limits': raw.pop('uncovered_conditions')
    if defect == 'foreign_score': raw['C'] = 'C1'
    context = {'plan_key': 'invalid-envelope', 'task_scope': 'CONTROL_INPUTS_ONLY',
               'evidence': {'MF.description': 'Automatic command fault'}}
    if defect == 'wrong_task': context.pop('task_scope')
    before = deepcopy(raw)
    class Client:
        def complete_json(self, request): return LLMResponse(raw, 'OFFLINE')
    with pytest.raises(ValueError, match='schema repair exhausted'):
        PhysicalScenarioAgent(Client()).propose(context)
    assert raw == before


def test_failed_control_safe_state_is_rejected_locally_without_losing_motion():
    raw = raw_plan(assessed=True)
    h = raw['hypotheses'][0]
    h['failed_controls'] = ['drive']
    h['safe_state_proposal'] = {'description': 'Maintain position using drive torque',
        'required_controls': ['drive'], 'basis_refs': ['MF.description'],
        'conditions': ['Drive torque available']}
    before = deepcopy(raw)
    class Client:
        calls = 0
        def complete_json(self, request):
            self.calls += 1
            return LLMResponse(raw, 'OFFLINE')
    client = Client()
    plan, audit = PhysicalScenarioAgent(client).propose(
        {'plan_key': 'failed-safe-state', 'evidence': {'MF.description': 'Drive failure'}})
    assert client.calls == 1 and raw == before
    proposed = plan.hypotheses[0]
    assert proposed['parameters'] == h['parameters'] and proposed['geometry'] == h['geometry']
    assert proposed['failed_controls'] == ['drive']
    assert proposed['remaining_controls'] == h['remaining_controls']
    assert 'safe_state_proposal' not in proposed
    assert audit['deferred_safe_states'] == [{'hypothesis_id': h['hypothesis_id'],
        'reason': 'SAFE_STATE_REQUIRES_FAILED_CONTROL', 'proposal': h['safe_state_proposal']}]
    assert '不采用' in proposed['uncovered_conditions'][-1]
    # This does not relax contradictory primary controls.
    raw['hypotheses'][0]['remaining_controls'].append('drive')
    with pytest.raises(ValueError, match='failed control cannot be a remaining'):
        PhysicalScenarioAgent(client).propose({'plan_key': 'contradiction', 'evidence': {'MF.description': 'Drive failure'}})
