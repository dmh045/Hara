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


def test_physical_json_repair_keeps_entire_reply_budget_usage_and_production_recovery(method, tmp_path):
    from test_geometry_completion import run
    from test_llm_malfunction_envelope_repair import _sequenced_scenario_client
    from hara_agent.infrastructure.llm.provider_budget import ProviderAttemptBudget
    from hara_agent.workflow import CheckpointRepository

    raw = raw_plan(assessed=True)
    broken = json.dumps(raw)[:-1]  # Nested geometry remains valid, outer world does not.
    calls = []
    client = _sequenced_scenario_client([broken, json.dumps(raw)], calls)
    ledger = tmp_path/'attempts.jsonl'
    client.attempt_budget = ProviderAttemptBudget(ledger, run_id='syntax-recovery', limit=2)
    agent = PhysicalScenarioAgent(client, response_checkpoint_dir=tmp_path/'responses')
    repo = CheckpointRepository(tmp_path/'states')
    state, children = run(method, agent, checkpoint=repo.save)
    assert len(calls) == client.attempt_budget.attempts == 2
    repair = json.loads(calls[1])['messages'][-1]['content']
    correction = json.loads(repair.split('FORMAT_CORRECTION_INPUT=', 1)[1])
    assert correction['previous_invalid_response'] == broken
    assert 'hypotheses' in correction['previous_invalid_response']
    assert correction['parse_error']['json_error_message'] == "Expecting ',' delimiter"
    assert ' <-- parse error --> ' in correction['parse_error']['json_error_excerpt']
    assert repair.startswith('FORMAT_CORRECTION_INPUT=')
    assert 'not repeating the analysis' in json.loads(calls[1])['messages'][0]['content']
    entry = next(iter(state.item_definition['physical_analysis_plans'].values()))
    assert entry['status'] == 'VALIDATED' and entry['plan'] == raw
    assert entry['audit']['request_count'] == 2
    usage = entry['responses'][0]['response']['usage']
    prompt_chars = sum(len(m['content']) for call in calls for m in json.loads(call)['messages'])
    assert usage['transport_attempts'] == 2 and usage['input_characters'] == prompt_chars
    assert state.item_definition['physical_analysis_metrics']['cumulative_uncached_requests'] == 2
    assert state.item_definition['physical_analysis_metrics']['cumulative_uncached_prompt_characters'] == prompt_chars
    assert state.item_definition['physical_analysis_metrics']['token_usage'] == {'prompt_tokens': 20, 'completion_tokens': 40}
    assert children['MF'][0].facts['motion_model_outcome']['status'] == 'CONTACT'
    original = json.loads((tmp_path/'attempts.responses/000001.json').read_text())
    assert original['response']['choices'][0]['message']['content'] == broken
    _, restored = run(method, agent, state=repo.load(state.run_id), checkpoint=repo.save)
    assert restored == children and len(calls) == 2 and client.attempt_budget.attempts == 2


@pytest.mark.parametrize('syntax_first', [True, False])
def test_physical_format_and_contract_corrections_have_separate_bounded_attempts(syntax_first):
    from test_llm_malfunction_envelope_repair import _sequenced_scenario_client
    raw = raw_plan()
    broken = json.dumps(raw)[:-1]
    invalid = json.dumps({'geometry': raw['hypotheses'][0]['geometry']})
    calls = []
    client = _sequenced_scenario_client([broken, invalid, invalid] if syntax_first else [invalid, broken], calls)
    with pytest.raises(ValueError):
        PhysicalScenarioAgent(client).propose({'plan_key':'bounded-syntax', 'evidence':{'MF.description':'fault'}})
    assert len(calls) == (3 if syntax_first else 2)


def test_retry_transport_policy_does_not_invalidate_an_exact_successful_response(tmp_path):
    from dataclasses import replace
    from hara_agent.infrastructure.llm.response_checkpoint import ResponseCheckpoint
    from hara_agent.infrastructure.llm.protocol import LLMRequest
    from test_geometry_completion import Client
    raw = raw_plan()
    client = Client([raw, raw])
    cache = ResponseCheckpoint(tmp_path)
    request = LLMRequest('construct_post_fault_physical_scenario', 'system', 'same physics',
        'PhysicalScenarioPlan', 'test', metadata={'schema_repair':True})
    first = cache.complete(client, request)
    stricter = replace(request, metadata={**request.metadata, 'strict_no_format_retry':True})
    reused = cache.complete(client, stricter)
    assert reused.data == first.data and reused.usage['response_checkpoint_hit']
    assert len(client.requests) == 1
    cache.complete(client, replace(stricter, user_prompt='different physics'))
    assert len(client.requests) == 2


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
            'timing_analysis': {'supported_models': ['GRADE_ROLLAWAY_TO_IMPACT_DISTANCE'],
                'status': 'CONDITIONAL_REFERENCE', 'limitations': 'Selected method scope only'},
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
            assert request.prompt_version == PhysicalScenarioAgent.PROMPT_VERSION
            assert PhysicalScenarioAgent.RESPONSE_TIMING_POLICY in request.system_prompt
            assert payload['output_contract']['safe_state_proposal']['transition']['post_stop_motion'].startswith('HOLD_POSITION')
            assert 'Unmitigated fault trajectory ONLY' in payload['output_contract']['unmitigated_post_stop_motion']['meaning']
            assert '允许的最大响应延迟' in request.system_prompt
            assert '程序由坡度与阻力推导' in request.system_prompt
            assert '半尺寸求和后不得再除以2' in request.system_prompt
            assert 'LEFT使y增加，RIGHT使y减小' in request.system_prompt
            assert '不得仅凭标线认定车辆/行人无法跨越' in request.system_prompt
            assert 'driver_control_assessments' in request.system_prompt
            assert payload['output_contract']['driver_control_assessments']['entry_shape']['failed_controls']==[]
            assert 'driver_control_assessments' not in payload['output_shape']['hypotheses'][0]
            assert '人员缺席' in request.system_prompt
            assert payload['output_contract']['response_timing'] == policy['timing_analysis']
            assert payload['output_contract']['control_arrays']['shape_example'] == ['driver_braking', 'braking']
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
            assert 'hypotheses必须保持空数组[]' in request.system_prompt
            assert '每个hypotheses成员都必须' not in request.system_prompt
            assert '末尾不得再次输出hypotheses空数组' not in request.system_prompt
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


@pytest.mark.parametrize('explicit_zero', [True, False])
def test_constant_speed_notation_reuses_same_motion_solver_and_retains_raw_reply(method, tmp_path, explicit_zero):
    from test_geometry_completion import Client
    raw = raw_plan(initial=3.6, acceleration=0)
    h = raw['hypotheses'][0]
    h['model'] = 'CONSTANT_SPEED_REPRESENTATIVE'
    if not explicit_zero:
        h['parameters'] = [p for p in h['parameters'] if p['field'] != 'longitudinal_acceleration_mps2']
    before = deepcopy(raw)
    client = Client([raw])
    agent = PhysicalScenarioAgent(client, response_checkpoint_dir=tmp_path)
    context = {'plan_key':'constant-notation', 'evidence':{'MF.description':'conditional constant motion'}}
    plan, audit = agent.propose(context)
    child = instantiate(method, plan.to_dict(), make_candidate(3.6))[0]
    assert child.facts['ego_impact_speed_kph'] == pytest.approx(3.6)
    assert child.facts['ttc_s'] == pytest.approx(2)
    assert child.fact_provenance['longitudinal_acceleration_mps2']['approval'] == 'PENDING'
    assert audit['motion_model_normalizations'][0]['zero_acceleration_from_model_definition'] is (not explicit_zero)
    assert raw == before == json.loads(next(tmp_path.glob('*.json')).read_text())['response']['data']
    restored, resumed = agent.propose(context)
    assert restored == plan and resumed['response_cache_hits'] == 1 and len(client.requests) == 1


@pytest.mark.parametrize('acceleration', [
    {'kind':'POINT','value':1}, {'kind':'POINT','value':False},
    {'kind':'SYMBOLIC','symbol':'a_fault'}, {'kind':'RANGE','lower':0,'value':0,'upper':1},
])
def test_constant_speed_notation_does_not_replace_an_explicit_acceleration(acceleration):
    from test_geometry_completion import Client
    raw = raw_plan(initial=3.6)
    raw['hypotheses'][0]['model'] = 'CONSTANT_SPEED_REPRESENTATIVE'
    p = raw['hypotheses'][0]['parameters'][-1]
    p.pop('value'); p.update(acceleration)
    before = deepcopy(raw)
    with pytest.raises(ValueError, match='Unknown physical model'):
        PhysicalScenarioAgent(Client([raw, raw])).propose({'plan_key':'contradictory-constant',
            'evidence':{'MF.description':'conditional fault'}})
    assert raw == before


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


def test_failed_command_explanation_preserves_failure_and_valid_response(method):
    raw = raw_plan()
    h = raw['hypotheses'][0]
    h['failed_controls'] = ['automatic_drive_command']
    h['remaining_controls'] = ['braking', 'driver_braking']
    h['unresolved_controls'] = {'automatic_drive_command': 'The failed command has no proven independent fallback.'}
    h['safe_state_proposal'] = {'description': 'Stop using the independent pedal path',
        'required_controls': ['driver_braking'], 'basis_refs': ['MF.description'],
        'conditions': ['Driver seated; independent brake path assumed'],
        'post_stop_motion': 'HOLD_POSITION',
        'transition': {'model': 'CONSTANT_NET_DECELERATION', 'net_deceleration_mps2': 3,
                       'rationale': 'Conditional net braking', 'basis_refs': ['MF.description']}}
    h['hazard_timing'] = {'time_origin': 'FAULT_ONSET', 'fault_to_model_start_s': 0,
        'hazard_criterion': 'LOSS_OF_BRAKING_AVOIDABILITY',
        'rationale': 'Initial state is fault onset', 'basis_refs': ['MF.description']}
    before = deepcopy(raw)
    class Frozen:
        calls = 0
        def complete_json(self, request):
            self.calls += 1
            return LLMResponse(raw, 'OFFLINE')
    client = Frozen()
    plan, audit = PhysicalScenarioAgent(client).propose({'plan_key': 'failed-explanation',
        'evidence': {'MF.description': 'Drive output exceeds demand'}})
    assert client.calls == 1 and raw == before and len(audit['control_scope_normalizations']) == 2
    normalized = plan.hypotheses[0]
    assert normalized['failed_controls'] == h['failed_controls']
    assert normalized['remaining_controls'] == h['remaining_controls']
    assert h['unresolved_controls']['automatic_drive_command'] in normalized['uncovered_conditions'][-1]
    child = instantiate(method, plan.to_dict())[0]
    from hara_agent.services.analysis.risk_calculation_input_service import RiskCalculationInputService
    timing = RiskCalculationInputService().ftti('MF', child.scenario_id, '',
        {**child.facts, '_fact_provenance': child.fact_provenance}, method=method)
    assert timing.hazard_time_limit_s > 0 and timing.calculation_nature == 'CONDITIONAL'
    # A declaration of availability still conflicts; normalization cannot
    # revive the failed command or reconcile two different hold behaviors.
    h['remaining_controls'].append('automatic_drive_command')
    with pytest.raises(ValueError, match='failed control cannot be a remaining'):
        PhysicalScenarioAgent(client).propose({'plan_key': 'conflicting', 'evidence': {'MF.description': 'fault'}})


def test_member_limits_can_supply_missing_plan_envelope_without_new_claims():
    raw = raw_plan()
    del raw['uncovered_conditions']
    original = deepcopy(raw)
    class Frozen:
        calls = 0
        def complete_json(self, request):
            self.calls += 1
            return LLMResponse(raw, 'OFFLINE')
    client = Frozen()
    plan, audit = PhysicalScenarioAgent(client).propose({'plan_key': 'member-limits',
        'evidence': {'MF.description': 'fault'}})
    assert client.calls == 1 and raw == original
    assert plan.to_dict()['uncovered_conditions'] == raw['hypotheses'][0]['uncovered_conditions']
    assert audit['control_envelope_normalization'] == 'PROJECTED_HYPOTHESIS_LIMITS'


def test_scoped_optional_response_gap_does_not_discard_shared_motion():
    raw = raw_plan()
    h = raw['hypotheses'][0]
    h['driver_control_assessments'] = {'outside_driver_seat': {
        'rationale': 'No pedal outside; hardware braking remains conditionally usable',
        'basis_refs': ['MF.description'], 'conditions': ['Same drive-output fault'],
        'uncovered_conditions': ['No holding-command assessment'],
        'failed_controls': [], 'remaining_controls': ['braking'],
        'unresolved_controls': {'drive': 'Failed drive output cannot supply holding torque'},
        'safe_state_proposal': {'description': 'Hold with the parking brake',
            'required_controls': ['parking_hold'], 'basis_refs': ['MF.description'],
            'conditions': ['Unassessed parking path']}}}
    before = deepcopy(raw)
    class Frozen:
        def complete_json(self, request): return LLMResponse(raw, 'OFFLINE')
    plan, audit = PhysicalScenarioAgent(Frozen()).propose({'plan_key': 'scoped-gap', 'evidence': {'MF.description': 'fault'}})
    from hara_agent.contracts.physical_scenario_plan import controls_for_driver
    outside = controls_for_driver(plan.hypotheses[0], 'outside_driver_seat')
    assert raw == before and outside['failed_controls'] == ['drive']
    assert outside['remaining_controls'] == ['braking'] and 'safe_state_proposal' not in outside
    assert outside['parameters'] == h['parameters'] and outside['geometry'] == h['geometry']
    assert audit['deferred_safe_states'][0]['driver_position'] == 'outside_driver_seat'
    assert audit['deferred_safe_states'][0]['proposal'] == before['hypotheses'][0]['driver_control_assessments']['outside_driver_seat']['safe_state_proposal']


@pytest.mark.parametrize('response_update', [False, True])
@pytest.mark.parametrize('braking_available', [False, True])
def test_parking_hold_cannot_supply_dynamic_braking_or_discard_valid_driver_branch(tmp_path, response_update, braking_available):
    from test_geometry_completion import Client
    from test_scoped_timing_and_avoidance import timed_plan
    from hara_agent.contracts.physical_scenario_plan import CONTROL_ASSESSMENT_REQUIRED, controls_for_driver
    raw = timed_plan()
    h = raw['hypotheses'][0]
    inside = {k: deepcopy(v) for k, v in h.items()
              if k in CONTROL_ASSESSMENT_REQUIRED or k in {'safe_state_proposal', 'hazard_timing'}}
    outside = deepcopy(inside)
    outside['remaining_controls'] = ['automatic_parking_command', 'parking_hold']
    outside['safe_state_proposal']['required_controls'] = list(outside['remaining_controls'])
    if braking_available:
        outside['remaining_controls'].append('braking')  # Available is not a declared dependency of this proposal.
    outside['safe_state_proposal']['driver_positions'] = ['outside_driver_seat']
    h['driver_control_assessments'] = {'in_driver_seat': inside, 'outside_driver_seat': outside}
    before = deepcopy(raw)
    context = {'plan_key': 'parking-hold-is-not-braking', 'evidence': {'MF.description': 'drive failure'}}
    if response_update:
        context.update(previous_plan=timed_plan(), correction_diagnostics=[{
            'hypothesis_id': h['hypothesis_id'], 'reason': 'SAFE_STATE_RESPONSE_PATH_REVIEW'}])
        raw = {'response_updates': [{'hypothesis_id': h['hypothesis_id'], 'control_assessment': {
            **{k: deepcopy(h[k]) for k in CONTROL_ASSESSMENT_REQUIRED},
            'driver_control_assessments': deepcopy(h['driver_control_assessments'])}}]}
        before = deepcopy(raw)
    client = Client([raw])
    agent = PhysicalScenarioAgent(client, response_checkpoint_dir=tmp_path)
    plan, audit = agent.propose(context)
    actual = controls_for_driver(plan.hypotheses[0], 'outside_driver_seat')
    assert 'safe_state_proposal' not in actual
    assert actual['remaining_controls'] == outside['remaining_controls']
    assert controls_for_driver(plan.hypotheses[0], 'in_driver_seat')['safe_state_proposal'] == inside['safe_state_proposal']
    assert plan.hypotheses[0]['geometry'] == h['geometry'] and plan.hypotheses[0]['parameters'] == h['parameters']
    assert audit['deferred_safe_states'][0]['reason'] == 'SAFE_STATE_MODEL_REQUIRES_BRAKING'
    assert audit['deferred_safe_states'][0]['hypothesis_id'] == h['hypothesis_id']
    assert audit['deferred_safe_states'][0]['driver_position'] == 'outside_driver_seat'
    assert audit['deferred_safe_states'][0]['proposal'] == outside['safe_state_proposal']
    assert raw == before and audit['responses'][0]['response']['data'] == before
    restored, _ = agent.propose(context)
    assert restored == plan and len(client.requests) == 1


def test_explicit_local_response_binds_shared_controls_and_reaches_both_driver_timings(method, tmp_path):
    from test_geometry_completion import Client, run
    from test_scoped_timing_and_avoidance import timed_plan
    from test_autonomous_physics import make_candidate
    from hara_agent.contracts.physical_scenario_plan import CONTROL_ASSESSMENT_REQUIRED
    from hara_agent.services.analysis.risk_calculation_input_service import RiskCalculationInputService
    raw = timed_plan(); h = raw['hypotheses'][0]
    local = {k: deepcopy(v) for k, v in h.items()
             if k in CONTROL_ASSESSMENT_REQUIRED or k in {'safe_state_proposal', 'hazard_timing'}}
    local['safe_state_proposal']['required_controls'] = ['driver_braking', 'braking']
    outside = deepcopy(local)
    outside['remaining_controls'] = []
    outside['safe_state_proposal']['required_controls'] = ['automatic_braking_command', 'braking']
    outside['safe_state_proposal']['driver_positions'] = ['outside_driver_seat']
    h['remaining_controls'] = ['automatic_braking_command', 'braking']
    h.pop('safe_state_proposal')
    h['driver_control_assessments'] = {'in_driver_seat': local, 'outside_driver_seat': outside}
    original = deepcopy(raw)
    inside_parent = make_candidate(3.6)
    outside_parent = deepcopy(inside_parent)
    outside_parent.scenario_id += '-outside'
    outside_parent.facts['driver_in_vehicle'] = False
    client = Client([raw]); agent = PhysicalScenarioAgent(client, response_checkpoint_dir=tmp_path)
    state, generated = run(method, agent, candidates={'MF': [inside_parent, outside_parent]})
    assert len(generated['MF']) == 2 and len(client.requests) == 1
    for child in generated['MF']:
        timing = RiskCalculationInputService().ftti('MF', child.scenario_id, '',
            {**child.facts, '_fact_provenance': child.fact_provenance}, method=method)
        assert timing.hazard_time_limit_s == pytest.approx(.9364916731)
        assert timing.status.value == 'CONDITIONAL'
        h = child.facts['physical_analysis']['hypothesis']
        assert 'braking' in h['remaining_controls']
        if not child.facts['driver_in_vehicle']:
            assert 'driver_braking' not in h['remaining_controls']
        assert set(original['hypotheses'][0]['conditions']) <= set(h['conditions'])
    entry = next(iter(state.item_definition['physical_analysis_plans'].values()))
    records = [r for r in entry['audit']['control_scope_normalizations']
               if r['normalization'] == 'BOUND_EXPLICIT_SHARED_RESPONSE_DEPENDENCY']
    assert len(records) == 2 and not entry['audit']['deferred_safe_states']
    assert raw == original and entry['responses'][0]['response']['data'] == original
    _, resumed = run(method, agent, state=state, candidates={'MF': [inside_parent, outside_parent]})
    assert resumed == generated and len(client.requests) == 1


@pytest.mark.parametrize('block', ['failed', 'unresolved', 'personnel', 'not_declared'])
def test_shared_dependency_binding_never_overrules_a_local_gap_or_creates_a_control(block):
    from test_geometry_completion import Client
    from test_scoped_timing_and_avoidance import timed_plan
    from hara_agent.contracts.physical_scenario_plan import CONTROL_ASSESSMENT_REQUIRED, controls_for_driver
    raw = timed_plan(); h = raw['hypotheses'][0]
    local = {k: deepcopy(v) for k, v in h.items()
             if k in CONTROL_ASSESSMENT_REQUIRED or k in {'safe_state_proposal', 'hazard_timing'}}
    h['remaining_controls'] = ['automatic_braking_command', 'braking']
    h.pop('safe_state_proposal')
    local['remaining_controls'] = []
    local['safe_state_proposal']['required_controls'] = ['braking']
    local['safe_state_proposal']['driver_positions'] = ['outside_driver_seat']
    if block == 'failed':
        local['failed_controls'] = ['braking']
    elif block == 'unresolved':
        local['unresolved_controls'] = {'braking': 'This configuration lacks an established actuator path'}
    elif block == 'personnel':
        h['remaining_controls'] = ['driver_braking']
        local['safe_state_proposal']['required_controls'] = ['driver_braking']
    else:
        h['remaining_controls'] = []
    h['driver_control_assessments'] = {'outside_driver_seat': local}
    original = deepcopy(raw)
    plan, audit = PhysicalScenarioAgent(Client([raw])).propose({'plan_key':block, 'evidence':{'MF.description':'fault'}})
    effective = controls_for_driver(plan.hypotheses[0], 'outside_driver_seat')
    assert not effective['remaining_controls'] and 'safe_state_proposal' not in effective
    assert audit['deferred_safe_states'] and raw == original


@pytest.mark.parametrize('shape', ['object', 'pair'])
def test_control_format_repair_preserves_qualifiers_and_original_reply(tmp_path, shape):
    control = dict(rationale='Conditional independent manual channel', basis_refs=['MF.description'],
        conditions=['Manual channel remains independent of automatic command'],
        uncovered_conditions=['Shared actuator failure not covered'],
        failed_controls=['automatic_braking_command'], remaining_controls=['driver_braking'])
    corrected = dict(hypotheses=[], uncovered_conditions=[], control_assessment=control)
    raw = deepcopy(corrected)
    details = {key: deepcopy(control[key]) for key in ('rationale', 'basis_refs', 'conditions', 'uncovered_conditions')}
    raw['control_assessment']['remaining_controls'] = [
        {'control': 'driver_braking', **details} if shape == 'object' else ['driver_braking', details]]
    original = deepcopy(raw)

    class Client:
        calls = 0
        def complete_json(self, request):
            self.calls += 1
            payload = json.loads(request.user_prompt)
            if request.metadata['schema_repair']:
                assert payload['previous_invalid_response'] == original
                diagnostic = payload['repair_errors'][0]
                assert '"remaining_controls":["driver_braking"]' in diagnostic
                assert 'do not discard or broaden' in diagnostic
                assert 'do not restrict the whole motion hypothesis' in diagnostic
                return LLMResponse(deepcopy(corrected), 'OFFLINE')
            return LLMResponse(deepcopy(raw), 'OFFLINE')

    client = Client()
    agent = PhysicalScenarioAgent(client, response_checkpoint_dir=tmp_path)
    context = {'plan_key': 'control-shape', 'task_scope': 'CONTROL_INPUTS_ONLY',
               'evidence': {'MF.description': 'Automatic command fault'}}
    records = []
    plan, audit = agent.propose(context, on_response=records.append)
    assert plan.control_assessment == control
    assert original == raw == records[0]['response']['data']
    assert client.calls == 2 and audit['schema_repairs'] == 1
    resumed, resumed_audit = agent.propose(context)
    assert resumed == plan and resumed_audit['response_cache_hits'] == 2 and client.calls == 2


def test_single_control_array_wrapper_is_lossless_and_avoids_a_format_request(tmp_path):
    from test_geometry_completion import Client
    raw = raw_plan()
    h = raw['hypotheses'][0]
    h['remaining_controls'] = [['driver_braking', 'braking']]
    h['failed_controls'] = [['drive']]
    before = deepcopy(raw)
    client = Client([raw])
    agent = PhysicalScenarioAgent(client, response_checkpoint_dir=tmp_path)
    context = {'plan_key':'control-wrapper', 'evidence':{'MF.description':'drive fault'}}
    plan, audit = agent.propose(context)
    assert len(client.requests) == 1 and audit['schema_repairs'] == 0
    assert audit['responses'][0]['response']['data'] == before == raw
    expected = deepcopy(raw['hypotheses'][0])
    expected.update(remaining_controls=['driver_braking','braking'], failed_controls=['drive'])
    assert plan.hypotheses[0] == expected
    assert {r['field'] for r in audit['control_array_normalizations']} == {'remaining_controls','failed_controls'}
    agent.propose(context)
    assert len(client.requests) == 1


@pytest.mark.parametrize('values', [
    [['driver_braking'], ['braking']], [['braking','unproven rationale']],
    [['braking', {'rationale':'may work'}]],
])
def test_control_alternatives_and_details_are_not_flattened(values):
    from test_geometry_completion import Client
    raw = raw_plan(); raw['hypotheses'][0]['remaining_controls'] = values
    before = deepcopy(raw)
    client = Client([raw,raw])
    with pytest.raises(ValueError, match='remaining_controls'):
        PhysicalScenarioAgent(client).propose({'plan_key':'ambiguous-controls','evidence':{'MF.description':'fault'}})
    assert len(client.requests) == 2 and raw == before


def test_control_reference_is_repaired_after_syntax_with_same_budget_and_exact_resume(tmp_path):
    from test_llm_malfunction_envelope_repair import _sequenced_scenario_client
    from hara_agent.infrastructure.llm.provider_budget import ProviderAttemptBudget
    raw=raw_plan(assessed=True)
    raw['hypotheses'][0]['driver_control_assessments']={'outside_driver_seat':dict(
        rationale='No conditional outside response established', basis_refs=['ITEM.P-0123'],
        conditions=['outside'], uncovered_conditions=['No independent outside control'],
        failed_controls=['drive'], remaining_controls=[], unresolved_controls={'braking':'No response established'})}
    invalid=deepcopy(raw)
    invalid['hypotheses'][0]['driver_control_assessments']['outside_driver_seat']['basis_refs']=['ITEM.T-0123']
    calls=[]
    client=_sequenced_scenario_client([json.dumps(invalid)[:-1],json.dumps(invalid),json.dumps(raw)],calls)
    client.attempt_budget=ProviderAttemptBudget(tmp_path/'ledger.jsonl',run_id='separate-repairs',limit=3)
    agent=PhysicalScenarioAgent(client,response_checkpoint_dir=tmp_path/'responses')
    context={'plan_key':'bounded-source-repair','evidence':{'MF.description':'fault','ITEM.P-0123':'outside control context'}}
    plan,audit=agent.propose(context)
    assert plan.to_dict()==raw
    assert len(calls)==client.attempt_budget.attempts==audit['request_count']==3
    semantic=json.loads(json.loads(calls[2])['messages'][-1]['content'])
    assert semantic['previous_invalid_response']==invalid
    assert 'ITEM.T-0123' in semantic['repair_errors'][0] and 'ITEM.P-0123' in semantic['repair_errors'][0]
    resumed,replay=agent.propose(context)
    assert resumed==plan and replay['response_cache_hits']==2
    assert len(calls)==client.attempt_budget.attempts==3


def test_active_method_vocabulary_uses_existing_bounded_repair_without_guessing(tmp_path):
    from test_geometry_completion import Client
    invalid = raw_plan(); invalid['hypotheses'][0]['geometry']['collision_type']='VEHICLE_TO_ROAD'
    valid = raw_plan()
    client = Client([invalid,valid])
    agent = PhysicalScenarioAgent(client,response_checkpoint_dir=tmp_path/'responses')
    context={'plan_key':'method-vocabulary','evidence':{'MF.description':'fault'},
        'risk_vocabulary':{'road_user_type':{'VEHICLE':'vehicle'},'collision_type':{'FRONTAL':'front'}}}
    plan,audit = agent.propose(context)
    assert len(client.requests)==2 and audit['schema_repairs']==1
    assert 'active method vocabulary' in client.requests[1].user_prompt
    assert plan.hypotheses[0]['geometry']['collision_type']=='FRONTAL'
    assert audit['responses'][0]['response']['data']==invalid
    agent.propose(context)
    assert len(client.requests)==2
