"""G1/P1–P3: complete geometry, finite correction and durable production reuse."""
from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
import subprocess
import sys

import pytest

from test_autonomous_physics import raw_plan, make_candidate, mf, score
from test_p7_closure import method, source
from hara_agent.config.llm_config import LLMConfig
from hara_agent.infrastructure.llm.protocol import LLMResponse
from hara_agent.services.analysis.physical_hypothesis_instantiation_service import PhysicalHypothesisInstantiationService
from hara_agent.services.semantic.physical_scenario_agent import PhysicalScenarioAgent
from hara_agent.workflow import HARAState, CheckpointRepository
from hara_agent.workflow.physical_analysis import construct_physical_scenarios


class Client:
    def __init__(self, responses):
        self.responses = responses
        self.requests = []

    def complete_json(self, request):
        self.requests.append(request)
        return LLMResponse(deepcopy(self.responses[len(self.requests) - 1]), 'OFFLINE',
            usage={'prompt_tokens': 11, 'completion_tokens': 7})


def run(method, agent, state=None, candidates=None, checkpoint=None, workers=1, malfunctions=None):
    state = state or HARAState('geometry-test')
    result = construct_physical_scenarios(state, candidates or {'MF': [make_candidate()]},
        malfunctions or [mf()], method=method, agent=agent, project_facts=None,
        checkpoint=checkpoint, max_workers=workers)
    return state, result


@pytest.mark.parametrize('model,field,initial', [
    ('CONSTANT_ACCELERATION_TO_IMPACT_DISTANCE', 'relative_distance_m', 0),
    ('CONSTANT_LATERAL_ACCELERATION_TO_CONTACT', 'lateral_clearance_m', 3.6),
    ('GRADE_ROLLAWAY_TO_IMPACT_DISTANCE', 'relative_distance_m', 0),
])
def test_missing_geometry_is_one_shared_correction_not_silent_symbol(method, tmp_path, model, field, initial):
    good = raw_plan(model, initial=initial)
    bad = deepcopy(good)
    bad['hypotheses'][0]['parameters'] = [p for p in bad['hypotheses'][0]['parameters'] if p['field'] != field]
    client = Client([bad, good])
    agent = PhysicalScenarioAgent(client, response_checkpoint_dir=tmp_path/'responses')
    parent = make_candidate(initial)
    other = replace(deepcopy(parent), scenario_id='SC-outside')
    other.facts['driver_in_vehicle'] = False
    state, result = run(method, agent, candidates={'MF': [parent, other]})
    assert len(client.requests) == 2 and len(result['MF']) == 2
    entry = next(iter(state.item_definition['physical_analysis_plans'].values()))
    assert entry['responses'][0]['response']['data'] == bad
    assert entry['semantic_correction_completed']
    assert all(field in e['fields'] for e in entry['correction_diagnostics'])
    context = json.loads(client.requests[0].user_prompt)['context']
    assert field in context['model_input_status']['models'][model]['needs_hypothesis']
    for child in result['MF']:
        info = child.facts['physical_analysis']
        assert info['sensitivity']['input_completeness']['classification'] == 'COMPUTABLE_REPRESENTATIVE'
        assert info['engineering_approval'] == 'NOT_GRANTED'
        assert child.facts['motion_model_outcome']['status'] == 'CONTACT'
        assert score(method, child)['severity']['severity_score']
        assert child.scenario_id not in {parent.scenario_id, other.scenario_id}
    assert state.item_definition['physical_analysis_metrics']['token_usage'] == {'prompt_tokens': 22, 'completion_tokens': 14}


def test_scoped_existing_dynamics_are_in_request_and_can_be_inherited(method):
    parent = make_candidate()
    field = 'longitudinal_acceleration_mps2'
    parent.facts[field] = .7
    parent.fact_provenance[field] = {**source(), 'object_id': 'EGO', 'analysis_value_semantic': 'POST_FAULT_MOTION_PARAMETER'}
    raw = raw_plan(); raw['hypotheses'][0]['parameters'].pop()
    client = Client([raw])
    _, result = run(method, PhysicalScenarioAgent(client), candidates={'MF': [parent]})
    status = json.loads(client.requests[0].user_prompt)['context']['model_input_status']
    assert status['models'][raw['hypotheses'][0]['model']]['available'][field]['value'] == .7
    assert result['MF'][0].facts[field] == .7
    parent.fact_provenance[field]['analysis_value_semantic'] = 'PRE_FAULT_OPERATING_ACCELERATION'
    status = PhysicalHypothesisInstantiationService(method).input_status(parent, 'MF')
    assert field in status['models'][raw['hypotheses'][0]['model']]['excluded_parent_inputs']


def test_symbolic_geometry_and_no_contact_are_valid_without_correction(method):
    symbolic = raw_plan()
    p = symbolic['hypotheses'][0]['parameters'][1]
    p.pop('value'); p.update(kind='SYMBOLIC', symbol='d', rationale='No target extent or clearance constraint bounds a defensible representative; retain v(d), T(d).')
    client = Client([symbolic])
    _, result = run(method, PhysicalScenarioAgent(client))
    child = result['MF'][0]
    assert len(client.requests) == 1 and 'relative_distance_m' not in child.facts
    assert child.facts['physical_analysis']['sensitivity']['parameter_relations']
    assert child.facts['physical_analysis']['sensitivity']['input_completeness']['classification'] == 'EXPLAINED_SYMBOLIC_RELATION'
    no_contact = raw_plan(initial=3.6, acceleration=-2)
    client = Client([no_contact])
    _, result = run(method, PhysicalScenarioAgent(client), candidates={'MF': [make_candidate(3.6)]})
    assert len(client.requests) == 1
    assert result['MF'][0].facts['motion_model_outcome']['status'] == 'STOPPED_BEFORE_CONTACT'
    assert 'ttc_s' not in result['MF'][0].facts and 'collision_branches' not in result['MF'][0].facts


def test_fixed_moving_target_skips_proposal_without_dropping_causal_identity(method):
    parent = make_candidate()
    parent.facts.update(object_speed_kph=10, object_id='moving-target')
    parent.fact_provenance['object_speed_kph'] = source()
    client = Client([])
    state, result = run(method, PhysicalScenarioAgent(client), candidates={'MF': [parent]})
    assert result['MF'] == [parent] and not client.requests
    assert parent.analysis_instance['physical_plan_diagnostics'][0]['object_speed_kph'] == 10
    assert not parent.facts.get('collision_branches')
    assert state.item_definition['physical_analysis_metrics']['known_model_gap_skipped'] == 1
    parent.fact_provenance['object_speed_kph']['approval'] = 'PENDING'
    assert PhysicalHypothesisInstantiationService.known_model_gap(parent, 'MF') is None


def test_partial_conflict_does_not_correct_forever(method):
    bad = raw_plan(); bad['hypotheses'][0]['parameters'].pop(1)
    client = Client([bad, bad])
    state, result = run(method, PhysicalScenarioAgent(client))
    assert len(client.requests) == 2 and result['MF'][0].scenario_id == 'SC'
    assert result['MF'][0].analysis_instance['physical_plan_diagnostics'][0]['reason'] == 'REQUIRED_MODEL_INPUT_OMITTED'


def test_response_survives_full_snapshot_failure_and_scope_change_invalidates(method, tmp_path):
    client = Client([raw_plan(), raw_plan()])
    agent = PhysicalScenarioAgent(client, response_checkpoint_dir=tmp_path/'responses')
    def fail(_):
        raise OSError('full snapshot disk failure')
    with pytest.raises(OSError, match='snapshot'):
        run(method, agent, checkpoint=fail)
    assert len(list((tmp_path/'responses').glob('*.json'))) == 1
    repo = CheckpointRepository(tmp_path/'states')
    state, result = run(method, agent, checkpoint=repo.save)
    assert len(client.requests) == 1
    restored = repo.load(state.run_id)
    _, resumed = run(method, agent, state=restored, checkpoint=repo.save)
    assert resumed == result and len(client.requests) == 1
    parent = make_candidate()
    parent.fact_provenance['ego_speed_kph']['source_refs'][0]['location'] = 'new-source-context'
    run(method, agent, candidates={'MF': [parent]})
    assert len(client.requests) == 2


def test_abrupt_process_exit_after_response_does_not_pay_again(tmp_path):
    payload = tmp_path/'plan.json'; payload.write_text(json.dumps(raw_plan()))
    cache = tmp_path/'responses'
    code = '''import json, os, sys
from pathlib import Path
from hara_agent.infrastructure.llm.protocol import LLMResponse
from hara_agent.services.semantic.physical_scenario_agent import PhysicalScenarioAgent
class Client:
 def complete_json(self, request): return LLMResponse(json.loads(Path(sys.argv[1]).read_text()), 'OFFLINE')
PhysicalScenarioAgent(Client(), response_checkpoint_dir=Path(sys.argv[2])).propose(
 {'plan_key':'process-exit','evidence':{'MF.description':'fault'}}, on_response=lambda _:os._exit(19))
'''
    exited = subprocess.run([sys.executable, '-c', code, str(payload), str(cache)], check=False)
    assert exited.returncode == 19
    client = Client([])
    plan, audit = PhysicalScenarioAgent(client, response_checkpoint_dir=cache).propose(
        {'plan_key': 'process-exit', 'evidence': {'MF.description': 'fault'}})
    assert not client.requests and audit['response_cache_hits'] == 1
    assert plan.to_dict() == raw_plan()


def test_batch_defaults_are_one_config_authority_and_env_is_preserved(monkeypatch):
    from hara_agent.services.semantic.scenario_batching import DEFAULT_SCENARIO_BATCH_MAX_CHARS, DEFAULT_SCENARIO_BATCH_MAX_ITEMS
    monkeypatch.delenv('HARA_LLM_SCENARIO_BATCH_MAX_CHARS', raising=False)
    monkeypatch.delenv('HARA_LLM_SCENARIO_BATCH_MAX_ITEMS', raising=False)
    config = LLMConfig.from_env()
    assert config.scenario_batch_max_chars == DEFAULT_SCENARIO_BATCH_MAX_CHARS == 24000
    assert config.scenario_batch_max_items == DEFAULT_SCENARIO_BATCH_MAX_ITEMS == 2
    monkeypatch.setenv('HARA_LLM_SCENARIO_BATCH_MAX_CHARS', '18000')
    assert LLMConfig.from_env().scenario_batch_max_chars == 18000


def test_parallel_partial_failure_keeps_successful_plan_and_unique_usage(method, tmp_path):
    from threading import Barrier
    barrier = Barrier(2)
    class PartialClient:
        def complete_json(self, request):
            barrier.wait(timeout=5)
            if json.loads(request.user_prompt)['context']['malfunction']['malfunction_id'] == 'MF-broken':
                raise RuntimeError('independent task failed')
            return LLMResponse(raw_plan(), 'OFFLINE', usage={'prompt_tokens': 11, 'completion_tokens': 7})
    state = HARAState('partial-physical')
    repo = CheckpointRepository(tmp_path/'state')
    malfunctions = [mf(), replace(mf(), malfunction_id='MF-broken')]
    with pytest.raises(RuntimeError, match='independent task'):
        run(method, PhysicalScenarioAgent(PartialClient(), response_checkpoint_dir=tmp_path/'responses'),
            state=state, candidates={m.malfunction_id: [make_candidate()] for m in malfunctions},
            malfunctions=malfunctions, checkpoint=repo.save, workers=2)
    restored = repo.load(state.run_id)
    assert sum(bool(e.get('instantiations')) for e in restored.item_definition['physical_analysis_plans'].values()) == 1
    assert restored.item_definition['physical_analysis_metrics']['token_usage'] == {'prompt_tokens': 11, 'completion_tokens': 7}
    assert len(list((tmp_path/'responses').glob('*.json'))) == 1


@pytest.mark.parametrize('value', [None, True, float('nan')])
def test_unusable_parent_number_is_not_reported_as_complete_input(method, value):
    parent = make_candidate()
    parent.facts['relative_distance_m'] = value
    parent.fact_provenance['relative_distance_m'] = source()
    status = PhysicalHypothesisInstantiationService(method).input_status(parent, 'MF')
    model = status['models']['CONSTANT_ACCELERATION_TO_IMPACT_DISTANCE']
    assert 'relative_distance_m' in model['needs_hypothesis']
    assert model['excluded_parent_inputs']['relative_distance_m'] == 'PARENT_INPUT_NOT_FINITE_NUMBER'
