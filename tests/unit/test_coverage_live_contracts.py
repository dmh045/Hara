"""Unedited Provider failures: repair representation, never physical conclusions."""
from copy import deepcopy
from dataclasses import fields
import gzip
import json
from pathlib import Path

import pytest

from test_autonomous_physics import method
from hara_agent.contracts.physical_scenario_plan import PhysicalScenarioPlan
from hara_agent.infrastructure.llm import LLMResponse
from hara_agent.models import MalfunctionCandidate, ReviewStatus, SourceRef
from hara_agent.services.analysis.physical_hypothesis_instantiation_service import PhysicalHypothesisInstantiationService
from hara_agent.services.analysis.scenario_physics import causal_physics_conflicts
from hara_agent.services.semantic.scenario_agent import ScenarioFeasibilityAgent, ScenarioSchemaContractError
from hara_agent.services.semantic.physical_scenario_agent import PhysicalScenarioAgent
from hara_agent.workflow import HARAState


def recorded():
    with gzip.open(Path(__file__).parents[1] / 'fixtures/coverage_live_contracts_20261026.json.gz', 'rt') as stream:
        return json.load(stream)


def malfunction(raw):
    value = {f.name: raw[f.name] for f in fields(MalfunctionCandidate) if f.name in raw}
    value['status'] = ReviewStatus(raw['status'])
    value['sources'] = [SourceRef(**s) for s in raw['sources']]
    return MalfunctionCandidate(**value)


def scenario(raw):
    return HARAState._scenario_from_dict(raw)


def test_exact_supplied_target_id_does_not_cost_item_repair_requests():
    data = recorded()['causal']
    response = json.loads(data['response']['choices'][0]['message']['content'])
    original = deepcopy(response)
    mf = malfunction(data['malfunction'])
    candidates = [scenario(c) for c in data['scenarios']]
    class Frozen:
        calls = 0
        def complete_json(self, request):
            self.calls += 1
            return LLMResponse(deepcopy(response), 'FROZEN_REAL')
    client = Frozen()
    results, audit = ScenarioFeasibilityAgent(client).assess(mf, candidates)
    assert client.calls == 1 and audit['item_repair_calls'] == 0
    assert len(results) == 2 and all(a.retain for a in results)
    for result, raw in zip(results, response['assessments']):
        assert result.physical_interaction['object_id'] == raw['physical_interaction']['target']
        assert result.physical_interaction['target'] == 'SCENARIO_OBJECT'
        assert result.hazardous_event == raw['hazardous_event']
        assert result.causal_chain == raw['causal_chain']
    assert response == original


@pytest.mark.parametrize('mutation', ['foreign_target', 'conflicting_object'])
def test_target_identity_normalization_does_not_accept_another_object(mutation):
    data = recorded()['causal']
    item = json.loads(data['response']['choices'][0]['message']['content'])['assessments'][0]
    if mutation == 'foreign_target': item['physical_interaction']['target'] += '-other'
    else: item['physical_interaction']['object_id'] = 'other'
    with pytest.raises(ScenarioSchemaContractError):
        ScenarioFeasibilityAgent._parse(malfunction(data['malfunction']), item, scenario=scenario(data['scenarios'][0]))


def test_fallback_target_identity_still_checks_same_trajectory():
    c = scenario(recorded()['causal']['scenarios'][0])
    c.facts['motion_model_outcome'] = {'status': 'NO_CONTACT_WITHIN_MODEL'}
    claim = {'target': 'SCENARIO_OBJECT', 'object_id': c.scenario_id + ':target',
             'object_position': c.facts['object_position'], 'relation': 'CONTACT',
             'assessment_reference': 'MODEL_TRAJECTORY'}
    assert 'NO_CONTACT_TRAJECTORY_CONTRADICTS_CONTACT_CLAIM' in causal_physics_conflicts(c, {'physical_interaction': claim})


def test_unresolved_safe_state_does_not_hide_remaining_physical_conflict(method):
    data = recorded()['physical']
    raw = data['response']['data']; original = deepcopy(raw)
    with pytest.raises(ValueError, match='Safe state proposal'):
        PhysicalScenarioPlan.from_dict(raw, evidence_ids=set(data['context']['evidence']))
    class Frozen:
        calls = 0
        def complete_json(self, request):
            self.calls += 1
            return LLMResponse(**deepcopy(data['response']))
    client = Frozen(); records = []
    plan, audit = PhysicalScenarioAgent(client).propose(data['context'], on_response=records.append)
    assert client.calls == 1 and audit['schema_repairs'] == 0
    assert audit['deferred_safe_states'][0]['proposal'] == raw['hypotheses'][0]['safe_state_proposal']
    assert records[0]['response']['data'] == original == raw
    h = plan.hypotheses[0]
    assert 'safe_state_proposal' not in h and h['remaining_controls'] == []
    assert h['parameters'] == raw['hypotheses'][0]['parameters']
    assert h['geometry'] == raw['hypotheses'][0]['geometry']
    children, errors = PhysicalHypothesisInstantiationService(method).instantiate(
        scenario(data['parent']), malfunction(data['malfunction']), plan,
        plan_key=data['context']['plan_key'], response_model='FROZEN_REAL',
        project_context=data['context']['project_context'], function=data['context']['function'])
    # The unedited real reply also contradicts its road-to-body orientation.
    # Local SG deferral must expose this to physical validation, not accept it.
    assert not children
    assert {e['reason'] for e in errors} == {'GRADE_HEADING_CONFLICT:heading_relative_to_downhill'}


def test_local_safe_state_deferral_keeps_a_consistent_motion_computable(method):
    from test_autonomous_physics import raw_plan, make_candidate, mf
    raw = raw_plan(); h = raw['hypotheses'][0]
    h['remaining_controls'] = []
    h['unresolved_controls'] = {'braking': 'Fault independence is unresolved'}
    h['safe_state_proposal'] = {'description': 'Stop if independent braking is available',
        'required_controls': ['braking'], 'basis_refs': ['MF.description'],
        'conditions': ['independent braking is available']}
    class Frozen:
        def complete_json(self, request):
            return LLMResponse(deepcopy(raw), 'OFFLINE_SYNTHETIC')
    plan, _ = PhysicalScenarioAgent(Frozen()).propose({'plan_key': 'local-state-gap', 'evidence': {'MF.description': 'drive fault'}})
    children, errors = PhysicalHypothesisInstantiationService(method).instantiate(
        make_candidate(), mf(), plan, plan_key='local-state-gap', response_model='OFFLINE_SYNTHETIC')
    assert children and not errors, errors
    assert children[0].facts['motion_model_outcome']['status'] == 'CONTACT'
    assert children[0].facts.get('emergency_braking_available') is None
