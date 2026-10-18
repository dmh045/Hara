"""Current-world symbolic evidence and the actual malformed live envelope."""
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_autonomous_physics import raw_plan, make_candidate, instantiate, mf
from test_physical_input_scope import symbol
from test_p7_closure import method
from hara_agent.config import LLMConfig
from hara_agent.infrastructure.llm import LLMRequest, LLMJSONContractError
from hara_agent.infrastructure.llm.openai_compatible import OpenAICompatibleClient
from hara_agent.infrastructure.llm.provider_budget import ProviderAttemptBudget
from hara_agent.services.semantic.scenario_agent import ScenarioFeasibilityAgent
from hara_agent.services.semantic.scenario_batching import _scenario_json
from hara_agent.services.semantic.scenario_evidence import build_fact_registry


def symbolic_child(method):
    raw = raw_plan(initial=7)
    symbol(raw, 'longitudinal_acceleration_mps2')
    symbol(raw, 'relative_distance_m')
    return instantiate(method, raw, make_candidate(7))[0]


def test_symbolic_relation_reaches_causal_prompt_and_validator_without_numeric_approval(method):
    child = symbolic_child(method)
    before = deepcopy(child)
    registry = build_fact_registry(mf(), child)
    ref = 'DERIVED.parameter_relations'
    record = registry.resolve(ref)
    assert record['derivation_type'] == 'SYMBOLIC_MODEL_RELATION'
    assert record['approval_status'] == 'PENDING'
    assert record['value']['object_id'] == child.facts['object_id']
    assert record['value']['impact_time_id'] == child.facts['impact_time_id']
    assert record['value']['numeric_result_available'] is False
    prompt = json.loads(_scenario_json(mf(), child, causal_evidence_budget=1))
    assert prompt['conditional_physical_analysis']['parameter_relations_ref'] == ref
    assert 'v_impact(d)' in prompt['causal_evidence_view']
    assert 'T(d)' in prompt['causal_evidence_view']
    assert '首次接触发生在停止前及模型有效期内' in prompt['causal_evidence_view']
    assert 'ttc_s' not in child.facts and 'ego_impact_speed_kph' not in child.facts
    payload = dict(scenario_id=child.scenario_id, physically_feasible=True, functionally_relevant=True,
        causally_relevant=True, breakpoint='NONE', hazardous_event='Within the stated model domain, contact with the same front object',
        causal_chain={'m_to_b': dict(claim='defined excessive drive', basis_type='DIRECT_FACT', evidence_refs=['MF.description']),
            'b_to_i': dict(claim='conditional forward motion', basis_type='DERIVED_PHYSICS', evidence_refs=[ref]),
            'i_to_h': dict(claim='contact only when the stated positive-distance/root/domain conditions hold', basis_type='DERIVED_PHYSICS', evidence_refs=[ref])},
        risk_dimension_changes=[], confidence=.8, status='PENDING')
    assessment = ScenarioFeasibilityAgent._parse(mf(), payload, scenario=child)
    assert assessment.causally_relevant
    assert child == before


def test_symbolic_relation_from_another_pair_is_not_registered(method):
    child = symbolic_child(method)
    assert build_fact_registry(replace(mf(), malfunction_id='OTHER'), child).resolve('DERIVED.parameter_relations') is None
    child.scenario_id = 'OTHER'
    assert build_fact_registry(mf(), child).resolve('DERIVED.parameter_relations') is None


def test_point_model_does_not_gain_symbolic_evidence(method):
    child = instantiate(method)[0]
    assert build_fact_registry(mf(), child).resolve('DERIVED.parameter_relations') is None
    assert child.facts['ego_impact_speed_kph'] > 0


def test_actual_missing_brace_is_not_hand_repaired_and_json_mode_survives_one_retry(tmp_path):
    malformed = (Path(__file__).parents[1] / 'fixtures/physical_live_invalid_causal_20261019.txt').read_text()
    assert hashlib.sha256(malformed.encode()).hexdigest() == 'ce92856ecc34d36d3b79c91169935154e44699b8c3568766a9071bf214e6f7f4'
    calls = []
    def transport(_url, _headers, body, _timeout):
        calls.append(json.loads(body))
        assert calls[-1]['response_format'] == {'type':'json_object'}
        return {'model':'OFFLINE_RAW_REPLAY','choices':[{'message':{'content':malformed},'finish_reason':'stop'}]}
    budget = ProviderAttemptBudget(tmp_path/'run.jsonl', run_id='review', limit=8)
    config = LLMConfig('openai-compatible','https://example.invalid','offline','unused', max_retries=0,
                       scenario_response_format='json_object')
    client = OpenAICompatibleClient(config, transport=transport, attempt_budget=budget)
    agent = ScenarioFeasibilityAgent(client)
    request = LLMRequest('assess_scenario_feasibility',agent.system_prompt,'JSON test',agent.schema_name,
                         agent.prompt_version,response_format=agent.response_format)
    with pytest.raises(LLMJSONContractError):
        client.complete_json(request)
    assert len(calls) == budget.attempts == 2
    assert 'FORMAT_CORRECTION_INPUT=' in calls[1]['messages'][1]['content']
    for path in (tmp_path/'run.responses').glob('*.json'):
        assert json.loads(path.read_text())['response']['choices'][0]['message']['content'] == malformed


def test_json_mode_is_explicit_and_minimal_output_keeps_valid_driver_proof(monkeypatch):
    monkeypatch.setenv('HARA_LLM_PROVIDER','volcengine-agent-plan')
    monkeypatch.delenv('HARA_LLM_SCENARIO_RESPONSE_FORMAT', raising=False)
    assert LLMConfig.from_env().scenario_response_format == 'json_object'
    monkeypatch.setenv('HARA_LLM_SCENARIO_RESPONSE_FORMAT','none')
    assert LLMConfig.from_env().scenario_response_format == 'none'
    monkeypatch.setenv('HARA_LLM_SCENARIO_RESPONSE_FORMAT','json_object')
    assert LLMConfig.from_env().scenario_response_format == 'json_object'
    assert ScenarioFeasibilityAgent(SimpleNamespace()).response_format is None
    assert '缺少任一项引用时省略' in ScenarioFeasibilityAgent.SYSTEM_PROMPT
    assert 'driver_noninterference' in ScenarioFeasibilityAgent.SYSTEM_PROMPT


@pytest.mark.parametrize('wrong_scope', [False, True])
def test_risk_fact_extraction_requests_only_missing_current_world_inputs(method, wrong_scope):
    from hara_agent.contracts import FactOrigin, FactType
    from hara_agent.infrastructure.llm import LLMResponse
    from hara_agent.services.semantic import ScenarioRiskFactAgent
    from test_scenario_risk_fact_agent import _spec
    child = instantiate(method)[0]
    if wrong_scope:
        child.scenario_id = 'different-world'
    before = deepcopy(child)
    calls = []
    class Client:
        def complete_json(self, request):
            raw = request.user_prompt.split('fact_requests=', 1)[1]
            requests, _ = json.JSONDecoder().raw_decode(raw)
            calls.extend(requests)
            return LLMResponse({'results':[{**{k:r[k] for k in ('malfunction_id','scenario_id','fact_type')},
                'status':'NOT_FOUND','evidence_ids':[]} for r in requests]}, 'OFFLINE')
    specs = [_spec(t, FactOrigin.SCENARIO_FACT) for t in
             (FactType.COLLISION_TYPE, FactType.VEHICLE_SPEED, FactType.INTERVENTION_AVAILABLE)]
    facts, audit = ScenarioRiskFactAgent(Client()).interpret(
        [dict(malfunction_id=mf().malfunction_id, scenario_id=child.scenario_id,
              physically_feasible=True, functionally_relevant=True, causally_relevant=True,
              risk_dimensions_changed=[], status='FINALIZED')], [child], [mf()], specs)
    assert not facts and child == before
    # The global initial-speed fact is still valid; the scoped hypothesis is not.
    assert len(calls) == audit['llm_requested_fact_pair_count'] == (2 if wrong_scope else 1)
    assert audit['existing_fact_count'] == (1 if wrong_scope else 2)
    if not wrong_scope:
        assert calls[0]['fact_type'] == 'INTERVENTION_AVAILABLE'
        assert child.fact_provenance['collision_type']['approval'] == 'PENDING'


def test_equal_method_binding_cannot_approve_a_conditional_collision_world(method):
    from dataclasses import asdict
    from hara_agent.models import ItemDefinitionFacts, RiskFact, SourceRef, FactProvenance, ReviewStatus
    from hara_agent.services.analysis import StructuredRiskScoringService, MethodContractASILService, MethodRiskFactBindingService
    from hara_agent.workflow import HARAState
    from hara_agent.workflow.nodes.scoring import score_structured_scenarios
    child = instantiate(method)[0]
    original = deepcopy(child.fact_provenance['collision_type'])
    assessment = ScenarioFeasibilityAgent._parse(mf(), dict(scenario_id=child.scenario_id,
        physically_feasible=True, functionally_relevant=True, causally_relevant=True,
        breakpoint='NONE', hazardous_event='conditional contact', risk_dimension_changes=[], confidence=.9,
        causal_chain={'m_to_b':dict(claim='defined fault',basis_type='DIRECT_FACT',evidence_refs=['MF.description']),
            'b_to_i':dict(claim='computed motion',basis_type='DERIVED_PHYSICS',evidence_refs=['DERIVED.ego_impact_speed_kph']),
            'i_to_h':dict(claim='conditional contact',basis_type='DERIVED_PHYSICS',evidence_refs=['DERIVED.relative_speed_kph'])}), scenario=child)
    fact = RiskFact('EXTRACTED', 'COLLISION_TYPE', child.facts['collision_type'],
        source_refs=[SourceRef('item_definition','synthetic','p1','parking')],
        context={'malfunction_id':mf().malfunction_id,'scenario_id':child.scenario_id},
        provenance=FactProvenance.DERIVED, approval=ReviewStatus.FINALIZED)
    state = HARAState('offline-source-review',scenarios=[child],malfunctions=[asdict(mf())],
        item_definition={'scenario_assessments':[assessment.to_dict()],
            'typed':asdict(ItemDefinitionFacts('synthetic','synthetic',risk_facts=[fact],sources=fact.source_refs))})
    score_structured_scenarios(state,StructuredRiskScoringService(method),MethodContractASILService(method),
                              risk_fact_binding=MethodRiskFactBindingService(method))
    assert state.scenarios[0].fact_provenance['collision_type'] == original
    audit = next(e for e in state.audit_trail if e['event']=='structured_risk_scoring_completed')
    assert audit['risk_fact_binding_audits'][0]['preserved_conditional_equal_facts'] == ['collision_type']
