"""Physical-only wire policy, bounded escalation, cache and frozen-input checks."""
from copy import deepcopy
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import json

import pytest

from hara_agent.config.llm_config import LLMConfig, physical_reasoning_options
from hara_agent.infrastructure.llm import LLMRequest
from hara_agent.infrastructure.llm.openai_compatible import OpenAICompatibleClient
from hara_agent.infrastructure.llm.provider_budget import ProviderAttemptBudget, ProviderAttemptBudgetExceeded
from hara_agent.infrastructure.llm.response_checkpoint import ResponseCheckpoint
from hara_agent.services.semantic.physical_scenario_agent import (
    PhysicalScenarioAgent, compact_physical_request, expand_physical_prompt_view, physical_prompt_view,
)
from test_autonomous_physics import raw_plan
from test_p7_closure import method


def config(**kwargs):
    return LLMConfig(provider="volcengine-agent-plan", base_url="https://example.invalid", model="test",
        api_key="unused", max_retries=0, scenario_thinking="enabled", **kwargs)


def request(task="construct_post_fault_physical_scenario"):
    return LLMRequest(task, "system", "user", "PhysicalScenarioPlan" if task.startswith("construct")
        else "ScenarioFeasibilityAssessmentList", "test")


def envelope(data):
    return {"model": "test", "choices": [{"message": {"content": json.dumps(data)}, "finish_reason": "stop"}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150,
                  "completion_tokens_details": {"reasoning_tokens": 30}}}


@pytest.mark.parametrize("mode,effort,expected", [
    ("", "", {"thinking": {"type": "enabled"}}),
    ("enabled", "low", {"thinking": {"type": "enabled"}, "reasoning_effort": "low"}),
    ("disabled", "low", {"thinking": {"type": "disabled"}}),
])
def test_physics_wire_options_do_not_change_causality_and_are_recorded(tmp_path, mode, effort, expected):
    bodies = []
    def transport(_url, _headers, body, _timeout):
        bodies.append(json.loads(body))
        return envelope(raw_plan() if len(bodies) == 1 else {"assessments": []})
    cfg = config(physics_thinking=mode, physics_reasoning_effort=effort)
    budget = ProviderAttemptBudget(tmp_path/"ledger.jsonl", run_id="test", limit=2)
    client = OpenAICompatibleClient(cfg, transport, budget)
    response = client.complete_json(request())
    client.complete_json(request("assess_scenario_feasibility"))
    assert {k: bodies[0][k] for k in ("thinking", "reasoning_effort") if k in bodies[0]} == expected
    assert bodies[1]["thinking"] == {"type": "enabled"} and "reasoning_effort" not in bodies[1]
    assert client.config == cfg
    events = [json.loads(line) for line in budget.path.read_text().splitlines()]
    sent = {**expected, "max_tokens": cfg.max_tokens}
    assert events[0]["request_options"] == response.usage["request_options"] == sent
    assert events[1]["usage"] == envelope({})["usage"]
    raw = json.loads((budget.path.with_suffix(".responses")/"000001.json").read_text())
    assert raw["request"]["metadata"]["_provider_options"] == sent
    assert "unused" not in budget.path.read_text()


def test_env_is_opt_in_and_unsupported_effort_is_not_silently_accepted(monkeypatch):
    for key in ("PHYSICS_THINKING", "PHYSICS_REASONING_EFFORT"):
        monkeypatch.delenv("HARA_LLM_" + key, raising=False)
    default = LLMConfig.from_env()
    assert default.physics_thinking == default.physics_reasoning_effort == ""
    monkeypatch.setenv("HARA_LLM_PHYSICS_THINKING", " enabled ")
    monkeypatch.setenv("HARA_LLM_PHYSICS_REASONING_EFFORT", " low ")
    assert LLMConfig.from_env().physics_reasoning_effort == "low"
    with pytest.raises(ValueError, match="PHYSICS_REASONING_EFFORT"):
        config(physics_reasoning_effort="typo").validate()
    with pytest.raises(ValueError, match="仅验证"):
        replace(config(physics_reasoning_effort="low"), provider="openai-compatible").validate()


@pytest.mark.parametrize("kind", ["schema", "format", "semantic"])
def test_existing_repair_escalates_low_without_adding_attempts(tmp_path, kind):
    bodies = []
    bad = raw_plan(); bad["hypotheses"][0].pop("basis_refs")
    def transport(_url, _headers, body, _timeout):
        bodies.append(json.loads(body))
        if kind == "format" and len(bodies) == 1:
            data = envelope(raw_plan()); data["choices"][0]["message"]["content"] = '{"hypotheses":[]'
            return data
        return envelope(bad if kind == "schema" and len(bodies) == 1 else raw_plan())
    budget = ProviderAttemptBudget(tmp_path/"ledger.jsonl", run_id="test", limit=2)
    client = OpenAICompatibleClient(config(physics_reasoning_effort="low"), transport, budget)
    if kind == "semantic":
        client.complete_json(replace(request(), metadata={"physical_semantic_correction": True}))
        assert bodies[0]["reasoning_effort"] == "high" and budget.attempts == 1
    elif kind == "format":
        client.complete_json(request())
    else:
        context = {"plan_key": "key", "evidence": {"MF.description": "fault"}}
        agent = PhysicalScenarioAgent(client, response_checkpoint_dir=tmp_path/"responses")
        plan, audit = agent.propose(context)
        assert audit["schema_repairs"] == 1 and plan.to_dict() == raw_plan()
        assert agent.propose(context)[0] == plan and budget.attempts == 2
    if kind != "semantic":
        assert [b["reasoning_effort"] for b in bodies] == ["low", "high"]
        assert len(bodies) == budget.attempts == 2
    assert client.config.physics_reasoning_effort == "low"


def test_budget_stop_before_escalation_resumes_same_raw_reply(tmp_path):
    bodies = []
    bad = raw_plan(); bad["hypotheses"][0].pop("basis_refs")
    def transport(_url, _headers, body, _timeout):
        bodies.append(json.loads(body)); return envelope(bad if len(bodies) == 1 else raw_plan())
    ledger = tmp_path/"ledger.jsonl"
    client = OpenAICompatibleClient(config(physics_reasoning_effort="low"), transport,
        ProviderAttemptBudget(ledger, run_id="test", limit=1))
    agent = PhysicalScenarioAgent(client, response_checkpoint_dir=tmp_path/"responses")
    context = {"plan_key": "key", "evidence": {"MF.description": "fault"}}
    with pytest.raises(ProviderAttemptBudgetExceeded):
        agent.propose(context)
    first = (ledger.with_suffix(".responses")/"000001.json").read_bytes()
    client.attempt_budget = ProviderAttemptBudget(ledger, run_id="test", limit=2)
    plan, audit = agent.propose(context)
    assert plan.to_dict() == raw_plan() and audit["response_cache_hits"] == 1
    assert [b["reasoning_effort"] for b in bodies] == ["low", "high"]
    assert (ledger.with_suffix(".responses")/"000001.json").read_bytes() == first


def test_modes_isolate_physical_cache_but_not_causal_cache_and_concurrent_hits(tmp_path):
    bodies = []
    def transport(_url, _headers, body, _timeout):
        bodies.append(json.loads(body)); return envelope({"hypotheses": [], "uncovered_conditions": []})
    cache = ResponseCheckpoint(tmp_path)
    clients = [OpenAICompatibleClient(cfg, transport) for cfg in [config(),
        config(physics_reasoning_effort="low"), config(physics_thinking="disabled")]]
    for client in clients:
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(lambda _: cache.complete(client, request()), range(4)))
        assert sum(not r.usage.get("response_checkpoint_hit") for r in results) == 1
    assert len(bodies) == 3
    # Physical configuration does not invalidate unrelated tasks.
    plain = replace(request(), task="unrelated")
    for client in clients: cache.complete(client, plain)
    assert len(bodies) == 4
    # Exact pre-change cache hash stays valid without either new setting.
    from dataclasses import asdict
    import hashlib
    cfg = clients[0].config
    material = {"request": asdict(request()), "route": {k: getattr(cfg, k)
        for k in ("provider", "base_url", "model", "scenario_thinking")}}
    old_key = hashlib.sha256(json.dumps(material, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), default=str).encode()).hexdigest()
    assert (tmp_path/f"{old_key}.json").exists()


def test_diagnostic_grouping_is_lossless_and_keeps_driver_and_raw_numeric_differences():
    diagnostics = [{"scenario_id": "s1", "hypothesis_id": "h1", "instance_hypothesis_id": "i1",
        "driver_position": "in_driver_seat", "reason": "SAFE_STOP_RESPONSE_UNREACHABLE",
        "evidence": {"ttc_s": .154286, "object": "a", "stage": "fault", "direction": "REVERSE"},
        "resolution": "Keep the negative result, not a new geometry."}]
    diagnostics += [dict(diagnostics[0], scenario_id="s2", instance_hypothesis_id="i2")]
    diagnostics += [dict(diagnostics[0], scenario_id="s3", driver_position="outside_driver_seat")]
    changed = deepcopy(diagnostics[0]); changed["evidence"]["ttc_s"] = .154287; diagnostics.append(changed)
    context = {"correction_diagnostics": diagnostics, "evidence": {"MF.description": "scope; not approved"}}
    view, texts = physical_prompt_view(context)
    body = {"context": view, "source_texts": texts}
    compact = compact_physical_request(body, response_only=False, controls_only=False)
    groups = compact["context"]["correction_diagnostics"]["$diagnostic_groups"]
    assert len(groups) == 3 and len(groups[0]["occurrences"]) == 2
    assert expand_physical_prompt_view(compact["context"], compact["source_texts"]) == context
    assert body["context"]["correction_diagnostics"] == diagnostics
    # Long generated identities may themselves share text/structure references.
    # The audit reconstruction must expand occurrences as well as diagnostics.
    for d in diagnostics:
        d['instance_hypothesis_id'] = 'long-world-identity-' * 8
    view, texts = physical_prompt_view(context)
    compact = compact_physical_request({'context': view, 'source_texts': texts},
        response_only=False, controls_only=False)
    assert expand_physical_prompt_view(compact['context'], compact['source_texts']) == context


def test_disabled_omits_effort_but_explicit_bounded_repair_uses_high():
    cfg = config(physics_thinking="disabled", physics_reasoning_effort="low")
    assert physical_reasoning_options(cfg) == {"thinking": {"type": "disabled"}}
    assert physical_reasoning_options(cfg, repair=True) == {"thinking": {"type": "enabled"}, "reasoning_effort": "high"}


def test_workflow_plan_cache_isolated_by_physical_policy_and_old_entry_survives(method, tmp_path):
    from test_geometry_completion import Client, run
    reply = {"hypotheses": [], "uncovered_conditions": ["No applicable model for this frozen boundary case"]}
    client = Client([reply, reply, reply]); client.config = config()
    agent = PhysicalScenarioAgent(client, response_checkpoint_dir=tmp_path)
    state, result = run(method, agent)
    original = deepcopy(state.item_definition['physical_analysis_plans'])
    assert run(method, agent, state=state)[1] == result and len(client.requests) == 1
    for cfg in (config(physics_reasoning_effort='low'), config(physics_thinking='disabled')):
        client.config = cfg
        run(method, agent, state=state)
    assert len(client.requests) == len(state.item_definition['physical_analysis_plans']) == 3
    for key, entry in original.items():
        assert state.item_definition['physical_analysis_plans'][key] == entry
    assert {json.loads(r.user_prompt)['context']['route'].get('physical_reasoning', {}).get(
        'initial', {}).get('reasoning_effort', 'not_sent') for r in client.requests} == {'low', 'not_sent'}


def test_bounded_application_resume_context_binds_physics_policy_without_changing_old_default(tmp_path):
    from types import SimpleNamespace
    from hara_agent.application import HARAApplication
    from hara_agent.config import RunConfig
    item = tmp_path / 'item.docx'
    item.write_bytes(b'unchanged source')
    run = RunConfig(item, None, tmp_path / 'report.xlsx', tmp_path / 'checkpoints', provider_attempt_limit=80)
    client = SimpleNamespace(config=config())
    app = HARAApplication(run, client)
    method_ref = {'method_source_hash': 'method', 'report_template_hash': 'report'}
    old = app.bounded_source_context(method_ref)
    assert 'physical_reasoning' not in old
    client.config = config(physics_reasoning_effort='low')
    low = app.bounded_source_context(method_ref)
    assert low != old
    assert {k: v for k, v in low.items() if k != 'physical_reasoning'} == old
    client.config = config(physics_thinking='disabled')
    assert app.bounded_source_context(method_ref) != low


def test_scoped_projection_keeps_direct_authority_and_only_moves_compiler_provenance_to_audit():
    compiler = {'source_type': 'method_contract', 'source_id': 'active',
        'location': 'yaml!dimensions.ROAD', 'excerpt': 'complete compiler taxonomy ' * 80}
    project = {'source_type': 'item_definition', 'source_id': 'item',
        'location': 'scope', 'excerpt': 'No automatic braking after this fault; downhill is behind ego.'}
    authority = {'approval': 'PENDING', 'source_refs': [compiler, project,
        {**compiler, 'source_id': 'different-contract'}, {**compiler, 'location': 'numeric-bound'}]}
    conditions = {'facts': {'method_scenario_dimensions': {'ROAD': {'atom_id': 'road',
        'method_semantics': {'slope_percent': [-15, 0], 'reference': 'BODY_FORWARD'}}}},
        'fact_authority': {'method_scenario_dimensions': authority, 'slope_percent': deepcopy(authority)}}
    context = {'conditions': conditions, 'evidence': {'SCENARIO.conditions': deepcopy(conditions)},
        'method_identity': {'method_source_hash': 'active'}}
    before = deepcopy(context)
    view, texts = physical_prompt_view(context)
    payload = compact_physical_request({'context': view, 'source_texts': texts},
        response_only=False, controls_only=False)
    actual = expand_physical_prompt_view(payload['context'], payload['source_texts'])
    expected = deepcopy(before)
    for c in [expected['conditions'], expected['evidence']['SCENARIO.conditions']]:
        s = c['fact_authority']['method_scenario_dimensions']['source_refs'][0]
        s.pop('excerpt'); s['excerpt_in_plan_audit'] = True
    assert actual == expected and context == before
    assert actual['conditions']['fact_authority']['slope_percent'] == authority


def test_control_task_omits_motion_choices_but_retains_actual_motion_sources_and_diagnostics():
    context = {'evidence': {'MF.description': 'braking command failed, actuator not proven failed'},
        'conditions': {'facts': {'ego_speed_kph': 7.000001, 'object_id': 'rear', 'time_role': 'POST_FAULT'}},
        'previous_plan': raw_plan(), 'correction_diagnostics': [{'reason': 'CONTROL_MISSING', 'value': .154286}],
        'model_input_status': {'models': {'unused-motion': {}}}, 'operating_atom_options': {'EGO_ACTION': []},
        'driver_configurations': ['outside_driver_seat'], 'intervention_inputs': ['emergency_braking_available']}
    view, texts = physical_prompt_view(context)
    payload = compact_physical_request({'context': view, 'source_texts': texts, 'output_contract': {}},
        response_only=False, controls_only=True)
    actual = expand_physical_prompt_view(payload['context'], payload['source_texts'])
    assert actual == {k: v for k, v in context.items() if k not in {'model_input_status', 'operating_atom_options'}}


def test_two_worlds_share_fixed_request_prefix_but_keep_different_evidence():
    from test_geometry_completion import Client
    client = Client([raw_plan(), raw_plan()])
    agent = PhysicalScenarioAgent(client)
    for key in ('a', 'b'):
        agent.propose({'plan_key': key, 'evidence': {'MF.description': key, key: 'scoped source'},
                       'conditions': {'facts': {'object_id': key}}})
    a, b = [r.user_prompt for r in client.requests]
    assert a != b and a.split('"basis_ref_ids":')[0] == b.split('"basis_ref_ids":')[0]
    assert len(a.split('"basis_ref_ids":')[0]) > 7000
    assert json.loads(a)['context']['conditions']['facts']['object_id'] == 'a'
    assert json.loads(b)['context']['conditions']['facts']['object_id'] == 'b'


def test_v3_paid_response_recovers_without_network_or_rewriting_history(tmp_path):
    import gzip
    from pathlib import Path
    from test_geometry_completion import Client
    frozen = json.loads(gzip.decompress((Path(__file__).parents[1]
        / 'fixtures/physical_request_full_resume.json.gz').read_bytes()))
    old_request = LLMRequest(**frozen['v3_request'])
    ResponseCheckpoint(tmp_path).complete(Client([raw_plan()]), old_request)
    before = {p.name: p.read_bytes() for p in tmp_path.glob('*.json')}
    no_calls = Client([])
    plan, audit = PhysicalScenarioAgent(no_calls, response_checkpoint_dir=tmp_path).propose(frozen['context'])
    assert plan.to_dict() == raw_plan() and not no_calls.requests
    assert audit['response_cache_hits'] == 1
    assert audit['responses'][0]['request']['user_prompt'] == old_request.user_prompt
    assert before == {p.name: p.read_bytes() for p in tmp_path.glob('*.json')}


def test_motion_correction_only_replaces_diagnosed_world_then_validates_full_plan(tmp_path):
    from test_geometry_completion import Client
    previous = raw_plan()
    second = deepcopy(previous['hypotheses'][0]); second['hypothesis_id'] = 'h2'
    second['geometry']['object_id'] = 'second-target'
    previous['hypotheses'].append(second)
    corrected = deepcopy(second); corrected['parameters'][1]['value'] = 3.154286
    client = Client([{'hypothesis_updates': [corrected]}])
    agent = PhysicalScenarioAgent(client, response_checkpoint_dir=tmp_path)
    context = {'plan_key': 'two-worlds', 'evidence': {'MF.description': 'fault'}, 'previous_plan': previous,
        'correction_diagnostics': [{'hypothesis_id': 'h2', 'reason': 'GEOMETRY_PARAMETER_CONFLICT'}]}
    original = deepcopy(context)
    plan, _ = agent.propose(context)
    body = json.loads(client.requests[0].user_prompt)
    view = expand_physical_prompt_view(body['context'], body['source_texts'])
    assert body['hypothesis_update_identities'] == ['h2']
    assert view['previous_plan']['hypotheses'] == [second]
    assert plan.hypotheses == (previous['hypotheses'][0], corrected)
    assert plan.uncovered_conditions == tuple(previous['uncovered_conditions'])
    assert context == original and agent.propose(context)[0] == plan and len(client.requests) == 1


@pytest.mark.parametrize('bad', ['other', 'duplicate', 'partial'])
def test_local_motion_update_cannot_touch_other_identity_or_bypass_validation(bad):
    from hara_agent.contracts.physical_scenario_plan import PhysicalScenarioPlan
    original = raw_plan(); second = deepcopy(original['hypotheses'][0]); second['hypothesis_id'] = 'h2'
    original['hypotheses'].append(second)
    row = deepcopy(second)
    if bad == 'other': row['hypothesis_id'] = 'h1'
    if bad == 'partial': row.pop('parameters')
    rows = [row, deepcopy(row)] if bad == 'duplicate' else [row]
    with pytest.raises(ValueError):
        merged = PhysicalScenarioPlan.apply_hypothesis_updates(original, {'hypothesis_updates': rows}, {'h2'})
        PhysicalScenarioPlan.from_dict(merged, evidence_ids={'MF.description'})
