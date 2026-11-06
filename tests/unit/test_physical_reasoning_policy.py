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
