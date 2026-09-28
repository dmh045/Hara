from __future__ import annotations

import json
from argparse import Namespace

import pytest

from hara_agent.config import LLMConfig, RunConfig
from hara_agent.cli import _shared_provider_budget
from hara_agent.infrastructure.llm.openai_compatible import (
    OpenAICompatibleClient, TransientLLMError,
)
from hara_agent.infrastructure.llm.provider_budget import (
    ProviderAttemptBudget, ProviderAttemptBudgetExceeded,
)
from hara_agent.infrastructure.llm.protocol import LLMRequest
from hara_agent.workflow.checkpoints import CheckpointRepository
from hara_agent.workflow.graph import WorkflowGraph
from hara_agent.workflow.state import HARAState, WorkflowStage


def _request() -> LLMRequest:
    return LLMRequest(
        task="assess_scenario_feasibility", system_prompt="system",
        user_prompt="user", schema_name="ScenarioFeasibilityAssessmentList",
        prompt_version="test",
    )


def _config() -> LLMConfig:
    return LLMConfig(
        provider="openai-compatible", base_url="https://example.invalid",
        model="test-model", api_key="unused", max_retries=1,
        retry_backoff_seconds=0,
    )


def test_transport_retries_consume_durable_provider_budget(tmp_path):
    ledger = tmp_path / "sample.provider-attempts.jsonl"
    budget = ProviderAttemptBudget(ledger, run_id="sample", limit=2)
    calls = 0

    def transport(*_args):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise TransientLLMError("retry", category="throttle")
        return {
            "model": "test-model",
            "choices": [{"message": {"content": '{"assessments":[]}'},
                         "finish_reason": "stop"}],
        }

    client = OpenAICompatibleClient(_config(), transport=transport,
                                    attempt_budget=budget)
    client.complete_json(_request())
    assert calls == 2
    assert budget.attempts == 2
    events = [json.loads(line) for line in ledger.read_text().splitlines()]
    assert [event["event"] for event in events] == [
        "attempt_started", "attempt_finished", "attempt_started", "attempt_finished",
    ]
    assert events[-1]["finish_reason"] == "stop"
    assert all("unused" not in line and "system" not in line and "user" not in line
               for line in ledger.read_text().splitlines())

    resumed = ProviderAttemptBudget(ledger, run_id="sample", limit=2)
    blocked = OpenAICompatibleClient(_config(), transport=transport,
                                     attempt_budget=resumed)
    with pytest.raises(ProviderAttemptBudgetExceeded) as error:
        blocked.complete_json(_request())
    assert error.value.attempts == 2
    assert calls == 2


def test_bounded_config_requires_all_limits_and_single_worker(tmp_path):
    base = dict(
        item_path=tmp_path / "item.docx",
        template_path=None,
        method_baseline_path=tmp_path / "baseline.yaml",
        report_template_path=tmp_path / "template.xlsx",
        output_path=tmp_path / "output.xlsx",
        run_dir=tmp_path / "runtime",
    )
    for name in ("item.docx", "baseline.yaml", "template.xlsx"):
        (tmp_path / name).write_bytes(b"fixture")
    with pytest.raises(ValueError, match="Function, Malfunction"):
        RunConfig(**base, provider_attempt_limit=48, max_workers=1).validate()
    with pytest.raises(ValueError, match="max-workers 1"):
        RunConfig(
            **base, provider_attempt_limit=48, sample_function_limit=2,
            sample_malfunction_limit=2, sample_parent_scenario_limit=8,
        ).validate()
    bounded = RunConfig(
        **base, provider_attempt_limit=48, sample_function_limit=2,
        sample_malfunction_limit=2, sample_parent_scenario_limit=8,
        max_workers=1,
    )
    bounded.validate()
    assert bounded.sample_scope()["provider_attempt_limit"] == 48


def test_budget_stop_returns_last_committed_checkpoint(tmp_path):
    repository = CheckpointRepository(tmp_path)
    graph = WorkflowGraph(repository)

    def initialize(state):
        state.stage = WorkflowStage.EXTRACT
        state.record("initialized")
        return state

    def interrupted(state):
        state.item_definition["uncommitted"] = True
        raise ProviderAttemptBudgetExceeded(
            limit=2, attempts=2, ledger_path=tmp_path / "attempts.jsonl",
        )

    graph.add_node(WorkflowStage.INITIALIZE, initialize)
    graph.add_node(WorkflowStage.EXTRACT, interrupted)
    result = graph.run(HARAState(run_id="sample"))
    assert result.interrupted
    assert result.reason == "provider_attempt_budget_exhausted"
    assert result.state.stage is WorkflowStage.EXTRACT
    assert "uncommitted" not in result.state.item_definition
    assert repository.load("sample").audit_trail[-1]["attempts"] == 2


def test_child_stage_reuses_parent_attempt_ledger(tmp_path):
    path = tmp_path / "sample.provider-attempts.jsonl"
    parent = ProviderAttemptBudget(path, run_id="sample", limit=2)
    first = parent.begin(_request(), model="test-model")
    parent.finish(first, response={"choices": [{"finish_reason": "stop"}]})
    child = _shared_provider_budget(Namespace(
        provider_attempt_limit=2, provider_budget_run_id="sample", run_dir=tmp_path,
        source_run_id="sample",
    ))
    assert child.attempts == 1
    second = child.begin(_request(), model="test-model")
    child.finish(second, response={"choices": [{"finish_reason": "stop"}]})
    with pytest.raises(ProviderAttemptBudgetExceeded):
        child.begin(_request(), model="test-model")


def test_bounded_child_requires_same_parent_ledger(tmp_path):
    parent_checkpoint = tmp_path / "sample.checkpoint.json"
    parent_checkpoint.write_text(json.dumps({"audit_trail": [{
        "event": "bounded_sample_configured",
        "scope": {"provider_attempt_limit": 48},
    }]}))
    (tmp_path / "child.checkpoint.json").write_text(json.dumps({
        "audit_trail": [{
            "event": "scenario_synthesis_child_run_materialized",
            "source_run_id": "sample",
        }],
    }))
    args = Namespace(
        provider_attempt_limit=None, provider_budget_run_id=None,
        run_dir=tmp_path, source_run_id="child",
    )
    with pytest.raises(ValueError, match="must reuse"):
        _shared_provider_budget(args)
    args.provider_attempt_limit = 48
    args.provider_budget_run_id = "wrong"
    with pytest.raises(ValueError, match="differs"):
        _shared_provider_budget(args)
