from __future__ import annotations

import io
import urllib.error
from unittest.mock import patch

import pytest

from hara_agent.config import LLMConfig
from hara_agent.infrastructure.llm import LLMQuotaExceededError, LLMRequest
from hara_agent.infrastructure.llm.openai_compatible import (
    OpenAICompatibleClient,
    TransientLLMError,
)


def _http_error(detail: str) -> urllib.error.HTTPError:
    return urllib.error.HTTPError(
        "https://example.invalid/v1/chat/completions",
        429,
        "Too Many Requests",
        {},
        io.BytesIO(detail.encode("utf-8")),
    )


def test_account_quota_exhaustion_is_non_transient_and_exposes_reset_time():
    detail = (
        '{"error":{"code":"AccountQuotaExceeded","message":'
        '"You have exceeded the weekly usage quota. It will reset at '
        '2026-08-24 00:00:00 +0800 CST. We recommend waiting."}}'
    )
    with patch("urllib.request.urlopen", side_effect=_http_error(detail)):
        with pytest.raises(LLMQuotaExceededError) as raised:
            OpenAICompatibleClient._http_transport(
                "https://example.invalid/v1/chat/completions", {}, b"{}", 1.0,
            )

    assert raised.value.provider_code == "AccountQuotaExceeded"
    assert raised.value.reset_at == "2026-08-24 00:00:00 +0800 CST"


def test_normal_http_429_remains_transient():
    detail = '{"error":{"code":"RateLimitExceeded","message":"Try again later."}}'
    with patch("urllib.request.urlopen", side_effect=_http_error(detail)):
        with pytest.raises(TransientLLMError) as raised:
            OpenAICompatibleClient._http_transport(
                "https://example.invalid/v1/chat/completions", {}, b"{}", 1.0,
            )

    assert raised.value.category == "http_429"


def test_quota_exhaustion_bypasses_retries_and_later_tasks_on_same_client(tmp_path):
    from dataclasses import replace
    from hara_agent.infrastructure.llm.provider_budget import ProviderAttemptBudget
    calls = 0

    def transport(_url, _headers, _body, _timeout):
        nonlocal calls
        calls += 1
        raise LLMQuotaExceededError(
            provider_code="AccountQuotaExceeded",
            reset_at="2026-08-24 00:00:00 +0800 CST",
        )

    budget = ProviderAttemptBudget(tmp_path/'attempts.jsonl', run_id='quota-test', limit=20)
    client = OpenAICompatibleClient(LLMConfig(
        provider="openai-compatible",
        base_url="https://example.invalid/v1",
        model="fake",
        api_key="fake",
        max_retries=5,
    ), transport=transport, attempt_budget=budget)
    request = LLMRequest(
        task="assess_scenario_feasibility",
        system_prompt="system",
        user_prompt="user",
        schema_name="ScenarioFeasibilityAssessmentV2List",
        prompt_version="test",
    )

    with pytest.raises(LLMQuotaExceededError, match="reset_at=2026-08-24"):
        client.complete_json(request)

    with pytest.raises(LLMQuotaExceededError, match="reset_at=2026-08-24"):
        client.complete_json(replace(request, task="construct_post_fault_physical_scenario"))

    assert calls == 1
    assert budget.attempts == 1


def test_quota_latch_keeps_cached_responses_readable_and_allows_new_client_resume(tmp_path):
    from dataclasses import replace
    from hara_agent.infrastructure.llm.response_checkpoint import ResponseCheckpoint
    calls = 0
    exhausted = False

    def transport(*_):
        nonlocal calls
        calls += 1
        if exhausted:
            raise LLMQuotaExceededError(provider_code="AccountQuotaExceeded")
        return {"model": "fake", "choices": [{"message": {"content": '{"ok":true}'}, "finish_reason": "stop"}]}

    config = LLMConfig(provider="openai-compatible", base_url="https://example.invalid/v1",
                       model="fake", api_key="fake", max_retries=0)
    client = OpenAICompatibleClient(config, transport=transport)
    cache = ResponseCheckpoint(tmp_path)
    request = LLMRequest(task="test", system_prompt="system", user_prompt="cached",
                         schema_name="Test", prompt_version="test")
    assert cache.complete(client, request).data == {"ok": True}
    exhausted = True
    with pytest.raises(LLMQuotaExceededError):
        cache.complete(client, replace(request, user_prompt="new"))
    assert cache.complete(client, request).data == {"ok": True}
    assert calls == 2
    exhausted = False
    resumed = OpenAICompatibleClient(config, transport=transport)
    assert cache.complete(resumed, replace(request, user_prompt="new")).data == {"ok": True}
    assert calls == 3
