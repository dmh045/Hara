import json

import pytest

from hara_agent.config import LLMConfig
from hara_agent.infrastructure.llm import LLMJSONContractError, LLMRequest
from hara_agent.infrastructure.llm.openai_compatible import (
    LLMSchemaContractError,
    OpenAICompatibleClient,
)


def _candidate() -> dict:
    return {
        "malfunction_id": "M1",
        "guideword": "loss",
        "description": "Function output is lost.",
        "functional_effect": "The requested function is unavailable.",
        "vehicle_level_hazard": "The vehicle can no longer perform the requested maneuver.",
        "causal_chain": ["Output lost", "Maneuver unavailable"],
        "confidence": 0.8,
        "status": "PENDING",
    }


def _request() -> LLMRequest:
    return LLMRequest(
        task="derive_malfunctions_and_hazards",
        system_prompt="system",
        user_prompt="user",
        schema_name="MalfunctionHazardCandidateList",
        prompt_version="test",
        metadata={"function_id": "F009"},
        max_tokens=1024,
    )


def _schema_request(task: str, schema_name: str) -> LLMRequest:
    return LLMRequest(
        task=task,
        system_prompt="system",
        user_prompt="user",
        schema_name=schema_name,
        prompt_version="test",
        max_tokens=1024,
    )


def _client(content: object, calls: list[bytes]) -> OpenAICompatibleClient:
    def transport(_url, _headers, body, _timeout):
        calls.append(body)
        return {
            "id": "response-1",
            "model": "fake",
            "choices": [{
                "message": {"content": json.dumps(content)},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    return OpenAICompatibleClient(LLMConfig(
        provider="openai-compatible",
        base_url="https://example.invalid/v1",
        model="fake",
        api_key="fake",
        max_retries=0,
    ), transport=transport)


@pytest.mark.parametrize("content", [_candidate(), [_candidate()]])
def test_complete_single_candidate_or_direct_array_is_wrapped_without_second_llm_call(content):
    calls: list[bytes] = []

    response = _client(content, calls).complete_json(_request())

    assert response.data == {"candidates": [_candidate()]}
    assert response.usage["schema_repair_count"] == 1
    assert len(calls) == 1


def test_incomplete_single_candidate_is_not_wrapped():
    calls: list[bytes] = []
    incomplete = _candidate()
    incomplete.pop("causal_chain")

    with pytest.raises(LLMSchemaContractError, match="candidates"):
        _client(incomplete, calls).complete_json(_request())

    assert len(calls) == 1


@pytest.mark.parametrize("as_array", [False, True])
def test_scenario_single_item_or_direct_array_is_wrapped(as_array):
    calls: list[bytes] = []
    item = {
        "scenario_id": "SC1",
        "physically_feasible": True,
        "functionally_relevant": True,
        "causally_relevant": False,
        "rationale": "No causal path.",
        "confidence": 0.7,
    }
    content = [item] if as_array else item

    response = _client(content, calls).complete_json(_schema_request(
        "assess_scenario_feasibility", "ScenarioFeasibilityAssessmentList",
    ))

    assert response.data == {"assessments": [item]}
    assert response.usage["schema_repair_count"] == 1
    assert len(calls) == 1


def test_scenario_risk_fact_single_item_is_wrapped():
    calls: list[bytes] = []
    item = {
        "malfunction_id": "M1",
        "scenario_id": "SC1",
        "fact_type": "ego_speed",
        "status": "NOT_FOUND",
    }

    response = _client(item, calls).complete_json(_schema_request(
        "interpret_scenario_risk_facts", "ScenarioRiskFacts",
    ))

    assert response.data == {"results": [item]}
    assert response.usage["schema_repair_count"] == 1
    assert len(calls) == 1


def test_core_item_artifacts_rejects_wrong_top_level_field_types():
    calls: list[bytes] = []

    with pytest.raises(LLMSchemaContractError, match="functions"):
        _client({
            "item_definition": {},
            "functions": "not-an-array",
        }, calls).complete_json(_schema_request(
            "extract_core_item_artifacts", "CoreItemArtifacts",
        ))

    assert len(calls) == 1


def _sequenced_scenario_client(contents: list[str], calls: list[bytes]):
    def transport(_url, _headers, body, _timeout):
        calls.append(body)
        content = contents[len(calls) - 1]
        return {
            "id": f"response-{len(calls)}",
            "model": "fake",
            "choices": [{
                "message": {"content": content},
                "finish_reason": "stop",
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20},
        }

    return OpenAICompatibleClient(LLMConfig(
        provider="openai-compatible",
        base_url="https://example.invalid/v1",
        model="fake",
        api_key="fake",
        max_retries=0,
    ), transport=transport)


def test_scenario_adapter_recovers_first_malformed_json_once():
    calls: list[bytes] = []
    client = _sequenced_scenario_client([
        '{"assessments": [',
        '{"assessments": []}',
    ], calls)

    response = client.complete_json(_schema_request(
        "assess_scenario_feasibility", "ScenarioFeasibilityAssessmentList",
    ))

    assert response.data == {"assessments": []}
    assert len(calls) == 2
    assert response.usage["format_retry_calls"] == 1
    assert response.usage["format_retry_successes"] == 1


def test_scenario_adapter_does_not_attempt_third_json_recovery_call():
    calls: list[bytes] = []
    client = _sequenced_scenario_client([
        '{"assessments": [',
        '{"assessments": [',
    ], calls)

    with pytest.raises(LLMJSONContractError):
        client.complete_json(_schema_request(
            "assess_scenario_feasibility", "ScenarioFeasibilityAssessmentList",
        ))

    assert len(calls) == 2


@pytest.mark.parametrize('content,key', [
    ('{"hypotheses":[{"hypothesis_id":"h1"}],"hypotheses":[],"uncovered_conditions":[]}', 'hypotheses'),
    ('{"hypotheses":[{"parameters":[{"value":2,"value":0}]}]}', 'value'),
    ('{"assessments":[{"causally_relevant":true,"causally_relevant":false}]}', 'causally_relevant'),
])
def test_duplicate_keys_cannot_discard_a_plan_or_select_a_different_value(content, key):
    calls = []
    client = _sequenced_scenario_client([content], calls)
    with pytest.raises(LLMJSONContractError) as error:
        client.complete_json(_schema_request('construct_post_fault_physical_scenario', 'PhysicalScenarioPlan'))
    assert error.value.diagnostics['duplicate_keys'] == [key]
    assert len(calls) == 1  # Ambiguous content is not a syntax-only repair.


def test_same_key_in_distinct_objects_is_not_a_duplicate():
    calls = []
    content = '{"hypotheses":[{"parameters":[{"value":2},{"value":0}]}]}'
    response = _sequenced_scenario_client([content], calls).complete_json(
        _schema_request('construct_post_fault_physical_scenario', 'PhysicalScenarioPlan'))
    assert response.data == json.loads(content) and len(calls) == 1
