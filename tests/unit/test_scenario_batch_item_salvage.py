from __future__ import annotations

from collections import deque
from types import SimpleNamespace

import pytest

from hara_agent.infrastructure.llm import LLMJSONContractError, LLMResponse
from hara_agent.models import MalfunctionCandidate, ScenarioCandidate
from hara_agent.services.semantic import ScenarioFeasibilityAgent
from hara_agent.services.semantic.scenario_batching import build_scenario_batches


def _malfunction() -> MalfunctionCandidate:
    return MalfunctionCandidate(
        "MF-1", "FUN-1", "loss", "braking command lost", "no deceleration",
        "vehicle continues moving", ["command lost", "vehicle continues moving"],
    )


def _scenarios() -> list[ScenarioCandidate]:
    return [
        ScenarioCandidate(
            item, "Parking", "object ahead", "explicit object",
            {"object_position": "ahead"}, semantic_fingerprint=f"fp-{item}",
        )
        for item in ("A", "B", "C")
    ]


def test_oversized_single_scenario_is_kept_as_a_singleton_batch():
    scenario = ScenarioCandidate(
        "SCN-LONG", "Parking", "object ahead", "explicit object",
        {"object_position": "ahead"},
        analysis_instance={"assumptions": ["x" * 1000]},
    )

    batches = build_scenario_batches(
        _malfunction(), [scenario], max_chars=100, max_items=12,
    )

    assert batches == [[scenario]]


def _negative(item: str) -> dict:
    return {
        "scenario_id": item,
        "physically_feasible": False,
        "functionally_relevant": False,
        "causally_relevant": False,
        "breakpoint": "M_TO_B",
        "causal_chain": {
            "m_to_b": {
                "claim": "the required behavior is not established",
                "basis_type": "ASSUMPTION",
                "evidence_refs": [],
            },
        },
        "risk_dimension_changes": [],
        "hazardous_event": "",
        "rationale": "the causal chain stops at the first unsupported transition",
        "confidence": 0.7,
    }


def _positive(item: str) -> dict:
    result = _negative(item)
    result.update({
        "physically_feasible": True,
        "functionally_relevant": True,
        "causally_relevant": True,
        "breakpoint": "NONE",
        "causal_chain": {
            "m_to_b": {
                "claim": "lost braking command removes deceleration",
                "basis_type": "DIRECT_FACT",
                "evidence_refs": ["MF.functional_effect"],
            },
            "b_to_i": {
                "claim": "the continuing behavior interacts with the object ahead",
                "basis_type": "DIRECT_FACT",
                "evidence_refs": [f"SCN.object_position"],
            },
            "i_to_h": {
                "claim": "the interaction forms the hazardous vehicle state",
                "basis_type": "DIRECT_FACT",
                "evidence_refs": ["SCN.object_position"],
            },
        },
        "hazardous_event": "vehicle continues toward the object ahead",
        "rationale": "the supplied facts support the complete M to B to I to H chain",
    })
    return result


def _downstream_negative(item: str) -> dict:
    result = _negative(item)
    result["breakpoint"] = "B_TO_I"
    result["causal_chain"] = {
        "m_to_b": {
            "claim": "the defined malfunction removes deceleration",
            "basis_type": "DIRECT_FACT",
            "evidence_refs": ["MF.functional_effect"],
        },
        "b_to_i": {
            "claim": "the required interaction context is not established",
            "basis_type": "ASSUMPTION",
            "evidence_refs": [],
        },
    }
    return result


def _self_referential(item: str) -> dict:
    result = _positive(item)
    result["causal_chain"]["i_to_h"]["evidence_refs"] = [
        "MF.vehicle_level_hazard"
    ]
    return result


class _QueueClient:
    def __init__(self, responses):
        self.responses = deque(responses)
        self.requests = []

    def complete_json(self, request):
        self.requests.append(request)
        return LLMResponse(data=self.responses.popleft(), model="fake")


def _run(responses):
    client = _QueueClient(responses)
    assessments, audit = ScenarioFeasibilityAgent(
        client, batch_max_chars=50000, batch_max_items=12,
    ).assess(_malfunction(), _scenarios())
    return client, assessments, audit


def test_all_valid_items_use_one_root_call():
    client, assessments, audit = _run([{
        "assessments": [_negative(item) for item in ("A", "B", "C")]
    }])

    assert [item.scenario_id for item in assessments] == ["A", "B", "C"]
    assert len(client.requests) == 1
    assert audit["adaptive_split_count"] == 0
    assert audit["item_repair_calls"] == 0
    assert audit["valid_initial_count"] == 3


def test_scenario_request_uses_configured_budget_with_safety_cap():
    client = _QueueClient([{
        "assessments": [_negative(item) for item in ("A", "B", "C")]
    }])
    client.config = SimpleNamespace(max_tokens=32768)

    ScenarioFeasibilityAgent(
        client, batch_max_chars=50000, batch_max_items=12,
    ).assess(_malfunction(), _scenarios())

    assert client.requests[0].max_tokens == 16384


@pytest.mark.parametrize('configured,still_truncated', [(65536,False),(65536,True),(16384,True)])
def test_singleton_reasoning_limit_escalates_once_and_resumes_exact_cache(tmp_path,configured,still_truncated):
    from hara_agent.infrastructure.llm import LLMOutputLimitError
    from hara_agent.services.semantic.scenario_batching import ScenarioAdaptiveBatchError
    class Client:
        config = SimpleNamespace(max_tokens=configured,model='reasoning-test')
        def __init__(self):self.requests=[]
        def complete_json(self,request):
            self.requests.append(request)
            if request.max_tokens==16384 or still_truncated:
                raise LLMOutputLimitError('reasoning used allowance',{'completion_tokens':request.max_tokens})
            return LLMResponse({'assessments':[_negative('A')]},'reasoning-test')
    client=Client();agent=ScenarioFeasibilityAgent(client,response_checkpoint_dir=tmp_path)
    if still_truncated:
        with pytest.raises(ScenarioAdaptiveBatchError):agent.assess(_malfunction(),_scenarios()[:1])
        assert [r.max_tokens for r in client.requests]==([16384,65536] if configured>16384 else [16384])
    else:
        result,audit=agent.assess(_malfunction(),_scenarios()[:1])
        assert [r.max_tokens for r in client.requests]==[16384,65536]
        assert audit['actual_llm_calls']==2 and audit['output_limit_count']==1 and audit['retry_calls']==1
        restored,resume_audit=ScenarioFeasibilityAgent(client,response_checkpoint_dir=tmp_path).assess(_malfunction(),_scenarios()[:1])
        assert restored==result and len(client.requests)==2
        assert resume_audit['actual_llm_calls']==0 and resume_audit['response_checkpoint_hits']==1


def test_normal_small_causal_response_is_reused_with_large_configured_allowance(tmp_path):
    client=_QueueClient([{'assessments':[_negative('A')]}]);client.config=SimpleNamespace(max_tokens=65536)
    agent=ScenarioFeasibilityAgent(client,response_checkpoint_dir=tmp_path)
    first,_=agent.assess(_malfunction(),_scenarios()[:1])
    second,audit=agent.assess(_malfunction(),_scenarios()[:1])
    assert second==first and len(client.requests)==1
    assert client.requests[0].max_tokens==16384 and audit['actual_llm_calls']==0


def test_one_invalid_item_repairs_only_that_item():
    client, assessments, audit = _run([
        {"assessments": [_negative("A"), _self_referential("B"), _negative("C")]},
        {"assessments": [_downstream_negative("B")]},
    ])

    assert [item.scenario_id for item in assessments] == ["A", "B", "C"]
    assert len(client.requests) == 2
    assert audit["adaptive_split_count"] == 0
    assert audit["item_repair_calls"] == 1
    assert audit["valid_items_salvaged"] == 2
    assert audit["repair_success_count"] == 1
    assert audit["error_counts_by_code"]["SELF_REFERENTIAL_CAUSAL_EVIDENCE"] == 1
    repair_prompt = client.requests[1].user_prompt
    assert '"scenario_id":"B"' in repair_prompt
    assert '"scenario_id":"A"' not in repair_prompt
    assert '"scenario_id":"C"' not in repair_prompt


def test_two_invalid_items_do_not_recall_valid_middle_item():
    client, assessments, audit = _run([
        {"assessments": [_self_referential("A"), _negative("B"), _self_referential("C")]},
        {"assessments": [_negative("A")]},
        {"assessments": [_negative("C")]},
    ])

    assert [item.scenario_id for item in assessments] == ["A", "B", "C"]
    assert len(client.requests) == 3
    assert audit["adaptive_split_count"] == 0
    assert audit["item_repair_calls"] == 2
    assert audit["valid_items_salvaged"] == 1


def test_repair_failure_is_pending_and_valid_siblings_are_preserved():
    client, assessments, audit = _run([
        {"assessments": [_negative("A"), _self_referential("B"), _negative("C")]},
        {"assessments": [_self_referential("B")]},
    ])

    assert [item.scenario_id for item in assessments] == ["A", "B", "C"]
    assert len(client.requests) == 2
    assert assessments[1].status.value == "PENDING"
    assert audit["repair_failure_count"] == 1
    assert audit["repair_success_count"] == 0
    assert audit["adaptive_split_count"] == 0


def test_malformed_repair_json_is_contained_and_does_not_fail_stage():
    class Client:
        def __init__(self):
            self.calls = 0

        def complete_json(self, request):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(data={
                    "assessments": [_negative("A"), _self_referential("B"), _negative("C")],
                }, model="fake", request_id="root-1")
            raise LLMJSONContractError(
                "malformed repair JSON",
                diagnostics={"request_id": "repair-2"},
            )

    client = Client()
    assessments, audit = ScenarioFeasibilityAgent(
        client, batch_max_chars=50000, batch_max_items=12,
    ).assess(_malfunction(), _scenarios())

    assert [item.scenario_id for item in assessments] == ["A", "B", "C"]
    assert [item.status.value for item in assessments] == ["PENDING", "PENDING", "PENDING"]
    assert client.calls == 2
    assert audit["repair_failed_count"] == 1
    assert audit["repair_failure_by_code"] == {"LLM_JSON_CONTRACT_ERROR": 1}
    failure = next(
        item for item in audit["item_salvage_audit"]
        if item["outcome"] == "repair_failed"
    )
    assert failure["error_stage"] == "repair_provider_error"
    assert failure["repair_exception_class"] == "LLMJSONContractError"
    assert failure["repair_error_code"] == "LLM_JSON_CONTRACT_ERROR"
    assert failure["repair_provider_request_id"] == "repair-2"
    assert failure["repair_raw_available"] is False


def test_unexpected_repair_exception_still_propagates():
    class Client:
        def __init__(self):
            self.calls = 0

        def complete_json(self, _request):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(data={
                    "assessments": [_negative("A"), _self_referential("B"), _negative("C")],
                }, model="fake")
            raise RuntimeError("authentication failure")

    with pytest.raises(RuntimeError, match="authentication failure"):
        ScenarioFeasibilityAgent(
            Client(), batch_max_chars=50000, batch_max_items=12,
        ).assess(_malfunction(), _scenarios())


def test_unknown_and_duplicate_ids_are_audited_by_identity():
    client, assessments, audit = _run([
        {"assessments": [_negative("A"), _negative("A"), _negative("X")]},
        {"assessments": [_negative("B")]},
        {"assessments": [_negative("C")]},
    ])

    assert [item.scenario_id for item in assessments] == ["A", "B", "C"]
    assert len(client.requests) == 3
    assert audit["unknown_ids"] == ["X"]
    assert audit["duplicate_ids"] == ["A"]
    assert audit["missing_ids"] == ["B", "C"]
    assert audit["adaptive_split_count"] == 0


def test_envelope_failure_alone_uses_adaptive_split():
    class EnvelopeClient:
        def __init__(self):
            self.calls = 0

        def complete_json(self, request):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(data={"assessments": "unparseable envelope"}, model="fake")
            ids = [item for item in ("A", "B", "C") if f'"scenario_id":"{item}"' in request.user_prompt]
            return LLMResponse(data={"assessments": [_negative(item) for item in ids]}, model="fake")

    client = EnvelopeClient()
    assessments, audit = ScenarioFeasibilityAgent(
        client, batch_max_chars=50000, batch_max_items=12,
    ).assess(_malfunction(), _scenarios())

    assert [item.scenario_id for item in assessments] == ["A", "B", "C"]
    assert client.calls == 3
    assert audit["adaptive_split_count"] == 1
    assert audit["adaptive_split_reason"] == {"provider_schema": 1}
    assert audit["item_repair_calls"] == 0


def test_provider_order_is_merged_in_input_order():
    client, assessments, audit = _run([{
        "assessments": [_negative("C"), _negative("A"), _negative("B")]
    }])

    assert [item.scenario_id for item in assessments] == ["A", "B", "C"]
    assert len(client.requests) == 1
    assert audit["adaptive_split_count"] == 0
