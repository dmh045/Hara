import json

import pytest

from hara_agent.contracts import FactOrigin, FactType, RequiredFactSpec
from hara_agent.config import LLMConfig
from hara_agent.infrastructure.llm import LLMResponse
from hara_agent.infrastructure.llm.openai_compatible import OpenAICompatibleClient
from hara_agent.models import (
    FactProvenance, MalfunctionCandidate, ReviewStatus, ScenarioCandidate, SourceRef,
)
from hara_agent.services.semantic import ScenarioRiskFactAgent


def _spec(fact_type, origin, constraints=(), unit=""):
    return RequiredFactSpec(
        fact_type=fact_type,
        required_for=("severity",),
        unit=unit,
        constraints=constraints,
        condition="when referenced",
        origin=origin,
        source_rule_ids=(),
        source_refs=(),
    )


@pytest.mark.parametrize("primary_valid,repair_valid", [
    (False, True), (False, False), (True, False),
])
def test_schema_envelope_error_uses_existing_bounded_coverage_repair(
    primary_valid, repair_valid,
):
    calls = []
    invalid = {"fact_type": "COLLISION_TYPE", "malfunction_id": "MF-1"}
    found = {"results": [{
        "malfunction_id": "MF-1", "scenario_id": "SCN-1",
        "fact_type": "COLLISION_TYPE", "status": "FOUND",
        "value": "FRONTAL", "unit": "", "evidence_ids": ["E0001"],
    }]}
    contents = [
        {"results": []} if primary_valid else invalid,
        found if repair_valid else invalid,
    ]

    def transport(_url, _headers, body, _timeout):
        calls.append(json.loads(body))
        return {
            "id": f"response-{len(calls)}", "model": "fake",
            "choices": [{"message": {"content": json.dumps(contents[len(calls) - 1])},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }

    client = OpenAICompatibleClient(LLMConfig(
        provider="openai-compatible", base_url="https://example.invalid/v1",
        model="fake", api_key="fake", max_retries=0,
    ), transport=transport)
    source = SourceRef("item_definition", "item.docx", "p1", "frontal collision")
    facts, audit = ScenarioRiskFactAgent(client).interpret(
        [{
            "malfunction_id": "MF-1", "scenario_id": "SCN-1",
            "physically_feasible": True, "functionally_relevant": True,
            "causally_relevant": True, "risk_dimensions_changed": ["severity"],
            "status": "FINALIZED",
        }],
        [ScenarioCandidate("SCN-1", "road", "collision", "detail", sources=[source])],
        [MalfunctionCandidate(
            "MF-1", "FUN-1", "loss", "lost", "absent", "collision",
            ["lost", "collision"], sources=[source],
        )],
        [_spec(FactType.COLLISION_TYPE, FactOrigin.SCENARIO_FACT,
               ("COLLISION_TYPE IN ('FRONTAL',)",))],
    )

    assert len(calls) == audit["llm_call_count"] == 2
    assert "single bounded coverage repair" in calls[1]["messages"][1]["content"]
    assert audit["coverage_repair_count"] == 1
    assert audit["validation_failure_count"] == int(not primary_valid) + int(not repair_valid)
    assert audit["unresolved_contract_count"] == int(not repair_valid)
    assert audit["resolution_counts"]["ABSENT_IN_EVIDENCE"] == 0
    if repair_valid:
        assert len(facts) == 1
        assert facts[0].value == "FRONTAL"
        assert facts[0].source_refs == [source]
    else:
        assert facts == []
        assert audit["resolution_counts"]["EXTRACTION_FAILED"] == 1


def test_scenario_risk_facts_are_validated_neutral_and_evidence_grounded():
    class Client:
        def complete_json(self, request):
            assert request.task == "interpret_scenario_risk_facts"
            assert "AVOIDABILITY_PERCENT" not in request.user_prompt
            return LLMResponse(data={"results": [{
                "malfunction_id": "MF-1",
                "scenario_id": "SCN-1",
                "fact_type": "COLLISION_TYPE",
                "status": "FOUND",
                "value": "FRONTAL",
                "unit": "",
                "evidence_ids": ["E0001"],
            }]}, model="fake")

    source = SourceRef("item_definition", "item.docx", "p1", "frontal collision")
    candidate = ScenarioCandidate(
        "SCN-1", "road", "frontal collision", "vehicle approaches target",
        sources=[source],
    )
    malfunction = MalfunctionCandidate(
        "MF-1", "FUN-1", "loss", "output lost", "control absent",
        "vehicle continues toward target", ["output lost", "control absent"],
        sources=[source],
    )
    facts, audit = ScenarioRiskFactAgent(Client()).interpret(
        [{
            "malfunction_id": "MF-1", "scenario_id": "SCN-1",
            "physically_feasible": True, "functionally_relevant": True,
            "causally_relevant": True, "risk_dimensions_changed": ["severity"],
            "hazardous_event": "frontal collision", "potential_harm": "injury",
            "status": "FINALIZED",
        }],
        [candidate],
        [malfunction],
        [
            _spec(FactType.COLLISION_TYPE, FactOrigin.SCENARIO_FACT,
                  ("COLLISION_TYPE IN ('FRONTAL',)",)),
            _spec(FactType.AVOIDABILITY_PERCENT, FactOrigin.HUMAN_EVIDENCE, unit="%"),
        ],
    )

    assert len(facts) == 1
    fact = facts[0]
    assert fact.parameter == "COLLISION_TYPE"
    assert fact.context == {"malfunction_id": "MF-1", "scenario_id": "SCN-1"}
    assert fact.provenance is FactProvenance.DERIVED
    assert fact.approval is ReviewStatus.FINALIZED
    assert fact.source_refs == [source]
    assert audit["found_fact_count"] == 1


def test_scenario_risk_fact_agent_keeps_invalid_value_unresolved_after_repair():
    class Client:
        def complete_json(self, request):
            return LLMResponse(data={"results": [{
                "malfunction_id": "MF-1", "scenario_id": "SCN-1",
                "fact_type": "COLLISION_TYPE", "status": "FOUND",
                "value": "UNKNOWN_KIND", "unit": "", "evidence_ids": ["E0001"],
            }]}, model="fake")

    source = SourceRef("item_definition", "item.docx", "p1", "collision")
    candidate = ScenarioCandidate("SCN-1", "road", "collision", "detail", sources=[source])
    malfunction = MalfunctionCandidate(
        "MF-1", "FUN-1", "loss", "lost", "absent", "collision",
        ["lost", "collision"], sources=[source],
    )
    facts, audit = ScenarioRiskFactAgent(Client()).interpret(
        [{
            "malfunction_id": "MF-1", "scenario_id": "SCN-1",
            "physically_feasible": True, "functionally_relevant": True,
            "causally_relevant": True, "risk_dimensions_changed": ["severity"],
            "status": "FINALIZED",
        }],
        [candidate], [malfunction],
        [_spec(FactType.COLLISION_TYPE, FactOrigin.SCENARIO_FACT,
               ("COLLISION_TYPE IN ('FRONTAL',)",))],
    )

    assert facts == []
    assert audit["coverage_repair_count"] == 1
    assert audit["unresolved_contract_count"] == 1
    assert audit["validation_failure_count"] == 2


def test_scenario_risk_fact_agent_repairs_only_missing_contract_results():
    class Client:
        def __init__(self):
            self.calls = 0

        def complete_json(self, request):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(data={"results": []}, model="fake")
            assert request.metadata["coverage_repair"] is True
            return LLMResponse(data={"results": [{
                "malfunction_id": "MF-1",
                "scenario_id": "SCN-1",
                "fact_type": "COLLISION_TYPE",
                "status": "FOUND",
                "value": "FRONTAL",
                "unit": "",
                "evidence_ids": ["E0001"],
            }]}, model="fake")

    source = SourceRef("item_definition", "item.docx", "p1", "frontal collision")
    candidate = ScenarioCandidate("SCN-1", "road", "collision", "detail", sources=[source])
    malfunction = MalfunctionCandidate(
        "MF-1", "FUN-1", "loss", "lost", "absent", "collision",
        ["lost", "collision"], sources=[source],
    )
    client = Client()
    facts, audit = ScenarioRiskFactAgent(client).interpret(
        [{
            "malfunction_id": "MF-1", "scenario_id": "SCN-1",
            "physically_feasible": True, "functionally_relevant": True,
            "causally_relevant": True, "risk_dimensions_changed": ["severity"],
            "status": "FINALIZED",
        }],
        [candidate], [malfunction],
        [_spec(FactType.COLLISION_TYPE, FactOrigin.SCENARIO_FACT,
               ("COLLISION_TYPE IN ('FRONTAL',)",))],
    )

    assert len(facts) == 1
    assert audit["coverage_repair_count"] == 1
    assert audit["llm_call_count"] == 2


def test_scenario_risk_fact_agent_repairs_result_with_missing_status():
    class Client:
        def __init__(self):
            self.calls = 0

        def complete_json(self, request):
            self.calls += 1
            result = {
                "malfunction_id": "MF-1",
                "scenario_id": "SCN-1",
                "fact_type": "COLLISION_TYPE",
            }
            if self.calls == 1:
                return LLMResponse(data={"results": [result]}, model="fake")
            result.update({
                "status": "FOUND", "value": "FRONTAL", "unit": "",
                "evidence_ids": ["E0001"],
            })
            return LLMResponse(data={"results": [result]}, model="fake")

    source = SourceRef("item_definition", "item.docx", "p1", "frontal collision")
    candidate = ScenarioCandidate(
        "SCN-1", "road", "collision", "detail", sources=[source]
    )
    malfunction = MalfunctionCandidate(
        "MF-1", "FUN-1", "loss", "lost", "absent", "collision",
        ["lost", "collision"], sources=[source],
    )
    client = Client()

    facts, audit = ScenarioRiskFactAgent(client).interpret(
        [{
            "malfunction_id": "MF-1", "scenario_id": "SCN-1",
            "physically_feasible": True, "functionally_relevant": True,
            "causally_relevant": True, "status": "FINALIZED",
        }],
        [candidate], [malfunction],
        [_spec(FactType.COLLISION_TYPE, FactOrigin.SCENARIO_FACT,
               ("COLLISION_TYPE IN ('FRONTAL',)",))],
    )

    assert client.calls == 2
    assert len(facts) == 1
    assert audit["coverage_repair_count"] == 1
    assert audit["unresolved_contract_count"] == 0
    assert audit["validation_failure_count"] == 1


def test_scenario_risk_fact_completed_batch_is_reused_from_cache():
    class Client:
        def __init__(self):
            self.calls = 0

        def complete_json(self, request):
            self.calls += 1
            return LLMResponse(data={"results": [{
                "malfunction_id": "MF-1", "scenario_id": "SCN-1",
                "fact_type": "COLLISION_TYPE", "status": "NOT_FOUND",
            }]}, model="fake")

    source = SourceRef("item_definition", "item.docx", "p1", "collision")
    candidate = ScenarioCandidate(
        "SCN-1", "road", "collision", "detail", sources=[source]
    )
    malfunction = MalfunctionCandidate(
        "MF-1", "FUN-1", "loss", "lost", "absent", "collision",
        ["lost", "collision"], sources=[source],
    )
    assessments = [{
        "malfunction_id": "MF-1", "scenario_id": "SCN-1",
        "physically_feasible": True, "functionally_relevant": True,
        "causally_relevant": True, "status": "FINALIZED",
    }]
    specs = [_spec(
        FactType.COLLISION_TYPE, FactOrigin.SCENARIO_FACT,
        ("COLLISION_TYPE IN ('FRONTAL',)",),
    )]
    cache = {}
    client = Client()
    agent = ScenarioRiskFactAgent(client)

    agent.interpret(
        assessments, [candidate], [malfunction], specs, batch_cache=cache,
    )
    _, second_audit = agent.interpret(
        assessments, [candidate], [malfunction], specs, batch_cache=cache,
    )

    assert client.calls == 1
    assert len(cache) == 1
    assert second_audit["batch_cache_hit_count"] == 1


def test_scenario_risk_fact_agent_batches_by_expected_result_count():
    class Client:
        def __init__(self):
            self.calls = 0

        def complete_json(self, request):
            self.calls += 1
            fact_request = request.user_prompt.split("fact_requests=", 1)[1]
            scenario_id = "SCN-1" if '"scenario_id":"SCN-1"' in fact_request else "SCN-2"
            return LLMResponse(data={"results": [{
                "malfunction_id": "MF-1",
                "scenario_id": scenario_id,
                "fact_type": "COLLISION_TYPE",
                "status": "NOT_FOUND",
            }]}, model="fake")

    source = SourceRef("item_definition", "item.docx", "p1", "collision")
    candidates = [
        ScenarioCandidate(scenario_id, "road", "collision", "detail", sources=[source])
        for scenario_id in ("SCN-1", "SCN-2")
    ]
    malfunction = MalfunctionCandidate(
        "MF-1", "FUN-1", "loss", "lost", "absent", "collision",
        ["lost", "collision"], sources=[source],
    )
    assessments = [{
        "malfunction_id": "MF-1", "scenario_id": scenario_id,
        "physically_feasible": True, "functionally_relevant": True,
        "causally_relevant": True, "risk_dimensions_changed": ["severity"],
        "status": "FINALIZED",
    } for scenario_id in ("SCN-1", "SCN-2")]
    client = Client()
    facts, audit = ScenarioRiskFactAgent(client, batch_max_results=1).interpret(
        assessments, candidates, [malfunction],
        [_spec(FactType.COLLISION_TYPE, FactOrigin.SCENARIO_FACT,
               ("COLLISION_TYPE IN ('FRONTAL',)",))],
    )

    assert facts == []
    assert client.calls == 2
    assert audit["batch_count"] == 2
    assert audit["llm_call_count"] == 2


def test_batch_fact_cannot_cite_evidence_owned_only_by_another_pair():
    class Client:
        def __init__(self):
            self.calls = 0

        def complete_json(self, request):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse(data={"results": [
                    {
                        "malfunction_id": "MF-1", "scenario_id": "SCN-1",
                        "fact_type": "COLLISION_TYPE", "status": "FOUND",
                        "value": "FRONTAL", "unit": "",
                        # E0002 belongs only to SCN-2 in this batch.
                        "evidence_ids": ["E0002"],
                    },
                    {
                        "malfunction_id": "MF-2", "scenario_id": "SCN-2",
                        "fact_type": "COLLISION_TYPE", "status": "NOT_FOUND",
                    },
                ]}, model="fake")
            return LLMResponse(data={"results": [{
                "malfunction_id": "MF-1", "scenario_id": "SCN-1",
                "fact_type": "COLLISION_TYPE", "status": "NOT_FOUND",
            }]}, model="fake")

    sources = [
        SourceRef("item_definition", "item.docx", "p1", "front object"),
        SourceRef("item_definition", "item.docx", "p2", "frontal collision"),
    ]
    candidates = [
        ScenarioCandidate("SCN-1", "road", "first", "detail", sources=[sources[0]]),
        ScenarioCandidate("SCN-2", "road", "second", "detail", sources=[sources[1]]),
    ]
    malfunctions = [
        MalfunctionCandidate(
            "MF-1", "FUN-1", "loss", "lost 1", "absent 1", "hazard 1",
            ["lost 1", "hazard 1"], sources=[sources[0]],
        ),
        MalfunctionCandidate(
            "MF-2", "FUN-2", "loss", "lost 2", "absent 2", "hazard 2",
            ["lost 2", "hazard 2"], sources=[sources[1]],
        ),
    ]
    assessments = [{
        "malfunction_id": malfunction_id, "scenario_id": scenario_id,
        "physically_feasible": True, "functionally_relevant": True,
        "causally_relevant": True, "status": "FINALIZED",
    } for malfunction_id, scenario_id in (("MF-1", "SCN-1"), ("MF-2", "SCN-2"))]

    facts, audit = ScenarioRiskFactAgent(Client()).interpret(
        assessments, candidates, malfunctions,
        [_spec(FactType.COLLISION_TYPE, FactOrigin.SCENARIO_FACT,
               ("COLLISION_TYPE IN ('FRONTAL',)",))],
    )

    assert facts == []
    assert audit["validation_failure_count"] == 1
    assert audit["coverage_repair_count"] == 1
    assert audit["resolution_counts"]["ABSENT_IN_EVIDENCE"] == 2


def test_dependent_risk_facts_skip_irrelevant_road_user_and_unresolved_speed():
    class Client:
        def __init__(self):
            self.calls = 0

        def complete_json(self, request):
            self.calls += 1
            assert "ROAD_USER_TYPE" not in request.user_prompt
            assert "SPEED_UNSPECIFIED" not in request.user_prompt
            return LLMResponse(data={"results": [{
                "malfunction_id": "MF-1", "scenario_id": "SCN-1",
                "fact_type": "COLLISION_TYPE", "status": "FOUND",
                "value": "FRONTAL", "unit": "", "evidence_ids": ["E0001"],
            }]}, model="fake")

    source = SourceRef("item_definition", "item.docx", "p1", "frontal collision")
    candidate = ScenarioCandidate("SCN-1", "road", "collision", "detail", sources=[source])
    malfunction = MalfunctionCandidate(
        "MF-1", "FUN-1", "loss", "lost", "absent", "collision",
        ["lost", "collision"], sources=[source],
    )
    specs = [
        _spec(FactType.COLLISION_TYPE, FactOrigin.SCENARIO_FACT,
              ("COLLISION_TYPE IN ('FRONTAL','VEHICLE_TO_ROAD_USER')",)),
        _spec(FactType.ROAD_USER_TYPE, FactOrigin.SCENARIO_FACT,
              ("ROAD_USER_TYPE IN ('PEDESTRIAN','CYCLIST')",)),
        _spec(FactType.SPEED_UNSPECIFIED, FactOrigin.SCENARIO_FACT, unit="km/h"),
    ]

    _, audit = ScenarioRiskFactAgent(Client()).interpret(
        [{
            "malfunction_id": "MF-1", "scenario_id": "SCN-1",
            "physically_feasible": True, "functionally_relevant": True,
            "causally_relevant": True, "status": "FINALIZED",
        }],
        [candidate], [malfunction], specs,
    )

    assert audit["llm_call_count"] == 1
    assert audit["candidate_fact_pair_count"] == 3
    assert audit["llm_requested_fact_pair_count"] == 1
    assert audit["resolution_counts"] == {
        "FOUND": 1,
        "NOT_APPLICABLE": 1,
        "ABSENT_IN_EVIDENCE": 0,
        "UNRESOLVED_DEPENDENCY": 0,
        "UNRESOLVED_METHOD_SEMANTICS": 1,
        "EXTRACTION_FAILED": 0,
    }


def test_road_user_type_is_requested_only_after_vru_collision_is_grounded():
    class Client:
        def __init__(self):
            self.calls = 0

        def complete_json(self, request):
            self.calls += 1
            fact_type = "COLLISION_TYPE" if self.calls == 1 else "ROAD_USER_TYPE"
            value = "VEHICLE_TO_ROAD_USER" if self.calls == 1 else "PEDESTRIAN"
            return LLMResponse(data={"results": [{
                "malfunction_id": "MF-1", "scenario_id": "SCN-1",
                "fact_type": fact_type, "status": "FOUND",
                "value": value, "unit": "", "evidence_ids": ["E0001"],
            }]}, model="fake")

    source = SourceRef("item_definition", "item.docx", "p1", "pedestrian collision")
    candidate = ScenarioCandidate("SCN-1", "road", "collision", "detail", sources=[source])
    malfunction = MalfunctionCandidate(
        "MF-1", "FUN-1", "loss", "lost", "absent", "collision",
        ["lost", "collision"], sources=[source],
    )
    client = Client()
    facts, audit = ScenarioRiskFactAgent(client).interpret(
        [{
            "malfunction_id": "MF-1", "scenario_id": "SCN-1",
            "physically_feasible": True, "functionally_relevant": True,
            "causally_relevant": True, "status": "FINALIZED",
        }], [candidate], [malfunction], [
            _spec(FactType.COLLISION_TYPE, FactOrigin.SCENARIO_FACT,
                  ("COLLISION_TYPE IN ('VEHICLE_TO_ROAD_USER',)",)),
            _spec(FactType.ROAD_USER_TYPE, FactOrigin.SCENARIO_FACT,
                  ("ROAD_USER_TYPE IN ('PEDESTRIAN','CYCLIST')",)),
        ],
    )

    assert client.calls == 2
    assert {item.parameter for item in facts} == {"COLLISION_TYPE", "ROAD_USER_TYPE"}
    assert audit["resolution_counts"]["FOUND"] == 2


def test_scenario_risk_fact_task_uses_scenario_thinking_policy():
    client = OpenAICompatibleClient(LLMConfig(
        provider="volcengine-agent-plan",
        base_url="https://example.invalid/v3",
        model="fake",
        api_key="fake",
        scenario_thinking="disabled",
    ))

    assert client._resolve_thinking_mode("interpret_scenario_risk_facts") == "disabled"
