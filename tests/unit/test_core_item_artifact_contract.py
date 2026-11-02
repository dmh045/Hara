from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest

from hara_agent.infrastructure.llm import LLMResponse
from hara_agent.models import ReviewStatus
from hara_agent.services.extraction import DocumentBlock, DocumentReader, ValidatedArtifactCache
from hara_agent.services.semantic.core_item_artifact_contract import (
    CORE_ITEM_ARTIFACTS_SCHEMA,
    CoreItemArtifactsContractError,
    normalize_core_item_artifacts,
)
from hara_agent.services.semantic.item_artifact_agent import (
    ItemArtifactExtractionAgent,
)
from hara_agent.services.semantic.function_source_guard import (
    FunctionSourceMismatchError,
)


ROOT = Path(__file__).resolve().parents[2]
EXPLICIT_FUNCTION_NAMES = [
    "输出制动扭矩",
    "输出驱动扭矩",
    "输出转向扭矩",
    "开启功能",
    "激活功能",
    "退出功能",
    "关闭功能",
    "输出驻车制动",
    "报警提示",
]
PSEUDO_PHASE_NAMES = [
    "AVP界面进入",
    "AVP泊入激活",
    "AVP泊入巡航",
    "AVP泊入车位",
    "AVP泊入结束",
    "AVP泊出准备",
    "AVP泊出激活",
    "AVP泊出车位",
    "AVP泊出巡航",
    "AVP泊出结束",
    "AVP功能取消",
    "AVP状态管理",
]


def _payload() -> dict:
    return {
        "item_definition": {
            "system_description": "AVP system",
            "item_boundary": "ADS ECU and vehicle interfaces",
            "operating_modes": ["Active"],
            "odd": {
                "locations": ["parking area"],
                "road_types": ["parking road"],
                "weather_conditions": ["clear"],
                "road_surfaces": ["dry"],
                "speed_range_kph": [0, 20],
            },
            "source_location": "paragraph[1]",
            "source_excerpt": "AVP system definition.",
            "confidence": 0.9,
            "status": "PENDING",
        },
        "functions": [{
            "function_id": "F01",
            "name": "Parking control",
            "output": "Vehicle motion request",
            "description": "Control vehicle motion in the parking area.",
            "preconditions": ["AVP active"],
            "triggers": ["Parking requested"],
            "odd_constraints": ["Parking area"],
            "fallback_behavior": "Stop vehicle",
            "consequences": ["Vehicle follows the planned path"],
            "source_location": "paragraph[2]",
            "source_excerpt": "Control vehicle motion in the parking area.",
            "confidence": 0.9,
            "status": "PENDING",
        }],
    }


def _current_item_artifact():
    return DocumentReader().read(ROOT / "input/ItemDef.docx")


def _payload_with_function_names(names: list[str]) -> dict:
    payload = _payload()
    payload["item_definition"].update({
        "system_description": "自主代客泊车相关项定义",
        "source_location": "paragraph[70]",
        "source_excerpt": (
            "本文档为奇瑞EH架构中智驾域的自主代客泊车在功能安全开发中的相关项定义。"
            "该文档的目的是定义和描述ADS的自主代客泊车功能，及其与环境和其它相关项的依赖性和相互影响。"
        ),
    })
    table_locations = {
        name: f"table[6].row[{index + 2}]"
        for index, name in enumerate(EXPLICIT_FUNCTION_NAMES)
    }
    payload["functions"] = [{
        "function_id": f"F-{index + 1:03d}",
        "name": name,
        "output": f"{name}对应的车辆输出",
        "description": f"执行{name}",
        "preconditions": [],
        "triggers": [],
        "odd_constraints": [],
        "fallback_behavior": None,
        "consequences": [],
        "source_location": table_locations.get(name, "paragraph[75]"),
        "source_excerpt": name,
        "confidence": 0.9,
        "status": "PENDING",
    } for index, name in enumerate(names)]
    return payload


def test_contract_mechanically_normalizes_single_string_list_fields() -> None:
    payload = _payload()
    payload["item_definition"]["operating_modes"] = "Active"
    payload["functions"][0]["consequences"] = "Vehicle follows the planned path"
    payload["functions"][0]["fallback_behavior"] = ["Stop", "Hold"]

    normalized, diagnostics = normalize_core_item_artifacts(payload)

    assert normalized["item_definition"]["operating_modes"] == ["Active"]
    assert normalized["functions"][0]["consequences"] == [
        "Vehicle follows the planned path"
    ]
    assert normalized["functions"][0]["fallback_behavior"] == "Stop; Hold"
    assert {item["path"] for item in diagnostics} >= {
        "item_definition.operating_modes",
        "functions[0].consequences",
        "functions[0].fallback_behavior",
    }


def test_contract_aggregates_unsafe_nested_types() -> None:
    payload = _payload()
    payload["functions"][0]["consequences"] = {"result": "stopped"}
    payload["functions"][0]["triggers"] = ["request", {"bad": True}]

    with pytest.raises(CoreItemArtifactsContractError) as raised:
        normalize_core_item_artifacts(payload)

    assert any("functions[0].consequences" in value for value in raised.value.errors)
    assert any("functions[0].triggers[1]" in value for value in raised.value.errors)


class _Client:
    def __init__(self, responses: list[dict]):
        self.responses = list(responses)
        self.requests = []
        self.config = SimpleNamespace(max_tokens=32768)

    def complete_json(self, request):
        self.requests.append(request)
        return LLMResponse(
            data=self.responses.pop(0),
            model="fake",
            request_id=f"request-{len(self.requests)}",
            usage={},
        )


def test_explicit_itemdef_function_table_rejects_pseudo_phase_functions() -> None:
    artifact = _current_item_artifact()
    client = _Client([_payload_with_function_names(PSEUDO_PHASE_NAMES)])

    with pytest.raises(FunctionSourceMismatchError) as raised:
        ItemArtifactExtractionAgent(client).extract(
            artifact.text, artifact.source_id, artifact.blocks,
        )

    assert raised.value.code == "FUNCTION_SOURCE_MISMATCH"
    assert raised.value.details["expected_count"] == 9
    assert raised.value.details["actual_count"] == 12
    assert raised.value.details["missing_names"] == EXPLICIT_FUNCTION_NAMES
    assert raised.value.details["unexpected_names"] == PSEUDO_PHASE_NAMES
    assert raised.value.details["authoritative_source_location"].startswith(
        "table[6].row[1:10]"
    )


def test_explicit_itemdef_function_table_accepts_exact_nine_functions() -> None:
    artifact = _current_item_artifact()
    client = _Client([_payload_with_function_names(EXPLICIT_FUNCTION_NAMES)])

    _, functions, audit = ItemArtifactExtractionAgent(client).extract(
        artifact.text, artifact.source_id, artifact.blocks,
    )

    assert [function.name for function in functions] == EXPLICIT_FUNCTION_NAMES
    assert audit["function_source_guard"] == {
        "status": "PASS",
        "expected_count": 9,
        "actual_count": 9,
        "authoritative_source_location": (
            "table[6].row[1:10] (header=table[6].row[1])"
        ),
        "membership_source_bindings": 9,
    }
    assert [function.sources[0].location for function in functions] == [
        f"table[6].row[{index}]" for index in range(2, 11)
    ]
    assert [function.sources[0].excerpt for function in functions] == (
        EXPLICIT_FUNCTION_NAMES
    )


def test_explicit_function_rows_replace_joined_provider_source_references() -> None:
    artifact = _current_item_artifact()
    payload = _payload_with_function_names(EXPLICIT_FUNCTION_NAMES)
    payload["functions"][4].update({
        "source_location": "整车功能表第5项、泊入开始章节",
        "source_excerpt": (
            "开启功能后，系统判断激活功能。当车速低于20km/h时，"
            "驾驶员双击拨杆激活AVP。"
        ),
    })
    payload["functions"][5].update({
        "source_location": "整车功能表第6项、状态转换章节",
        "source_excerpt": "系统判断退出功能。AVP任务将被取消。",
    })
    client = _Client([payload])

    _, functions, audit = ItemArtifactExtractionAgent(client).extract(
        artifact.text, artifact.source_id, artifact.blocks,
    )

    assert functions[4].sources[0].location == "table[6].row[6]"
    assert functions[4].sources[0].excerpt == "激活功能"
    assert functions[5].sources[0].location == "table[6].row[7]"
    assert functions[5].sources[0].excerpt == "退出功能"
    assert audit["source_reference_repair_count"] == 0
    assert audit["llm_call_count"] == 1


def test_explicit_itemdef_function_table_rejects_missing_function() -> None:
    artifact = _current_item_artifact()
    client = _Client([_payload_with_function_names(EXPLICIT_FUNCTION_NAMES[:-1])])

    with pytest.raises(FunctionSourceMismatchError) as raised:
        ItemArtifactExtractionAgent(client).extract(
            artifact.text, artifact.source_id, artifact.blocks,
        )

    assert raised.value.details["expected_count"] == 9
    assert raised.value.details["actual_count"] == 8
    assert raised.value.details["missing_names"] == ["报警提示"]
    assert raised.value.details["unexpected_names"] == []


def test_process_phase_text_cannot_expand_explicit_function_table() -> None:
    artifact = _current_item_artifact()
    phase_blocks = [
        DocumentBlock(
            f"P-PHASE-{index:02d}",
            "paragraph",
            f"paragraph[{1000 + index}]",
            name,
        )
        for index, name in enumerate(PSEUDO_PHASE_NAMES, start=1)
    ]
    blocks = [*artifact.blocks, *phase_blocks]
    document_text = "\n".join(block.text for block in blocks)
    client = _Client([_payload_with_function_names(EXPLICIT_FUNCTION_NAMES)])

    _, functions, audit = ItemArtifactExtractionAgent(client).extract(
        document_text, artifact.source_id, blocks,
    )

    assert [function.name for function in functions] == EXPLICIT_FUNCTION_NAMES
    assert not set(PSEUDO_PHASE_NAMES) & {function.name for function in functions}
    assert audit["function_source_guard"]["status"] == "PASS"


def test_item_agent_accepts_string_consequences_without_second_llm_call() -> None:
    payload = _payload()
    payload["functions"][0]["consequences"] = "Vehicle follows the planned path"
    document = (
        "AVP system definition.\n"
        "Control vehicle motion in the parking area."
    )
    client = _Client([payload])

    facts, functions, audit = ItemArtifactExtractionAgent(client).extract(
        document,
        "ItemDef.docx",
        [
            {"kind": "paragraph", "location": "paragraph[1]", "text": "AVP system definition."},
            {"kind": "paragraph", "location": "paragraph[2]", "text": "Control vehicle motion in the parking area."},
        ],
    )

    assert facts.status is ReviewStatus.FINALIZED
    assert functions[0].status is ReviewStatus.FINALIZED
    assert functions[0].consequences == ["Vehicle follows the planned path"]
    assert audit["schema_repair_count"] == 0
    assert audit["llm_call_count"] == 1
    assert audit["function_source_guard"] == {
        "status": "NOT_APPLICABLE",
        "authoritative_source_count": 0,
    }
    assert audit["contract_normalizations"] == [{
        "path": "functions[0].consequences",
        "from_type": "string",
        "to_type": "array",
    }]
    assert client.requests[0].response_schema == CORE_ITEM_ARTIFACTS_SCHEMA
    assert client.requests[0].prompt_version == (
        "item-artifacts-v7-explicit-function-source-binding"
    )
    assert "only membership authority for functions[]" in (
        client.requests[0].system_prompt
    )


def test_item_agent_repairs_unsafe_schema_once() -> None:
    invalid = _payload()
    invalid["functions"][0]["consequences"] = {"result": "stopped"}
    repaired = _payload()
    client = _Client([invalid, repaired])
    document = (
        "AVP system definition.\n"
        "Control vehicle motion in the parking area."
    )

    _, functions, audit = ItemArtifactExtractionAgent(client).extract(
        document, "ItemDef.docx",
    )

    assert functions[0].consequences == ["Vehicle follows the planned path"]
    assert [request.task for request in client.requests] == [
        "extract_core_item_artifacts",
        "repair_core_item_artifact_schema",
    ]
    assert audit["schema_repair_count"] == 1
    assert audit["llm_call_count"] == 2


def test_item_agent_repairs_repeated_output_from_cached_candidate() -> None:
    invalid = _payload()
    invalid["functions"][0]["output"] = invalid["functions"][0]["name"]
    client = _Client([_payload()])
    recorded = []
    document = "AVP system definition.\nControl vehicle motion in the parking area."
    _, functions, audit = ItemArtifactExtractionAgent(client).extract(
        document, "ItemDef.docx",
        candidate_response=LLMResponse(data=invalid, model="fake", request_id="first", usage={}),
        candidate_recorder=recorded.append,
    )
    assert functions[0].output == "Vehicle motion request"
    assert functions[0].function_id == "F01"
    assert audit["function_output_repair_count"] == 1
    assert audit["llm_call_count"] == 1
    assert [r.task for r in client.requests] == ["repair_core_item_function_outputs"]
    assert json.loads(client.requests[0].user_prompt)["item_definition_document"] == document
    # A subsequent resume revalidates the accepted candidate with no new request.
    resumed = _Client([])
    _, resumed_functions, resumed_audit = ItemArtifactExtractionAgent(resumed).extract(
        document, "ItemDef.docx", candidate_response=recorded[-1],
    )
    assert resumed_functions == functions
    assert resumed_audit["llm_call_count"] == 0


@pytest.mark.parametrize("change", ["membership", "name", "source", "status", "unchanged"])
def test_output_repair_is_bounded_and_preserves_other_fields(change: str) -> None:
    invalid = _payload()
    invalid["functions"][0]["output"] = invalid["functions"][0]["name"]
    repaired = _payload()
    if change == "membership":
        repaired["functions"] = []
    elif change == "unchanged":
        repaired = invalid
    else:
        key = {"name": "name", "source": "source_excerpt", "status": "status"}[change]
        repaired["functions"][0][key] = "Changed"
    client = _Client([invalid, repaired])
    recorded = []
    with pytest.raises(ValueError, match="完全相同|non-output"):
        ItemArtifactExtractionAgent(client).extract(
            "AVP system definition.\nControl vehicle motion in the parking area.",
            "ItemDef.docx", candidate_recorder=recorded.append,
        )
    assert len(client.requests) == 2
    if change != "unchanged":
        assert len(recorded) == 1  # A rejected rewrite cannot become a resume candidate.


def test_item_agent_repairs_ungrounded_sources_without_changing_content() -> None:
    invalid = _payload()
    invalid["functions"][0]["source_excerpt"] = "Control parking motion."
    repaired = _payload()
    client = _Client([invalid, repaired])
    document = (
        "AVP system definition.\n"
        "Control vehicle motion in the parking area."
    )

    _, functions, audit = ItemArtifactExtractionAgent(client).extract(
        document, "ItemDef.docx",
    )

    assert functions[0].description == "Control vehicle motion in the parking area."
    assert [request.task for request in client.requests] == [
        "extract_core_item_artifacts",
        "repair_core_item_artifact_sources",
    ]
    assert audit["source_reference_repair_count"] == 1
    assert audit["llm_call_count"] == 2


def test_item_agent_rejects_source_repair_that_changes_content() -> None:
    invalid = _payload()
    invalid["functions"][0]["source_excerpt"] = "Control parking motion."
    repaired = _payload()
    repaired["functions"][0]["description"] = "Changed engineering content."
    client = _Client([invalid, repaired])

    with pytest.raises(
        ValueError,
        match="source reference repair altered non-source Item Definition content",
    ):
        ItemArtifactExtractionAgent(client).extract(
            "AVP system definition.\nControl vehicle motion in the parking area.",
            "ItemDef.docx",
        )


def test_item_agent_revalidates_quarantined_candidate_without_llm_call() -> None:
    payload = _payload()
    payload["functions"][0]["consequences"] = "Vehicle follows the planned path"
    document = (
        "AVP system definition.\n"
        "Control vehicle motion in the parking area."
    )
    first_client = _Client([payload])
    recorded: list[LLMResponse] = []
    ItemArtifactExtractionAgent(first_client).extract(
        document, "ItemDef.docx", candidate_recorder=recorded.append,
    )
    assert len(recorded) == 1

    second_client = _Client([])
    _, functions, audit = ItemArtifactExtractionAgent(second_client).extract(
        document,
        "ItemDef.docx",
        candidate_response=recorded[0],
    )

    assert second_client.requests == []
    assert functions[0].consequences == ["Vehicle follows the planned path"]
    assert audit["candidate_cache_hit"] is True
    assert audit["llm_call_count"] == 0


def test_candidate_cache_is_isolated_from_validated_cache() -> None:
    directory = Path("runtime/agent/cache-contract-tests")
    key = f"candidate-{uuid4().hex}"
    candidate_path = directory / "candidates" / f"{key}.json"
    cache = ValidatedArtifactCache(directory, "readwrite")
    payload = {"schema_version": "raw", "data": _payload()}
    try:
        cache.save_candidate(key, payload)

        assert cache.load(key) is None
        assert cache.load_candidate_with_status(key) == (payload, "hit")
    finally:
        candidate_path.unlink(missing_ok=True)
