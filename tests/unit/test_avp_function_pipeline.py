"""Migrate P001 expansion invariants through the ordinary production stages."""
from copy import deepcopy
from dataclasses import asdict
import json
from types import SimpleNamespace

import pytest

from hara_agent.infrastructure.llm import LLMResponse
from hara_agent.method_sources import MethodSourceResolver
from hara_agent.models import GuidewordAssessment, GuidewordDisposition, ReviewStatus, SourceRef
from hara_agent.services.semantic import (
    GuidewordApplicabilityAgent, ItemArtifactExtractionAgent, MalfunctionHazardAgent,
)
from hara_agent.workflow.nodes.hazop import assess_guidewords
from hara_agent.workflow.nodes.malfunctions import derive_malfunctions
from hara_agent.workflow.state import HARAState
from test_core_item_artifact_contract import (
    EXPLICIT_FUNCTION_NAMES, _Client, _avp_semantic_payload, _current_item_artifact,
)


@pytest.mark.parametrize("max_workers", [1, 4])
def test_avp_function_semantics_survive_bounded_guideword_and_malfunction_derivation(max_workers):
    artifact = _current_item_artifact()
    facts, functions, _ = ItemArtifactExtractionAgent(
        _Client([_avp_semantic_payload(artifact)]),
    ).extract(artifact.text, artifact.source_id, artifact.blocks)
    before = deepcopy(functions)
    method = MethodSourceResolver().resolve(
        template_path=None,
        baseline_manifest_path="method_assets/fusa_baseline_v1/manifest.yaml",
        report_template_path="references/HARA_Template_AI_20260327.xlsx",
    ).method
    guidewords = list(method.guidewords.guidewords)
    assert len(guidewords) >= 3  # Exercise retained, not-applicable and no-hazard identities.

    class GuidewordClient:
        config = SimpleNamespace(max_tokens=8192)

        def complete_json(self, request):
            context, _ = json.JSONDecoder().raw_decode(request.user_prompt.split("FunctionContext=", 1)[1])
            function = next(item for item in functions if item.function_id == context["function_id"])
            for field in ("output", "description", "preconditions", "triggers",
                          "odd_constraints", "fallback_behavior", "consequences"):
                assert context[field] == getattr(function, field)
            return LLMResponse(data={"assessments": [
                {"guideword": word.guideword_id, "applicable": index != 1,
                 "disposition": ("DOWNSTREAM_CANDIDATE" if index == 0 else
                                 "NOT_APPLICABLE" if index == 1 else "NO_CREDIBLE_HAZARD"),
                 "rationale": "Offline transport fixture disposition", "confidence": 0.9}
                for index, word in enumerate(guidewords)
            ]}, model="offline-scripted")

    class MalfunctionClient:
        config = SimpleNamespace(max_tokens=8192)

        def complete_json(self, request):
            context, _ = json.JSONDecoder().raw_decode(request.user_prompt.split("FunctionContext=", 1)[1])
            return LLMResponse(data={"candidates": [{
                "malfunction_id": "MODEL-LOCAL", "guideword": guidewords[0].name,
                "description": context["output"] + "输出缺失",
                "functional_effect": "当前功能输出不可用",
                "vehicle_level_hazard": "车辆预期运行受到影响",
                "causal_chain": ["功能输出缺失", "车辆行为发生偏差"],
                "confidence": 0.9, "status": "PENDING",
                "derivation": {"function_refs": ["output"], "subject": context["output"],
                    "quantity": "输出", "basis": "Scripted source-output deviation", "assumptions": []},
            }]}, model="offline-scripted")

    state = HARAState(run_id="avp-production-stages", functions=[asdict(item) for item in functions],
                      item_definition={"typed": asdict(facts)})
    assess_guidewords(state, GuidewordApplicabilityAgent(GuidewordClient()), functions,
                     guidewords, max_workers=max_workers)
    expected_pairs = {(function.function_id, word.guideword_id)
                      for function in functions for word in guidewords}
    assert len(state.guideword_assessments) == len(expected_pairs)
    assert {(item["function_id"], item["guideword_id"])
            for item in state.guideword_assessments} == expected_pairs
    assessments = []
    for value in state.guideword_assessments:
        payload = dict(value)
        payload["sources"] = [SourceRef(**item) for item in payload["sources"]]
        payload["status"] = ReviewStatus(payload["status"])
        assessments.append(GuidewordAssessment(**payload))
    derive_malfunctions(state, MalfunctionHazardAgent(MalfunctionClient()), functions,
                        assessments, max_workers=max_workers)
    downstream = {(item.function_id, item.guideword_id) for item in assessments
                  if item.disposition is GuidewordDisposition.DOWNSTREAM_CANDIDATE}
    assert len(state.malfunctions) == len(downstream) == len(functions)
    assert {(item["function_id"], item["guideword_id"]) for item in state.malfunctions} == downstream
    assert len({item["malfunction_id"] for item in state.malfunctions}) == len(state.malfunctions)
    assert all(item["model_local_id"] == "MODEL-LOCAL" for item in state.malfunctions)
    assert all(item["malfunction_id"] == f"MF-{item['function_id']}-001" for item in state.malfunctions)
    assert all(item["status"] == "PENDING" and item["sources"] for item in state.malfunctions)
    assert functions == before
    assert state.functions == [asdict(item) for item in functions]
    assert [item.name for item in functions] == EXPLICIT_FUNCTION_NAMES
    summary = next(item for item in reversed(state.audit_trail)
                   if item["event"] == "malfunction_derivation_summary")
    assert summary["actual_llm_calls"] == len(functions)
    assert summary["coverage_repair_count"] == 0
