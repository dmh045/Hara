import pytest

from hara_agent.infrastructure.llm import LLMResponse
from hara_agent.contracts import FailureModeTaxonomyValue
from hara_agent.models import (
    FunctionDefinition, GuidewordAssessment, GuidewordDisposition,
    ReviewStatus, SourceRef,
)
from hara_agent.services.semantic import GuidewordApplicabilityAgent, MalfunctionHazardAgent
from hara_agent.workflow.nodes.malfunctions import derive_malfunctions
from hara_agent.workflow.state import HARAState


def _candidate(identifier: str, guideword: str) -> dict:
    return {
        "malfunction_id": identifier,
        "guideword": guideword,
        "description": f"function evidence output {guideword} deviation",
        "functional_effect": f"{guideword} effect",
        "vehicle_level_hazard": f"{guideword} hazardous vehicle state",
        "causal_chain": [f"{guideword} deviation", f"{guideword} vehicle effect"],
        "confidence": 0.8,
        "status": "PENDING",
        "derivation": {"function_refs": ["source[0]"], "subject": "function evidence",
            "quantity": "output", "basis": "Deviation of the source-defined functional output", "assumptions": []},
    }


def _inputs():
    source = SourceRef("item_definition", "item.docx", "p1", "function evidence")
    function = FunctionDefinition(
        "F001", "localization", "vehicle pose", "provide vehicle pose",
        sources=[source], status=ReviewStatus.FINALIZED,
    )
    assessments = [
        GuidewordAssessment(
            "F001", guideword, True, "semantically applicable",
            sources=[source], status=ReviewStatus.FINALIZED, confidence=0.8,
        )
        for guideword in ("loss", "too late")
    ]
    return function, assessments


def _failure_type_contract():
    source = SourceRef("method", "failure_type_taxonomy.yaml", "types[0]", "loss")
    return (
        FailureModeTaxonomyValue("loss", (), source, "complete loss"),
        FailureModeTaxonomyValue("degraded", (), source, "degraded performance"),
    )


def test_provider_contract_requires_canonical_failure_type_without_extra_call():
    class Client:
        def __init__(self):
            self.requests = []

        def complete_json(self, request):
            self.requests.append(request)
            payload = _candidate("M1", "loss")
            payload["failure_type"] = "loss"
            return LLMResponse(data={"candidates": [payload]}, model="fake")

    function, assessments = _inputs()
    client = Client()
    candidates, audit = MalfunctionHazardAgent(
        client, failure_types=_failure_type_contract(),
    ).generate(function, [assessments[0]])

    assert candidates[0].failure_type == "loss"
    assert audit["llm_call_count"] == 1
    schema = client.requests[0].response_schema
    assert schema["properties"]["candidates"]["items"]["properties"]["failure_type"] == {
        "type": "string", "enum": ["loss", "degraded"],
    }
    assert "failure_type" in schema["properties"]["candidates"]["items"]["required"]


@pytest.mark.parametrize("failure_type", ["", "invented_type"])
def test_missing_or_invalid_failure_type_fails_closed(failure_type):
    class Client:
        def __init__(self):
            self.calls = 0

        def complete_json(self, _request):
            self.calls += 1
            payload = _candidate("M1", "loss")
            if failure_type:
                payload["failure_type"] = failure_type
            return LLMResponse(data={"candidates": [payload]}, model="fake")

    function, assessments = _inputs()
    client = Client()
    with pytest.raises(ValueError, match="MALFUNCTION_FAILURE_TYPE_CONTRACT_VIOLATION"):
        MalfunctionHazardAgent(
            client, failure_types=_failure_type_contract(),
        ).generate(function, [assessments[0]])
    assert client.calls == 1


def test_missing_guideword_is_repaired_once_and_then_strictly_validated():
    class Client:
        def __init__(self):
            self.requests = []

        def complete_json(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                assert "每一个Guideword至少返回一个" in request.user_prompt
                return LLMResponse(
                    data={"candidates": [_candidate("M1", "loss")]}, model="fake",
                )
            assert request.metadata["coverage_repair"] is True
            assert "MissingGuidewords=['too late']" in request.user_prompt
            return LLMResponse(
                data={"candidates": [_candidate("M2", "too late")]}, model="fake",
            )

    function, assessments = _inputs()
    client = Client()
    candidates, audit = MalfunctionHazardAgent(client).generate(function, assessments)

    assert {item.guideword for item in candidates} == {"loss", "too late"}
    assert audit["coverage_repair_count"] == 1
    assert audit["coverage_missing_before_repair"] == ["too late"]
    assert audit["llm_call_count"] == 2
    assert all(item.status is ReviewStatus.PENDING and item.is_analysis_premise for item in candidates)


def test_coverage_repair_fails_closed_when_missing_identity_remains_missing():
    class Client:
        def __init__(self):
            self.requests = []

        def complete_json(self, request):
            self.requests.append(request)
            if request.metadata.get("coverage_repair"):
                return LLMResponse(data={"candidates": []}, model="fake")
            return LLMResponse(
                data={"candidates": [_candidate("M1", "loss")]}, model="fake",
            )

    function, assessments = _inputs()
    client = Client()
    with pytest.raises(ValueError, match="缺少Malfunction候选"):
        MalfunctionHazardAgent(client).generate(function, assessments)

    # Initial generation, grouped repair, then one targeted repair for the
    # still-missing guideword. Only a genuinely unresolved identity fails.
    assert len(client.requests) == 3


def test_grouped_repair_keeps_partial_success_and_targets_only_remaining_identity():
    class Client:
        def __init__(self):
            self.requests = []

        def complete_json(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                return LLMResponse(data={"candidates": []}, model="fake")
            if len(self.requests) == 2:
                return LLMResponse(
                    data={"candidates": [_candidate("M1", "loss")]}, model="fake",
                )
            assert request.metadata["guideword_count"] == 1
            assert "MissingGuidewords=['too late']" in request.user_prompt
            return LLMResponse(
                data={"candidates": [_candidate("M2", "too late")]}, model="fake",
            )

    function, assessments = _inputs()
    candidates, audit = MalfunctionHazardAgent(Client()).generate(function, assessments)

    assert {item.guideword for item in candidates} == {"loss", "too late"}
    assert audit["coverage_missing_after_grouped_repair"] == ["too late"]
    assert audit["coverage_repair_count"] == 2
    assert audit["llm_call_count"] == 3


@pytest.mark.parametrize(
    ("chain", "expected"),
    [
        ("output lost -> maneuver unavailable", ["output lost", "maneuver unavailable"]),
        ('["output lost", "maneuver unavailable"]', ["output lost", "maneuver unavailable"]),
    ],
)
def test_explicit_scalar_causal_chain_is_safely_normalized(chain, expected):
    function, assessments = _inputs()
    raw = _candidate("M1", "loss")
    raw["causal_chain"] = chain

    candidate = MalfunctionHazardAgent._parse(function, raw, assessments[0])

    assert candidate.causal_chain == expected


def test_unsegmented_scalar_causal_chain_is_rejected_without_character_splitting():
    function, assessments = _inputs()
    raw = _candidate("M1", "loss")
    raw["causal_chain"] = "output loss causes maneuver unavailability"

    with pytest.raises(ValueError, match="no explicit segment boundary"):
        MalfunctionHazardAgent._parse(function, raw, assessments[0])


def test_f01_style_batch_normalizes_format_only_variations_without_repair_call():
    guidewords = [
        "always active", "as well as", "different to", "incomplete",
        "loss", "too early", "too late", "unintended",
    ]
    source = SourceRef("item_definition", "item.docx", "p1", "function evidence")
    function = FunctionDefinition(
        "F01", "parking control", "vehicle motion", "control parking motion",
        sources=[source], status=ReviewStatus.FINALIZED,
    )
    assessments = [
        GuidewordAssessment(
            "F01", guideword, True, "semantically applicable",
            sources=[source], status=ReviewStatus.FINALIZED, confidence=0.8,
        )
        for guideword in guidewords
    ]

    class Client:
        def __init__(self):
            self.requests = []

        def complete_json(self, request):
            self.requests.append(request)
            candidates = []
            for index, guideword in enumerate(guidewords):
                item = _candidate("" if index == 0 else "M", guideword.upper())
                item["description"] = f"function evidence output {guideword} deviation"
                item["causal_chain"] = (
                    f"{guideword} function deviation → {guideword} vehicle hazard"
                )
                candidates.append(item)
            return LLMResponse(data={"candidates": candidates}, model="fake")

    client = Client()
    candidates, audit = MalfunctionHazardAgent(client).generate(function, assessments)

    assert len(candidates) == len(guidewords)
    assert {item.guideword for item in candidates} == set(guidewords)
    assert len({item.malfunction_id for item in candidates}) == len(guidewords)
    assert all(len(item.causal_chain) == 2 for item in candidates)
    assert audit["coverage_repair_count"] == 0
    assert audit["llm_call_count"] == 1
    assert len(client.requests) == 1


def test_blanket_guideword_selection_is_audited_without_overriding_semantics():
    class Client:
        def complete_json(self, request):
            return LLMResponse(data={"assessments": [{
                "guideword": guideword,
                "applicable": True,
                "disposition": "DOWNSTREAM_CANDIDATE",
                "rationale": "the output has this deviation dimension",
                "confidence": 0.8,
                "status": "PENDING",
            } for guideword in ("loss", "too late")]}, model="fake")

    function, _ = _inputs()
    assessments, audit = GuidewordApplicabilityAgent(Client()).assess(
        function, ["loss", "too late"],
    )

    assert all(item.applicable for item in assessments)
    assert all(item.status is ReviewStatus.FINALIZED for item in assessments)
    assert audit["blanket_applicability"] is True


def test_no_credible_hazard_skips_malfunction_llm_call():
    class Client:
        def complete_json(self, request):
            raise AssertionError("no-hazard disposition must not call the LLM")

    function, _ = _inputs()
    assessment = GuidewordAssessment(
        "F001",
        "different to",
        True,
        "the difference cannot change vehicle-level behavior",
        sources=list(function.sources),
        status=ReviewStatus.FINALIZED,
        confidence=0.8,
        disposition=GuidewordDisposition.NO_CREDIBLE_HAZARD,
    )

    candidates, audit = MalfunctionHazardAgent(Client()).generate(
        function, [assessment],
    )

    assert candidates == []
    assert audit["skipped"] is True
    assert audit["skip_reason"] == "no_credible_hazard_guidewords"
    assert audit["llm_call_count"] == 0


def test_not_applicable_skips_malfunction_llm_call():
    class Client:
        def complete_json(self, request):
            raise AssertionError("not-applicable disposition must not call the LLM")

    function, _ = _inputs()
    assessment = GuidewordAssessment(
        "F001",
        "different to",
        False,
        "the output has no alternate direction or target",
        sources=list(function.sources),
        status=ReviewStatus.FINALIZED,
        confidence=0.8,
        disposition=GuidewordDisposition.NOT_APPLICABLE,
    )

    candidates, audit = MalfunctionHazardAgent(Client()).generate(
        function, [assessment],
    )

    assert candidates == []
    assert audit["skipped"] is True
    assert audit["skip_reason"] == "no_applicable_guidewords"
    assert audit["llm_call_count"] == 0


def test_downstream_candidate_generates_malfunction_with_bound_guideword_id():
    class Client:
        def __init__(self):
            self.calls = 0

        def complete_json(self, request):
            self.calls += 1
            return LLMResponse(data={"candidates": [_candidate("M1", "loss")]}, model="fake")

    function, assessments = _inputs()
    assessments[0].guideword_id = "GW-LOSS"
    client = Client()
    candidates, audit = MalfunctionHazardAgent(client).generate(
        function, [assessments[0]],
    )

    assert client.calls == 1
    assert audit["llm_call_count"] == 1
    assert candidates[0].guideword_id == "GW-LOSS"


def test_provider_output_for_no_credible_hazard_fails_closed_instead_of_filtering():
    class Client:
        def complete_json(self, request):
            return LLMResponse(
                data={"candidates": [_candidate("M1", "different to")]},
                model="fake",
            )

    function, assessments = _inputs()
    assessments[0].guideword = "loss"
    assessments[0].guideword_id = "GW-LOSS"
    assessments[1].guideword = "different to"
    assessments[1].guideword_id = "GW-DIFFERENT"
    assessments[1].disposition = GuidewordDisposition.NO_CREDIBLE_HAZARD

    with pytest.raises(ValueError, match="MALFUNCTION_GUIDEWORD_GATE_VIOLATION"):
        MalfunctionHazardAgent(Client()).generate(function, assessments)


def test_composite_output_less_considers_partial_safety_actuation_output():
    source = SourceRef("item_definition", "item.docx", "p1", "parking completion")
    function = FunctionDefinition(
        "F05",
        "parking completion",
        "engage P; apply EPB; show HMI completion; return to home screen",
        "complete the parking-end sequence",
        sources=[source],
        status=ReviewStatus.FINALIZED,
    )

    class Client:
        def __init__(self):
            self.request = None

        def complete_json(self, request):
            self.request = request
            return LLMResponse(data={"assessments": [{
                "guideword": "Less",
                "applicable": True,
                "disposition": "DOWNSTREAM_CANDIDATE",
                "rationale": "partial completion can engage P while omitting EPB actuation",
                "confidence": 0.8,
            }]}, model="fake")

    client = Client()
    assessments, _ = GuidewordApplicabilityAgent(client).assess(function, ["Less"])

    assert assessments[0].disposition is GuidewordDisposition.DOWNSTREAM_CANDIDATE
    assert assessments[0].guideword_id == "Less"
    assert "canonical_output_components" in client.request.user_prompt
    assert "engage P" in client.request.user_prompt
    assert "apply EPB" in client.request.user_prompt
    assert "missing partial output" in client.request.system_prompt


def test_workflow_fails_closed_if_an_alternate_agent_bypasses_guideword_gate():
    function, _ = _inputs()
    blocked_assessment = GuidewordAssessment(
        "F001",
        "loss",
        True,
        "the deviation is applicable but cannot create a credible vehicle hazard",
        sources=list(function.sources),
        status=ReviewStatus.FINALIZED,
        confidence=0.8,
        disposition=GuidewordDisposition.NO_CREDIBLE_HAZARD,
        guideword_id="GW-LOSS",
    )

    class BypassingAgent:
        def generate(self, generated_function, generated_assessments):
            return [
                MalfunctionHazardAgent._parse(
                    generated_function,
                    _candidate("M1", "loss"),
                    generated_assessments[0],
                )
            ], {"function_id": generated_function.function_id, "skipped": False}

    state = HARAState(run_id="guideword-gate-test")
    with pytest.raises(ValueError, match="MALFUNCTION_GUIDEWORD_GATE_VIOLATION"):
        derive_malfunctions(
            state,
            BypassingAgent(),
            [function],
            [blocked_assessment],
        )

    assert state.audit_trail[-1]["event"] == "malfunction_guideword_gate_violation"
