import pytest

from hara_agent.infrastructure.llm import LLMResponse
from hara_agent.models import (
    FunctionDefinition,
    GuidewordAssessment,
    GuidewordDisposition,
    ReviewStatus,
    SourceRef,
)
from hara_agent.services.semantic import MalfunctionHazardAgent
from hara_agent.services.semantic import malfunction_guideword_gate as gate
from hara_agent.workflow.nodes.malfunctions import derive_malfunctions
from hara_agent.workflow.state import HARAState


def _assessment(
    *,
    applicable: bool = True,
    disposition: GuidewordDisposition = GuidewordDisposition.DOWNSTREAM_CANDIDATE,
    status: ReviewStatus = ReviewStatus.FINALIZED,
    guideword_id: str = "GW-LOSS",
) -> GuidewordAssessment:
    return GuidewordAssessment(
        function_id="F001",
        guideword="loss",
        guideword_id=guideword_id,
        applicable=applicable,
        disposition=disposition,
        rationale="governed guideword assessment rationale",
        sources=[SourceRef("item", "item.docx", "p1", "function")],
        status=status,
        confidence=0.8,
    )


def _validate(assessment: GuidewordAssessment, *, guideword_id: str = "GW-LOSS", scheduled=()):
    return gate.validate_malfunction_guideword_gate(
        function_id="F001",
        guideword="loss",
        guideword_id=guideword_id,
        assessment=assessment,
        scheduled_guideword_ids=scheduled or {"GW-LOSS"},
    )


def test_shared_gate_allows_exact_finalized_downstream_candidate():
    assessment = _assessment()

    assert _validate(assessment) is assessment


@pytest.mark.parametrize(
    "assessment, guideword_id, scheduled",
    [
        (_assessment(disposition=GuidewordDisposition.NO_CREDIBLE_HAZARD), "GW-LOSS", {"GW-LOSS"}),
        (_assessment(applicable=False, disposition=GuidewordDisposition.NOT_APPLICABLE), "GW-LOSS", {"GW-LOSS"}),
        (_assessment(status=ReviewStatus.PENDING), "GW-LOSS", {"GW-LOSS"}),
        (_assessment(), "GW-LOSS", {"GW-OTHER"}),
    ],
)
def test_shared_gate_fails_closed_for_non_downstream_or_unscheduled_assessment(
    assessment,
    guideword_id,
    scheduled,
):
    with pytest.raises(gate.MalfunctionGuidewordGateViolation, match="MALFUNCTION_GUIDEWORD_GATE_VIOLATION"):
        _validate(assessment, guideword_id=guideword_id, scheduled=scheduled)


def test_agent_and_workflow_call_the_same_shared_gate_helper(monkeypatch):
    calls = []
    original = gate.validate_malfunction_guideword_gate

    def spy(**kwargs):
        calls.append(kwargs)
        return original(**kwargs)

    monkeypatch.setattr(gate, "validate_malfunction_guideword_gate", spy)
    assessment = _assessment()
    source = assessment.sources[0]
    function = FunctionDefinition(
        "F001", "localization", "vehicle pose", "provide vehicle pose",
        sources=[source], status=ReviewStatus.FINALIZED,
    )

    class Client:
        def complete_json(self, request):
            return LLMResponse(data={"candidates": [{
                "malfunction_id": "M1",
                "guideword": "loss",
                "description": "vehicle pose output loss deviation",
                "functional_effect": "loss effect",
                "vehicle_level_hazard": "loss hazardous vehicle state",
                "causal_chain": ["loss deviation", "loss vehicle effect"],
                "confidence": 0.8,
                "status": "PENDING",
                "derivation": {"function_refs": ["output"], "subject": "vehicle pose",
                    "quantity": "output", "basis": "Loss of the stated output", "assumptions": []},
            }]}, model="fake")

    MalfunctionHazardAgent(Client()).generate(function, [assessment])
    agent_call_count = len(calls)
    assert agent_call_count >= 1

    class BypassingAgent:
        def generate(self, generated_function, generated_assessments):
            candidate = MalfunctionHazardAgent._parse(
                generated_function,
                {
                    "malfunction_id": "M2",
                    "guideword": "loss",
                    "description": "loss deviation from alternate agent",
                    "functional_effect": "loss effect",
                    "vehicle_level_hazard": "loss hazardous vehicle state",
                    "causal_chain": ["loss deviation", "loss vehicle effect"],
                    "confidence": 0.8,
                    "status": "PENDING",
                },
                generated_assessments[0],
            )
            return [candidate], {"function_id": generated_function.function_id, "skipped": False}

    derive_malfunctions(
        HARAState(run_id="shared-guideword-gate"),
        BypassingAgent(),
        [function],
        [assessment],
    )

    assert len(calls) == agent_call_count + 1
