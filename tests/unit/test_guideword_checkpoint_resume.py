from __future__ import annotations

import pytest

from hara_agent.models import (
    FunctionDefinition, GuidewordAssessment, ReviewStatus, SourceRef,
)
from hara_agent.workflow.checkpoints import CheckpointRepository
from hara_agent.workflow.nodes.hazop import assess_guidewords
from hara_agent.workflow.review_artifacts import (
    ReviewArtifactReader, ReviewArtifactWriter,
)
from hara_agent.workflow.state import HARAState, WorkflowStage


def _function(function_id: str) -> FunctionDefinition:
    return FunctionDefinition(
        function_id=function_id, name=f"Function {function_id}",
        output="motion request", sources=[SourceRef(
            "item_definition", "item.docx", f"function/{function_id}", "motion",
        )], status=ReviewStatus.FINALIZED,
    )


class _Agent:
    PROMPT_VERSION = "test-guidewords-v1"

    def __init__(self, *, fail_on: str = ""):
        self.fail_on = fail_on
        self.calls: list[str] = []
        self.client = object()

    def assess(self, function, guidewords):
        self.calls.append(function.function_id)
        if function.function_id == self.fail_on:
            raise RuntimeError("interrupted")
        return [GuidewordAssessment(
            function_id=function.function_id, guideword=guidewords[0],
            applicable=True, rationale="the output can be lost",
            sources=list(function.sources), status=ReviewStatus.FINALIZED,
            confidence=0.9,
        )], {"function_id": function.function_id}


def test_guideword_resume_uses_committed_batch_and_reconciles_review(tmp_path):
    repository = CheckpointRepository(tmp_path / "agent")
    review_root = tmp_path / "review"
    writer = ReviewArtifactWriter("sample", review_root)
    original_record = writer.record_guideword_assessment

    def record_after_commit(assessment):
        committed = repository.load("sample")
        assert assessment.function_id in committed.item_definition[
            "guideword_assessment_batches"
        ]
        return original_record(assessment)

    writer.record_guideword_assessment = record_after_commit
    state = HARAState(run_id="sample", stage=WorkflowStage.FUNCTIONS)
    functions = [_function("F01"), _function("F02")]
    first_agent = _Agent(fail_on="F02")

    with pytest.raises(RuntimeError, match="interrupted"):
        assess_guidewords(
            state, first_agent, functions, ["loss"], checkpoint=repository.save,
            review_artifact_writer=writer,
        )
    committed = repository.load("sample")
    assert committed.stage is WorkflowStage.FUNCTIONS
    assert set(committed.item_definition["guideword_assessment_batches"]) == {"F01"}
    assert len(ReviewArtifactReader("sample", review_root).read_all()["guideword_assessment"]) == 1
    # Simulate an interruption after the atomic checkpoint but before its
    # separate review JSONL append; resume must repair the projection.
    (review_root / "sample" / "guideword_assessments.jsonl").unlink()

    second_agent = _Agent()
    resumed = assess_guidewords(
        committed, second_agent, functions, ["loss"], checkpoint=repository.save,
        review_artifact_writer=ReviewArtifactWriter("sample", review_root),
    )
    assert second_agent.calls == ["F02"]
    assert resumed.stage is WorkflowStage.HAZOP
    assert "guideword_assessment_batches" not in resumed.item_definition
    assert {item["function_id"] for item in resumed.guideword_assessments} == {"F01", "F02"}
    records = ReviewArtifactReader("sample", review_root).read_all()["guideword_assessment"]
    assert {item["function_id"] for item in records} == {"F01", "F02"}
