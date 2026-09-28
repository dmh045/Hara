from __future__ import annotations

from dataclasses import dataclass
from time import monotonic
from typing import Callable

from hara_agent.services.validation import ReleaseGateValidator
from hara_agent.infrastructure.llm.provider_budget import ProviderAttemptBudgetExceeded

from .checkpoints import CheckpointRepository
from .state import HARAState, WorkflowStage
from .review_artifacts import ReviewArtifactWriter


Node = Callable[[HARAState], HARAState]


@dataclass(frozen=True)
class WorkflowRunResult:
    state: HARAState
    interrupted: bool
    reason: str = ""


class WorkflowGraph:
    """Dependency-free graph runner; nodes remain reusable by LangGraph later."""

    TERMINAL = {WorkflowStage.COMPLETE, WorkflowStage.BLOCKED}

    def __init__(self, checkpoint_repository: CheckpointRepository | None = None,
                 progress: Callable[[str, str, float], None] | None = None,
                 review_artifact_writer: ReviewArtifactWriter | None = None):
        self.nodes: dict[WorkflowStage, Node] = {}
        self.checkpoints = checkpoint_repository
        self.progress = progress
        self.review_artifact_writer = review_artifact_writer

    def add_node(self, stage: WorkflowStage, node: Node) -> "WorkflowGraph":
        if stage in self.nodes:
            raise ValueError(f"WorkflowStage重复注册: {stage.value}")
        self.nodes[stage] = node
        return self

    def run(self, state: HARAState, stop_on_review: bool = True,
            max_steps: int = 50,
            stop_before: set[WorkflowStage] | None = None) -> WorkflowRunResult:
        for _ in range(max_steps):
            if state.stage in self.TERMINAL:
                return WorkflowRunResult(state, False)
            if stop_before and state.stage in stop_before:
                self._save(state)
                return WorkflowRunResult(state, True, f"planned_stop:{state.stage.value}")
            if stop_on_review and state.stage is WorkflowStage.QUALITY_GATE and not state.can_publish:
                validation = ReleaseGateValidator().evaluate(state)
                state.record(
                    "quality_gate_blocked",
                    blockers=list(validation.blockers),
                    checks=validation.checks,
                )
                self._save(state)
                return WorkflowRunResult(state, True, "pending_engineering_review")
            node = self.nodes.get(state.stage)
            if node is None:
                self._save(state)
                return WorkflowRunResult(state, True, f"node_not_registered:{state.stage.value}")
            previous = state.stage
            started = monotonic()
            if self.progress:
                self.progress("started", previous.value, 0.0)
            try:
                state = node(state)
            except ProviderAttemptBudgetExceeded as exc:
                if self.checkpoints is None:
                    raise
                # A stage may have partially mutated its in-memory state.
                # Resume only from the last atomic committed checkpoint.
                state = self.checkpoints.load(state.run_id)
                state.record(
                    "provider_attempt_budget_exhausted",
                    attempts=exc.attempts, limit=exc.limit,
                    ledger_path=str(exc.ledger_path),
                    committed_stage=state.stage.value,
                )
                self._save(state)
                if self.review_artifact_writer is not None:
                    self.review_artifact_writer.write_summary(state)
                return WorkflowRunResult(state, True, "provider_attempt_budget_exhausted")
            except Exception as exc:
                if self.review_artifact_writer is not None:
                    self.review_artifact_writer.mark_failed(state, exc)
                if self.progress:
                    self.progress("failed", previous.value, monotonic() - started)
                raise
            if self.progress:
                self.progress("completed", previous.value, monotonic() - started)
            if state.stage is previous:
                state.errors.append({"type": "workflow_no_progress", "stage": previous.value})
                state.stage = WorkflowStage.BLOCKED
            if self.review_artifact_writer is not None:
                self.review_artifact_writer.write_summary(state)
            self._save(state)
        state.errors.append({"type": "workflow_step_limit", "max_steps": max_steps})
        state.stage = WorkflowStage.BLOCKED
        self._save(state)
        return WorkflowRunResult(state, False, "step_limit")

    def _save(self, state: HARAState):
        if self.checkpoints:
            self.checkpoints.save(state)
