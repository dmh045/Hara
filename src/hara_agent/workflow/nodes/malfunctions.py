from __future__ import annotations

import sys

from hara_agent.models import (
    FunctionDefinition,
    GuidewordAssessment,
    MalfunctionCandidate,
)
from hara_agent.services.semantic import MalfunctionHazardAgent
from hara_agent.services.semantic import malfunction_guideword_gate
from hara_agent.services.analysis import FailureModeSelectorResolver
from hara_agent.workflow.state import HARAState, WorkflowStage
from hara_agent.workflow.review_artifacts import ReviewArtifactWriter

from .parallel import ordered_parallel_map


def _validate_malfunction_guideword_gate(
    state: HARAState,
    candidates: list[MalfunctionCandidate],
    assessments: list[GuidewordAssessment],
) -> None:
    """Protect the stage boundary even if an alternate agent bypasses its parser."""
    by_identity: dict[tuple[str, str], GuidewordAssessment] = {}
    for assessment in assessments:
        key = (assessment.function_id, assessment.guideword_id)
        if key in by_identity:
            state.record(
                "malfunction_guideword_gate_violation",
                reason="duplicate_guideword_assessment_identity",
                function_id=assessment.function_id,
                guideword_id=assessment.guideword_id,
            )
            raise ValueError(
                "MALFUNCTION_GUIDEWORD_GATE_VIOLATION "
                f"duplicate assessment function_id={assessment.function_id} "
                f"guideword_id={assessment.guideword_id}"
            )
        by_identity[key] = assessment

    for candidate in candidates:
        guideword_id = candidate.guideword_id
        assessment = by_identity.get((candidate.function_id, guideword_id))
        scheduled_guideword_ids = {
            item.guideword_id
            for item in assessments
            if item.function_id == candidate.function_id
        }
        try:
            malfunction_guideword_gate.validate_malfunction_guideword_gate(
                function_id=candidate.function_id,
                guideword=candidate.guideword,
                guideword_id=candidate.guideword_id,
                assessment=assessment,
                scheduled_guideword_ids=scheduled_guideword_ids,
            )
        except malfunction_guideword_gate.MalfunctionGuidewordGateViolation:
            state.record(
                "malfunction_guideword_gate_violation",
                reason="candidate_not_downstream_candidate",
                function_id=candidate.function_id,
                guideword=candidate.guideword,
                guideword_id=guideword_id,
                assessment_applicable=(assessment.applicable if assessment else None),
                assessment_disposition=(
                    assessment.disposition.value if assessment else None
                ),
                assessment_status=(assessment.status.value if assessment else None),
            )
            raise ValueError(
                "MALFUNCTION_GUIDEWORD_GATE_VIOLATION "
                f"function_id={candidate.function_id} guideword={candidate.guideword!r} "
                f"guideword_id={guideword_id!r}"
            )
    state.record(
        "malfunction_guideword_gate_validated",
        candidate_count=len(candidates),
    )


def derive_malfunctions(state: HARAState, agent: MalfunctionHazardAgent,
                        functions: list[FunctionDefinition],
                        assessments: list[GuidewordAssessment],
                        max_workers: int = 1,
                        progress=None,
                        review_artifact_writer: ReviewArtifactWriter | None = None,
                        selector_resolver: FailureModeSelectorResolver | None = None) -> HARAState:
    candidates = []
    def generate(function):
        related = [item for item in assessments if item.function_id == function.function_id]
        return agent.generate(function, related)

    batches = ordered_parallel_map(
        functions,
        generate,
        max_workers=max_workers,
        on_progress=(lambda done, total: progress("malfunction", done, total)) if progress else None,
    )
    audits = []
    for items, audit in batches:
        candidates.extend(items)
        audits.append(audit)
        state.record("malfunction_hazard_candidates_generated", **audit)
        for issue in audit.get("unresolved_candidate_issues", []):
            state.pending_reviews.append({
                "field": "malfunction_derivation",
                "function_id": audit.get("function_id", ""),
                **issue,
            })
    _validate_malfunction_guideword_gate(state, candidates, assessments)
    normalized = []
    seen_ids = set()
    function_counts = {}
    for item in candidates:
        model_local_id = item.malfunction_id
        function_counts[item.function_id] = function_counts.get(item.function_id, 0) + 1
        authoritative_id = f"MF-{item.function_id}-{function_counts[item.function_id]:03d}"
        if authoritative_id in seen_ids:
            raise ValueError(f"确定性Malfunction ID冲突: {authoritative_id}")
        seen_ids.add(authoritative_id)
        item.malfunction_id = authoritative_id
        item.model_local_id = model_local_id
        if selector_resolver is not None:
            item.selector_resolution = selector_resolver.resolve(item).to_dict()
        value = item.to_dict()
        normalized.append(value)
        if review_artifact_writer is not None:
            review_artifact_writer.record_malfunction(item)
        state.record(
            "malfunction_identity_normalized",
            function_id=item.function_id,
            model_local_id=model_local_id,
            authoritative_id=authoritative_id,
        )
    state.malfunctions = normalized
    pending_count = sum(item.status.value == "PENDING" for item in candidates)
    if pending_count:
        state.pending_reviews.append({
            "field": "malfunctions",
            "reason": f"{pending_count}个Malfunction/Hazard候选尚未完成工程确认",
        })
    skip_reasons = {
        reason: sum(audit.get("skip_reason") == reason for audit in audits)
        for reason in (
            "no_applicable_guidewords",
            "no_complete_applicable_guidewords",
            "no_credible_hazard_guidewords",
        )
    }
    llm_called = sum(not audit.get("skipped", False) for audit in audits)
    coverage_repairs = sum(
        int(audit.get("coverage_repair_count", 0)) for audit in audits
    )
    actual_llm_calls = sum(
        int(audit.get("llm_call_count", int(not audit.get("skipped", False))))
        for audit in audits
    )
    print(
        "[HARA] malfunction derivation summary "
        f"functions={len(functions)} llm_called={llm_called} "
        f"actual_llm_calls={actual_llm_calls} coverage_repairs={coverage_repairs} "
        f"skipped_no_applicable={skip_reasons['no_applicable_guidewords']} "
        f"skipped_incomplete={skip_reasons['no_complete_applicable_guidewords']} "
        f"skipped_no_hazard={skip_reasons['no_credible_hazard_guidewords']} "
        f"candidate_count={len(candidates)}",
        file=sys.stderr,
        flush=True,
    )
    state.record(
        "malfunction_derivation_summary",
        function_count=len(functions),
        llm_called=llm_called,
        actual_llm_calls=actual_llm_calls,
        coverage_repair_count=coverage_repairs,
        skipped_no_applicable=skip_reasons["no_applicable_guidewords"],
        skipped_incomplete=skip_reasons["no_complete_applicable_guidewords"],
        skipped_no_hazard=skip_reasons["no_credible_hazard_guidewords"],
        candidate_count=len(candidates),
    )
    state.stage = WorkflowStage.MALFUNCTIONS
    return state
