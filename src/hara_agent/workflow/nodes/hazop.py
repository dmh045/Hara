from __future__ import annotations

from dataclasses import asdict, is_dataclass
import hashlib
import json
from typing import Callable

from hara_agent.contracts import Guideword
from hara_agent.models import (
    FunctionDefinition, GuidewordAssessment, GuidewordDisposition,
    ReviewStatus, SourceRef,
)
from hara_agent.services.semantic import GuidewordApplicabilityAgent
from hara_agent.workflow.state import HARAState, WorkflowStage
from hara_agent.workflow.review_artifacts import ReviewArtifactWriter

from .parallel import ordered_parallel_map


GUIDEWORD_BATCH_CHECKPOINT_VERSION = "guideword-function-batch-v1"


def _batch_key(agent: GuidewordApplicabilityAgent, function: FunctionDefinition,
               guidewords: list[str | Guideword]) -> str:
    client = getattr(agent, "client", None)
    config = getattr(client, "config", None)
    material = {
        "version": GUIDEWORD_BATCH_CHECKPOINT_VERSION,
        "prompt_version": getattr(agent, "PROMPT_VERSION", ""),
        "function": asdict(function),
        "guidewords": [asdict(item) if is_dataclass(item) else str(item)
                       for item in guidewords],
        "provider": getattr(config, "provider", type(client).__name__),
        "model": getattr(config, "model", ""),
        "guideword_thinking": getattr(config, "guideword_thinking", ""),
    }
    return hashlib.sha256(json.dumps(
        material, ensure_ascii=False, sort_keys=True, default=str,
    ).encode("utf-8")).hexdigest()


def _restore_batch(entry: object, *, key: str, function_id: str):
    if not isinstance(entry, dict) or entry.get("key") != key:
        return None
    try:
        items = []
        for value in entry["assessments"]:
            payload = dict(value)
            payload["sources"] = [SourceRef(**item) for item in payload.get("sources", [])]
            payload["status"] = ReviewStatus(payload.get("status", ReviewStatus.PENDING.value))
            disposition = payload.get("disposition")
            payload["disposition"] = (
                GuidewordDisposition(disposition) if disposition else None
            )
            items.append(GuidewordAssessment(**payload))
        audit = dict(entry["audit"])
    except (KeyError, TypeError, ValueError):
        return None
    if any(item.function_id != function_id for item in items):
        return None
    return items, audit


def assess_guidewords(state: HARAState, agent: GuidewordApplicabilityAgent,
                      functions: list[FunctionDefinition],
                      guidewords: list[str | Guideword],
                      max_workers: int = 1,
                      progress=None,
                      review_artifact_writer: ReviewArtifactWriter | None = None,
                      checkpoint: Callable[[HARAState], object] | None = None) -> HARAState:
    assessments = []
    cache = state.item_definition.setdefault("guideword_assessment_batches", {})
    if not isinstance(cache, dict):
        cache = {}
        state.item_definition["guideword_assessment_batches"] = cache
    batch_by_function = {}
    pending_functions = []

    def record_result(_function, result) -> None:
        if review_artifact_writer is None:
            return
        items, _audit = result
        for assessment in items:
            review_artifact_writer.record_guideword_assessment(assessment)

    for function in functions:
        key = _batch_key(agent, function, guidewords)
        restored = _restore_batch(
            cache.get(function.function_id), key=key,
            function_id=function.function_id,
        )
        if restored is None:
            if function.function_id in cache:
                raise ValueError(
                    "Committed guideword Function batch no longer matches its "
                    "inputs or prompt; restart with a new run ID"
                )
            pending_functions.append(function)
        else:
            batch_by_function[function.function_id] = restored
            record_result(function, restored)

    def save_batch(function, result) -> None:
        cache[function.function_id] = {
            "version": GUIDEWORD_BATCH_CHECKPOINT_VERSION,
            "key": _batch_key(agent, function, guidewords),
            "assessments": [asdict(item) for item in result[0]],
            "audit": result[1],
        }
        batch_by_function[function.function_id] = result
        state.record(
            "guideword_function_batch_checkpointed",
            function_id=function.function_id,
            completed_function_count=len(batch_by_function),
            total_function_count=len(functions),
        )
        if checkpoint is not None:
            checkpoint(state)
        record_result(function, result)

    ordered_parallel_map(
        pending_functions,
        lambda function: agent.assess(function, guidewords),
        max_workers=max_workers,
        on_progress=(lambda done, _total: progress(
            "guideword", len(batch_by_function), len(functions),
        )) if progress else None,
        on_result=save_batch,
    )
    batches = [batch_by_function[function.function_id] for function in functions]
    for items, audit in batches:
        assessments.extend(items)
        state.record("guideword_applicability_assessed", **audit)
    state.guideword_assessments = [asdict(item) for item in assessments]
    state.item_definition.pop("guideword_assessment_batches", None)
    # HAZOP applicability is its own durable artifact.  Do not overload the
    # downstream Malfunction collection or lose negative decisions later.
    state.malfunctions = []
    incomplete_count = sum(not item.is_semantically_complete for item in assessments)
    review_pending_count = sum(
        item.status.value == "PENDING" and item.is_semantically_complete
        for item in assessments
    )
    if incomplete_count:
        state.pending_reviews.append({
            "field": "guideword_assessments",
            "issue_type": "semantic_incomplete",
            "reason": f"{incomplete_count}个Guideword适用性结论语义字段不完整",
        })
    if review_pending_count:
        state.pending_reviews.append({
            "field": "guideword_assessments",
            "issue_type": "pending_review",
            "reason": f"{review_pending_count}个完整Guideword适用性结论尚未完成工程审批",
        })
    blanket_function_ids = [
        str(audit.get("function_id", ""))
        for _, audit in batches
        if audit.get("blanket_applicability") is True
    ]
    if blanket_function_ids:
        state.record(
            "guideword_blanket_applicability_observed",
            function_ids=blanket_function_ids,
            function_count=len(blanket_function_ids),
            action="continue_to_malfunction_and_causal_filters",
        )
    near_blanket_function_ids = [
        str(audit.get("function_id", ""))
        for _, audit in batches
        if audit.get("near_blanket_applicability") is True
    ]
    if near_blanket_function_ids:
        state.record(
            "guideword_high_applicability_reviewed",
            function_ids=near_blanket_function_ids,
            function_count=len(near_blanket_function_ids),
            review_mode="same_request_self_check_plus_deterministic_audit",
            additional_llm_calls=0,
        )

    guideword_id_to_name = {
        (value.guideword_id if isinstance(value, Guideword) else str(value).strip()): (
            value.name if isinstance(value, Guideword) else str(value).strip()
        )
        for value in guidewords
    }
    applicable_ids = {
        item.guideword_id
        for item in assessments
        if item.applicable
        and item.is_semantically_complete
        and item.status.value == "FINALIZED"
    }
    unmatched_ids = [
        guideword_id for guideword_id in guideword_id_to_name
        if guideword_id not in applicable_ids
    ]
    unmatched = [guideword_id_to_name[guideword_id] for guideword_id in unmatched_ids]
    state.record(
        "guideword_global_coverage_audited",
        guideword_count=len(guideword_id_to_name),
        matched_count=len(guideword_id_to_name) - len(unmatched_ids),
        unmatched_guideword_ids=unmatched_ids,
        unmatched_guidewords=unmatched,
        action="audit_only_no_forced_match",
    )
    state.stage = WorkflowStage.HAZOP
    return state
