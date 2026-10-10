from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import sys
from typing import Callable, Sequence

from hara_agent.contracts import FactOrigin, RequiredFactSpec
from hara_agent.contracts import MethodContract
from hara_agent.models import (
    ItemDefinitionFacts,
    MalfunctionCandidate,
    ScenarioCandidate,
    ScenarioFeasibilityAssessment, evaluate_risk_eligibility,
)
from hara_agent.services.semantic import (
    DEFAULT_CAUSAL_EVIDENCE_BUDGET, ScenarioFeasibilityAgent,
    ScenarioRiskFactAgent, build_project_evidence_registry,
)
from hara_agent.services.analysis import RiskExecutionTraceService, ScenarioMethodService
from hara_agent.services.analysis.malfunction_situation_selection import selection_function
from hara_agent.workflow.state import HARAState, WorkflowStage

from .parallel import ordered_parallel_map
from ..review_artifacts import ReviewArtifactWriter


SCENARIO_BATCH_CHECKPOINT_VERSION = "scenario-malfunction-batch-v5"


def _scenario_batch_cache_key(
    agent: ScenarioFeasibilityAgent,
    malfunction: MalfunctionCandidate,
    candidates: Sequence[ScenarioCandidate],
) -> str:
    config = getattr(agent.client, "config", None)
    material = {
        "cache_version": SCENARIO_BATCH_CHECKPOINT_VERSION,
        "prompt_version": agent.prompt_version,
        "assessment_contract_version": agent.assessment_contract_version,
        "causal_evidence_budget": getattr(
            agent, "causal_evidence_budget", DEFAULT_CAUSAL_EVIDENCE_BUDGET,
        ),
        "provider": getattr(config, "provider", type(agent.client).__name__),
        "base_url": getattr(config, "base_url", ""),
        "model": getattr(config, "model", type(agent.client).__name__),
        "malfunction": asdict(malfunction),
        "scenarios": [{
            "scenario_id": item.scenario_id,
            "semantic_fingerprint": item.semantic_fingerprint,
            "scenario_contract_version": item.scenario_contract_version,
            "facts": item.facts,
            "fact_provenance": item.fact_provenance,
            "analysis_instance": item.analysis_instance,
        } for item in candidates],
    }
    serialized = json.dumps(
        material, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        default=str,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _load_cached_scenario_batch(
    entry: object,
    *,
    cache_key: str,
    malfunction_id: str,
    expected_scenario_ids: Sequence[str],
) -> tuple[list, dict] | None:
    if not isinstance(entry, dict) or entry.get("cache_key") != cache_key:
        return None
    try:
        assessments = [
            ScenarioFeasibilityAssessment.from_dict(item)
            for item in entry.get("assessments", [])
        ]
        audit = dict(entry["audit"])
    except (KeyError, TypeError, ValueError):
        return None
    actual_ids = [item.scenario_id for item in assessments]
    if (
        actual_ids != list(expected_scenario_ids)
        or any(item.malfunction_id != malfunction_id for item in assessments)
        or audit.get("malfunction_id") != malfunction_id
    ):
        return None
    return assessments, audit


def _assess_instantiated_scenarios(
    agent: ScenarioFeasibilityAgent,
    malfunction: MalfunctionCandidate,
    candidates: list[ScenarioCandidate],
    instantiation: dict,
    project_registry,
) -> tuple[list[ScenarioFeasibilityAssessment], dict]:
    """Assess only the candidate instances selected for one malfunction."""
    assessments, audit = agent.assess(
        malfunction, candidates, project_registry=project_registry,
    )
    audit["analytical_scenario_instantiation"] = instantiation
    return assessments, audit


def assess_scenarios(state: HARAState, agent: ScenarioFeasibilityAgent,
                     malfunctions: list[MalfunctionCandidate],
                     candidates: list[ScenarioCandidate],
                     max_workers: int = 1,
                     progress=None,
                     risk_fact_agent: ScenarioRiskFactAgent | None = None,
                     required_fact_specs: Sequence[RequiredFactSpec] = (),
                     method: MethodContract | None = None,
                     checkpoint: Callable[[HARAState], object] | None = None,
                     review_artifact_writer: ReviewArtifactWriter | None = None) -> HARAState:
    assessments = []
    typed = state.item_definition.get("typed", {})
    project_facts = (
        ItemDefinitionFacts.from_dict(typed)
        if isinstance(typed, dict) and typed else None
    )
    project_registry = (
        build_project_evidence_registry(project_facts, method)
        if project_facts is not None else None
    )
    cache = state.item_definition.setdefault("scenario_assessment_batches", {})
    if not isinstance(cache, dict):
        cache = {}
        state.item_definition["scenario_assessment_batches"] = cache
    batch_by_malfunction: dict[str, tuple[list, dict]] = {}
    pending_malfunctions = []
    template_service = ScenarioMethodService(method) if method is not None else None
    contextual_candidates: dict[str, list[ScenarioCandidate]] = {}
    instantiation_audit: dict[str, dict] = {}
    for malfunction in malfunctions:
        if template_service is None:
            contextual_candidates[malfunction.malfunction_id] = list(candidates)
            instantiation_audit[malfunction.malfunction_id] = {
                "malfunction_id": malfunction.malfunction_id,
                "selection_mode": "BASE_CANDIDATES_ONLY",
                "selection_basis": "NO_SCENARIO_METHOD_CONTRACT",
                "base_candidate_count": len(candidates),
                "instance_count": 0,
                "options": [],
            }
        else:
            instantiated, audit = template_service.instantiate_analytical_candidates(
                malfunction, candidates, project_facts=project_facts,
                function=selection_function(next(
                    (item for item in state.functions if item.get("function_id") == malfunction.function_id), None,
                )),
            )
            contextual_candidates[malfunction.malfunction_id] = instantiated
            instantiation_audit[malfunction.malfunction_id] = audit
            if review_artifact_writer is not None:
                for candidate in instantiated:
                    review_artifact_writer.record_scenario_candidate(
                        candidate,
                        generated_for_malfunction_ids=[malfunction.malfunction_id],
                    )
    runtime_candidates = [
        candidate
        for malfunction in malfunctions
        for candidate in contextual_candidates[malfunction.malfunction_id]
    ]
    candidate_by_id = {
        candidate.scenario_id: candidate for candidate in runtime_candidates
    }

    selection_audits = {
        key: {**value["situation_selection"], "causal_attempted_count": None,
              "causal_retained_count": None, "causal_status": "NOT_COMPLETED"}
        for key, value in instantiation_audit.items() if "situation_selection" in value
    }

    def persist_selection_audit():
        if selection_audits:
            payload = {"per_malfunction": list(selection_audits.values()), "provider_calls_added_by_selection": 0}
            state.item_definition["malfunction_situation_selection"] = payload
            if review_artifact_writer is not None:
                review_artifact_writer.write_malfunction_situation_selection(payload)

    persist_selection_audit()

    def persist_review_batch(malfunction, result, *, cached=False) -> None:
        """Persist one canonical malfunction result before other workers finish."""

        batch_assessments, audit = result
        if malfunction.malfunction_id in selection_audits:
            selection_audits[malfunction.malfunction_id].update({
                "causal_attempted_count": 0 if cached else len(batch_assessments),
                "causal_assessed_count": len(batch_assessments),
                "causal_retained_count": sum(evaluate_risk_eligibility(x).eligible for x in batch_assessments),
                "causal_status": "CACHE_REUSED" if cached else "COMPLETED",
            })
            persist_selection_audit()
        if review_artifact_writer is None:
            return
        for assessment in batch_assessments:
            review_artifact_writer.record_scenario_feasibility(
                assessment,
                function_id=malfunction.function_id,
                guideword=malfunction.guideword,
                audit=audit,
            )
        # Counts are rebuilt from successfully appended JSONL records, not
        # workflow-local progress, so review remains correct after failure.
        review_artifact_writer.write_summary(state)

    for malfunction in malfunctions:
        expected_scenario_ids = [
            item.scenario_id
            for item in contextual_candidates[malfunction.malfunction_id]
        ]
        cache_key = _scenario_batch_cache_key(
            agent, malfunction, contextual_candidates[malfunction.malfunction_id],
        )
        restored = _load_cached_scenario_batch(
            cache.get(malfunction.malfunction_id),
            cache_key=cache_key,
            malfunction_id=malfunction.malfunction_id,
            expected_scenario_ids=expected_scenario_ids,
        )
        if restored is None:
            pending_malfunctions.append(malfunction)
        else:
            batch_by_malfunction[malfunction.malfunction_id] = restored
            persist_review_batch(malfunction, restored, cached=True)
    cached_count = len(batch_by_malfunction)
    if cached_count:
        print(
            "[HARA] scenario checkpoint restore "
            f"completed_malfunctions={cached_count} "
            f"remaining_malfunctions={len(pending_malfunctions)}",
            file=sys.stderr,
            flush=True,
        )

    def save_batch(malfunction, result) -> None:
        batch_assessments, audit = result
        # ordered_parallel_map calls on_result for every completed worker
        # before waiting for the remaining workers.  This preserves prior
        # completed malfunction projections if a later provider call fails.
        persist_review_batch(malfunction, result)
        cache_key = _scenario_batch_cache_key(
            agent, malfunction, contextual_candidates[malfunction.malfunction_id],
        )
        cache[malfunction.malfunction_id] = {
            "cache_version": SCENARIO_BATCH_CHECKPOINT_VERSION,
            "cache_key": cache_key,
            "assessments": [item.to_dict() for item in batch_assessments],
            "audit": audit,
        }
        batch_by_malfunction[malfunction.malfunction_id] = result
        state.record(
            "scenario_malfunction_batch_checkpointed",
            malfunction_id=malfunction.malfunction_id,
            assessment_count=len(batch_assessments),
            completed_malfunction_count=len(batch_by_malfunction),
            total_malfunction_count=len(malfunctions),
        )
        if checkpoint is not None:
            checkpoint(state)

    new_batches = ordered_parallel_map(
        pending_malfunctions,
        lambda malfunction: _assess_instantiated_scenarios(
            agent, malfunction, contextual_candidates[malfunction.malfunction_id],
            instantiation_audit[malfunction.malfunction_id], project_registry,
        ),
        max_workers=max_workers,
        on_progress=(
            lambda done, _total: progress(
                "scenario", cached_count + done, len(malfunctions)
            )
        ) if progress else None,
        on_result=save_batch,
    )
    for malfunction, result in zip(pending_malfunctions, new_batches):
        batch_by_malfunction[malfunction.malfunction_id] = result
    batches = [
        batch_by_malfunction[malfunction.malfunction_id]
        for malfunction in malfunctions
    ]
    audits = []
    for items, audit in batches:
        assessments.extend(items)
        audits.append(audit)
        state.record("scenario_feasibility_assessed", **audit)
    retained_ids = {
        item.scenario_id for item in assessments
        if evaluate_risk_eligibility(item).eligible
    }
    state.scenarios = [
        candidate_by_id[scenario_id]
        for scenario_id in sorted(retained_ids)
        if scenario_id in candidate_by_id
    ]
    serialized_assessments = [item.to_dict() for item in assessments]
    state.item_definition["scenario_assessments"] = serialized_assessments
    if risk_fact_agent is not None:
        risk_batch_cache = state.item_definition.setdefault(
            "scenario_risk_fact_batches", {}
        )
        if not isinstance(risk_batch_cache, dict):
            risk_batch_cache = {}
            state.item_definition["scenario_risk_fact_batches"] = risk_batch_cache

        def save_risk_batch(cache_key, entry) -> None:
            risk_batch_cache[cache_key] = entry
            state.record(
                "scenario_risk_fact_batch_checkpointed",
                cache_key=cache_key,
                completed_batch_count=len(risk_batch_cache),
            )
            if checkpoint is not None:
                checkpoint(state)

        risk_facts, risk_fact_audit = risk_fact_agent.interpret(
            serialized_assessments,
            state.scenarios,
            malfunctions,
            required_fact_specs,
            batch_cache=risk_batch_cache,
            on_batch=save_risk_batch,
        )
        typed_facts = ItemDefinitionFacts.from_dict(state.item_definition["typed"])
        requested_types = {
            item.fact_type.value
            for item in required_fact_specs
            if item.origin is FactOrigin.SCENARIO_FACT
        }
        requested_pairs = {
            (str(item.get("malfunction_id", "")), str(item.get("scenario_id", "")))
            for item in serialized_assessments
            if item.get("malfunction_id") and item.get("scenario_id")
        }
        replaced_ids = {
            item.fact_id
            for item in typed_facts.risk_facts
            if item.parameter in requested_types
            and (
                item.context.get("malfunction_id", ""),
                item.context.get("scenario_id", ""),
            ) in requested_pairs
        }
        if replaced_ids or risk_facts:
            typed_facts.risk_facts = [
                item for item in typed_facts.risk_facts if item.fact_id not in replaced_ids
            ]
            typed_facts.method_risk_fact_bindings = [
                item for item in typed_facts.method_risk_fact_bindings
                if item.source_fact_id not in replaced_ids
            ]
            typed_facts.risk_facts.extend(risk_facts)
            state.item_definition["typed"] = asdict(typed_facts)
        pending_risk_facts = [
            item for item in risk_facts
            if item.approval.value == "PENDING"
        ]
        unresolved_risk_fact_contracts = int(
            risk_fact_audit.get("unresolved_contract_count", 0)
        )
        if pending_risk_facts or unresolved_risk_fact_contracts:
            state.pending_reviews.append({
                "field": "scenario_risk_facts",
                "reason": (
                    f"{len(pending_risk_facts)} scenario facts remain PENDING and "
                    f"{unresolved_risk_fact_contracts} requested facts remained "
                    "unresolved after bounded contract repair"
                ),
            })
        state.record(
            "scenario_risk_facts_interpreted",
            replaced_stale_fact_count=len(replaced_ids),
            **risk_fact_audit,
        )
        state.item_definition.pop("scenario_risk_fact_batches", None)
    pending_assessments = sum(item.status.value == "PENDING" for item in assessments)
    pending_candidates = sum(item.status.value == "PENDING" for item in state.scenarios)
    if pending_assessments or pending_candidates:
        state.pending_reviews.append({
            "field": "scenarios",
            "reason": (
                f"{pending_candidates}个场景候选和{pending_assessments}个可行性结论尚未完成工程确认"
            ),
        })
    state.stage = WorkflowStage.SCORING
    state.record(
        "scenario_candidates_selected",
        candidate_count=len(candidates),
        retained_count=len(state.scenarios),
        assessment_count=len(assessments),
    )
    if review_artifact_writer is not None:
        binding_gaps = []
        for event in reversed(state.audit_trail):
            if event.get("event") == "scenario_candidates_prepared":
                binding_gaps = list(event.get("scenario_binding_gaps", []))
                break
        review_artifact_writer.write_risk_execution_trace(
            RiskExecutionTraceService(method).project(
                run_id=state.run_id,
                assessments=serialized_assessments,
                candidates=runtime_candidates,
                committed=True,
                binding_gaps=binding_gaps,
            )
        )
        review_artifact_writer.write_summary(state)
    state.item_definition.pop("scenario_assessment_batches", None)
    initial_batches = sum(item.get("initial_batch_count", 0) for item in audits)
    adaptive_splits = sum(item.get("adaptive_split_count", 0) for item in audits)
    leaf_batches = sum(item.get("leaf_batch_count", 0) for item in audits)
    total_calls = sum(item.get("llm_calls", 0) for item in audits)
    retry_calls = sum(item.get("retry_calls", 0) for item in audits)
    total_timeouts = sum(item.get("timeout_count", 0) for item in audits)
    output_limits = sum(item.get("output_limit_count", 0) for item in audits)
    schema_errors = sum(item.get("schema_error_count", 0) for item in audits)
    coverage_errors = sum(item.get("coverage_error_count", 0) for item in audits)
    single_failures = sum(item.get("single_item_failures", 0) for item in audits)
    valid_initial_count = sum(item.get("valid_initial_count", 0) for item in audits)
    invalid_initial_count = sum(item.get("invalid_initial_count", 0) for item in audits)
    valid_items_salvaged = sum(item.get("valid_items_salvaged", 0) for item in audits)
    invalid_items_repaired = sum(item.get("invalid_items_repaired", 0) for item in audits)
    repair_success_count = sum(item.get("repair_success_count", 0) for item in audits)
    repair_failed_count = sum(item.get("repair_failed_count", 0) for item in audits)
    prompt_tokens_total = sum(item.get("prompt_tokens_total", 0) for item in audits)
    completion_tokens_total = sum(item.get("completion_tokens_total", 0) for item in audits)
    missing_ids = []
    unknown_ids = []
    duplicate_ids = []
    adaptive_split_reasons = {}
    for audit in audits:
        for target, key in (
            (missing_ids, "missing_ids"),
            (unknown_ids, "unknown_ids"),
            (duplicate_ids, "duplicate_ids"),
        ):
            for identity in audit.get(key, []):
                if identity not in target:
                    target.append(identity)
        for reason, count in audit.get("adaptive_split_reason", {}).items():
            adaptive_split_reasons[reason] = (
                adaptive_split_reasons.get(reason, 0) + int(count)
            )
    all_chars = [value for item in audits for value in item.get("leaf_input_chars", [])]
    all_items = [value for item in audits for value in item.get("leaf_items", [])]
    completion_tokens = [value for item in audits for value in item.get("completion_tokens", [])]
    reasoning_chars = [value for item in audits for value in item.get("reasoning_characters", [])]
    candidate_evidence_count = sum(
        item.get("candidate_evidence_count", 0) for item in audits
    )
    selected_evidence_count = sum(
        item.get("selected_evidence_count", 0) for item in audits
    )
    causal_prompt_evidence_chars = sum(
        item.get("causal_prompt_evidence_chars", 0) for item in audits
    )
    causal_prompt_evidence_tokens = sum(
        item.get("causal_prompt_evidence_tokens", 0) for item in audits
    )
    dropped_evidence_count = sum(
        item.get("dropped_evidence_count", 0) for item in audits
    )
    mandatory_evidence_refs = []
    selected_context_evidence_refs = []
    total_prompt_evidence_count = sum(
        item.get("total_prompt_evidence_count", 0) for item in audits
    )
    evidence_selection_audit = []
    m_to_b_anchor_refs = []
    m_to_b_anchor_available = True
    evidence_selection_reason = {}
    for audit in audits:
        for target, key in (
            (mandatory_evidence_refs, "mandatory_evidence_refs"),
            (selected_context_evidence_refs, "selected_context_evidence_refs"),
        ):
            for ref in audit.get(key, []):
                if ref not in target:
                    target.append(ref)
        evidence_selection_audit.extend(audit.get("evidence_selection_audit", []))
        m_to_b_anchor_available = m_to_b_anchor_available and bool(
            audit.get("m_to_b_anchor_available", False)
        )
        for ref in audit.get("m_to_b_anchor_refs", []):
            if ref not in m_to_b_anchor_refs:
                m_to_b_anchor_refs.append(ref)
        for reason, count in audit.get("evidence_selection_reason", {}).items():
            evidence_selection_reason[reason] = (
                evidence_selection_reason.get(reason, 0) + int(count)
            )
    transport_counts = {}
    for audit in audits:
        for category, count in audit.get("transport_error_counts", {}).items():
            transport_counts[category] = transport_counts.get(category, 0) + int(count)
    format_metrics = {
        key: sum(item.get(key, 0) for item in audits)
        for key in (
            "json_contract_errors", "format_retry_calls", "format_retry_successes",
            "format_retry_failures", "markdown_fence_normalizations",
        )
    }
    print(
        "[HARA] scenario feasibility summary "
        f"malfunctions={len(malfunctions)} scenario_candidates={len(runtime_candidates)} "
        f"scenario_pairs={len(runtime_candidates)} "
        f"initial_batches={initial_batches} adaptive_splits={adaptive_splits} "
        f"leaf_batches={leaf_batches} llm_calls={total_calls} retry_calls={retry_calls} "
        f"timeouts={total_timeouts} output_limits={output_limits} "
        f"incomplete_reads={transport_counts.get('incomplete_read', 0)} "
        f"remote_disconnects={transport_counts.get('remote_disconnect', 0)} "
        f"connection_resets={transport_counts.get('connection_reset', 0)} "
        f"transport_retries={sum(transport_counts.values())} "
        f"json_contract_errors={format_metrics['json_contract_errors']} "
        f"format_retry_calls={format_metrics['format_retry_calls']} "
        f"max_leaf_items={max(all_items, default=0)} max_leaf_chars={max(all_chars, default=0)}",
        file=sys.stderr,
        flush=True,
    )
    state.record(
        "scenario_feasibility_summary",
        malfunction_count=len(malfunctions),
        scenario_candidate_count=len(runtime_candidates),
        scenario_pair_count=len(runtime_candidates),
        llm_calls=total_calls,
        actual_llm_calls=total_calls,
        retry_calls=retry_calls,
        initial_batch_count=initial_batches,
        adaptive_split_count=adaptive_splits,
        leaf_batch_count=leaf_batches,
        timeout_count=total_timeouts,
        output_limit_count=output_limits,
        schema_error_count=schema_errors,
        coverage_error_count=coverage_errors,
        single_item_failures=single_failures,
        incomplete_reads=transport_counts.get("incomplete_read", 0),
        remote_disconnects=transport_counts.get("remote_disconnect", 0),
        connection_resets=transport_counts.get("connection_reset", 0),
        transport_retries=sum(transport_counts.values()),
        transport_failures=sum(item.get("transport_failures", 0) for item in audits),
        transport_error_counts=transport_counts,
        valid_initial_count=valid_initial_count,
        invalid_initial_count=invalid_initial_count,
        valid_items_salvaged=valid_items_salvaged,
        invalid_items_repaired=invalid_items_repaired,
        repair_success_count=repair_success_count,
        repair_failed_count=repair_failed_count,
        repair_failure_count=repair_failed_count,
        provider_calls_total=total_calls,
        prompt_tokens_total=prompt_tokens_total,
        completion_tokens_total=completion_tokens_total,
        missing_ids=missing_ids,
        unknown_ids=unknown_ids,
        duplicate_ids=duplicate_ids,
        adaptive_split_used=bool(adaptive_splits),
        adaptive_split_reason=dict(sorted(adaptive_split_reasons.items())),
        candidate_evidence_count=candidate_evidence_count,
        selected_evidence_count=selected_evidence_count,
        causal_prompt_evidence_chars=causal_prompt_evidence_chars,
        causal_prompt_evidence_tokens=causal_prompt_evidence_tokens,
        evidence_selection_reason=dict(sorted(evidence_selection_reason.items())),
        dropped_evidence_count=dropped_evidence_count,
        mandatory_evidence_refs=mandatory_evidence_refs,
        selected_context_evidence_refs=selected_context_evidence_refs,
        total_prompt_evidence_count=total_prompt_evidence_count,
        evidence_selection_audit=evidence_selection_audit,
        m_to_b_anchor_available=m_to_b_anchor_available,
        m_to_b_anchor_refs=m_to_b_anchor_refs,
        **format_metrics,
        max_leaf_items=max(all_items, default=0),
        min_leaf_items=min(all_items, default=0),
        average_leaf_items=(round(sum(all_items) / len(all_items), 1) if all_items else 0),
        max_leaf_chars=max(all_chars, default=0),
        min_leaf_chars=min(all_chars, default=0),
        average_leaf_chars=(round(sum(all_chars) / len(all_chars), 1) if all_chars else 0),
        max_completion_tokens=max(completion_tokens, default=0),
        average_completion_tokens=(
            round(sum(completion_tokens) / len(completion_tokens), 1) if completion_tokens else 0
        ),
        max_reasoning_characters=max(reasoning_chars, default=0),
        average_reasoning_characters=(
            round(sum(reasoning_chars) / len(reasoning_chars), 1) if reasoning_chars else 0
        ),
    )
    return state
