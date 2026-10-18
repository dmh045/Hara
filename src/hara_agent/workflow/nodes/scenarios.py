from __future__ import annotations

from dataclasses import asdict, replace
from copy import deepcopy
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
from hara_agent.services.analysis import RiskExecutionTraceService, MethodScenarioCandidateService
from hara_agent.services.analysis.malfunction_situation_selection import selection_function
from hara_agent.workflow.state import HARAState, WorkflowStage

from .parallel import ordered_parallel_map
from ..review_artifacts import ReviewArtifactWriter
from hara_agent.services.analysis.scenario_equivalence import (
    driver_families, driver_reuse_proof, exact_task_key, rebind_assessment_payload,
)


SCENARIO_BATCH_CHECKPOINT_VERSION = "scenario-malfunction-batch-v10-exact-local-reuse"


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
        "method_execution_identity": getattr(agent, "method_execution_identity", {}),
        "causal_evidence_budget": getattr(
            agent, "causal_evidence_budget", DEFAULT_CAUSAL_EVIDENCE_BUDGET,
        ),
        "provider": getattr(config, "provider", type(agent.client).__name__),
        "base_url": getattr(config, "base_url", ""),
        "model": getattr(config, "model", type(agent.client).__name__),
        "response_format": getattr(agent, "response_format", None),
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
    prior_batch=None,
) -> tuple[list[ScenarioFeasibilityAssessment], dict]:
    """Deduplicate before batching; parse every reused result in the child world."""
    exact_groups = {}
    for candidate in candidates:
        key = exact_task_key(agent, malfunction, candidate, project_registry)
        exact_groups.setdefault(key, []).append(candidate)
    unique = [members[0] for members in exact_groups.values()]
    restored, cache_records = {}, []
    prior = prior_batch or {}
    old_assessments = {row["scenario_id"]: row for row in prior.get("assessments", [])}
    old_tasks = {row["task_key"]: row for row in prior.get("audit", {}).get("causal_reuse", {}).get("exact_mapping", [])}
    for key, members in exact_groups.items():
        old = old_tasks.get(key)
        if not old:
            continue
        current = members[0]
        old_ids = [identity for identity in old["scenario_ids"] if identity in old_assessments]
        payloads = []
        for identity in old_ids:
            context = old.get("identity_contexts", {}).get(identity, {})
            parent = replace(current, scenario_id=identity, facts={**current.facts,
                "object_id": context.get("object_id", identity + ":target")})
            payloads.append(rebind_assessment_payload(old_assessments[identity], parent, current))
        if not payloads:
            continue
        from hara_agent.services.analysis.scenario_equivalence import signature
        if len({signature(payload) for payload in payloads}) != 1:
            cache_records.append({"scenario_id": current.scenario_id, "kind": "exact_input",
                "reused": False, "reason": "HISTORICAL_EXACT_RESULT_CONFLICT", "task_key": key})
            continue
        try:
            restored[current.scenario_id] = ScenarioFeasibilityAgent._parse(
                malfunction, payloads[0], scenario=current, project_registry=project_registry)
            cache_records.append({"scenario_id": current.scenario_id, "kind": "exact_input",
                "reused": True, "child_revalidated": True, "task_key": key,
                "representative_scenario_id": old_ids[0], "reason": "EXACT_TASK_CHECKPOINT_HIT"})
        except ValueError as error:
            cache_records.append({"scenario_id": current.scenario_id, "kind": "exact_input",
                "reused": False, "reason": "CACHED_CHILD_EVIDENCE_REVALIDATION_REQUIRED", "detail": str(error)})
    families = driver_families(unique) if any(
        candidate.analysis_instance.get("p6_generation") for candidate in unique
    ) else [[candidate] for candidate in unique]
    representatives = [family[0] for family in families]
    first_wave = []
    prebatched = set()
    for family in families:
        if family[0].scenario_id in restored:
            continue
        # If an unproven driver pair already fits one request, speculatively
        # splitting it cannot save an HTTP call and can double cost. Keep the
        # pair together; still evaluate its proof for later members/audit.
        from hara_agent.services.semantic.scenario_batching import estimate_scenario_prompt_chars
        together = (len(family) > 1 and family[1].scenario_id not in restored
                    and getattr(agent, "paired_driver_requests", False)
                    and agent.batch_max_items >= 2
                    and estimate_scenario_prompt_chars(malfunction, family[:2],
                        system_prompt=agent.system_prompt + "\n" + agent.MACHINE_OUTPUT_RULE,
                        project_registry=project_registry, causal_evidence_budget=agent.causal_evidence_budget)
                    <= agent.batch_max_chars)
        first_wave.extend(family[:2] if together else family[:1])
        if together:
            prebatched.add(family[1].scenario_id)

    def assess_batch(items):
        # A local copy retains paired transmission without mutating a shared agent.
        from copy import copy
        local = copy(agent) if getattr(agent, "paired_driver_requests", False) else agent
        if getattr(local, "paired_driver_requests", False):
            local.batch_max_items = min(local.batch_max_items, 2)
        return local.assess(malfunction, items, project_registry=project_registry)

    assessments, audit = assess_batch(first_wave) if first_wave else ([], {
        "task": "assess_scenario_feasibility", "malfunction_id": malfunction.malfunction_id,
        "actual_llm_calls": 0, "llm_calls": 0, "usage": {"batches": []}})
    by_id = {**restored, **{item.scenario_id: item for item in assessments}}
    pending, reuse_records, partial_contexts = [], list(cache_records), {}

    def revalidate(assessed, parent, child, changed=()):
        payload = rebind_assessment_payload(assessed.to_dict(), parent, child, changed)
        return ScenarioFeasibilityAgent._parse(
            malfunction, payload, scenario=child, project_registry=project_registry)

    def shared_prefix(assessed, parent, child):
        from hara_agent.services.semantic.scenario_evidence import build_fact_registry
        first = build_fact_registry(malfunction, parent, project_registry)
        second = build_fact_registry(malfunction, child, project_registry)
        # Only the upstream malfunction-to-behavior step: interaction is never
        # inherited just because two parameter points happen to have equal grades.
        hop = assessed.causal_chain.get("m_to_b", {})
        refs = hop.get("evidence_refs", [])
        if (hop.get("basis_type") != "ASSUMPTION" and refs
                and all(ref.startswith("MF.") and first.resolve(ref) == second.resolve(ref)
                        and first.resolve(ref) is not None for ref in refs)):
            return {"reused_from": parent.scenario_id, "m_to_b": deepcopy(hop)}
        return {}

    for family in families:
        parent = family[0]
        assessed = by_id[parent.scenario_id]
        for child in family[1:]:
            if child.scenario_id in restored:
                continue
            proof = driver_reuse_proof(assessed.to_dict(), parent, child)
            if child.scenario_id in prebatched:
                reuse_records.append({"scenario_id": child.scenario_id,
                    "representative_scenario_id": parent.scenario_id, "kind": "complete_causal",
                    **proof, "reused": False, "dispatch": "COBATCHED_INDEPENDENT_JUDGMENT"})
                continue
            reused = None
            if proof["reusable"]:
                try:
                    reused = revalidate(assessed, parent, child, proof.get("changed_fields", []))
                except ValueError as error:
                    proof = {"reusable": False, "reason": "CHILD_EVIDENCE_REVALIDATION_REQUIRED", "detail": str(error)}
            if reused is None:
                shared = shared_prefix(assessed, parent, child)
                if shared:
                    partial_contexts[child.scenario_id] = shared
                    child = replace(child, analysis_instance={**child.analysis_instance,
                                    "shared_causal_evidence": shared})
                pending.append(child)
            else:
                by_id[child.scenario_id] = reused
            reuse_records.append({"scenario_id": child.scenario_id,
                "representative_scenario_id": parent.scenario_id,
                "kind": "complete_causal", **proof, "reused": reused is not None,
                "child_revalidated": reused is not None})
    independent_count = len(first_wave) + len(pending)
    if pending:
        additional, additional_audit = assess_batch(pending)
        by_id.update({item.scenario_id: item for item in additional})
        _merge_scenario_audits(audit, additional_audit)

    # Exact tasks can reuse retained, excluded or Pending diagnoses. The compiler
    # determines child status again; complete=false is not an exact-task blocker.
    exact_fallback = []
    for key, members in exact_groups.items():
        parent = members[0]
        for child in members[1:]:
            record = {"scenario_id": child.scenario_id,
                "representative_scenario_id": parent.scenario_id, "task_key": key,
                "kind": "exact_input", "reason": "IDENTICAL_EFFECTIVE_TASK",
                "reused": False, "child_revalidated": False}
            try:
                by_id[child.scenario_id] = revalidate(by_id[parent.scenario_id], parent, child)
                record.update(reused=True, child_revalidated=True)
            except ValueError as error:
                record.update(reason="CHILD_EVIDENCE_REVALIDATION_REQUIRED", detail=str(error))
                exact_fallback.append(child)
            reuse_records.append(record)
    if exact_fallback:
        additional, additional_audit = assess_batch(exact_fallback)
        by_id.update({item.scenario_id: item for item in additional})
        _merge_scenario_audits(audit, additional_audit)
        independent_count += len(exact_fallback)
    partial_records = {record["scenario_id"]: record
                       for record in audit.get("partial_evidence_reuse_records", [])}
    # Non-Provider test agents may expose only compiled results, not batch audit.
    for identity, shared in partial_contexts.items():
        partial_records.setdefault(identity, {"scenario_id": identity,
            "representative_scenario_id": shared["reused_from"], "kind": "partial_causal",
            "shared_steps": ["m_to_b"], "remaining_steps": "independently validated",
            "reused": by_id[identity].causal_chain.get("m_to_b") == shared["m_to_b"]})
    reuse_records.extend(partial_records.values())
    assessments = [by_id[candidate.scenario_id] for candidate in candidates]
    audit.update(candidate_count=len(candidates), expected_count=len(candidates),
                 returned_count=len(assessments), retained_count=sum(item.retain for item in assessments))
    counts = {kind: sum(r["kind"] == kind and r["reused"] for r in reuse_records)
              for kind in ("exact_input", "complete_causal", "partial_causal")}
    audit["causal_reuse"] = {
        "mode": "EXACT_THEN_REPRESENTATIVE_AND_SHARED_EVIDENCE",
        "candidate_count": len(candidates), "unique_exact_inputs": len(unique),
        "representative_count": len(representatives),
        "co_batched_independent_members": len(prebatched),
        "independently_assessed_count": independent_count,
        "reused_count": counts["exact_input"] + counts["complete_causal"],
        "exact_input_reuse": counts["exact_input"],
        "complete_causal_reuse": counts["complete_causal"],
        "partial_causal_reuse": counts["partial_causal"],
        "partial_context_offered": len(partial_records),
        "common_evidence_batches": audit.get("shared_context_batch_count", 0),
        "requests": audit.get("actual_llm_calls", 0),
        "response_checkpoint_hits": audit.get("response_checkpoint_hits", 0),
        "exact_task_checkpoint_hits": len(restored),
        "records": reuse_records,
        "exact_mapping": [{"task_key": key, "representative_scenario_id": members[0].scenario_id,
                           "scenario_ids": [member.scenario_id for member in members],
                           "identity_contexts": {member.scenario_id: {"object_id": member.facts.get("object_id", member.scenario_id + ":target")}
                                                 for member in members}}
                          for key, members in exact_groups.items()],
    }
    audit["analytical_scenario_instantiation"] = instantiation
    return assessments, audit


def _merge_scenario_audits(audit, additional):
    """Account every actual request/repair/cache event across scheduling waves."""
    for key, value in additional.items():
        previous = audit.get(key)
        if previous is None:
            audit[key] = deepcopy(value)
        elif isinstance(value, list) and isinstance(previous, list):
            audit[key] = previous + value
        elif isinstance(value, (int, float)) and not isinstance(value, bool) and isinstance(previous, (int, float)):
            audit[key] = max(previous, value) if key.startswith("max_") else previous + value
        elif key in {"error_counts_by_code", "transport_error_counts", "repair_failure_by_code", "evidence_selection_reason"}:
            audit[key] = {code: previous.get(code, 0) + value.get(code, 0) for code in previous.keys() | value.keys()}
    audit["usage"] = {"batches": [*audit.get("usage", {}).get("batches", []),
                                 *additional.get("usage", {}).get("batches", [])]}


def assess_scenarios(state: HARAState, agent: ScenarioFeasibilityAgent,
                     malfunctions: list[MalfunctionCandidate],
                     candidates: list[ScenarioCandidate],
                     max_workers: int = 1,
                     progress=None,
                     risk_fact_agent: ScenarioRiskFactAgent | None = None,
                     physical_agent=None,
                     selected_generated_scenario_ids=(),
                     required_fact_specs: Sequence[RequiredFactSpec] = (),
                     method: MethodContract | None = None,
                     checkpoint: Callable[[HARAState], object] | None = None,
                     review_artifact_writer: ReviewArtifactWriter | None = None) -> HARAState:
    assessments = []
    agent.method_execution_identity = ({
        "compiler_version": method.compiler_version,
        "contract_version": method.contract_version,
        "method_hash": method.metadata.get("method_source_hash"),
        "policy": method.metadata.get("project_analysis_policy", {}),
    } if method is not None else {})
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
    template_service = MethodScenarioCandidateService(method) if method is not None else None
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
            instantiated, audit = template_service.generate_for_malfunction(
                malfunction, candidates, project_facts=project_facts,
                function=selection_function(next(
                    (item for item in state.functions if item.get("function_id") == malfunction.function_id), None,
                )),
                record_generated=(lambda candidate: review_artifact_writer.record_scenario_candidate(
                    candidate, generated_for_malfunction_ids=[malfunction.malfunction_id],
                )) if review_artifact_writer is not None else None,
            )
            contextual_candidates[malfunction.malfunction_id] = instantiated
            instantiation_audit[malfunction.malfunction_id] = audit
            if review_artifact_writer is not None:
                for candidate in instantiated:
                    review_artifact_writer.record_scenario_candidate(
                        candidate,
                        generated_for_malfunction_ids=[malfunction.malfunction_id],
                    )
    if selected_generated_scenario_ids:
        requested = set(selected_generated_scenario_ids)
        available = {candidate.scenario_id for values in contextual_candidates.values() for candidate in values}
        if requested - available:
            raise ValueError(f"Selected generated scenario IDs unavailable: {sorted(requested - available)}")
        for identity, values in contextual_candidates.items():
            selected = [candidate for candidate in values if candidate.scenario_id in requested]
            contextual_candidates[identity] = selected
            instantiation_audit[identity]["bounded_generated_selection"] = {
                "available": len(values), "selected_ids": [candidate.scenario_id for candidate in selected]}
        state.record("bounded_generated_scenarios_selected", selected_ids=sorted(requested), available_count=len(available))
    if physical_agent is not None and method is not None and method.structured_risk_method is not None:
        from hara_agent.workflow.physical_analysis import construct_physical_scenarios
        contextual_candidates = construct_physical_scenarios(state, contextual_candidates, malfunctions,
            method=method, agent=physical_agent, project_facts=project_facts,
            checkpoint=checkpoint)
        for malfunction_id, generated in contextual_candidates.items():
            instantiation_audit[malfunction_id]["physical_analysis"] = {
                "instance_count": len(generated), "stage": "BEFORE_CAUSAL_ASSESSMENT"}
            if review_artifact_writer is not None:
                for candidate in generated:
                    review_artifact_writer.record_scenario_candidate(candidate,
                        generated_for_malfunction_ids=[malfunction_id])
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
            prior_batch=cache.get(malfunction.malfunction_id),
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
