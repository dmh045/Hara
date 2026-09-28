from __future__ import annotations

import threading
import sys
import time
from dataclasses import asdict, replace
from typing import Sequence

from hara_agent.contracts import RequiredFactSpec
from hara_agent.infrastructure.llm import LLMResponse
from hara_agent.models import (
    FunctionDefinition, ItemDefinitionFacts, ReviewStatus, SourceRef,
)
from hara_agent.services.extraction import ValidatedArtifactCache
from hara_agent.services.extraction import build_project_fact_spec_batches
from hara_agent.services.semantic import (
    ItemArtifactExtractionAgent,
    ItemEvidenceRouter,
    ItemSupplementAgent,
    TargetedProjectFactExtractionAgent,
)
from hara_agent.services.semantic.item_definition_agent import ItemDefinitionNormalizer
from hara_agent.services.semantic.function_source_guard import (
    validate_function_source_parity,
)
from hara_agent.workflow.state import HARAState, WorkflowStage
from hara_agent.workflow.review_artifacts import ReviewArtifactWriter

from .parallel import ordered_parallel_map


ARTIFACT_SCHEMA_VERSION = "validated-item-artifact-v9-source-selection"


def _speed_extraction_modes(
    requested_operating_modes: Sequence[str],
    extracted_operating_modes: Sequence[str],
) -> tuple[str, ...]:
    """Extract speed only for the caller's modes when they are explicit.

    Declared Item modes remain the fallback for workflows that do not request a
    particular context.  Mixing both sets made a single production request ask
    for every state-machine mode and allowed unrelated NOT_FOUND results to
    obscure the one speed envelope needed by downstream preflight.
    """

    requested = tuple(
        str(mode).strip() for mode in requested_operating_modes
        if str(mode).strip()
    )
    if requested:
        return requested
    return tuple(
        str(mode).strip() for mode in extracted_operating_modes
        if str(mode).strip()
    )


def _facts_from_dict(value: dict) -> ItemDefinitionFacts:
    return ItemDefinitionFacts.from_dict(value)


def _function_from_dict(value: dict) -> FunctionDefinition:
    payload = dict(value)
    payload["sources"] = [SourceRef(**item) for item in payload.get("sources", [])]
    payload["status"] = ReviewStatus(payload.get("status", ReviewStatus.PENDING.value))
    return FunctionDefinition(**payload)


def _missing_core_fields(facts: ItemDefinitionFacts) -> list[str]:
    missing = []
    if not facts.operating_modes:
        missing.append("operating_modes")
    if not facts.odd_locations:
        missing.append("odd.locations")
    if not facts.odd_road_types:
        missing.append("odd.road_types")
    if not facts.odd_weather_conditions:
        missing.append("odd.weather_conditions")
    if not facts.odd_road_surfaces:
        missing.append("odd.road_surfaces")
    if facts.speed_min_kph is None:
        missing.append("odd.speed_range_kph")
    return missing


def _merge_supplements(
    facts: ItemDefinitionFacts, patches: dict[str, dict],
) -> tuple[ItemDefinitionFacts, list[str]]:
    warnings = []
    values = {
        "operating_modes": list(facts.operating_modes),
        "odd_locations": list(facts.odd_locations),
        "odd_road_types": list(facts.odd_road_types),
        "odd_weather_conditions": list(facts.odd_weather_conditions),
        "odd_road_surfaces": list(facts.odd_road_surfaces),
        "speed_min_kph": facts.speed_min_kph,
        "speed_max_kph": facts.speed_max_kph,
        "speed_envelopes": list(facts.speed_envelopes),
        "risk_facts": list(facts.risk_facts),
        "method_risk_fact_bindings": list(facts.method_risk_fact_bindings),
    }
    odd_patch = patches.get("odd_repair", {})
    odd = odd_patch.get("odd", {}) if isinstance(odd_patch, dict) else {}
    if not isinstance(odd, dict):
        odd = {}
        warnings.append("局部ODD补抽取返回的odd不是object，已忽略")
    list_mappings = (
        ("operating_modes", odd_patch.get("operating_modes") if isinstance(odd_patch, dict) else None),
        ("odd_locations", odd.get("locations")),
        ("odd_road_types", odd.get("road_types")),
        ("odd_weather_conditions", odd.get("weather_conditions")),
        ("odd_road_surfaces", odd.get("road_surfaces")),
    )
    for field, candidate in list_mappings:
        if not values[field] and candidate is not None:
            values[field] = [str(item) for item in ItemDefinitionNormalizer._list(candidate, field)]
    if values["speed_min_kph"] is None and "speed_range_kph" in odd:
        speed_min, speed_max, warning = ItemDefinitionNormalizer._speed_range(
            odd.get("speed_range_kph")
        )
        values["speed_min_kph"], values["speed_max_kph"] = speed_min, speed_max
        if warning:
            warnings.append(warning)

    return replace(facts, **values), warnings


def _merge_targeted_project_facts(
    facts: ItemDefinitionFacts, payloads: dict[str, dict],
) -> ItemDefinitionFacts:
    serialized = asdict(facts)
    for payload in payloads.values():
        for field in (
            "speed_envelopes", "risk_facts",
        ):
            for item in payload.get(field, []):
                if item not in serialized[field]:
                    serialized[field].append(item)
    return ItemDefinitionFacts.from_dict(serialized)


def extract_item_artifacts(
    state: HARAState,
    artifact_agent: ItemArtifactExtractionAgent,
    supplement_agent: ItemSupplementAgent,
    router: ItemEvidenceRouter,
    *,
    targeted_agent: TargetedProjectFactExtractionAgent | None = None,
    max_workers: int = 2,
    progress=None,
    cache: ValidatedArtifactCache | None = None,
    required_fact_specs: Sequence[RequiredFactSpec] = (),
    requested_operating_modes: Sequence[str] = (),
    review_artifact_writer: ReviewArtifactWriter | None = None,
) -> HARAState:
    """Run one full-document extraction, then only targeted repair calls."""
    stage_started = time.monotonic()
    cache_save_elapsed = 0.0
    document_text = str(state.item_definition.get("text", ""))
    source_id = str(state.item_definition.get("source_id", ""))
    blocks = list(state.item_definition.get("blocks", []))
    source_selection = dict(state.item_definition.get("source_selection", {}))
    client_config = getattr(artifact_agent.client, "config", None)
    cache_material = {
        "document_text": document_text,
        "source_id": source_id,
        "source_selection_fingerprint": source_selection.get("fingerprint", ""),
        "source_selection_policy_id": source_selection.get("policy_id", ""),
        "source_selection_policy_version": source_selection.get("policy_version", ""),
        "project_policy_id": source_selection.get("project_policy_id", ""),
        "project_policy_version": source_selection.get("project_policy_version", ""),
        "main_prompt_version": artifact_agent.PROMPT_VERSION,
        "supplement_prompt_version": supplement_agent.PROMPT_VERSION,
        "targeted_prompt_version": (
            targeted_agent.PROMPT_VERSION if targeted_agent else "disabled"
        ),
        "router_version": "item-evidence-router-v2-schema-guided",
        "artifact_schema_version": ARTIFACT_SCHEMA_VERSION,
        "provider": getattr(client_config, "provider", "custom-client"),
        "base_url": getattr(client_config, "base_url", ""),
        "model": getattr(client_config, "model", type(artifact_agent.client).__name__),
        "extraction_thinking": getattr(client_config, "extraction_thinking", "default"),
        "required_project_fact_specs": [
            {
                "fact_type": spec.fact_type.value,
                "required_for": list(spec.required_for),
                "unit": spec.unit,
                "constraints": list(spec.constraints),
                "condition": spec.condition,
                "origin": spec.origin.value,
            }
            for spec in required_fact_specs
        ],
        "requested_operating_modes": [
            str(mode) for mode in requested_operating_modes if str(mode).strip()
        ],
    }
    candidate_material = {
        "document_text": document_text,
        "source_id": source_id,
        "source_selection_fingerprint": source_selection.get("fingerprint", ""),
        "source_selection_policy_id": source_selection.get("policy_id", ""),
        "source_selection_policy_version": source_selection.get("policy_version", ""),
        "project_policy_id": source_selection.get("project_policy_id", ""),
        "project_policy_version": source_selection.get("project_policy_version", ""),
        "main_prompt_version": artifact_agent.PROMPT_VERSION,
        "provider": cache_material["provider"],
        "base_url": cache_material["base_url"],
        "model": cache_material["model"],
        "extraction_thinking": cache_material["extraction_thinking"],
    }
    cache_key = cache.key(cache_material) if cache else ""
    candidate_key = cache.key(candidate_material) if cache else ""
    cache_lookup_started = time.monotonic()
    if cache and cache_key:
        cached_payload, cache_status = cache.load_with_status(cache_key)
    else:
        cached_payload, cache_status = None, "cache_not_configured"
    cache_lookup_elapsed = time.monotonic() - cache_lookup_started
    if cached_payload is None:
        print(
            f"[HARA] artifact cache miss key={cache_key or 'none'} reason={cache_status}",
            file=sys.stderr,
            flush=True,
        )
    else:
        print(
            f"[HARA] artifact cache hit key={cache_key}",
            file=sys.stderr,
            flush=True,
        )
    core_cached = False
    cached_core = cached_payload.get("core") if isinstance(cached_payload, dict) else None
    if isinstance(cached_core, dict):
        try:
            facts = _facts_from_dict(cached_core["facts"])
            functions = [_function_from_dict(item) for item in cached_core["functions"]]
            artifact_agent.validator.ensure_valid(functions)
            function_source_guard = validate_function_source_parity(
                functions, blocks, source_id=source_id,
            )
            artifact_agent._ensure_source_grounded(
                facts, functions, document_text, blocks,
            )
            core_cached = True
            main_audit = {
                "task": "extract_core_item_artifacts",
                "prompt_version": artifact_agent.PROMPT_VERSION,
                "model": cache_material["model"],
                "cache_hit": True,
                "cache_key": cache_key,
                "elapsed_seconds": 0.0,
                "llm_call_count": 0,
                "input_characters": len(document_text),
                "function_source_guard": function_source_guard,
            }
        except (KeyError, TypeError, ValueError):
            core_cached = False
            print(
                f"[HARA] artifact cache miss key={cache_key} "
                "reason=invalid_cached_artifact",
                file=sys.stderr,
                flush=True,
            )
    core_started = time.monotonic()
    if not core_cached:
        candidate_response = None
        candidate_status = "candidate_cache_not_configured"
        if cache and candidate_key:
            candidate_payload, candidate_status = cache.load_candidate_with_status(
                candidate_key
            )
            if isinstance(candidate_payload, dict):
                try:
                    candidate_data = candidate_payload["data"]
                    if not isinstance(candidate_data, dict):
                        raise TypeError("candidate data is not an object")
                    candidate_response = LLMResponse(
                        data=candidate_data,
                        model=str(candidate_payload.get("model", cache_material["model"])),
                        request_id=str(candidate_payload.get("request_id", "")),
                        usage=dict(candidate_payload.get("usage", {}) or {}),
                    )
                    print(
                        f"[HARA] core candidate cache hit key={candidate_key}",
                        file=sys.stderr,
                        flush=True,
                    )
                except (KeyError, TypeError, ValueError):
                    candidate_response = None
                    candidate_status = "candidate_payload_invalid"
                    print(
                        f"[HARA] core candidate cache ignored key={candidate_key} "
                        f"reason={candidate_status}",
                        file=sys.stderr,
                        flush=True,
                    )

        def record_candidate(response: LLMResponse) -> None:
            if cache and candidate_key:
                cache.save_candidate(candidate_key, {
                    "schema_version": "raw-core-item-response-v1-quarantine",
                    "data": response.data,
                    "model": response.model,
                    "request_id": response.request_id,
                    "usage": response.usage,
                })

        facts, functions, main_audit = artifact_agent.extract(
            document_text,
            source_id,
            blocks,
            candidate_response=candidate_response,
            candidate_recorder=record_candidate,
        )
        main_audit.update({
            "cache_hit": False,
            "cache_key": cache_key,
            "candidate_cache_key": candidate_key,
            "candidate_cache_status": candidate_status,
        })
        cached_payload = {
            "schema_version": ARTIFACT_SCHEMA_VERSION,
            "core": {
                "facts": asdict(facts),
                "functions": [asdict(item) for item in functions],
            },
            "supplements": {},
            "targeted_project_facts": {},
        }
        if cache and cache_key:
            save_started = time.monotonic()
            cache.save(cache_key, cached_payload)
            cache_save_elapsed += time.monotonic() - save_started
    core_elapsed = time.monotonic() - core_started
    if progress:
        progress("item_artifacts_main", 1, 1)

    routing_started = time.monotonic()
    missing_before = _missing_core_fields(facts)
    routed = []
    routing_diagnostics = []
    cached_supplements = (
        dict(cached_payload.get("supplements", {}))
        if isinstance(cached_payload, dict) else {}
    )
    if missing_before and "odd_repair" not in cached_supplements:
        odd_result = router.retrieve(blocks, "odd_repair")
        routing_diagnostics.append(odd_result.diagnostics.to_dict())
        if odd_result.routed:
            routed.append(odd_result.routed)
    routing_elapsed = time.monotonic() - routing_started

    patches, supplement_audits = dict(cached_supplements), []
    supplement_wall_started = time.monotonic()
    if routed:
        cache_lock = threading.Lock()

        def extract_and_cache(item):
            result = supplement_agent.extract(item, source_id)
            if cache and cache_key and isinstance(cached_payload, dict):
                with cache_lock:
                    cached_payload.setdefault("supplements", {})[item.task] = result[0]
                    save_started = time.monotonic()
                    cache.save(cache_key, cached_payload)
                    nonlocal_cache_save[0] += time.monotonic() - save_started
            return result

        nonlocal_cache_save = [0.0]
        results = ordered_parallel_map(
            routed,
            extract_and_cache,
            max_workers=min(max_workers, 2),
            on_progress=(
                lambda done, total: progress("item_artifacts_supplement", done, total)
            ) if progress else None,
        )
        for route, (patch, audit) in zip(routed, results):
            patches[route.task] = patch
            supplement_audits.append(audit)
        cache_save_elapsed += nonlocal_cache_save[0]
    supplement_wall_elapsed = time.monotonic() - supplement_wall_started
    merge_started = time.monotonic()
    if patches:
        facts, merge_warnings = _merge_supplements(facts, patches)
    else:
        merge_warnings = []
    merge_elapsed = time.monotonic() - merge_started

    project_fact_batches = build_project_fact_spec_batches(
        _speed_extraction_modes(requested_operating_modes, facts.operating_modes),
        required_fact_specs,
    )
    targeted_audits = []
    targeted_payloads = (
        dict(cached_payload.get("targeted_project_facts", {}))
        if isinstance(cached_payload, dict) else {}
    )
    if targeted_agent is not None:
        targeted_routes = []
        for category, specs in project_fact_batches.items():
            if category in targeted_payloads:
                continue
            route_result = router.retrieve(
                blocks,
                "project_evidence",
                required_specs=tuple(spec.retrieval_spec() for spec in specs),
            )
            routing_diagnostics.append(route_result.diagnostics.to_dict())
            if route_result.routed is not None:
                targeted_routes.append((category, specs, route_result.routed))
        if targeted_routes:
            targeted_lock = threading.Lock()
            targeted_save_elapsed = [0.0]

            def extract_targeted(item):
                category, specs, route = item
                normalized, audit = targeted_agent.extract(
                    category, specs, route, source_id
                )
                payload = normalized.to_cache_dict()
                if cache and cache_key and isinstance(cached_payload, dict):
                    with targeted_lock:
                        cached_payload.setdefault("targeted_project_facts", {})[category] = payload
                        save_started = time.monotonic()
                        cache.save(cache_key, cached_payload)
                        targeted_save_elapsed[0] += time.monotonic() - save_started
                return category, payload, audit

            targeted_results = ordered_parallel_map(
                targeted_routes,
                extract_targeted,
                max_workers=min(max_workers, 3),
                on_progress=(
                    lambda done, total: progress("targeted_project_facts", done, total)
                ) if progress else None,
            )
            for category, payload, audit in targeted_results:
                targeted_payloads[category] = payload
                targeted_audits.append(audit)
            cache_save_elapsed += targeted_save_elapsed[0]
        facts = _merge_targeted_project_facts(facts, targeted_payloads)
    # Approved input is accepted after bounded normalization.  Missing core
    # facts and degraded repairs remain fail-closed at the specific boundary.
    facts.status = (
        ReviewStatus.FINALIZED
        if not _missing_core_fields(facts) and not merge_warnings
        else ReviewStatus.PENDING
    )
    total_elapsed = time.monotonic() - stage_started
    supplement_elapsed = {
        audit["task"].removeprefix("supplement_"): audit.get("elapsed_seconds", 0.0)
        for audit in supplement_audits
    }
    llm_calls = int(main_audit.get("llm_call_count", 0)) + sum(
        int(audit.get("llm_call_count", 0)) for audit in supplement_audits
    ) + sum(int(audit.get("llm_call_count", 0)) for audit in targeted_audits)
    llm_elapsed = float(main_audit.get("elapsed_seconds", 0.0)) + sum(
        float(audit.get("elapsed_seconds", 0.0)) for audit in supplement_audits
    ) + sum(float(audit.get("elapsed_seconds", 0.0)) for audit in targeted_audits)
    transient_error_counts: dict[str, int] = {}
    for audit in [main_audit, *supplement_audits, *targeted_audits]:
        usage = audit.get("usage", {})
        counts = usage.get("transient_error_counts", {}) if isinstance(usage, dict) else {}
        if isinstance(counts, dict):
            for error_type, count in counts.items():
                transient_error_counts[str(error_type)] = (
                    transient_error_counts.get(str(error_type), 0) + int(count)
                )
    local_elapsed = max(0.0, total_elapsed - core_elapsed - supplement_wall_elapsed)
    print(
        "[HARA] extract timing "
        f"core_item_artifact={core_elapsed:.1f}s evidence_routing={routing_elapsed:.1f}s "
        f"supplement_odd_repair={supplement_elapsed.get('odd_repair', 0.0):.1f}s "
        f"supplement_parallel_wall={supplement_wall_elapsed:.1f}s "
        f"merge_validate={merge_elapsed:.1f}s cache_lookup={cache_lookup_elapsed:.1f}s "
        f"cache_save={cache_save_elapsed:.1f}s total={total_elapsed:.1f}s "
        f"llm_calls={llm_calls} transient_errors={transient_error_counts}",
        file=sys.stderr,
        flush=True,
    )

    state.item_definition["typed"] = asdict(facts)
    state.functions = [asdict(item) for item in functions]
    if review_artifact_writer is not None:
        for function in functions:
            review_artifact_writer.record_function(function)
    state.record(
        "item_artifacts_extracted",
        **main_audit,
        missing_core_fields_before_supplement=missing_before,
        missing_core_fields_after_supplement=_missing_core_fields(facts),
        supplement_task_count=len(routed),
        routing_diagnostics=routing_diagnostics,
        merge_warnings=merge_warnings,
    )
    for audit in supplement_audits:
        state.record("item_artifact_supplement_extracted", **audit)
    for audit in targeted_audits:
        state.record("targeted_project_facts_extracted", **audit)
    state.record(
        "extract_timing",
        elapsed_seconds=round(total_elapsed, 3),
        llm_calls=llm_calls,
        llm_elapsed_seconds=round(llm_elapsed, 3),
        local_elapsed_seconds=round(local_elapsed, 3),
        core_item_artifact_seconds=round(core_elapsed, 3),
        evidence_routing_seconds=round(routing_elapsed, 3),
        supplement_task_elapsed_seconds=supplement_elapsed,
        supplement_parallel_wall_seconds=round(supplement_wall_elapsed, 3),
        merge_validate_seconds=round(merge_elapsed, 3),
        cache_lookup_seconds=round(cache_lookup_elapsed, 3),
        cache_save_seconds=round(cache_save_elapsed, 3),
        transient_error_counts=transient_error_counts,
        cache_status=cache_status,
        cache_hit=core_cached,
        cache_key=cache_key,
    )

    if facts.status is ReviewStatus.PENDING:
        state.pending_reviews.append({
            "field": "item_definition",
            "reason": (
                "Approved Item Definition could not be normalized completely; "
                f"missing={_missing_core_fields(facts)} warnings={merge_warnings}"
            ),
        })
    pending_functions = [
        item for item in functions if item.status is ReviewStatus.PENDING
    ]
    if pending_functions:
        state.pending_reviews.append({
            "field": "functions",
            "reason": (
                f"{len(pending_functions)} Function candidates did not pass "
                "schema/role/source validation"
            ),
        })
    state.stage = WorkflowStage.FUNCTIONS
    return state
