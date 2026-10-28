"""Explicit pre-causal scheduling for bounded physical analysis proposals."""
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
import sys
from threading import Lock
from time import perf_counter

from hara_agent.contracts.physical_scenario_plan import PhysicalScenarioPlan, PHYSICAL_PLAN_VERSION, MODEL_CATALOG, braking_capability
from hara_agent.services.analysis.physical_hypothesis_instantiation_service import PhysicalHypothesisInstantiationService
from hara_agent.services.analysis.scenario_equivalence import scenario_conditions, signature
from hara_agent.services.semantic.physical_scenario_agent import PhysicalScenarioAgent
from hara_agent.services.extraction.evidence_retrieval import (
    CoverageFirstContextAssembler, DeterministicEvidenceRetriever, FactRetrievalSpec,
)


def _source_evidence(state, function):
    """Route bounded original passages, never infer a control or interaction."""
    specs = [
        FactRetrievalSpec("intervention", ("制动踏板", "接管", "驾驶员干预", "remote", "brake pedal", "override")),
        FactRetrievalSpec("interfaces", ("IPB", "EPS", "VDC", "EPB", "interface", "接口")),
        FactRetrievalSpec("participants", ("行人", "两轮车", "车辆", "pedestrian", "traffic")),
    ]
    terms = tuple(str(function[k]) for k in ("name", "output") if function.get(k))
    if terms:
        specs.append(FactRetrievalSpec("function", terms))
    blocks = state.item_definition.get("blocks", [])
    context = CoverageFirstContextAssembler().assemble(blocks, specs,
        DeterministicEvidenceRetriever().rank(blocks, specs), task="physical_analysis", max_characters=4000)
    evidence = {"ITEM." + block["block_id"]: {"value": block["text"], "sources": [{
        "source_type": "item_definition", "source_id": state.item_definition.get("source_id", ""),
        "location": block["location"], "excerpt": block["text"]}]} for block in context.source_blocks}
    return evidence, context.diagnostics.to_dict()


def _update_response_totals(metrics, responses, sign=1):
    """Each committed entry replaces its previous contribution, including resume."""
    for name in ("cumulative_proposal_responses", "cumulative_uncached_requests", "cumulative_schema_repair_responses", "cumulative_uncached_prompt_characters"):
        metrics.setdefault(name, 0)
    metrics.setdefault("token_usage", {"prompt_tokens": 0, "completion_tokens": 0})
    for r in responses:
        usage = r["response"]["usage"]
        for name, amount in (
            ("cumulative_proposal_responses", 1),
            ("cumulative_uncached_requests", 0 if usage.get("response_checkpoint_hit") else
                1 + usage.get("format_retry_calls", 0)),
            ("cumulative_schema_repair_responses", int(r.get("schema_repair", False))),
            ("cumulative_uncached_prompt_characters", 0 if usage.get("response_checkpoint_hit") else
                usage.get("input_characters", len(r["request"].get("system_prompt", "")) + len(r["request"].get("user_prompt", "")))),
        ):
            metrics[name] = metrics.get(name, 0) + sign * amount
        tokens = metrics.setdefault("token_usage", {})
        for name in ("prompt_tokens", "completion_tokens"):
            tokens[name] = tokens.get(name, 0) + sign * usage.get(name, 0)


def _construct_parallel(state, candidates, malfunctions, *, method, agent, project_facts,
                        checkpoint, progress, max_workers):
    """Private workers, changed-entry copies, and one ordered snapshot writer.

    Copy outside the lock; merge and save together inside it. A slower old
    snapshot can never overwrite a newer commit. No background writer or pool.
    """
    from .nodes.parallel import ordered_parallel_map
    from .state import HARAState
    lock = Lock()
    cache = state.item_definition.setdefault("physical_analysis_plans", {})
    reports, totals = {}, {}
    _update_response_totals(totals, [])
    for entry in cache.values():
        _update_response_totals(totals, entry.get("responses", []))
    budget = getattr(agent.client, "attempt_budget", None)
    starting = getattr(budget, "attempts", None)
    timing = {"merge_lock_wait_s": 0.0, "changed_entry_copy_s": 0.0,
              "checkpoint_s": 0.0, "checkpoint_count": 0, "copied_plan_entries": 0, "workers": {}}
    state.item_definition["physical_analysis_timing"] = timing
    needs_snapshot = False

    def save_snapshot():
        nonlocal needs_snapshot
        saving = perf_counter()
        checkpoint(state)
        timing["checkpoint_s"] += perf_counter() - saving
        timing["checkpoint_count"] += 1
        needs_snapshot = False

    def merge(malfunction_id, key, entry, report, *, cached_refresh=False):
        nonlocal needs_snapshot
        copying = perf_counter()
        isolated_entry = deepcopy(entry) if entry is not None else None
        isolated_report = deepcopy(report)
        copied = perf_counter()
        with lock:
            timing["merge_lock_wait_s"] += perf_counter() - copied
            timing["changed_entry_copy_s"] += copied - copying
            if key is not None:
                _update_response_totals(totals, cache.get(key, {}).get("responses", []), -1)
                cache[key] = isolated_entry
                _update_response_totals(totals, isolated_entry.get("responses", []))
                timing["copied_plan_entries"] += 1
                needs_snapshot = True
            reports[malfunction_id] = isolated_report
            metrics = {}
            for report in reports.values():
                for name, value in report.items():
                    if name.startswith("cumulative_"):
                        continue
                    if isinstance(value, int):
                        metrics[name] = metrics.get(name, 0) + value
                    elif isinstance(value, list):
                        metrics.setdefault(name, []).extend(value)
            metrics.update(deepcopy(totals))
            metrics.update(
                provider_transport_attempts_this_stage=budget.attempts - starting if starting is not None else None,
                metrics_scope="PHYSICAL_PROPOSALS_AND_SOLVES; causal reuse is accounted in scenario audits",
            )
            state.item_definition["physical_analysis_metrics"] = metrics
            if checkpoint and key is not None and not cached_refresh:
                save_snapshot()

    def worker(mf):
        started = perf_counter()
        with lock:
            # Parent entries are replaced, never mutated by another worker.
            references = {key: value for key, value in cache.items()
                if value.get("context", {}).get("malfunction", {}).get("malfunction_id") == mf.malfunction_id}
        local = HARAState(state.run_id, functions=deepcopy(state.functions), item_definition={
            "physical_analysis_plans": deepcopy(references),
            # Read-only source passages are shared; workers never change them.
            "blocks": state.item_definition.get("blocks", []),
            "source_id": state.item_definition.get("source_id", ""),
        })
        try:
            result = construct_physical_scenarios(local,
                {mf.malfunction_id: deepcopy(candidates[mf.malfunction_id])}, [mf], method=method,
                agent=agent, project_facts=project_facts,
                _commit_plan=lambda key, entry, report, **options: merge(mf.malfunction_id, key, entry, report, **options))
            return result[mf.malfunction_id]
        finally:
            # Include no-proposal cases and failures without copying all plans
            # or writing a duplicate full snapshot at every worker completion.
            merge(mf.malfunction_id, None, None, local.item_definition.get("physical_analysis_metrics", {}))
            with lock:
                timing["workers"][mf.malfunction_id] = {
                    "elapsed_s": perf_counter() - started,
                    "plan_groups": len(local.item_definition["physical_analysis_plans"]),
                    "candidate_count": len(candidates[mf.malfunction_id])}

    try:
        results = ordered_parallel_map(malfunctions, worker, max_workers=max_workers, on_progress=progress)
    finally:
        # Cached deterministic recalculation is reproducible without a request.
        # Flush it together after workers drain; new responses/status changes
        # still trigger immediate ordered snapshots inside merge().
        with lock:
            if checkpoint and needs_snapshot:
                save_snapshot()
    return {mf.malfunction_id: result for mf, result in zip(malfunctions, results)}


def construct_physical_scenarios(state, candidates, malfunctions, *, method, agent, project_facts,
                                 checkpoint=None, progress=None, max_workers=1, _commit_plan=None):
    if max_workers > 1 and len(malfunctions) > 1:
        return _construct_parallel(state, candidates, malfunctions, method=method, agent=agent,
            project_facts=project_facts, checkpoint=checkpoint, progress=progress, max_workers=max_workers)
    producer = PhysicalHypothesisInstantiationService(method)
    cache = state.item_definition.setdefault("physical_analysis_plans", {})
    metrics = {"existing_path_skipped": 0, "plan_requests": 0, "plan_cache_hits": 0,
               "physical_plan_reuse": 0, "generated_instances": 0, "schema_repairs": 0,
               "response_cache_hits": 0, "semantic_corrections": 0, "plan_reuse_records": [], "diagnostics": []}
    state.item_definition["physical_analysis_metrics"] = metrics
    budget = getattr(agent.client, "attempt_budget", None)
    starting_attempts = getattr(budget, "attempts", None)

    # Rebuild once on recovery; new responses update only their contribution.
    _update_response_totals(metrics, [])
    for entry in cache.values():
        _update_response_totals(metrics, entry.get("responses", []))

    def reconcile_metrics():
        metrics["provider_transport_attempts_this_stage"] = (budget.attempts - starting_attempts) if starting_attempts is not None else None
        metrics["physical_solve_points"] = producer.physics_solve_count
        metrics["exact_plan_input_reuse"] = sum(
            len(row["scenario_ids"]) - len(set(row["exact_input_keys"]))
            for row in metrics["plan_reuse_records"])
        metrics["metrics_scope"] = "PHYSICAL_PROPOSALS_AND_SOLVES; causal reuse is accounted in scenario audits"
    project = project_facts.to_dict() if hasattr(project_facts, "to_dict") else asdict(project_facts) if project_facts else {}
    functions = {f["function_id"]: f for f in state.functions}
    config = getattr(agent.client, "config", None)
    route = {key: getattr(config, key, "") for key in ("provider", "model", "base_url", "scenario_thinking")}
    for mf in malfunctions:
        source_evidence, source_routing = _source_evidence(state, functions.get(mf.function_id, {}))
        driver_configurations = sorted({str(c.facts.get("driver_position", ""))
                                        for c in candidates[mf.malfunction_id]
                                        if c.facts.get("driver_position")})
        groups, kept = defaultdict(list), []
        for candidate in candidates[mf.malfunction_id]:
            task_scope = producer.task_scope(candidate, mf.malfunction_id)
            if not task_scope:
                kept.append(candidate); metrics["existing_path_skipped"] += 1
            else:
                groups[(task_scope, signature(scenario_conditions(candidate, without_driver=True)))].append(candidate)
        # Distribute source-condition requests over existing motion groups;
        # never multiply groups by all parking layouts or weather conditions.
        source_conditions = [{"source_field": field, "value": value}
            for field in ("odd_road_types", "odd_weather_conditions")
            for value in dict.fromkeys(project.get(field, []))]
        coverage_offset = int(signature(asdict(mf))[:8], 16)
        for (task_scope, condition_key), family in sorted(groups.items()):
            first = family[0]
            # The same motion plan can be instantiated for multiple driver
            # configurations. Causality and risk are still evaluated per world.
            inputs = {"conditions": scenario_conditions(first, without_driver=True),
                "driver_policy": "Calculate the unmitigated trajectory, without subtracting a driver's action. Separately assess remaining control CAPABILITY and its conditional assumptions for the listed driver configurations; lack of intervention in the motion model does not mean lack of an available independent control path. Causality and controllability are assessed per instance.",
                "driver_configurations": driver_configurations,
                "malfunction": asdict(mf), "function": functions.get(mf.function_id, {}),
                "project_context": {k: v for k, v in project.items() if k not in {"risk_facts", "method_risk_fact_bindings", "sources"}},
                "project_physical_facts": [f for f in project.get("risk_facts", [])
                    if f.get("parameter") in {"longitudinal_acceleration_mps2", "slope_percent", "ego_speed_kph",
                        "relative_distance_m", "steering_control_available", "emergency_braking_available",
                        "remote_intervention_available", "other_road_user_avoidance_possible"}],
                "method_identity": {k: method.metadata.get(k) for k in ("method_source_hash", "project_analysis_policy")},
                "method_version": [method.contract_version, method.compiler_version],
                "intervention_inputs": sorted({condition.field for rule in method.structured_risk_method.controllability_overrides
                                               for condition in (*rule.all_of, *rule.any_of)} |
                    {condition.field for condition in method.structured_risk_method.controllability_profile.applicability}),
                "risk_vocabulary": {"road_user_type": dict(method.structured_risk_method.severity.road_user_groups),
                                    "collision_type": dict(method.structured_risk_method.severity.collision_types)},
                "analysis_strategy": PhysicalScenarioAgent.ANALYSIS_STRATEGY,
                "model_input_status": producer.input_status(first, mf.malfunction_id),
                "producer_version": PHYSICAL_PLAN_VERSION, "model_catalog_identity": signature(MODEL_CATALOG),
                "prompt_version": agent.PROMPT_VERSION, "route": route}
            if task_scope == "MOTION_AND_CONTROLS" and source_conditions:
                # Scope selection must not change the task for the same world.
                # A sample's ordinal is different from its full-run ordinal;
                # use the canonical condition identity, not enumeration order.
                focus_index = (coverage_offset + int(condition_key[:8], 16)) % len(source_conditions)
                inputs["source_condition_focus"] = source_conditions[focus_index]
                inputs["source_condition_policy"] = (
                    "Use this allowed source condition only if relevant and compatible; otherwise select a relevant "
                    "alternative from project_context or explain the uncovered condition. Return operating_condition "
                    "with physical_effect linked to your geometry, motion, visibility or intervention assumptions. "
                    "One source axis per hypothesis; no Cartesian expansion. A label alone does not prove coverage.")
            if task_scope == "MOTION_AND_CONTROLS":
                inputs["operating_atom_options"] = producer.operating_atom_options(
                    first, mf, project, functions.get(mf.function_id))
                object_id = first.facts.get("method_scenario_dimensions", {}).get("OBJECT", {}).get("atom_id")
                selected_object = next((a for a in method.metadata.get("scenario_atom_catalog", []) if a.get("atom_id") == object_id), None)
                if selected_object:
                    inputs["selected_object_scope"] = {k: selected_object[k] for k in ("atom_id", "label", "physical_semantics") if k in selected_object}
                    # Distribute explicitly named road-user classes over existing
                    # worlds, like the source-condition focus above. This is a
                    # task focus, not a fact binding or another expansion axis.
                    from hara_agent.services.analysis.scenario_selection_quality import ScenarioCandidateRanker
                    categories = ScenarioCandidateRanker.atom_categories(selected_object)
                    object_types = sorted(name for name in inputs["risk_vocabulary"]["road_user_type"]
                        if "OBJECT_" + name in categories)
                    if len(object_types) > 1 and not first.facts.get("road_user_type"):
                        inputs["road_user_focus"] = object_types[(coverage_offset + int(condition_key[:8], 16)) % len(object_types)]
                inputs["interaction_coverage_policy"] = (
                    "Compare physically compatible object motions before choosing representatives. "
                    "Existing specific objects/constraints remain binding. For a justified new target, "
                    "prefer distinct relevant interactions within this plan; do not repeat the same "
                    "stationary-target mechanism only with another distance. When road_user_focus is supplied, use at least one existing hypothesis slot for that explicitly source-named class, with its actual motion/interaction; if incompatible, give the specific uncovered condition instead. Other explicit classes remain alternatives, not mandatory extra instances. When the selected method object explicitly includes multiple road-user types, compare their distinct movement/escape mechanisms; do not silently narrow that scope to only the most familiar type. Replace redundant hypotheses rather than multiplying them. Keep at most 3 total instances. "
                    "If no distinct supported interaction is appropriate, explain why; never add an unrelated fault.")
            if task_scope == "CONTROL_INPUTS_ONLY":
                inputs.update(task_scope=task_scope, prompt_version=agent.CONTROL_PROMPT_VERSION)
            evidence = {"MF.description": {"value": mf.description, "sources": [vars(ref) for ref in mf.sources]},
                        "MF.functional_effect": {"value": mf.functional_effect, "sources": [vars(ref) for ref in mf.sources]},
                        "PROJECT.operating_context": {"value": inputs["project_context"], "sources": project.get("sources", [])},
                        "SCENARIO.conditions": inputs["conditions"], **source_evidence}
            inputs["source_routing"] = source_routing
            inputs["evidence"] = evidence
            key = signature(inputs)
            context = {**inputs, "plan_key": key}
            saved = cache.get(key, {})
            entry = saved or {"context": context, "responses": [], "status": "REQUESTED"}
            cache[key] = entry
            prior_status = entry.get("status")
            if prior_status == "VALIDATED" and entry.get("initial_plan"):
                try:
                    PhysicalScenarioPlan.validate_correction_scope(entry["initial_plan"], entry["plan"],
                        entry.get("correction_diagnostics", []), context.get("driver_configurations", []))
                except ValueError as error:
                    # Re-enter the same proposal/correction through raw response
                    # checkpoints. Preserve the old result; only an affected
                    # correction may need its existing bounded repair request.
                    entry.setdefault("correction_scope_history", []).append({"plan": entry["plan"], "error": str(error)})
                    entry["status"] = "REQUESTED"
                    entry.pop("semantic_correction_completed", None)
                    entry.pop("semantic_correction_error", None)
            prior_validation = entry.get("response_validation_version")
            validation_version = getattr(agent, "RESPONSE_VALIDATION_VERSION", None)
            if prior_status == "INVALID" and prior_validation != validation_version:
                # Revalidate the raw response through the existing checkpoint;
                # successful plans do not need another model request. Same-version
                # failures remain bounded, including across process restarts.
                entry.setdefault("validation_history", []).append({
                    "version": prior_validation, "error": entry.get("error", "")})
                entry["status"] = "REQUESTED"
                entry.pop("error", None)
            prior_responses = len(entry.get("responses", []))
            prior_instances = signature(entry.get("instantiations", {}))

            def save_plan(*, cached_refresh=False):
                if _commit_plan:
                    _commit_plan(key, entry, metrics, cached_refresh=cached_refresh)
                elif checkpoint:
                    checkpoint(state)

            def persist_response(record):
                # Replaying an already journaled response must not duplicate
                # cumulative usage. Request identity includes correction input.
                request_key = signature(record["request"])
                already_recorded_hit = record["response"]["usage"].get("response_checkpoint_hit") and any(
                    signature(r["request"]) == request_key for r in entry["responses"])
                if not already_recorded_hit:
                    entry["responses"].append(record)
                    _update_response_totals(metrics, [record])
                reconcile_metrics()
                # PhysicalScenarioAgent has already fsynced the raw response.
                # If no durable response store exists, retain the old snapshot
                # frequency. Derivation can always replay that exact request.
                if not getattr(agent, "responses", None):
                    save_plan()

            try:
                if entry.get("status") in {"VALIDATED", "CORRECTION_REQUESTED"}:
                    plan = PhysicalScenarioPlan.from_dict(entry["plan"], evidence_ids=set(evidence))
                    metrics["plan_cache_hits"] += 1
                elif entry.get("status") == "INVALID":
                    for candidate in family:
                        candidate.analysis_instance["physical_plan_diagnostics"] = [entry["error"]]
                    kept.extend(family)
                    metrics["diagnostics"].append({"plan_key": key, "reason": entry["error"]})
                    continue
                else:
                    entry["response_validation_version"] = validation_version
                    print(f"[HARA] physical-plan {mf.malfunction_id}: {len(family)} instances, key={key[:12]}", file=sys.stderr, flush=True)
                    plan, audit = agent.propose(context, on_response=persist_response)
                    entry.update(status="VALIDATED", plan=plan.to_dict(), audit=audit)
                    metrics["plan_requests"] += audit["request_count"]
                    metrics["schema_repairs"] += audit["schema_repairs"]
                    metrics["response_cache_hits"] += audit["response_cache_hits"]
                def instantiate_family(current_plan):
                    def control_gap(hypothesis):
                        if braking_capability(hypothesis) is not None:
                            return None
                        unresolved_controls = hypothesis.get("unresolved_controls", {})
                        unresolved = unresolved_controls.get("braking")
                        if unresolved and (not unresolved_controls.get("driver_braking")
                                           or entry.get("semantic_correction_completed")):
                            return None
                        return {"reason": "CONDITIONAL_CONTROL_BASIS_UNRESOLVED" if unresolved else "CONTROL_CAPABILITY_NOT_ASSESSED", "field": "emergency_braking_available",
                            "hypothesis_id": hypothesis.get("hypothesis_id", "control_assessment"),
                            "previous_reason": unresolved or "",
                            "resolution": "Assess braking against source and fault scope. Distinguish control availability from whether its response can avoid this particular contact: a short contact window does not itself establish unavailable braking. An independent manual braking path may be a PENDING conditional assumption when compatible with the declared fault; deterministic analysis evaluates its time window separately. Preserve valid motion. If a specific shared failure, source conflict or model limit prevents even that conditional case, retain unresolved_controls.braking and explain that blocker. Do not invent proven capability; one correction only."}

                    rows = [(candidate, *producer.instantiate(candidate, mf, current_plan,
                        plan_key=key, response_model=entry["audit"]["model"], project_context=project,
                        function=functions.get(mf.function_id))) for candidate in family]
                    # A declared target side conflicting with numeric coordinates
                    # is a proposal error, not a new no-contact mechanism. Feed
                    # the existing bounded correction; retain the original child.
                    for _, children, errors in rows:
                        for child in children:
                            outcome = child.facts.get("motion_model_outcome", {})
                            analysis = child.facts.get("physical_analysis", {})
                            response_hypothesis = (child.facts.get("control_analysis") or analysis).get("hypothesis", {})
                            proposal = response_hypothesis.get("safe_state_proposal", {})
                            if (child.facts.get("driver_in_vehicle") is False
                                    and "driver_braking" in proposal.get("required_controls", [])
                                    and not response_hypothesis.get("safe_state_unresolved_reason")):
                                errors.append({"reason": "SAFE_STATE_DRIVER_SCOPE_UNASSESSED",
                                    "hypothesis_id": response_hypothesis.get("hypothesis_id", "control_assessment"),
                                    "resolution": "Preserve the shared motion. In driver_control_assessments.outside_driver_seat assess a complete compatible command/execution path from source and fault scope, or give a concrete safe_state_unresolved_reason. Keep the in-seat path. Do not use a pedal outside, assume remote availability from an App, duplicate the trajectory, or alter parameters. Use the same one bounded correction."})
                            response_missing = (not proposal.get("transition") or
                                child.facts.get("fault_motion_model") == "RECTILINEAR_MOVING_TARGET_CONTACT"
                                and proposal.get("transition", {}).get("post_stop_motion") != "HOLD_POSITION")
                            if (child.facts.get("fault_motion_model") in method.metadata.get("project_analysis_policy", {}).get("timing_analysis", {}).get("supported_models", [])
                                    and set(proposal.get("required_controls", [])) & {"braking", "driver_braking"}
                                    and response_missing and not response_hypothesis.get("timing_unresolved_reason")):
                                errors.append({"reason": "SAFE_STOP_RESPONSE_NOT_MODELED",
                                    "hypothesis_id": response_hypothesis.get("hypothesis_id", "control_assessment"),
                                    "resolution": "Preserve this motion. Complete the proposed braking safe-state response with an explained conditional net-deceleration transition and, for a moving target, explicit post_stop_motion:HOLD_POSITION with a remaining-control basis; alternatively give timing_unresolved_reason for a concrete incompatibility. A missing measured value alone does not prohibit a conditional assumption. One existing correction only."})
                            if proposal and not response_missing:
                                from hara_agent.services.analysis.risk_calculation_input_service import RiskCalculationInputService
                                timing = RiskCalculationInputService().ftti(mf.malfunction_id, child.scenario_id, "",
                                    {**child.facts, "_fact_provenance": child.fact_provenance}, method=method)
                                if timing.timing_trace.get("status") in {"AVOIDABILITY_ALREADY_LOST", "NO_IMMEDIATE_SAFE_BRAKING_WINDOW"}:
                                    errors.append({"reason": "SAFE_STOP_RESPONSE_UNREACHABLE",
                                        "hypothesis_id": response_hypothesis.get("hypothesis_id", "control_assessment"),
                                        "evidence": timing.timing_trace,
                                        "resolution": "Preserve the unmitigated world. This proposed response does not avoid its target. Reconcile the safe-state claim with the available controls/model, or explicitly retain the unsupported strategy in safe_state_unresolved_reason. Do not move the target or increase braking just to force a safe result. No extra correction loop."})
                            avoidance = analysis.get("avoidance_validation", {})
                            if avoidance and avoidance.get("status") not in {"CONDITIONAL_ESCAPE_VALIDATED", "NO_CONTACT_TO_AVOID"}:
                                errors.append({"reason": avoidance["status"], "evidence": avoidance,
                                    "hypothesis_id": analysis.get("hypothesis_id"),
                                    "resolution": "Reconcile the proposed escape with this trajectory and corridor. Do not retain an unsupported positive avoidance claim or tune geometry just to obtain C1."})
                            counterfactual = analysis.get("stationary_ego_counterfactual", {})
                            if counterfactual.get("status") == "CONTACT":
                                errors.append({"reason": "CONTACT_WITH_STATIONARY_EGO", "evidence": counterfactual,
                                    "hypothesis_id": analysis.get("hypothesis_id"),
                                    "resolution": "This target also contacts the stationary ego. Establish the fault's incremental hazard (not merely contact), or propose a physically consistent distinct target path. Preserve the counterfactual; do not claim fault-induced contact without evidence."})
                            if outcome.get("reason") == "TARGET_OFFSET_POSITION_CONFLICT":
                                errors.append({"reason": outcome["reason"],
                                    "hypothesis_id": child.facts.get("physical_analysis", {}).get("hypothesis", {}).get("hypothesis_id"),
                                    "resolution": "Reconcile target position with its signed body-frame offsets. Preserve valid hypotheses; if the existing model cannot represent this trajectory, state that limitation rather than changing the fault mechanism."})
                    # A syntactically valid motion plan may have omitted the
                    # capability analysis. Use the existing ONE family-level
                    # correction, retaining useful motion even if still unknown.
                    if "emergency_braking_available" in inputs["intervention_inputs"]:
                        from hara_agent.services.analysis.scenario_physics import source_is_accepted_for
                        from hara_agent.services.analysis.hazardous_event_risk_context_service import HazardousEventRiskContextService
                        contexts = HazardousEventRiskContextService(method)
                        for parent, children, errors in rows:
                            # Geometry can fail before producing a contact child.
                            # Include that proposal's capability gap in the SAME
                            # correction, rather than discovering it afterwards.
                            failed_ids = {e.get("hypothesis_id") for e in errors}
                            if parent.facts.get("driver_in_vehicle") is True:
                                for hypothesis in current_plan.hypotheses:
                                    if (hypothesis["hypothesis_id"] in failed_ids
                                            and "in_driver_seat" in hypothesis.get("driver_positions", ["in_driver_seat"])
                                            and (gap := control_gap(hypothesis)) and gap not in errors):
                                        errors.append(gap)
                            for child in children:
                                if child.facts.get("driver_in_vehicle") is not True:
                                    continue
                                if (task_scope == "MOTION_AND_CONTROLS"
                                        and child.facts.get("motion_model_outcome", {}).get("status") != "CONTACT"):
                                    continue  # No numeric contact needing C, or a deliberately symbolic plan.
                                analysis = child.facts.get("control_analysis") or child.facts.get("physical_analysis", {})
                                hypothesis = analysis.get("hypothesis", {})
                                error = control_gap(hypothesis)
                                if error is None:
                                    continue
                                if (type(child.facts.get("emergency_braking_available")) is bool
                                        and source_is_accepted_for(child.fact_provenance.get("emergency_braking_available", {}),
                                            malfunction_id=mf.malfunction_id, scenario_id=child.scenario_id)):
                                    continue
                                context = contexts.build(malfunction_id=mf.malfunction_id,
                                    scenario_id=child.scenario_id, hazard_node_id="control-completeness",
                                    scenario={**child.facts, "_fact_provenance": child.fact_provenance})
                                if contexts.controllability_readiness(context)["status"] == "READY":
                                    continue  # A valid independent override already answers the active method.
                                if error not in errors:
                                    errors.append(error)
                    return rows

                instantiated = instantiate_family(plan)
                failures = [{"scenario_id": candidate.scenario_id, **failure}
                    for candidate, _, errors in instantiated for failure in errors]
                failures.extend({"reason": "SAFE_STATE_CONTROL_DEPENDENCY_UNRESOLVED", **deferred,
                    "resolution": "Reconcile the original safe-state proposal with failed/remaining controls. Assess the undeclared dependency from the provided source, or propose a reachable state using the remaining path (including continued service braking if physically supported). Do not silently assume a control works or delete a needed dependency. Preserve valid motion; one existing correction only."}
                    for deferred in entry.get("audit", {}).get("deferred_safe_states", []))
                if (failures or entry.get("status") == "CORRECTION_REQUESTED") and not entry.get("semantic_correction_completed"):
                    # One explicit, budgeted semantic correction. Persist the
                    # original response and feedback before transport so resume
                    # uses the identical response-checkpoint key.
                    entry.setdefault("initial_plan", plan.to_dict())
                    entry.setdefault("correction_diagnostics", failures)
                    # Keep already computed motion recoverable if the shared
                    # budget is exhausted during the capability correction.
                    entry["instantiations"] = {candidate.scenario_id: {
                        "generated_ids": [g.scenario_id for g in generated],
                        "generated_scenarios": [asdict(g) for g in generated], "failures": errors}
                        for candidate, generated, errors in instantiated}
                    entry.update(status="CORRECTION_REQUESTED")
                    if not getattr(agent, "responses", None):
                        save_plan()
                    correction_context = {**context, "previous_plan": entry["initial_plan"],
                        "correction_diagnostics": entry["correction_diagnostics"],
                        "correction_policy": "Correct only reported conflicts or omitted assessments; preserve valid hypotheses and motion. Explicitly unresolved controls are allowed. One semantic correction allowed."}
                    metrics["semantic_corrections"] += 1
                    try:
                        corrected_plan, audit = agent.propose(correction_context, on_response=persist_response)
                    except (ValueError, KeyError, TypeError) as error:
                        # A malformed correction cannot erase the previously
                        # validated proposal. Keep its valid motion and local
                        # input gaps; all failed replies remain in responses.
                        entry.update(status="VALIDATED", semantic_correction_completed=True,
                                     semantic_correction_error=str(error))
                        metrics["diagnostics"].append({"plan_key": key,
                            "reason": "PHYSICAL_CORRECTION_INVALID_ORIGINAL_RETAINED", "error": str(error)})
                    else:
                        plan = corrected_plan
                        entry.update(status="VALIDATED", plan=plan.to_dict(), audit=audit,
                                     semantic_correction_completed=True)
                        metrics["plan_requests"] += audit["request_count"]
                        metrics["schema_repairs"] += audit["schema_repairs"]
                        metrics["response_cache_hits"] += audit["response_cache_hits"]
                        instantiated = instantiate_family(plan)
                reuse_kind = "control_plan_reuse" if task_scope == "CONTROL_INPUTS_ONLY" else "physical_plan_reuse"
                metrics[reuse_kind] = metrics.get(reuse_kind, 0) + len(family) - 1
                metrics["plan_reuse_records"].append({"plan_key": key,
                    "task_scope": task_scope,
                    "scenario_ids": [candidate.scenario_id for candidate in family],
                    "exact_input_keys": [signature(scenario_conditions(candidate)) for candidate in family]})
                for candidate, generated, failures in instantiated:
                    if not plan.hypotheses and plan.control_assessment is None:
                        failures = [{"reason": "PHYSICAL_MODEL_NOT_APPLICABLE",
                                     "uncovered_conditions": list(plan.uncovered_conditions)}]
                    kept.extend(generated or [candidate])
                    if failures:
                        for item in generated or [candidate]:
                            item.analysis_instance["physical_plan_diagnostics"] = failures
                    metrics["generated_instances"] += len(generated)
                    for child in generated:
                        condition = child.facts.get("selected_source_condition")
                        if condition:
                            metrics.setdefault("source_condition_instances", []).append({
                                "scenario_id": child.scenario_id, **condition})
                        elif source_conditions and task_scope == "MOTION_AND_CONTROLS":
                            metrics.setdefault("source_condition_uncovered_ids", []).append(child.scenario_id)
                    if plan.control_assessment is not None:
                        metrics["control_only_instances"] = metrics.get("control_only_instances", 0) + len(generated)
                    metrics["diagnostics"].extend({"scenario_id": candidate.scenario_id, **f} for f in failures)
                    entry.setdefault("instantiations", {})[candidate.scenario_id] = {
                        "generated_ids": [g.scenario_id for g in generated],
                        "generated_scenarios": [asdict(g) for g in generated], "failures": failures}
            except (ValueError, KeyError, TypeError) as error:
                entry.update(status="INVALID", error=str(error))
                for candidate in family:
                    candidate.analysis_instance["physical_plan_diagnostics"] = [str(error)]
                metrics["diagnostics"].append({"plan_key": key, "reason": str(error)})
                kept.extend(family)
            finally:
                # Includes budget/transport exceptions. Successful earlier plans
                # and every parsed response remain durable on resume.
                reconcile_metrics()
                new_result = (not saved or entry.get("status") != prior_status
                              or len(entry.get("responses", [])) != prior_responses
                              or entry.get("response_validation_version") != prior_validation)
                if (checkpoint or _commit_plan) and (new_result
                        or signature(entry.get("instantiations", {})) != prior_instances):
                    save_plan(cached_refresh=not new_result)
        candidates[mf.malfunction_id] = kept
    return candidates
