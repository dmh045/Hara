"""Explicit pre-causal scheduling for bounded physical analysis proposals."""
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
import sys
import re
from threading import Lock
from time import perf_counter
from hara_agent.config.llm_config import physical_reasoning_identity
from hara_agent.services.analysis.project_fact_resolver import effective_project_policy

from hara_agent.contracts.physical_scenario_plan import PhysicalScenarioPlan, PHYSICAL_PLAN_VERSION, MODEL_CATALOG, braking_capability, declared_response_path, automatic_command_unassessed
from hara_agent.services.analysis.physical_hypothesis_instantiation_service import PhysicalHypothesisInstantiationService
from hara_agent.services.analysis.driver_configuration_service import DriverConfigurationBrancher
from hara_agent.services.analysis.scenario_equivalence import scenario_conditions, signature
from hara_agent.services.semantic.physical_scenario_agent import PhysicalScenarioAgent
from hara_agent.services.extraction.evidence_retrieval import (
    CoverageFirstContextAssembler, DeterministicEvidenceRetriever, FactRetrievalSpec,
    retrieval_blocks,
)
from hara_agent.infrastructure.llm import LLMOutputLimitError


def _unmodeled_declared_response(candidate, method):
    """One production diagnostic for all declared, model-supported response paths."""
    facts = candidate.facts
    h = (facts.get("control_analysis") or facts.get("physical_analysis", {})).get("hypothesis", {})
    outcome = facts.get("motion_model_outcome", {})
    driver = "in_driver_seat" if facts.get("driver_in_vehicle") is True else "outside_driver_seat" if facts.get("driver_in_vehicle") is False else ""
    contact = outcome.get("status") == "CONTACT" or not outcome and bool(facts.get("collision_branches"))
    supported = method.metadata.get("project_analysis_policy", {}).get("timing_analysis", {}).get("supported_models", [])
    if (not contact or h.get("safe_state_proposal") or facts.get("fault_motion_model") not in supported):
        return None
    if automatic_command_unassessed(h, driver):
        return {"reason": "CONTROL_AUTOMATIC_COMMAND_UNASSESSED",
            "hypothesis_id": h.get("hypothesis_id", "control_assessment"), "driver_position": driver,
            "resolution": "The actuator is explicitly retained, but its automatic command path has not been assessed. Evaluate the system as the responder using the supplied fault scope and interfaces, separately from manual or remote operation. Declare automatic_braking_command remaining, failed, or unresolved with a specific reason in the existing control fields. Absence of an outside person's remote interface does not resolve automatic response availability. Do not invent an interface or convert function disablement into braking. Use the existing single correction only."}
    if not declared_response_path(h, driver):
        if "braking" in h.get("remaining_controls", []) and driver == "outside_driver_seat":
            return {"reason": "SAFE_STATE_INDEPENDENT_RESPONSE_REQUIRED",
                "hypothesis_id": h.get("hypothesis_id", "control_assessment"), "driver_position": driver,
                "resolution": "An actuator is retained but no existing command path is established. Propose the independent system response as safe_state_proposal.required_command_path with REQUIRED_NOT_ESTABLISHED, so the same physics can calculate its design deadline. Keep the failed commands failed and the unmitigated world unchanged. Do not credit this required interface as existing capability or C intervention. Return a compatible response and fault clock, or a specific actuator/physical/model contradiction; unknown implementation delay is the quantity to bound, not a reason to omit the analysis. Use the same single correction."}
        return None
    return {"reason": "SAFE_STATE_DECLARED_PATH_UNMODELED",
        "hypothesis_id": h.get("hypothesis_id", "control_assessment"), "driver_position": driver,
        "resolution": "This world already declares a conditional response control path. Supply the compatible structured response for deterministic evaluation of its maximum allowed response delay, or identify a specific model/enable/fault contradiction. A safe_state_proposal is a candidate, not proof of safe stopping or engineering approval; a fault-release response can remain in motion. Missing measured latency, unverified independence or a predicted short/unsafe window is not a substitute for this calculation. Do not invent interfaces, credit failed controls, borrow an in-seat pedal outside or change this trajectory. Use the existing single correction only."}


def _source_evidence(state, function):
    """Route bounded original passages, never infer a control or interaction."""
    specs = [
        FactRetrievalSpec("intervention", ("制动踏板", "接管", "驾驶员干预", "remote", "brake pedal", "override")),
        FactRetrievalSpec("automatic_stop", ("自动停", "自动把车停", "automatic stop", "safe stop")),
        FactRetrievalSpec("brake_holding", ("制动保持", "制动hold", "保压", "brake holding", "hold pressure")),
        FactRetrievalSpec("fault_detection", ("故障诊断", "故障检测", "检测到故障", "检测到异常", "被检测到", "诊断通过", "fault detection", "diagnostic"),
            context_hints=("超时", "信号", "timeout", "signal")),
        FactRetrievalSpec("fault_reaction", ("故障处理", "处理措施", "功能禁用", "功能被禁用", "功能不可用", "fault reaction", "disable function", "fault response"),
            context_hints=("通信", "信号", "communication", "signal")),
        FactRetrievalSpec("control_transition", ("释放", "移交", "退出", "中止", "release control", "handover", "abort")),
        FactRetrievalSpec("interfaces", ("IPB", "EPS", "VDC", "EPB", "interface", "接口", "执行器", "actuator")),
        FactRetrievalSpec("participants", ("行人", "两轮车", "车辆", "pedestrian", "traffic")),
    ]
    terms = tuple(str(function[k]) for k in ("name", "output") if function.get(k))
    if terms:
        specs.append(FactRetrievalSpec("function", terms))
    # PDF pages remain the stored source identity; only the retrieval view is
    # split into exact page substrings so unrelated page text does not crowd
    # out the control and phase evidence needed by this task.
    blocks = retrieval_blocks(state.item_definition.get("blocks", []))
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


def _physical_families(producer, candidates, malfunction_id):
    groups, kept = defaultdict(list), []
    for candidate in candidates:
        scope = producer.task_scope(candidate, malfunction_id)
        if scope:
            groups[(scope, signature(scenario_conditions(candidate, without_driver=True)))].append(candidate)
        else:
            kept.append(candidate)
    return groups, kept


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
    producer = PhysicalHypothesisInstantiationService(method)
    tasks, kept, configurations, remaining = [], {}, {}, {}
    for mf in malfunctions:
        groups, kept[mf.malfunction_id] = _physical_families(producer, candidates[mf.malfunction_id], mf.malfunction_id)
        configurations[mf.malfunction_id] = sorted({producer.driver_position(c)
            for c in candidates[mf.malfunction_id] if producer.driver_position(c)})
        remaining[mf.malfunction_id] = len(groups)
        for group, family in sorted(groups.items()):
            tasks.append((mf, group, family))
    if not tasks:
        result = construct_physical_scenarios(state, candidates, malfunctions, method=method,
            agent=agent, project_facts=project_facts, checkpoint=checkpoint)
        if progress:
            progress(len(malfunctions), len(malfunctions))
        return result
    # Index immutable prior entries once; a worker copies only its own family.
    prior_groups = defaultdict(dict)
    for key, entry in cache.items():
        context = entry.get("context", {})
        prior_groups[(context.get("malfunction", {}).get("malfunction_id"),
            context.get("task_scope", "MOTION_AND_CONTROLS"), signature(context.get("conditions", {})))][key] = entry
    _update_response_totals(totals, [])
    for entry in cache.values():
        _update_response_totals(totals, entry.get("responses", []))
    budget = getattr(agent.client, "attempt_budget", None)
    starting = getattr(budget, "session_attempts", None)
    timing = {"merge_lock_wait_s": 0.0, "changed_entry_copy_s": 0.0,
              "checkpoint_s": 0.0, "checkpoint_count": 0, "copied_plan_entries": 0, "workers": {}}
    state.item_definition["physical_analysis_timing"] = timing
    needs_snapshot = False
    unsaved_new_plans = 0

    def save_snapshot():
        nonlocal needs_snapshot, unsaved_new_plans
        saving = perf_counter()
        checkpoint(state)
        timing["checkpoint_s"] += perf_counter() - saving
        timing["checkpoint_count"] += 1
        needs_snapshot = False
        unsaved_new_plans = 0

    def merge(task_id, key, entry, report, *, cached_refresh=False):
        nonlocal needs_snapshot, unsaved_new_plans
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
            reports[task_id] = isolated_report
            metrics = {"existing_path_skipped": sum(len(rows) for rows in kept.values())}
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
                provider_transport_attempts_this_stage=budget.session_attempts - starting if starting is not None else None,
                metrics_scope="PHYSICAL_PROPOSALS_AND_SOLVES; causal reuse is accounted in scenario audits",
            )
            state.item_definition["physical_analysis_metrics"] = metrics
            if checkpoint and key is not None and not cached_refresh:
                unsaved_new_plans += 1
                # Raw responses are already atomically durable. Batch the
                # reproducible full-state snapshots, not the paid responses.
                # A client without durable responses keeps immediate saving.
                if not getattr(agent, "responses", None) or unsaved_new_plans >= max_workers:
                    save_snapshot()

    def worker(task):
        mf, group, family = task
        task_id = (mf.malfunction_id, group)
        started = perf_counter()
        references = prior_groups.get((mf.malfunction_id, *group), {})
        local = HARAState(state.run_id, functions=deepcopy(state.functions), item_definition={
            "physical_analysis_plans": deepcopy(references),
            # Read-only source passages are shared; workers never change them.
            "blocks": state.item_definition.get("blocks", []),
            "source_id": state.item_definition.get("source_id", ""),
            "item_sha256": state.item_definition.get("item_sha256", ""),
        })
        try:
            result = construct_physical_scenarios(local,
                {mf.malfunction_id: deepcopy(family)}, [mf], method=method,
                agent=agent, project_facts=project_facts,
                _driver_configurations=configurations[mf.malfunction_id],
                _commit_plan=lambda key, entry, report, **options: merge(task_id, key, entry, report, **options))
            return result[mf.malfunction_id]
        finally:
            # Include no-proposal cases and failures without copying all plans
            # or writing a duplicate full snapshot at every worker completion.
            merge(task_id, None, None, local.item_definition.get("physical_analysis_metrics", {}))
            with lock:
                report = timing["workers"].setdefault(mf.malfunction_id,
                    {"elapsed_s": 0.0, "plan_groups": 0, "candidate_count": 0})
                report["elapsed_s"] += perf_counter() - started
                report["plan_groups"] += len(local.item_definition["physical_analysis_plans"])
                report["candidate_count"] += len(family)

    completed = sum(count == 0 for count in remaining.values())
    if progress and completed:
        progress(completed, len(malfunctions))
    def completed_task(task, result):
        nonlocal completed
        mid = task[0].malfunction_id
        remaining[mid] -= 1
        if remaining[mid] == 0:
            completed += 1
            if progress:
                progress(completed, len(malfunctions))

    try:
        results = ordered_parallel_map(tasks, worker, max_workers=max_workers, on_result=completed_task)
    finally:
        # Cached deterministic recalculation is reproducible without a request.
        # Flush every remaining change after workers drain, including errors.
        with lock:
            if checkpoint and needs_snapshot:
                save_snapshot()
    for (mf, _, _), result in zip(tasks, results):
        kept[mf.malfunction_id].extend(result)
    return kept


def _retain_rejected_response(candidate, generated, entry, *, producer, mf, method,
                              key, project, function):
    """Retain a withdrawn response only on its exact, unchanged physical world."""
    if entry.get("initial_plan") and any(not (g.facts.get("control_analysis") or
            g.facts.get("physical_analysis", {})).get("hypothesis", {}).get("safe_state_proposal") for g in generated):
        from hara_agent.services.analysis.deterministic_risk_executor import FTTIMethodExecutor
        from hara_agent.services.analysis.risk_calculation_input_service import RiskCalculationInputService
        original = PhysicalScenarioPlan.from_dict(entry["initial_plan"], evidence_ids=set(entry["context"]["evidence"]))
        originals, _ = producer.instantiate(candidate, mf, original, plan_key=key,
            response_model=entry["audit"]["model"], project_context=project,
            function=function)
        for child in generated:
            analysis = child.facts.get("control_analysis") or child.facts.get("physical_analysis", {})
            if analysis.get("hypothesis", {}).get("safe_state_proposal"):
                continue
            snapshot = FTTIMethodExecutor.response_input_snapshot(child.facts)
            for previous in originals:
                previous_analysis = previous.facts.get("control_analysis") or previous.facts.get("physical_analysis", {})
                if (previous_analysis.get("hypothesis", {}).get("hypothesis_id")
                        != analysis.get("hypothesis", {}).get("hypothesis_id")
                        or FTTIMethodExecutor.response_input_snapshot(previous.facts) != snapshot):
                    continue
                evaluated = RiskCalculationInputService().ftti(mf.malfunction_id, previous.scenario_id, "",
                    {**previous.facts, "_fact_provenance": previous.fact_provenance}, method=method)
                if evaluated.timing_trace.get("status") in {"NO_IMMEDIATE_SAFE_BRAKING_WINDOW", "AVOIDABILITY_ALREADY_LOST"}:
                    analysis["rejected_response_evaluation"] = {"input_values": snapshot,
                        "method_source_hash": method.metadata.get("method_source_hash"),
                        "original_scenario_id": previous.scenario_id,
                        "proposal": previous_analysis["hypothesis"]["safe_state_proposal"],
                        "timing_trace": evaluated.timing_trace}
                    break


def construct_physical_scenarios(state, candidates, malfunctions, *, method, agent, project_facts,
                                 checkpoint=None, progress=None, max_workers=1, _commit_plan=None,
                                 _driver_configurations=None):
    if max_workers > 1:
        return _construct_parallel(state, candidates, malfunctions, method=method, agent=agent,
            project_facts=project_facts, checkpoint=checkpoint, progress=progress, max_workers=max_workers)
    producer = PhysicalHypothesisInstantiationService(method)
    cache = state.item_definition.setdefault("physical_analysis_plans", {})
    metrics = {"existing_path_skipped": 0, "plan_requests": 0, "plan_cache_hits": 0,
               "physical_plan_reuse": 0, "generated_instances": 0, "schema_repairs": 0,
               "response_cache_hits": 0, "semantic_corrections": 0, "plan_reuse_records": [], "diagnostics": []}
    state.item_definition["physical_analysis_metrics"] = metrics
    budget = getattr(agent.client, "attempt_budget", None)
    starting_attempts = getattr(budget, "session_attempts", None)

    # Rebuild once on recovery; new responses update only their contribution.
    _update_response_totals(metrics, [])
    for entry in cache.values():
        _update_response_totals(metrics, entry.get("responses", []))

    def reconcile_metrics():
        metrics["provider_transport_attempts_this_stage"] = (budget.session_attempts - starting_attempts) if starting_attempts is not None else None
        metrics["physical_solve_points"] = producer.physics_solve_count
        metrics["exact_plan_input_reuse"] = sum(
            len(row["scenario_ids"]) - len(set(row["exact_input_keys"]))
            for row in metrics["plan_reuse_records"])
        metrics["metrics_scope"] = "PHYSICAL_PROPOSALS_AND_SOLVES; causal reuse is accounted in scenario audits"
    project = project_facts.to_dict() if hasattr(project_facts, "to_dict") else asdict(project_facts) if project_facts else {}
    functions = {f["function_id"]: f for f in state.functions}
    config = getattr(agent.client, "config", None)
    route = {key: getattr(config, key, "") for key in ("provider", "model", "base_url", "scenario_thinking")}
    route.update(physical_reasoning_identity(config))
    for mf in malfunctions:
        source_evidence, source_routing = _source_evidence(state, functions.get(mf.function_id, {}))
        driver_configurations = _driver_configurations if _driver_configurations is not None else sorted({producer.driver_position(c)
                                        for c in candidates[mf.malfunction_id]
                                        if producer.driver_position(c)})
        groups, kept = _physical_families(producer, candidates[mf.malfunction_id], mf.malfunction_id)
        metrics["existing_path_skipped"] += len(kept)
        # Distribute source-condition requests over existing motion groups;
        # never multiply groups by all parking layouts or weather conditions.
        source_conditions = [{"source_field": field, "value": value}
            for field in ("odd_road_types", "odd_weather_conditions")
            for value in dict.fromkeys(project.get(field, []))]
        coverage_offset = int(signature(mf.to_dict())[:8], 16)
        for (task_scope, condition_key), family in sorted(groups.items()):
            first = family[0]
            project_policy = effective_project_policy(method.metadata.get("project_analysis_policy", {}),
                state.item_definition.get("item_sha256", ""))
            if not DriverConfigurationBrancher(policy=project_policy).location_convention(first):
                project_policy.pop("driver_location_convention", None)
            focus_identity = signature({
                "source": first.facts.get("source_operating_conditions", {}),
                "operating_scenario": first.facts.get("operating_scenario"),
                "atoms": {k: v.get("atom_id") for k, v in first.facts.get("method_scenario_dimensions", {}).items()},
            })
            # The same motion plan can be instantiated for multiple driver
            # configurations. Causality and risk are still evaluated per world.
            inputs = {"conditions": scenario_conditions(first, without_driver=True),
                "driver_policy": "Calculate the unmitigated trajectory, without subtracting a driver's action. Separately assess remaining control CAPABILITY and its conditional assumptions for the listed driver configurations; lack of intervention in the motion model does not mean lack of an available independent control path. Causality and controllability are assessed per instance.",
                "driver_configurations": driver_configurations,
                "malfunction": mf.to_dict(), "function": functions.get(mf.function_id, {}),
                "project_context": {k: v for k, v in project.items() if k not in {"risk_facts", "method_risk_fact_bindings", "sources"}},
                "project_physical_facts": [f for f in project.get("risk_facts", [])
                    if f.get("parameter") in {"longitudinal_acceleration_mps2", "slope_percent", "ego_speed_kph",
                        "relative_distance_m", "steering_control_available", "emergency_braking_available",
                        "remote_intervention_available", "other_road_user_avoidance_possible"}],
                "method_identity": {"method_source_hash": method.metadata.get("method_source_hash"),
                                    "project_analysis_policy": project_policy},
                "method_version": [method.contract_version, method.compiler_version],
                "intervention_inputs": sorted(producer.intervention_inputs),
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
                focus_index = (coverage_offset + int(focus_identity[:8], 16)) % len(source_conditions)
                inputs["source_condition_focus"] = source_conditions[focus_index]
                inputs["source_condition_policy"] = (
                    "Use this allowed source condition only if relevant and compatible; otherwise select a relevant "
                    "alternative from project_context or explain the uncovered condition. Return operating_condition "
                    "with physical_effect linked to your geometry, motion, visibility or intervention assumptions. "
                    "One source axis per hypothesis; no Cartesian expansion. A label alone does not prove coverage.")
            if task_scope == "MOTION_AND_CONTROLS":
                inputs["operating_atom_options"] = producer.operating_atom_options(
                    first, mf, project, functions.get(mf.function_id))
                object_options = inputs["operating_atom_options"].pop("OBJECT", [])
                if object_options:
                    inputs["object_focus_options"] = object_options
                object_id = first.facts.get("method_scenario_dimensions", {}).get("OBJECT", {}).get("atom_id")
                selected_object = next((a for a in method.metadata.get("scenario_atom_catalog", []) if a.get("atom_id") == object_id), None)
                if selected_object:
                    inputs["selected_object_scope"] = {k: selected_object[k] for k in ("atom_id", "label", "physical_semantics") if k in selected_object}
                if (not first.facts.get("road_user_type") and not first.facts.get("object_type")
                        and (selected_object or not object_id)):
                    # Unbound targets are analytical choices too. Use the same
                    # source/method-filtered shortlist as later object binding,
                    # rather than leaving every such world biased to vehicles.
                    from hara_agent.services.analysis.scenario_selection_quality import ScenarioCandidateRanker
                    option_ids = {a["atom_id"] for a in object_options}
                    object_scope = [selected_object] if selected_object else [a
                        for a in method.metadata.get("scenario_atom_catalog", []) if a["atom_id"] in option_ids]
                    categories = set().union(*(ScenarioCandidateRanker.atom_categories(a)
                        for a in object_scope))
                    object_types = sorted(name for name in inputs["risk_vocabulary"]["road_user_type"]
                        if "OBJECT_" + name in categories)
                    if len(object_types) > 1:
                        inputs["road_user_focus"] = object_types[(coverage_offset + int(focus_identity[:8], 16)) % len(object_types)]
                    if "road_user_focus" in inputs and not selected_object:
                        inputs["road_user_focus_basis"] = (
                            "Compatible active-method OBJECT shortlist, not an extracted project object. "
                            "Instantiate this class only as a justified conditional interaction within this existing world; "
                            "preserve explicit source constraints and state a specific incompatibility if it cannot apply. "
                            "Replace a redundant representative; do not add a coverage axis or multiply instances.")
                inputs["interaction_coverage_policy"] = (
                    "Compare physically compatible object motions before choosing representatives. "
                    "Existing specific objects/constraints remain binding. For a justified new target, "
                    "prefer distinct relevant interactions within this plan; do not repeat the same "
                    "stationary-target mechanism only with another distance. When road_user_focus is supplied, use at least one existing hypothesis slot for that explicitly source-named class, with its actual motion/interaction; if incompatible, give the specific uncovered condition instead. Other explicit classes remain alternatives, not mandatory extra instances. When the selected method object explicitly includes multiple road-user types, compare their distinct movement/escape mechanisms; do not silently narrow that scope to only the most familiar type. Replace redundant hypotheses rather than multiplying them. Keep at most 3 total instances. "
                    "If no distinct supported interaction is appropriate, explain why; never add an unrelated fault.")
                if inputs.get("road_user_focus_basis"):
                    inputs["interaction_coverage_policy"] = inputs["interaction_coverage_policy"].replace(
                        "explicitly source-named class", "method-supported analytical class")
            if task_scope == "CONTROL_INPUTS_ONLY":
                inputs.update(task_scope=task_scope, prompt_version=agent.CONTROL_PROMPT_VERSION)
            evidence = {"MF.description": {"value": mf.description, "sources": [vars(ref) for ref in mf.sources]},
                        "MF.functional_effect": {"value": mf.functional_effect, "sources": [vars(ref) for ref in mf.sources]},
                        "PROJECT.operating_context": {"value": inputs["project_context"], "sources": project.get("sources", [])},
                        "SCENARIO.conditions": inputs["conditions"], **source_evidence}
            inputs["source_routing"] = source_routing
            inputs["evidence"] = evidence
            if state.item_definition.get("item_sha256"):
                inputs["project_input_sha256"] = state.item_definition["item_sha256"]
            key = signature(inputs)
            context = {**inputs, "plan_key": key}
            saved = cache.get(key, {})
            entry = saved or {"context": context, "responses": [], "status": "REQUESTED"}
            cache[key] = entry
            # Old geometry diagnostics used an instantiated hypothesis identity,
            # so the correction contract could not find the reported world.
            # Migrate only that proven, failed correction; keep raw responses.
            unresolved_geometry = any(f.get("reason") == "CONTACT_WITH_STATIONARY_EGO"
                for row in entry.get("instantiations", {}).values() for f in row.get("failures", []))
            if ("CORRECTION_CHANGED_UNREPORTED_WORLD" in entry.get("semantic_correction_error", "")
                    or unresolved_geometry):
                hypotheses = entry.get("initial_plan", {}).get("hypotheses", [])
                corrected_diagnostics = []
                for diagnostic in entry.get("correction_diagnostics", []):
                    identity = diagnostic.get("hypothesis_id", "")
                    raw_identity = next((h["hypothesis_id"] for h in hypotheses
                        if identity.startswith(f"{key}:{h['hypothesis_id']}:")), None)
                    corrected_diagnostics.append({**diagnostic, "instance_hypothesis_id": identity,
                        "hypothesis_id": raw_identity} if raw_identity else diagnostic)
                if corrected_diagnostics != entry.get("correction_diagnostics", []):
                    entry.setdefault("correction_scope_history", []).append({
                        "error": entry.pop("semantic_correction_error", "LEGACY_UNSCOPED_GEOMETRY_DIAGNOSTIC"),
                        "diagnostics": entry["correction_diagnostics"]})
                    entry["correction_diagnostics"] = corrected_diagnostics
                    entry.pop("semantic_correction_completed", None)
            prior_status = entry.get("status")
            if prior_status == "INVALID":
                previous_limit = entry.get("output_limit_tokens")
                # Older checkpoints retained only this specific transport error.
                # Recover its limit, without treating semantic failures as retryable.
                if previous_limit is None and "finish_reason=length" in entry.get("error", ""):
                    match = re.search(r"本次max_tokens=(\d+)", entry["error"])
                    previous_limit = int(match[1]) if match else None
                if previous_limit and agent.max_tokens > previous_limit:
                    entry.setdefault("validation_history", []).append({
                        "reason": "OUTPUT_LIMIT_INCREASED", "error": entry.get("error", ""),
                        "previous_max_tokens": previous_limit, "max_tokens": agent.max_tokens})
                    entry["status"] = "REQUESTED"
                    entry.pop("error", None)
            if (prior_status == "VALIDATED" and entry.get("initial_plan")
                    and not (entry.get("semantic_correction_error") and entry["plan"] == entry["initial_plan"])):
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
            if (prior_status == "VALIDATED" and prior_validation != validation_version
                    and entry.get("initial_plan") and (entry.get("semantic_correction_error")
                        or entry.get("audit", {}).get("deferred_safe_states"))):
                # A failed optional correction kept valid motion. Reconsider
                # its saved reply once after a validator repair, just as for an
                # INVALID initial proposal; do not retry on every resume.
                entry.setdefault("validation_history", []).append({
                    "version": prior_validation, "error": entry.pop("semantic_correction_error",
                        "OPTIONAL_SAFE_STATE_DEPENDENCY_REVALIDATION")})
                entry.update(status="CORRECTION_REQUESTED", response_validation_version=validation_version)
                entry.pop("semantic_correction_completed", None)
            if (prior_status == "VALIDATED" and prior_validation != validation_version
                    and entry.get("initial_plan") and any(
                        str(f.get("reason", "")).startswith(("UNSUPPORTED_METHOD_VOCABULARY:", "GRAVITY_DIRECTION_CONFLICT:"))
                        for row in entry.get("instantiations", {}).values() for f in row.get("failures", []))):
                entry.setdefault("validation_history", []).append({"version": prior_validation,
                    "error": "Instantiation feedback revised for a retained failed proposal",
                    "diagnostics": entry.get("correction_diagnostics", [])})
                entry.update(status="CORRECTION_REQUESTED", response_validation_version=validation_version)
                entry.pop("semantic_correction_completed", None)
                entry.pop("correction_diagnostics", None)  # Regenerate feedback with the current owning instantiator.
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
                        candidate.analysis_instance["physical_plan_diagnostics"] = [{
                            "reason": "PHYSICAL_PLAN_INVALID", "task_scope": task_scope, "detail": entry["error"]}]
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
                    for parent, children, errors in rows:
                        driver = ("in_driver_seat" if parent.facts.get("driver_in_vehicle") is True else
                                  "outside_driver_seat" if parent.facts.get("driver_in_vehicle") is False else None)
                        for hypothesis in current_plan.hypotheses or ([current_plan.control_assessment] if current_plan.control_assessment else []):
                            if driver is None or driver not in hypothesis.get("driver_positions", [driver]):
                                continue
                            local = hypothesis.get("driver_control_assessments", {}).get(driver, {})
                            conflicts = (set(hypothesis.get("remaining_controls", []))
                                         & set(local.get("unresolved_controls", {}))) - {"driver_braking", "remote_control"}
                            if conflicts:
                                errors.append({"reason": "SAFE_STATE_SHARED_CONTROL_SCOPE_CONFLICT",
                                    "hypothesis_id": hypothesis.get("hypothesis_id", "control_assessment"),
                                    "driver_position": driver, "controls": sorted(conflicts),
                                    "resolution": "The common world declares these controls remaining, but this personnel branch calls them unresolved. Reconcile the scope in this same correction: retain an applicable common conditional path, or move the capability into only the branch where its explicit enable conditions hold and retain the concrete gap elsewhere. Driver absence alone does not invalidate an automatic command or actuator. Unverified independence must be treated consistently across the common world; do not invent proven capability, borrow a pedal outside, or change the trajectory."})
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
                            response_gap = _unmodeled_declared_response(child, method)
                            if response_gap:
                                errors.append(response_gap)
                            response_missing = (not proposal.get("transition") or
                                child.facts.get("fault_motion_model") == "RECTILINEAR_MOVING_TARGET_CONTACT"
                                and proposal.get("transition", {}).get("model") == "CONSTANT_NET_DECELERATION"
                                and proposal.get("transition", {}).get("post_stop_motion") != "HOLD_POSITION")
                            if (child.facts.get("fault_motion_model") in method.metadata.get("project_analysis_policy", {}).get("timing_analysis", {}).get("supported_models", [])
                                    and set(proposal.get("required_controls", [])) & {"braking", "driver_braking"}
                                    and response_missing and not response_hypothesis.get("timing_unresolved_reason")):
                                errors.append({"reason": "SAFE_STOP_RESPONSE_NOT_MODELED",
                                    "hypothesis_id": response_hypothesis.get("hypothesis_id", "control_assessment"),
                                    "driver_position": driver, "motion_model": child.facts.get("fault_motion_model"),
                                    "resolution": "Preserve this motion. Complete the proposed braking safe-state response with an explained conditional net-deceleration transition and, for a moving target, explicit post_stop_motion:HOLD_POSITION with a remaining-control basis; alternatively give timing_unresolved_reason for a concrete incompatibility. A missing measured value alone does not prohibit a conditional assumption. One existing correction only."})
                            if proposal and not response_missing:
                                from hara_agent.services.analysis.risk_calculation_input_service import RiskCalculationInputService
                                timing = RiskCalculationInputService().ftti(mf.malfunction_id, child.scenario_id, "",
                                    {**child.facts, "_fact_provenance": child.fact_provenance}, method=method)
                                if timing.timing_trace.get("status") in {"AVOIDABILITY_ALREADY_LOST", "NO_IMMEDIATE_SAFE_BRAKING_WINDOW"}:
                                    errors.append({"reason": "SAFE_STOP_RESPONSE_UNREACHABLE",
                                        "hypothesis_id": response_hypothesis.get("hypothesis_id", "control_assessment"),
                                        "driver_position": driver,
                                        **({"existing_unresolved_reason": response_hypothesis["safe_state_unresolved_reason"]}
                                           if response_hypothesis.get("safe_state_unresolved_reason") else {}),
                                        "evidence": timing.timing_trace,
                                        "resolution": "Preserve the unmitigated world. This proposed response does not avoid its target. Reconcile the safe-state claim with the available controls/model, or explicitly retain the unsupported strategy in safe_state_unresolved_reason. Do not move the target or increase braking just to force a safe result. No extra correction loop."})
                                elif (timing.timing_trace.get("status") == "CONDITIONAL_BOUNDARY"
                                        and timing.hazard_time_limit_s is None
                                        and not response_hypothesis.get("timing_unresolved_reason")):
                                    errors.append({"reason": "SAFE_STOP_FAULT_CLOCK_NOT_BOUND",
                                        "hypothesis_id": response_hypothesis.get("hypothesis_id", "control_assessment"),
                                        "driver_position": driver,
                                        "evidence": timing.timing_trace,
                                        "resolution": "The existing calculator found a response boundary, but its fault origin and hazard criterion are not bound. Preserve motion, target, response and driver scope. Complete hazard_timing from the same conditional world's initial phase and active timing policy, with its basis; do not default the offset to zero or equate contact time with FTTI. If the clock cannot be related, give a concrete timing_unresolved_reason. Use this same one correction only."})
                            avoidance = analysis.get("avoidance_validation", {})
                            if avoidance and avoidance.get("status") not in {"CONDITIONAL_ESCAPE_VALIDATED", "NO_CONTACT_TO_AVOID"}:
                                errors.append({"reason": avoidance["status"], "evidence": avoidance,
                                    "hypothesis_id": response_hypothesis.get("hypothesis_id"),
                                    "instance_hypothesis_id": analysis.get("hypothesis_id"),
                                    "driver_position": driver,
                                    "resolution": "Reconcile the proposed escape with this trajectory and corridor. Do not retain an unsupported positive avoidance claim or tune geometry just to obtain C1."})
                            counterfactual = analysis.get("stationary_ego_counterfactual", {})
                            if counterfactual.get("status") == "CONTACT":
                                errors.append({"reason": "CONTACT_WITH_STATIONARY_EGO", "evidence": counterfactual,
                                    "hypothesis_id": response_hypothesis.get("hypothesis_id"),
                                    "instance_hypothesis_id": analysis.get("hypothesis_id"),
                                    "driver_position": driver,
                                    "resolution": "This target also contacts the stationary ego. Establish the fault's incremental hazard (not merely contact), or propose a physically consistent distinct target path. Preserve the counterfactual; do not claim fault-induced contact without evidence."})
                            if outcome.get("reason") == "TARGET_OFFSET_POSITION_CONFLICT":
                                errors.append({"reason": outcome["reason"],
                                    "hypothesis_id": child.facts.get("physical_analysis", {}).get("hypothesis", {}).get("hypothesis_id"),
                                    "evidence": {k: child.facts.get(k) for k in (
                                        "object_position", "object_offset_x_m", "object_offset_y_m", "object_longitudinal_direction")},
                                    "resolution": "Reconcile this target's position with the supplied signed offsets in body x-forward/y-left coordinates (LEFT is positive y; RIGHT negative y). Correct the conflicting geometry/parameter and its explanation together; repeating these same fields leaves the conflict unresolved. Preserve valid hypotheses and the fault mechanism; do not tune parameters to force contact. If this model cannot represent the trajectory, state that limitation."})
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
                if (prior_status == "VALIDATED" and prior_validation != validation_version
                        and entry.get("semantic_correction_completed")
                        and any(f.get("reason") in {"CONTROL_AUTOMATIC_COMMAND_UNASSESSED", "SAFE_STATE_INDEPENDENT_RESPONSE_REQUIRED"} for f in failures)):
                    entry.setdefault("validation_history", []).append({"version": prior_validation,
                        "error": "Retained actuator needs current command/response requirement assessment",
                        "initial_plan": entry.get("initial_plan"), "diagnostics": entry.get("correction_diagnostics", [])})
                    entry.update(response_validation_version=validation_version,
                        initial_plan=plan.to_dict(), correction_diagnostics=failures)
                    entry.pop("semantic_correction_completed", None)
                # A computed failure is retained even when the proposal already
                # disclaims a reachable safe state. Repeating that disclaimer
                # adds no new semantic input; other gaps still get the same
                # bounded correction. Keep all failures in the child/audit.
                correction_failures = [failure for failure in failures if not (
                    failure.get("reason") == "SAFE_STOP_RESPONSE_UNREACHABLE"
                    and failure.get("existing_unresolved_reason"))]
                if (correction_failures or entry.get("status") == "CORRECTION_REQUESTED") and not entry.get("semantic_correction_completed"):
                    # One explicit, budgeted semantic correction. Persist the
                    # original response and feedback before transport so resume
                    # uses the identical response-checkpoint key.
                    entry.setdefault("initial_plan", plan.to_dict())
                    entry.setdefault("correction_diagnostics", correction_failures)
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
                    _retain_rejected_response(candidate, generated, entry, producer=producer,
                        mf=mf, method=method, key=key, project=project,
                        function=functions.get(mf.function_id))
                    if not plan.hypotheses and plan.control_assessment is None:
                        failures = [{"reason": "PHYSICAL_MODEL_NOT_APPLICABLE",
                                     "uncovered_conditions": list(plan.uncovered_conditions)}]
                    elif task_scope == "MOTION_AND_CONTROLS" and plan.hypotheses and not generated:
                        failures.append({"reason": "PHYSICAL_INSTANTIATION_FAILED", "task_scope": task_scope,
                            "detail": "; ".join(dict.fromkeys(f["reason"] for f in failures))})
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
                if isinstance(error, LLMOutputLimitError):
                    entry["output_limit_tokens"] = error.diagnostics.get("max_tokens", agent.max_tokens)
                else:
                    entry.pop("output_limit_tokens", None)
                for candidate in family:
                    candidate.analysis_instance["physical_plan_diagnostics"] = [{
                        "reason": "PHYSICAL_PLAN_INVALID", "task_scope": task_scope, "detail": str(error)}]
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
