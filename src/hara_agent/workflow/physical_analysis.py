"""Explicit pre-causal scheduling for bounded physical analysis proposals."""
from collections import defaultdict
from dataclasses import asdict
import sys

from hara_agent.contracts.physical_scenario_plan import PhysicalScenarioPlan, PHYSICAL_PLAN_VERSION, MODEL_CATALOG
from hara_agent.services.analysis.physical_hypothesis_instantiation_service import PhysicalHypothesisInstantiationService
from hara_agent.services.analysis.scenario_equivalence import scenario_conditions, signature


def construct_physical_scenarios(state, candidates, malfunctions, *, method, agent, project_facts,
                                 checkpoint=None, progress=None):
    producer = PhysicalHypothesisInstantiationService(method)
    cache = state.item_definition.setdefault("physical_analysis_plans", {})
    metrics = {"existing_path_skipped": 0, "plan_requests": 0, "plan_cache_hits": 0,
               "physical_plan_reuse": 0, "generated_instances": 0, "schema_repairs": 0,
               "response_cache_hits": 0, "semantic_corrections": 0, "plan_reuse_records": [], "diagnostics": []}
    state.item_definition["physical_analysis_metrics"] = metrics
    budget = getattr(agent.client, "attempt_budget", None)
    starting_attempts = getattr(budget, "attempts", None)

    def reconcile_metrics():
        responses = [r for entry in cache.values() for r in entry.get("responses", [])]
        metrics["cumulative_proposal_responses"] = len(responses)
        metrics["cumulative_uncached_requests"] = sum(not r["response"]["usage"].get("response_checkpoint_hit") for r in responses)
        metrics["cumulative_schema_repair_responses"] = sum(r.get("schema_repair", False) for r in responses)
        metrics["token_usage"] = {key: sum(r["response"]["usage"].get(key, 0) for r in responses)
                                  for key in ("prompt_tokens", "completion_tokens")}
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
        groups, kept = defaultdict(list), []
        for candidate in candidates[mf.malfunction_id]:
            if not producer.needed(candidate, mf.malfunction_id):
                kept.append(candidate); metrics["existing_path_skipped"] += 1
            else:
                groups[signature(scenario_conditions(candidate, without_driver=True))].append(candidate)
        for condition_key, family in groups.items():
            first = family[0]
            # The same motion plan can be instantiated for multiple driver
            # configurations. Causality and risk are still evaluated per world.
            inputs = {"conditions": scenario_conditions(first, without_driver=True),
                "driver_policy": "Propose motion before driver intervention. Driver presence/enablement and causal relevance are evaluated independently for each instance; do not assume a driver's corrective action in the motion plan.",
                "malfunction": asdict(mf), "function": functions.get(mf.function_id, {}),
                "project_context": {k: v for k, v in project.items() if k not in {"risk_facts", "method_risk_fact_bindings", "sources"}},
                "project_physical_facts": [f for f in project.get("risk_facts", [])
                    if f.get("parameter") in {"longitudinal_acceleration_mps2", "slope_percent", "ego_speed_kph",
                        "relative_distance_m", "steering_control_available", "emergency_braking_available"}],
                "method_identity": {k: method.metadata.get(k) for k in ("method_source_hash", "project_analysis_policy")},
                "method_version": [method.contract_version, method.compiler_version],
                "risk_vocabulary": {"road_user_type": dict(method.structured_risk_method.severity.road_user_groups),
                                    "collision_type": dict(method.structured_risk_method.severity.collision_types)},
                "validation_sampling_policy": state.item_definition.get("physical_validation_policy", ""),
                "producer_version": PHYSICAL_PLAN_VERSION, "model_catalog_identity": signature(MODEL_CATALOG),
                "prompt_version": agent.PROMPT_VERSION, "route": route}
            evidence = {"MF.description": {"value": mf.description, "sources": [vars(ref) for ref in mf.sources]},
                        "MF.functional_effect": {"value": mf.functional_effect, "sources": [vars(ref) for ref in mf.sources]},
                        "PROJECT.operating_context": {"value": inputs["project_context"], "sources": project.get("sources", [])},
                        "SCENARIO.conditions": inputs["conditions"]}
            inputs["evidence"] = evidence
            key = signature(inputs)
            context = {**inputs, "plan_key": key}
            saved = cache.get(key, {})
            entry = saved or {"context": context, "responses": [], "status": "REQUESTED"}
            cache[key] = entry

            def persist_response(record):
                entry["responses"].append(record)
                reconcile_metrics()
                if checkpoint:
                    checkpoint(state)

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
                    print(f"[HARA] physical-plan {mf.malfunction_id}: {len(family)} instances, key={key[:12]}", file=sys.stderr, flush=True)
                    plan, audit = agent.propose(context, on_response=persist_response)
                    entry.update(status="VALIDATED", plan=plan.to_dict(), audit=audit)
                    metrics["plan_requests"] += audit["request_count"]
                    metrics["schema_repairs"] += audit["schema_repairs"]
                    metrics["response_cache_hits"] += audit["response_cache_hits"]
                def instantiate_family(current_plan):
                    return [(candidate, *producer.instantiate(candidate, mf, current_plan,
                        plan_key=key, response_model=entry["audit"]["model"], project_context=project,
                        function=functions.get(mf.function_id))) for candidate in family]

                instantiated = instantiate_family(plan)
                failures = [{"scenario_id": candidate.scenario_id, **failure}
                    for candidate, _, errors in instantiated for failure in errors]
                if (failures or entry.get("status") == "CORRECTION_REQUESTED") and not entry.get("semantic_correction_completed"):
                    # One explicit, budgeted semantic correction. Persist the
                    # original response and feedback before transport so resume
                    # uses the identical response-checkpoint key.
                    entry.setdefault("initial_plan", plan.to_dict())
                    entry.setdefault("correction_diagnostics", failures)
                    entry.update(status="CORRECTION_REQUESTED")
                    if checkpoint:
                        checkpoint(state)
                    correction_context = {**context, "previous_plan": entry["initial_plan"],
                        "correction_diagnostics": entry["correction_diagnostics"],
                        "correction_policy": "Correct only conflicts; preserve valid hypotheses. One semantic correction allowed."}
                    plan, audit = agent.propose(correction_context, on_response=persist_response)
                    entry.update(status="VALIDATED", plan=plan.to_dict(), audit=audit,
                                 semantic_correction_completed=True)
                    metrics["semantic_corrections"] += 1
                    metrics["plan_requests"] += audit["request_count"]
                    metrics["schema_repairs"] += audit["schema_repairs"]
                    metrics["response_cache_hits"] += audit["response_cache_hits"]
                    instantiated = instantiate_family(plan)
                metrics["physical_plan_reuse"] += len(family) - 1
                metrics["plan_reuse_records"].append({"plan_key": key,
                    "scenario_ids": [candidate.scenario_id for candidate in family],
                    "exact_input_keys": [signature(scenario_conditions(candidate)) for candidate in family]})
                for candidate, generated, failures in instantiated:
                    kept.extend(generated or [candidate])
                    if failures:
                        for item in generated or [candidate]:
                            item.analysis_instance["physical_plan_diagnostics"] = failures
                    metrics["generated_instances"] += len(generated)
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
                if checkpoint:
                    checkpoint(state)
        candidates[mf.malfunction_id] = kept
    return candidates
