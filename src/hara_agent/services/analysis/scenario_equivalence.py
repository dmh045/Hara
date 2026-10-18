"""Structured equivalence proofs shared by scheduling and report projection."""

from collections import defaultdict
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from typing import Any

from hara_agent.models import ScenarioCandidate


DRIVER_FIELDS = frozenset({"driver_position", "allowed_driver_position", "driver_in_vehicle"})


def signature(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":"), default=str).encode()).hexdigest()


def scenario_conditions(scenario: ScenarioCandidate, *, without_driver: bool = False) -> dict:
    """Identity from actual conditions, not IDs, display prose or ratings.

    Resolved atom IDs identify their governed semantics. Unresolved bindings
    remain in full: two unknown values are not silently declared equivalent.
    """
    facts = deepcopy(scenario.facts)
    if without_driver:
        for field in DRIVER_FIELDS:
            facts.pop(field, None)
    bindings = facts.get("method_scenario_dimensions", {})
    for dimension, binding in bindings.items() if isinstance(bindings, dict) else ():
        if isinstance(binding, dict) and binding.get("resolution_status") == "RESOLVED":
            bindings[dimension] = {key: binding.get(key) for key in (
                "atom_id", "canonical_atom_id", "filled_dimensions", "method_semantics",
                "applicability_status", "speed_intersection_kph",
            )}
    # This contains the same speed value plus extraction/selection history;
    # effective numeric constraints remain in facts and in the binding above.
    facts.pop("speed_context_resolution", None)
    authority = {
        field: {key: metadata.get(key) for key in (
            "provenance", "approval", "method_source_hash", "method_contract_hash",
            "analysis_value_semantic", "time_role", "unit", "source_refs",
            "applicable_scope", "validation_status", "input_fact_metadata",
            "object_id", "coordinate_frame", "fault_motion_model",
        ) if key in metadata}
        for field, metadata in scenario.fact_provenance.items()
        if field in facts and isinstance(metadata, dict)
    }
    local_ids = {getattr(scenario, "scenario_id", ""), getattr(scenario, "source_scenario_id", "")}
    instance = getattr(scenario, "analysis_instance", {})
    local_ids.update(instance.get("equivalent_parent_ids", []))
    local_ids.discard("")
    phases = {}

    def semantic(value, key=""):
        if isinstance(value, dict):
            result = {name: semantic(item, name) for name, item in sorted(value.items())
                    if name not in {"branch_id", "semantic_fingerprint", "display_label"}}
            if value.get("source_type") == "parent_scenario" and value.get("source_id") in local_ids:
                result["source_id"] = "$LOCAL_SCOPE"
            return result
        if isinstance(value, (list, tuple)):
            result = [semantic(item, key) for item in value]
            return sorted(result, key=lambda item: json.dumps(item, sort_keys=True, default=str)) if key in {"source_refs", "scenario_atom_ids"} else result
        if key in {"scenario_id", "parent_scenario_id"} and value in local_ids:
            return "$LOCAL_SCOPE"
        if key == "hazardous_event_id" and isinstance(value, str):
            return "::".join("$LOCAL_SCOPE" if part in local_ids else part for part in value.split("::"))
        if key in {"impact_time_id", "trajectory_id"} and value:
            phases.setdefault(str(value), f"$PHASE_{len(phases)}")
            return phases[str(value)]
        return value

    return semantic({"operating_mode": scenario.operating_mode, "facts": facts, "fact_authority": authority})


def coalesce_candidates(candidates: list[ScenarioCandidate]) -> tuple[list[ScenarioCandidate], list[dict]]:
    """Coalesce method-instantiated worlds with identical structured conditions.

    All generated identities are returned in the mapping. Callers persist the
    original candidates before this operation; this does not delete evidence.
    """
    groups = defaultdict(list)
    for candidate in candidates:
        if not candidate.facts.get("scenario_atom_ids"):
            key = candidate.scenario_id
        else:
            key = signature(scenario_conditions(candidate))
        groups[key].append(candidate)
    retained, mappings = [], []
    for key, members in groups.items():
        first = members[0]
        ids = [member.scenario_id for member in members]
        retained.append(replace(first, analysis_instance={
            **first.analysis_instance,
            "p6_generation": True,
            "equivalent_candidate_ids": ids,
            "equivalent_parent_ids": sorted({m.source_scenario_id for m in members}),
            "condition_signature": key,
        }))
        mappings.append({"representative_scenario_id": first.scenario_id,
                         "scenario_ids": ids, "condition_signature": key,
                         "basis": "IDENTICAL_METHOD_ATOMS_AND_STRUCTURED_CONDITIONS"})
    return retained, mappings


def driver_families(candidates: list[ScenarioCandidate]) -> list[list[ScenarioCandidate]]:
    groups = defaultdict(list)
    for candidate in candidates:
        groups[signature(scenario_conditions(candidate, without_driver=True))].append(candidate)
    return list(groups.values())


def driver_reuse_proof(assessment: dict, parent: ScenarioCandidate, child: ScenarioCandidate) -> dict:
    """Require explicit noninterference as well as identical non-driver inputs."""
    changed = sorted(key for key in DRIVER_FIELDS
                     if parent.facts.get(key) != child.facts.get(key))
    parent_conditions = scenario_conditions(parent, without_driver=True)
    child_conditions = scenario_conditions(child, without_driver=True)
    if parent_conditions["facts"] != child_conditions["facts"] or parent.operating_mode != child.operating_mode:
        return {"reusable": False, "reason": "NON_DRIVER_CONDITIONS_CHANGED"}
    if parent_conditions != child_conditions:
        return {"reusable": False, "reason": "NON_DRIVER_EVIDENCE_AUTHORITY_CHANGED"}
    metadata = assessment.get("dependency_metadata", {})
    dependencies = set().union(*(set(metadata.get(key, [])) for key in (
        "causal_evidence_fields", "physical_feasibility_fields", "scenario_identity_fields")))
    declared = set(metadata.get("noninterference_fields", []))
    if (metadata.get("complete") is not True or not metadata.get("child_subset_refinement")
            or not set(changed) <= declared or set(changed) & dependencies):
        return {"reusable": False, "reason": "DRIVER_NONINTERFERENCE_NOT_PROVEN"}
    causal_refs = {ref for hop in assessment.get("causal_chain", {}).values()
                   if isinstance(hop, dict) for ref in hop.get("evidence_refs", [])}
    if any(ref.removeprefix("SCN.") in changed for ref in causal_refs):
        return {"reusable": False, "reason": "DRIVER_FACT_USED_BY_CAUSAL_CHAIN"}
    return {"reusable": True, "reason": "VERIFIED_DRIVER_NONINTERFERENCE",
            "changed_fields": changed, "dependency_metadata": metadata}
