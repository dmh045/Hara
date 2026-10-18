"""Frozen 1e150b2 scheduler ONLY for same-input offline performance comparison."""
from __future__ import annotations
from hara_agent.services.analysis.scenario_equivalence import driver_families, driver_reuse_proof
from hara_agent.services.semantic import ScenarioFeasibilityAgent
from hara_agent.models import evaluate_risk_eligibility

def _assess_instantiated_scenarios(
    agent: ScenarioFeasibilityAgent,
    malfunction: MalfunctionCandidate,
    candidates: list[ScenarioCandidate],
    instantiation: dict,
    project_registry,
) -> tuple[list[ScenarioFeasibilityAssessment], dict]:
    """Assess only the candidate instances selected for one malfunction."""
    families = driver_families(candidates) if any(
        candidate.analysis_instance.get("p6_generation") for candidate in candidates
    ) else [[candidate] for candidate in candidates]
    if getattr(agent, "paired_driver_requests", False):
        ordered = [candidate for family in families for candidate in family]
        previous_max_items = agent.batch_max_items
        try:
            agent.batch_max_items = min(previous_max_items, 2)
            assessments, audit = agent.assess(malfunction, ordered, project_registry=project_registry)
        finally:
            agent.batch_max_items = previous_max_items
        by_id = {item.scenario_id: item for item in assessments}
        audit["causal_reuse"] = {
            "mode": "SHARED_REQUEST_INDEPENDENT_JUDGEMENTS",
            "candidate_count": len(candidates), "representative_count": len(families),
            "independently_assessed_count": len(candidates), "reused_count": 0,
            "records": [],
        }
        audit["analytical_scenario_instantiation"] = instantiation
        return [by_id[c.scenario_id] for c in candidates], audit
    representatives = [family[0] for family in families]
    assessments, audit = agent.assess(malfunction, representatives, project_registry=project_registry)
    by_id = {item.scenario_id: item for item in assessments}
    pending, reuse_records = [], []
    for family in families:
        parent = family[0]
        assessed = by_id[parent.scenario_id]
        for child in family[1:]:
            proof = driver_reuse_proof(assessed.to_dict(), parent, child)
            reused = None
            if proof["reusable"] and evaluate_risk_eligibility(assessed).eligible:
                payload = {key: value for key, value in assessed.to_dict().items() if key in {
                    "physically_feasible", "functionally_relevant", "causally_relevant",
                    "breakpoint", "causal_chain", "risk_dimension_changes", "hazardous_event",
                    "confidence", "status", "dependency_metadata",
                }}
                payload["scenario_id"] = child.scenario_id
                changed_refs = {"SCN." + field for field in proof["changed_fields"]}
                payload["risk_dimension_changes"] = [
                    change for change in payload["risk_dimension_changes"]
                    if not changed_refs.intersection(change.get("evidence_refs", []))
                ]
                # Recompile and revalidate every evidence reference against the
                # child's registry. No copied graph retains a parent's IDs.
                try:
                    reused = ScenarioFeasibilityAgent._parse(
                        malfunction, payload, scenario=child, project_registry=project_registry,
                    )
                except ValueError:
                    proof = {"reusable": False, "reason": "CHILD_EVIDENCE_REVALIDATION_REQUIRED"}
            if reused is None:
                pending.append(child)
            else:
                by_id[child.scenario_id] = reused
            reuse_records.append({"scenario_id": child.scenario_id,
                                 "representative_scenario_id": parent.scenario_id,
                                 **proof, "reused": reused is not None})
    if pending:
        additional, additional_audit = agent.assess(malfunction, pending, project_registry=project_registry)
        by_id.update({item.scenario_id: item for item in additional})
        # Preserve existing Provider accounting across both scheduling waves.
        for key, value in additional_audit.items():
            previous = audit.get(key)
            if isinstance(value, list) and isinstance(previous, list):
                audit[key] = previous + value
            elif isinstance(value, (int, float)) and not isinstance(value, bool) and isinstance(previous, (int, float)):
                audit[key] = max(previous, value) if key.startswith("max_") else previous + value
        audit["usage"] = {"batches": [
            *audit.get("usage", {}).get("batches", []),
            *additional_audit.get("usage", {}).get("batches", []),
        ]}
    assessments = [by_id[candidate.scenario_id] for candidate in candidates]
    audit.update(candidate_count=len(candidates), expected_count=len(candidates),
                 returned_count=len(assessments), retained_count=sum(item.retain for item in assessments))
    audit["causal_reuse"] = {
        "candidate_count": len(candidates), "representative_count": len(representatives),
        "independently_assessed_count": len(representatives) + len(pending),
        "reused_count": sum(record["reused"] for record in reuse_records),
        "records": reuse_records,
    }
    audit["analytical_scenario_instantiation"] = instantiation
    return assessments, audit
