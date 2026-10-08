"""Offline quality metrics for the P5-D2 Scenario selector."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from typing import Any, Iterable, Mapping

from hara_agent.contracts import (
    ScenarioBindingAuthority, ScenarioDimensionApplicability,
    ScenarioSynthesisAssessment, ScenarioSynthesisInput,
)
from hara_agent.models import ScenarioCandidate
from hara_agent.services.analysis.scenario_selection_quality import (
    ScenarioVariantDiversityValidator,
)


def _average(values: Iterable[int]) -> float:
    items = list(values)
    return round(sum(items) / len(items), 3) if items else 0.0


def _fingerprint(selected: Mapping[str, tuple[str, ...]], *, omit_environment: bool = False) -> str:
    material = {
        dimension: list(atom_ids)
        for dimension, atom_ids in sorted(selected.items())
        if not omit_environment or dimension not in {"WHERE", "ROAD"}
    }
    return json.dumps(material, sort_keys=True, separators=(",", ":"))


class ScenarioSelectorQualityAudit:
    """Measure candidate governance and optional realized sibling selections."""

    def build(
        self, inputs: Iterable[ScenarioSynthesisInput], *,
        assessments_by_group: Mapping[str, tuple[ScenarioSynthesisAssessment, ...]] | None = None,
        children: Iterable[ScenarioCandidate] = (),
        mode: str = "OFFLINE_CANDIDATE_PLAN",
    ) -> dict[str, Any]:
        synthesis_inputs = tuple(inputs)
        assessments = dict(assessments_by_group or {})
        input_by_group = {item.semantic_group_id: item for item in synthesis_inputs}
        child_items = tuple(children)
        all_assessments = [
            (group_id, item)
            for group_id, values in assessments.items() for item in values
        ]

        applicability = Counter(
            candidate_set.applicability.status.value
            for item in synthesis_inputs for candidate_set in item.dimension_candidate_sets
        )
        traffic_sets = [
            candidate_set for item in synthesis_inputs
            for candidate_set in item.dimension_candidate_sets
            if candidate_set.dimension == "TRAFFIC_PATTERN"
        ]
        traffic_required_groups = {
            item.semantic_group_id for item in synthesis_inputs
            if next(
                value for value in item.dimension_candidate_sets
                if value.dimension == "TRAFFIC_PATTERN"
            ).applicability.status is ScenarioDimensionApplicability.REQUIRED
        }
        traffic_resolved_groups = {
            group_id for group_id, item in all_assessments
            if item.selected_atoms.get("TRAFFIC_PATTERN")
        }
        dynamics_sets = [
            candidate_set for item in synthesis_inputs
            for candidate_set in item.dimension_candidate_sets
            if candidate_set.dimension == "EGO_DYNAMICS"
        ]
        child_bindings = [
            child.facts.get("method_scenario_dimensions", {}).get("EGO_DYNAMICS", {})
            for child in child_items
        ]
        child_bindings = [item for item in child_bindings if isinstance(item, dict)]

        exact_duplicates = 0
        trivial_groups = 0
        primary_diverse_groups = 0
        environment_only_groups = 0
        for group_id, values in assessments.items():
            plan = input_by_group[group_id].coverage_plan
            fingerprints = [_fingerprint(item.selected_atoms) for item in values]
            exact_duplicates += len(fingerprints) - len(set(fingerprints))
            reasons = ScenarioVariantDiversityValidator.reasons(
                plan, (item.selected_atoms for item in values),
            )
            trivial_groups += "TRIVIAL_VARIANT_DIVERSITY" in reasons
            varying = {
                dimension for dimension in {
                    key for item in values for key in item.selected_atoms
                }
                if len({tuple(item.selected_atoms.get(dimension, ())) for item in values}) > 1
            }
            primary_diverse_groups += bool(varying & set(plan.primary_variation_dimensions))
            environment_only_groups += bool(varying) and varying <= {"WHERE", "ROAD"}

        full_clusters: dict[str, list[tuple[str, str]]] = defaultdict(list)
        shortlists: dict[str, str] = {}
        outputs: dict[str, tuple[str, ...]] = {}
        for synthesis_input in synthesis_inputs:
            shortlists[synthesis_input.semantic_group_id] = json.dumps({
                item.dimension: [candidate.atom_id for candidate in item.candidates]
                for item in synthesis_input.dimension_candidate_sets
            }, sort_keys=True, separators=(",", ":"))
        for group_id, assessment in all_assessments:
            key = _fingerprint(assessment.selected_atoms)
            full_clusters[key].append((input_by_group[group_id].malfunction_id, group_id))
            outputs.setdefault(group_id, ())
            outputs[group_id] = (*outputs[group_id], key)
        duplicate_clusters = [items for items in full_clusters.values() if len(items) > 1]
        cross_malfunction = [
            items for items in duplicate_clusters if len({item[0] for item in items}) > 1
        ]
        same_shortlist_same_output = 0
        different_shortlist_same_output = 0
        output_groups: dict[tuple[str, ...], list[str]] = defaultdict(list)
        for group_id, value in outputs.items():
            output_groups[tuple(sorted(value))].append(group_id)
        for group_ids in output_groups.values():
            if len(group_ids) < 2:
                continue
            shortlist_count = len({shortlists[group_id] for group_id in group_ids})
            same_shortlist_same_output += shortlist_count == 1
            different_shortlist_same_output += shortlist_count > 1

        complete_fingerprints = [
            _fingerprint(item.selected_atoms) for _, item in all_assessments
        ]
        non_environment_fingerprints = [
            _fingerprint(item.selected_atoms, omit_environment=True)
            for _, item in all_assessments
        ]
        template_selected = 0
        for group_id, assessment in all_assessments:
            sets = {item.dimension: item for item in input_by_group[group_id].dimension_candidate_sets}
            for dimension, atom_ids in assessment.selected_atoms.items():
                by_id = {item.atom_id: item for item in sets[dimension].candidates}
                template_selected += any(
                    by_id[atom_id].template_relationship != "NONE" for atom_id in atom_ids
                )

        result = {
            "artifact_version": "p5-d2-scenario-selector-quality-audit-v1",
            "mode": mode,
            "provider_calls": 0,
            "parent_groups": len(synthesis_inputs),
            "child_count": len(all_assessments),
            "child_group_distribution": {
                str(count): sum(len(values) == count for values in assessments.values())
                for count in (1, 2, 3)
            },
            "unique_full_atom_sets": len(set(complete_fingerprints)),
            "unique_atom_sets_excluding_environment": len(set(non_environment_fingerprints)),
            "dimension_applicability": {
                status.value: applicability[status.value]
                for status in ScenarioDimensionApplicability
            },
            "traffic_pattern": {
                "required": len(traffic_required_groups),
                "resolved": len(traffic_required_groups & traffic_resolved_groups),
                "required_but_missing": len(traffic_required_groups - traffic_resolved_groups),
                "not_applicable": sum(
                    item.applicability.status is ScenarioDimensionApplicability.NOT_APPLICABLE
                    for item in traffic_sets
                ),
            },
            "ego_dynamics": {
                "exact_locks": sum(
                    bool(item.locked_atom_ids) for item in dynamics_sets
                ),
                "broad_refinable_parents": sum(
                    item.binding_decision.authority is ScenarioBindingAuthority.RANGE_CONTAINMENT
                    and item.binding_decision.refinable for item in dynamics_sets
                ),
                "refined_children": sum(bool(item.get("child_refined_atom")) for item in child_bindings),
                "broad_bucket_retained": sum(
                    bool(item.get("parent_method_atom"))
                    and not item.get("child_refined_atom") for item in child_bindings
                ),
                "unique_dynamics": len({
                    str(item.get("canonical_atom_id", "")) for item in child_bindings
                    if item.get("canonical_atom_id")
                }),
            },
            "coverage_plan": {
                "primary_axis_distribution": dict(sorted(Counter(
                    dimension for item in synthesis_inputs
                    for dimension in item.coverage_plan.primary_variation_dimensions
                ).items())),
                "secondary_axis_distribution": dict(sorted(Counter(
                    dimension for item in synthesis_inputs
                    for dimension in item.coverage_plan.secondary_variation_dimensions
                ).items())),
                "variant_count_distribution": {
                    str(count): sum(item.coverage_plan.desired_variant_count == count for item in synthesis_inputs)
                    for count in (1, 2, 3)
                },
            },
            "sibling_quality": {
                "exact_duplicates": exact_duplicates,
                "trivial_diversity_groups": trivial_groups,
                "primary_axis_diversity_groups": primary_diverse_groups,
                "environment_only_diversity_groups": environment_only_groups,
            },
            "cross_malfunction": {
                "duplicate_clusters": len(cross_malfunction),
                "largest_duplicate_cluster": max(map(len, duplicate_clusters), default=0),
                "same_shortlist_same_output": same_shortlist_same_output,
                "different_shortlist_same_output": different_shortlist_same_output,
            },
            "ranking": {
                "average_catalog_size": _average(
                    item.catalog_size for value in synthesis_inputs
                    for item in value.dimension_candidate_sets
                ),
                "average_hard_filtered_pool": _average(
                    item.hard_filtered_pool_size for value in synthesis_inputs
                    for item in value.dimension_candidate_sets
                ),
                "average_final_shortlist": _average(
                    len(item.candidates) for value in synthesis_inputs
                    for item in value.dimension_candidate_sets
                ),
                "dimension_shortlist_truncated_groups": sum(
                    any(item.shortlist_truncated for item in value.dimension_candidate_sets)
                    for value in synthesis_inputs
                ),
                "dimension_shortlist_truncated_by_dimension": dict(sorted(Counter(
                    item.dimension for value in synthesis_inputs
                    for item in value.dimension_candidate_sets if item.shortlist_truncated
                ).items())),
                "combination_beam_truncated_groups": 0,
                "template_authoritative_candidate_usage": template_selected,
            },
        }
        return result

    @staticmethod
    def markdown(audit: Mapping[str, Any]) -> str:
        traffic = audit["traffic_pattern"]
        dynamics = audit["ego_dynamics"]
        ranking = audit["ranking"]
        return "\n".join((
            "# P5-D2 Scenario Selector Quality Audit", "",
            f"- Mode: `{audit['mode']}`",
            f"- Provider calls: {audit['provider_calls']}",
            f"- Parent groups: {audit['parent_groups']}",
            f"- Realized child selections: {audit['child_count']}",
            f"- TRAFFIC_PATTERN required / resolved / missing: {traffic['required']} / {traffic['resolved']} / {traffic['required_but_missing']}",
            f"- EGO_DYNAMICS exact locks / broad refinable: {dynamics['exact_locks']} / {dynamics['broad_refinable_parents']}",
            f"- Average catalog / hard-filtered / shortlist: {ranking['average_catalog_size']} / {ranking['average_hard_filtered_pool']} / {ranking['average_final_shortlist']}",
            f"- Dimension shortlist truncated groups: {ranking['dimension_shortlist_truncated_groups']}",
            f"- Combination beam truncated groups: {ranking['combination_beam_truncated_groups']}",
            "",
            "Candidate-plan mode reports unresolved REQUIRED dimensions as missing; it does not claim that a Provider selection ran.",
        )) + "\n"


__all__ = ["ScenarioSelectorQualityAudit"]
