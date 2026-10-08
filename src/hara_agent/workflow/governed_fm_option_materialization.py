"""Materialize approved FM physical options from committed analytical children."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path

from hara_agent.method_sources import MethodSourceResolver
from hara_agent.method_sources.yaml_validation import validate_manifest
from hara_agent.models import MalfunctionCandidate, ReviewStatus, SourceRef
from hara_agent.services.analysis.scenario_method_service import ScenarioMethodService

from .checkpoints import CheckpointRepository
from .risk_rescoring import write_json
from .state import HARAState, WorkflowStage


def _malfunction(raw: dict) -> MalfunctionCandidate:
    payload = dict(raw)
    payload["sources"] = [SourceRef(**item) for item in payload.get("sources", [])]
    payload["status"] = ReviewStatus(payload.get("status", ReviewStatus.PENDING.value))
    return MalfunctionCandidate(**payload)


class GovernedFMOptionMaterializer:
    """Expand source options; changed physics always awaits causal validation."""

    def run(
        self, *, source_run_id: str, target_run_id: str,
        baseline: Path, source_baseline: Path, report_template: Path,
        run_dir: Path = Path("runtime/agent"),
        review_root: Path = Path("runtime/review"),
    ) -> dict:
        repository = CheckpointRepository(run_dir)
        source_path = repository.path_for(source_run_id)
        target_path = repository.path_for(target_run_id)
        review_dir = review_root / target_run_id
        if (
            source_run_id == target_run_id or target_path.exists()
            or review_dir.exists()
        ):
            raise ValueError("Use a new target run; committed artifacts are immutable")
        source = repository.load(source_run_id)
        old = MethodSourceResolver().resolve(
            template_path=None, baseline_manifest_path=source_baseline,
            report_template_path=report_template,
        ).method
        new = MethodSourceResolver().resolve(
            template_path=None, baseline_manifest_path=baseline,
            report_template_path=report_template,
        ).method
        old_manifest, _, old_hashes = validate_manifest(source_baseline.resolve())
        new_manifest, _, new_hashes = validate_manifest(baseline.resolve())
        policy_asset = new_manifest["normalized_sources"]["project_analysis_policy"]
        if not all((
            source.method_contract.get("method_source_hash") == old.metadata["method_source_hash"],
            old.contract_version == new.contract_version,
            old.compiler_version == new.compiler_version,
            old_manifest["sources"] == new_manifest["sources"],
            old_manifest["normalized_sources"] == new_manifest["normalized_sources"],
            set(old_hashes) == set(new_hashes),
            all(old_hashes[key] == new_hashes[key]
                for key in old_hashes if key != policy_asset),
            old_hashes[policy_asset] != new_hashes[policy_asset],
        )):
            raise ValueError("Method transition is not limited to the governed project policy")
        old_policy = old.metadata["project_analysis_policy"]
        new_policy = new.metadata["project_analysis_policy"]
        expected = deepcopy(old_policy)
        expected["engineering_decisions"] = new_policy.get("engineering_decisions", {})
        expected["source_ref"] = new_policy["source_ref"]
        if (
            old_policy.get("engineering_decisions")
            or not new_policy.get("engineering_decisions")
            or new_policy != expected
        ):
            raise ValueError("Project policy transition includes unrelated rules")
        decisions = new_policy["engineering_decisions"]
        template_decisions = [
            (key, value) for key, value in decisions.items()
            if isinstance(value, dict) and value.get("approved_template_id")
            and value.get("status") == "CONFIRMED_FOR_CURRENT_PROJECT"
        ]
        primary_decisions = [
            (key, value) for key, value in decisions.items()
            if isinstance(value, dict) and value.get("harm_mechanism_id")
            and value.get("status") == "CONFIRMED_FOR_CURRENT_PROJECT"
            and str(value.get("decision", "")).startswith("PRIMARY_")
        ]
        if len(template_decisions) != 1 or len(primary_decisions) != 1:
            raise ValueError("Expected one approved template and one primary harm decision")
        template_decision_id, d1 = template_decisions[0]
        harm_decision_id, d2 = primary_decisions[0]
        harm_type = str(d2["harm_mechanism_id"]).removeprefix("HARM_") + "_BRANCH"
        if not all((
            d1.get("malfunction_id") == d2.get("malfunction_id"),
            d1.get("effective_project_scope") == new_policy.get("project_scope"),
            d2.get("effective_project_scope") == new_policy.get("project_scope"),
            len(source.scenarios) == 6,
            all(item.analysis_instance.get("validation_status") == "VALIDATED"
                for item in source.scenarios),
        )):
            raise ValueError("Current source or approved decisions are incompatible")
        malfunction = next(
            (_malfunction(item) for item in source.malfunctions
             if item.get("malfunction_id") == d1["malfunction_id"]), None,
        )
        if malfunction is None:
            raise ValueError("Governed malfunction is absent from source")
        variants, option_audit = ScenarioMethodService(new).instantiate_analytical_candidates(
            malfunction, source.scenarios,
        )
        if (
            len(variants) != len(source.scenarios) * len(d1["approved_option_ids"])
            or option_audit.get("conflict_count")
            or option_audit.get("unmapped_value_count")
            or option_audit.get("selection_mode") != "STRONG_TEMPLATE_ANALYTICAL_INSTANCES"
        ):
            raise ValueError("Governed FM options did not materialize exactly")
        parents = {item.scenario_id: item for item in source.scenarios}
        policy_ref = new_policy["source_ref"]
        materialized = []
        for variant in variants:
            parent = parents[variant.source_scenario_id]
            facts = deepcopy(variant.facts)
            provenance = deepcopy(variant.fact_provenance)
            instance = deepcopy(variant.analysis_instance)
            scope = {
                "malfunction_id": malfunction.malfunction_id,
                "scenario_id": variant.scenario_id,
                "parent_scenario_id": parent.scenario_id,
            }
            for field, metadata in provenance.items():
                if not isinstance(metadata, dict):
                    continue
                prior_scope = metadata.get("applicable_scope", {})
                if (
                    isinstance(prior_scope, dict)
                    and prior_scope.get("scenario_id") == parent.scenario_id
                    and field in parent.facts
                    and parent.facts[field] == facts.get(field)
                ):
                    metadata["applicable_scope"] = scope
                    if metadata.get("analysis_assumption_scope"):
                        metadata["analysis_assumption_scope"] = scope
                    metadata["inherited_as_option_variant"] = True
                for ref in metadata.get("source_refs", []):
                    if (
                        isinstance(ref, dict)
                        and ref.get("source_type") == "method_contract"
                        and ref.get("source_id") == old.metadata["method_source_hash"]
                    ):
                        ref["source_id"] = new.metadata["method_source_hash"]
                        metadata["source_method_transition"] = old.metadata["method_source_hash"]
            facts["harm_mechanism_id"] = d2["harm_mechanism_id"]
            facts["harm_mechanism_type"] = harm_type
            for field in ("harm_mechanism_id", "harm_mechanism_type"):
                provenance[field] = {
                    "provenance": "SCENARIO_DEFINED", "origin": "SCENARIO_DEFINED",
                    "approval": "FINALIZED", "validation_status": "VALIDATED",
                    "applicable_scope": scope,
                    "source_refs": [{
                        **policy_ref,
                        "location": (
                            f"{policy_ref['location']}:"
                            f"engineering_decisions.{harm_decision_id}"
                        ),
                        "excerpt": d2["decision"],
                    }],
                    "selection_basis": "GOVERNED_PRIMARY_HARM_MECHANISM",
                }
            instance.update({
                "hazardous_event_id": parent.analysis_instance.get("hazardous_event_id", ""),
                "selected_atoms": parent.analysis_instance.get("selected_atoms", []),
                "original_parent_scenario_id": parent.analysis_instance.get("parent_scenario_id", ""),
                "original_child_scenario_id": parent.scenario_id,
                "driver_configuration_branch": parent.analysis_instance.get("driver_configuration_branch", {}),
                "harm_mechanism_id": d2["harm_mechanism_id"],
                "harm_mechanism_type": harm_type,
                f"{template_decision_id.split('_', 1)[0]}_decision_source_ref": policy_ref,
                f"{harm_decision_id.split('_', 1)[0]}_decision_source_ref": policy_ref,
            })
            materialized.append(replace(
                variant, facts=facts, fact_provenance=provenance,
                analysis_instance=instance,
            ))
        state = HARAState.read_committed(source.to_dict())
        state.run_id = target_run_id
        state.stage = WorkflowStage.BLOCKED
        state.scenarios = materialized
        state.item_definition = deepcopy(source.item_definition)
        state.item_definition["scenario_assessments"] = []
        state.risk_results = []
        state.safety_goals = []
        state.pending_reviews = []
        state.method_contract["template_hash"] = new.metadata["method_source_hash"]
        state.method_contract["method_source_hash"] = new.metadata["method_source_hash"]
        transition = {
            "source_method_hash": old.metadata["method_source_hash"],
            "target_method_hash": new.metadata["method_source_hash"],
            "changed_asset": policy_asset,
            "source_policy_sha256": old_hashes[policy_asset],
            "target_policy_sha256": new_hashes[policy_asset],
            "unchanged_asset_count": len(new_hashes) - 1,
        }
        state.audit_trail = [{
            "event": "scenario_synthesis_child_run_materialized",
            "source_run_id": source_run_id,
            "target_run_id": target_run_id,
            "child_count": len(materialized),
            "provider_calls_for_upstream_regeneration": 0,
            "method_transition": transition,
            "harm_mechanism_decision": d2["decision"],
        }]
        queue = {
            "artifact_version": "scenario-synthesis-causal-delta-queue-v1",
            "source_run_id": source_run_id,
            "run_id": target_run_id,
            "records": [{
                "malfunction_id": malfunction.malfunction_id,
                "parent_scenario_id": item.source_scenario_id,
                "child_scenario_id": item.scenario_id,
                "status": "CAUSAL_REVALIDATION_REQUIRED",
                "changed_fields": [
                    "harm_mechanism_id", "object_type", "object_position",
                    "relative_distance_m", "object_speed_kph", "collision_type",
                ],
                "provider_required": True,
                "reason": "New source-defined physical option and primary harm basis affect causal feasibility.",
            } for item in materialized],
            "summary": {
                "total": len(materialized), "deterministic_reuse": 0,
                "provider_revalidation": len(materialized),
                "causal_gap": 0, "source_conflict": 0,
            },
        }
        if len({item.scenario_id for item in materialized}) != len(materialized):
            raise ValueError("FM option variants do not have unique identities")
        review_dir.mkdir(parents=True, exist_ok=False)
        checkpoint = repository.save(state)
        write_json(review_dir / "causal_delta_queue.json", queue)
        audit = {
            "source_run_id": source_run_id, "target_run_id": target_run_id,
            "source_checkpoint_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
            "checkpoint": str(checkpoint), "method_transition": transition,
            "D1": d1, "D2": d2,
            "original_six_ids": [item.scenario_id for item in source.scenarios],
            "original_six_preserved_in_source": True,
            "option_audit": option_audit,
            "variant_count": len(materialized),
            "variants": [{
                "scenario_id": item.scenario_id,
                "original_child_id": item.source_scenario_id,
                "source_option_id": item.analysis_instance["source_option_id"],
                "driver_position": item.facts.get("driver_position"),
                "scenario_atom_ids": item.facts.get("scenario_atom_ids", []),
            } for item in materialized],
            "provider_calls": 0,
        }
        write_json(review_dir / "governed_fm_option_materialization.json", audit)
        return audit


__all__ = ["GovernedFMOptionMaterializer"]
