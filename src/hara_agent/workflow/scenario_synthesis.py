from __future__ import annotations

from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from hara_agent.contracts import (
    CoverageLabel, MethodContract, ScenarioSynthesisAssessment,
    SynthesisValidationStatus,
)
from hara_agent.models import (
    EvidenceValue, ReviewStatus, RiskAssessment, ScenarioCandidate,
    evaluate_risk_eligibility_payload,
)
from hara_agent.services.analysis import (
    AnalyticalPhysicsInstantiationService, ConstrainedScenarioSynthesisService,
    ExposureInputReadinessService, RiskExecutionTraceService,
)
from hara_agent.services.analysis.scenario_selection_quality import ScenarioShortlistPolicy
from hara_agent.services.analysis.driver_configuration_service import (
    DriverConfigurationBrancher,
)
from hara_agent.services.reporting import (
    OfflineReportRebuilder, ScenarioSelectorQualityAudit,
)
from hara_agent.services.semantic.scenario_synthesis_agent import (
    BoundedScenarioSynthesisAgent, SCENARIO_SYNTHESIS_PROMPT_VERSION,
)
from hara_agent.workflow.checkpoints import CheckpointRepository
from hara_agent.workflow.review_artifacts import ReviewArtifactWriter
from hara_agent.workflow.state import HARAState, WorkflowStage


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent,
        prefix=f".{path.stem}-", suffix=".tmp", delete=False,
    ) as stream:
        json.dump(payload, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", newline="\n", dir=path.parent,
        prefix=f".{path.stem}-", suffix=".tmp", delete=False,
    ) as stream:
        stream.write(value)
        temporary = Path(stream.name)
    try:
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


class ScenarioSynthesisRunner:
    """Execute a child-only P5-D run without mutating the accepted parent."""

    def __init__(
        self, *, method: MethodContract, client: Any | None,
        run_dir: str | Path = "runtime/agent",
        review_root: str | Path = "runtime/review",
    ):
        self.method = method
        self.client = client
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.review_root = Path(review_root).expanduser().resolve()
        self.synthesis = ConstrainedScenarioSynthesisService(method)
        self.physics = AnalyticalPhysicsInstantiationService(method)
        self.exposure = ExposureInputReadinessService(method)
        policy = method.metadata.get("project_analysis_policy", {})
        policy = policy if isinstance(policy, dict) else {}
        self.driver_brancher = DriverConfigurationBrancher(
            policy_id=str(policy.get("policy_id", "")),
        )

    def _parent_inventory(self, source_run_id: str) -> dict[str, str]:
        paths = [self.run_dir / f"{source_run_id}.checkpoint.json"]
        review_dir = self.review_root / source_run_id
        if review_dir.is_dir():
            paths.extend(sorted(path for path in review_dir.rglob("*") if path.is_file()))
        return {
            str(path): _sha256(path) for path in paths if path.is_file()
        }

    @staticmethod
    def _eligible_records(state: HARAState) -> list[dict[str, Any]]:
        return [
            item for item in state.item_definition.get("scenario_assessments", [])
            if isinstance(item, dict) and evaluate_risk_eligibility_payload(item).eligible
        ]

    @staticmethod
    def _project_context(state: HARAState) -> dict[str, Any]:
        typed = state.item_definition.get("typed", {})
        if not isinstance(typed, dict):
            return {}
        return {
            key: deepcopy(typed.get(key)) for key in (
                "system_description", "item_boundary", "operating_modes",
                "odd_locations", "odd_road_types", "odd_weather_conditions",
                "odd_road_surfaces", "speed_min_kph", "speed_max_kph",
                "speed_envelopes",
            ) if key in typed
        }

    def _prepare(
        self, state: HARAState,
    ) -> tuple[list[Any], dict[str, ScenarioCandidate], dict[str, dict[str, Any]],
               dict[tuple[str, str], dict[str, Any]], dict[str, Any]]:
        scenarios = {item.scenario_id: item for item in state.scenarios}
        malfunctions = {
            str(item.get("malfunction_id", "")): item
            for item in state.malfunctions if isinstance(item, dict)
        }
        functions = {
            str(item.get("function_id", "")): item
            for item in state.functions if isinstance(item, dict)
        }
        assessments = {
            (str(item.get("malfunction_id", "")), str(item.get("scenario_id", ""))): item
            for item in self._eligible_records(state)
        }
        inputs = []
        missing_foreign_keys = []
        project_context = self._project_context(state)
        for key, assessment in sorted(assessments.items()):
            malfunction = malfunctions.get(key[0])
            parent = scenarios.get(key[1])
            if malfunction is None or parent is None:
                missing_foreign_keys.append({
                    "malfunction_id": key[0], "scenario_id": key[1],
                })
                continue
            inputs.append(self.synthesis.build_input(
                malfunction=malfunction, parent=parent, assessment=assessment,
                project_context=project_context,
                function=functions.get(str(malfunction.get("function_id", "")), {}),
            ))
        before = Counter()
        for synthesis_input in inputs:
            for item in synthesis_input.dimension_candidate_sets:
                if item.resolution_status_before.upper() == "RESOLVED":
                    before[item.dimension] += 1
        shortlist_truncation_by_dimension = Counter(
            candidate_set.dimension
            for synthesis_input in inputs
            for candidate_set in synthesis_input.dimension_candidate_sets
            if candidate_set.shortlist_truncated
        )
        stats = {
            "eligible_parent_he": len(assessments),
            "unique_parent_scenarios": len({item.parent_scenario_id for item in inputs}),
            "unique_semantic_synthesis_groups": len({item.semantic_group_id for item in inputs}),
            "missing_foreign_keys": missing_foreign_keys,
            "dimension_resolution_before": {
                dimension: before[dimension] for dimension in self.synthesis.dimensions
            },
            "catalog_candidates_total": sum(
                item.catalog_size for synthesis_input in inputs
                for item in synthesis_input.dimension_candidate_sets
            ),
            "shortlisted_candidates_total": sum(
                len(item.candidates) for synthesis_input in inputs
                for item in synthesis_input.dimension_candidate_sets
            ),
            "hard_filtered_candidates_total": sum(
                item.hard_filtered_pool_size for synthesis_input in inputs
                for item in synthesis_input.dimension_candidate_sets
            ),
            "dimension_shortlist_truncated_groups": sum(
                any(item.shortlist_truncated for item in synthesis_input.dimension_candidate_sets)
                for synthesis_input in inputs
            ),
            "dimension_shortlist_truncated_by_dimension": {
                dimension: shortlist_truncation_by_dimension[dimension]
                for dimension in self.synthesis.dimensions
            },
            "combination_beam_truncated_groups": 0,
        }
        denominator = max(1, len(inputs) * len(self.synthesis.dimensions))
        stats["average_catalog_candidates_per_dimension"] = round(
            stats["catalog_candidates_total"] / denominator, 3
        )
        stats["average_shortlist_candidates_per_dimension"] = round(
            stats["shortlisted_candidates_total"] / denominator, 3
        )
        stats["average_hard_filtered_candidates_per_dimension"] = round(
            stats["hard_filtered_candidates_total"] / denominator, 3
        )
        stats["max_shortlist_candidates_per_dimension"] = max((
            len(item.candidates) for synthesis_input in inputs
            for item in synthesis_input.dimension_candidate_sets
        ), default=0)
        return inputs, scenarios, malfunctions, assessments, {"stats": stats}

    @staticmethod
    def _provider_ready(synthesis_input: Any) -> bool:
        if synthesis_input.contextual_speed.get("status") == "SOURCE_CONFLICT":
            return False
        return all(
            (
                bool(item.candidates) and item.generation_status != "METHOD_GAP"
                if item.applicability.status.value == "REQUIRED"
                else not item.candidates
                if item.applicability.status.value == "NOT_APPLICABLE"
                else True
            )
            for item in synthesis_input.dimension_candidate_sets
        )

    def _domain(self, malfunction: dict[str, Any]) -> str:
        structured = self.method.structured_risk_method
        if structured is None:
            return ""
        category = str(malfunction.get("component_category", ""))
        matches = [
            rule.domain.value for rule in structured.exposure.domain_rules
            if category in rule.component_categories
        ]
        return matches[0] if len(matches) == 1 else ""

    @staticmethod
    def _stable_identity(item: Any) -> tuple[str, str, str]:
        return (
            item.malfunction_id, item.parent_scenario_id, item.hazardous_event_id,
        )

    def _apply_smoke_plan_overrides(
        self, inputs: list[Any], identities: list[dict[str, Any]] | None,
    ) -> list[Any]:
        """Preserve a historical smoke's requested shape without changing Method scope."""
        if identities is None:
            return inputs
        requested = {}
        for record in identities:
            identity = (
                str(record.get("malfunction_id", "")),
                str(record.get("parent_scenario_id", "")),
                str(record.get("hazardous_event_id", "")),
            )
            raw_count = record.get("requested_variant_count")
            if raw_count in (None, ""):
                continue
            count = int(raw_count)
            if count not in {1, 2, 3}:
                raise ValueError(f"Invalid historical smoke variant count: {count}")
            requested[identity] = count
        result = []
        for item in inputs:
            count = requested.get(self._stable_identity(item))
            current = item.coverage_plan.desired_variant_count
            if count is None or count == current:
                result.append(item)
                continue
            if count > current:
                raise ValueError(
                    "Historical smoke variant count exceeds the current supported "
                    f"Coverage Plan for {self._stable_identity(item)}: {count}>{current}"
                )
            result.append(replace(
                item,
                coverage_plan=replace(
                    item.coverage_plan,
                    desired_variant_count=count,
                    variant_intents=item.coverage_plan.variant_intents[:count],
                ),
            ))
        return result

    def _smoke_groups(
        self, inputs: list[Any], count: int,
        identities: list[dict[str, Any]] | None = None,
    ) -> list[Any]:
        ready = sorted(
            (item for item in inputs if self._provider_ready(item)),
            key=lambda item: item.semantic_group_id,
        )
        if identities is not None:
            by_identity = {self._stable_identity(item): item for item in ready}
            requested = [(
                str(item.get("malfunction_id", "")),
                str(item.get("parent_scenario_id", "")),
                str(item.get("hazardous_event_id", "")),
            ) for item in identities]
            missing = [identity for identity in requested if identity not in by_identity]
            if missing:
                raise ValueError(
                    "Requested recovery-smoke identities are not Provider-ready: "
                    + json.dumps(missing, ensure_ascii=False)
                )
            if len(requested) != len(set(requested)):
                raise ValueError("Recovery-smoke identities must be unique")
            return [by_identity[identity] for identity in requested]
        target_strata = (
            "vehicle_interaction", "vru", "reversing", "parking_in_out",
            "slope_road_relation", "strong_fm_template", "no_fm_template",
            "coverage_2", "coverage_3", "primary_ego_action",
            "primary_ego_dynamics", "primary_object", "domain_z", "domain_f",
        )
        chosen: list[Any] = []
        counts: Counter[str] = Counter()
        chosen_malfunctions: set[str] = set()
        while len(chosen) < count:
            candidates = [item for item in ready if item not in chosen]
            if not candidates:
                break
            best = min(candidates, key=lambda item: (
                -sum(
                    1.0 / (1 + counts[feature])
                    for feature in self._smoke_features(item)
                    if feature in target_strata
                ),
                -(item.malfunction_id not in chosen_malfunctions),
                item.semantic_group_id,
            ))
            chosen.append(best)
            counts.update(self._smoke_features(best))
            chosen_malfunctions.add(best.malfunction_id)
        return chosen

    def _smoke_features(self, item: Any) -> set[str]:
        query = item.structured_semantic_query
        objects = set(map(str, query.get("object_categories", [])))
        actions = set(map(str, query.get("action_categories", [])))
        road = set(map(str, query.get("road_relations", [])))
        road.update(map(str, query.get("odd_road_categories", [])))
        text = json.dumps({
            "malfunction": item.malfunction,
            "parent": item.parent_scenario.get("facts", {}),
        }, ensure_ascii=False).casefold()
        primary = set(item.coverage_plan.primary_variation_dimensions)
        domain = self._domain(item.malfunction)
        result = set()
        if "OBJECT_VEHICLE" in objects:
            result.add("vehicle_interaction")
        if objects & {"OBJECT_PEDESTRIAN", "OBJECT_CYCLIST"}:
            result.add("vru")
        if "ACTION_REVERSE" in actions:
            result.add("reversing")
        if "ACTION_PARK" in actions or any(
            token in text for token in ("停车", "泊车", "parking", "park-in", "park-out")
        ):
            result.add("parking_in_out")
        if "ROAD_SLOPE" in road:
            result.add("slope_road_relation")
        result.add(
            "strong_fm_template" if item.fm_scenario_template else "no_fm_template"
        )
        result.add(f"coverage_{item.coverage_plan.desired_variant_count}")
        for dimension, stratum in (
            ("EGO_ACTION", "primary_ego_action"),
            ("EGO_DYNAMICS", "primary_ego_dynamics"),
            ("OBJECT", "primary_object"),
        ):
            if dimension in primary:
                result.add(stratum)
        if domain in {"Z", "F"}:
            result.add(f"domain_{domain.casefold()}")
        return result

    def _smoke_strata(self, inputs: list[Any]) -> dict[str, Any]:
        target = {
            "vehicle_interaction", "vru", "reversing", "parking_in_out",
            "slope_road_relation", "strong_fm_template", "no_fm_template",
            "coverage_2", "coverage_3", "primary_ego_action",
            "primary_ego_dynamics", "primary_object", "domain_z", "domain_f",
        }
        by_group = {
            item.semantic_group_id: sorted(self._smoke_features(item)) for item in inputs
        }
        covered = set().union(*(set(value) for value in by_group.values())) if by_group else set()
        return {
            "target": sorted(target), "covered": sorted(covered & target),
            "unavailable": sorted(target - covered), "by_group": by_group,
        }

    @staticmethod
    def _trace_call_count(trace: dict[str, Any]) -> int:
        return len(trace.get("calls", []))

    def _smoke_passed(self, traces: list[dict[str, Any]], expected_count: int) -> tuple[bool, list[str]]:
        failures = []
        if len(traces) != expected_count or any(item.get("status") != "PASS" for item in traces):
            failures.append("SMOKE_SELECTION_OR_SCHEMA_FAILURE")
        all_calls = [call for trace in traces for call in trace.get("calls", [])]
        successful_calls = [
            call for call in all_calls
            if call.get("schema_status") == "PASS"
        ]
        if any(call.get("schema_status") != "PASS" for call in all_calls):
            failures.append("SCHEMA_NOT_ALL_PASS")
        models = {str(call.get("resolved_model", "")) for call in successful_calls if call.get("resolved_model")}
        if len(models) != 1:
            failures.append("RESOLVED_MODEL_DRIFT")
        elif models != {"doubao-seed-2-1-turbo-260628"}:
            failures.append("RESOLVED_MODEL_UNEXPECTED")
        configured = {
            str(call.get("configured_model", "")) for call in all_calls
            if call.get("configured_model")
        }
        if configured != {"doubao-seed-2.0-pro"}:
            failures.append("CONFIGURED_MODEL_UNEXPECTED")
        if any(str(call.get("thinking", "")) != "disabled" for call in successful_calls):
            failures.append("THINKING_POLICY_CHANGED")
        if any(int(call.get("reasoning_characters", 0) or 0) != 0 for call in successful_calls):
            failures.append("REASONING_CONTENT_PRESENT")
        if any(str(call.get("finish_reason", "")) != "stop" for call in successful_calls):
            failures.append("FINISH_REASON_NOT_STOP")
        return not failures, failures

    def _run_provider_groups(
        self, *, inputs: list[Any], max_workers: int,
        progress_path: Path, phase: str,
        existing: dict[str, tuple[tuple[ScenarioSynthesisAssessment, ...], dict[str, Any]]],
    ) -> dict[str, tuple[tuple[ScenarioSynthesisAssessment, ...], dict[str, Any]]]:
        if self.client is None:
            raise ValueError("Provider execution requested without a configured client")
        agent = BoundedScenarioSynthesisAgent(self.client, self.synthesis)
        pending = [item for item in inputs if item.semantic_group_id not in existing]
        if not pending:
            return existing
        completed = 0
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            futures = {executor.submit(agent.select, item): item for item in pending}
            for future in as_completed(futures):
                item = futures[future]
                assessments, trace = future.result()
                existing[item.semantic_group_id] = (assessments, trace)
                completed += 1
                print(
                    f"[HARA][P5-D] {phase} {completed}/{len(pending)} "
                    f"group={item.semantic_group_id} status={trace.get('status')}",
                    flush=True,
                )
                _write_json(progress_path, {
                    "artifact_version": "scenario-synthesis-provider-trace-v1",
                    "phase": phase,
                    "completed_groups": completed,
                    "target_groups": len(pending),
                    "groups": [value[1] for _, value in sorted(existing.items())],
                })
        return existing

    def _recover_completed_smoke(
        self, *, trace_path: Path, audit_path: Path,
        smoke_inputs: list[Any], source_run_id: str, target_run_id: str,
        parent_inventory: dict[str, str],
    ) -> dict[str, tuple[tuple[ScenarioSynthesisAssessment, ...], dict[str, Any]]]:
        """Reuse a fully committed Provider trace after an offline-only failure."""
        if not trace_path.is_file() or not audit_path.is_file():
            raise ValueError("Completed smoke trace and source audit are required for resume")
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
        if (
            audit.get("source_run_id") != source_run_id
            or audit.get("run_id") != target_run_id
            or audit.get("parent_artifact_hashes_before") != parent_inventory
            or audit.get("parent_artifacts_mutated") is not False
        ):
            raise ValueError("Completed smoke trace source binding differs")
        smoke = trace.get("smoke", {})
        expected_ids = [item.semantic_group_id for item in smoke_inputs]
        expected_identities = [{
            "malfunction_id": item.malfunction_id,
            "parent_scenario_id": item.parent_scenario_id,
            "hazardous_event_id": item.hazardous_event_id,
            "requested_variant_count": item.coverage_plan.desired_variant_count,
        } for item in smoke_inputs]
        if (
            trace.get("artifact_version") != "scenario-synthesis-provider-trace-v1"
            or smoke.get("passed") is not True
            or smoke.get("requested") != len(smoke_inputs)
            or smoke.get("executed") != len(smoke_inputs)
            or smoke.get("semantic_group_ids") != expected_ids
            or smoke.get("stable_identities") != expected_identities
        ):
            raise ValueError("Completed smoke trace identities or status differ")
        group_values = trace.get("groups", [])
        record_values = smoke.get("selection_records", [])
        groups = {item.get("semantic_group_id"): item for item in group_values}
        records = {
            item.get("semantic_group_id"): item
            for item in record_values
        }
        if (
            len(group_values) != len(expected_ids)
            or len(record_values) != len(expected_ids)
            or set(groups) != set(expected_ids)
            or set(records) != set(expected_ids)
        ):
            raise ValueError("Completed smoke trace does not cover the selected groups")
        recovered = {}
        for item in smoke_inputs:
            group_id = item.semantic_group_id
            group = groups[group_id]
            variants = records[group_id].get("variants", [])
            if group.get("status") != "PASS" or len(variants) != item.coverage_plan.desired_variant_count:
                raise ValueError("Completed smoke trace has an invalid group result")
            assessments = []
            expected_labels = [
                str(intent.get("coverage_label", ""))
                for intent in item.coverage_plan.variant_intents
            ]
            for index, raw in enumerate(variants):
                if (
                    raw.get("semantic_group_id") != group_id
                    or raw.get("validation_status") != "VALIDATED"
                    or not isinstance(raw.get("selected_atoms"), dict)
                    or raw.get("coverage_label") != expected_labels[index]
                    or not str(raw.get("semantic_rationale", "")).strip()
                    or not isinstance(raw.get("context_refs"), list)
                    or not raw["context_refs"]
                    or any(not isinstance(ref, str) or not ref for ref in raw["context_refs"])
                    or any(
                        not isinstance(atom_ids, list)
                        or any(not isinstance(atom_id, str) or not atom_id for atom_id in atom_ids)
                        for atom_ids in raw["selected_atoms"].values()
                    )
                ):
                    raise ValueError("Completed smoke trace has an invalid assessment")
                assessments.append(ScenarioSynthesisAssessment(
                    semantic_group_id=group_id,
                    coverage_label=CoverageLabel(raw["coverage_label"]),
                    selected_atoms={
                        dimension: tuple(atom_ids)
                        for dimension, atom_ids in raw["selected_atoms"].items()
                    },
                    semantic_rationale=str(raw["semantic_rationale"]),
                    context_refs=tuple(raw["context_refs"]),
                    validation_status=SynthesisValidationStatus.VALIDATED,
                    validation_reasons=tuple(raw.get("validation_reasons", [])),
                    selection_authority=str(raw.get("selection_authority", "")),
                ))
            recovered[group_id] = (tuple(assessments), group)
        return recovered

    @staticmethod
    def _dependency_metadata(assessment: dict[str, Any]) -> dict[str, Any] | None:
        causal = assessment.get("causal_assessment", {})
        causal = causal if isinstance(causal, dict) else {}
        value = assessment.get("dependency_metadata", causal.get("dependency_metadata"))
        required = (
            "causal_evidence_fields", "physical_feasibility_fields",
            "scenario_identity_fields",
        )
        if (
            not isinstance(value, dict) or value.get("complete") is not True
            or not all(isinstance(value.get(field), list) for field in required)
            or not str(value.get("child_subset_refinement", "")).strip()
        ):
            return None
        return value

    def _causal_delta(
        self, *, synthesis_input: Any, parent_assessment: dict[str, Any],
        child: ScenarioCandidate,
    ) -> dict[str, Any]:
        changed_dimensions = sorted(
            dimension for dimension, binding in child.facts.get("method_scenario_dimensions", {}).items()
            if binding.get("resolution_status") == "RESOLVED"
            and dimension not in {
                item.dimension for item in synthesis_input.dimension_candidate_sets
                if item.locked_atom_ids
            }
        )
        driver_branch = child.analysis_instance.get("driver_configuration_branch", {})
        changed_driver_fields = (
            ["driver_position", "allowed_driver_position", "driver_in_vehicle"]
            if isinstance(driver_branch, dict) and driver_branch else []
        )
        metadata = self._dependency_metadata(parent_assessment)
        if metadata is None:
            return {
                "malfunction_id": synthesis_input.malfunction_id,
                "parent_scenario_id": synthesis_input.parent_scenario_id,
                "child_scenario_id": child.scenario_id,
                "status": "CAUSAL_REVALIDATION_REQUIRED",
                "changed_dimensions": changed_dimensions,
                "changed_fields": changed_driver_fields,
                "causal_reuse_basis": "",
                "reason": "DEPENDENCY_METADATA_INCOMPLETE",
                "provider_required": True,
            }
        dependencies = {
            str(item).removeprefix("SCN.")
            for field in (
                "causal_evidence_fields", "physical_feasibility_fields",
                "scenario_identity_fields",
            ) for item in metadata[field] if isinstance(item, str)
        }
        changed_keys = set(changed_dimensions) | {
            {
                "WHERE": "operating_scenario", "ROAD": "road_surface_conditions",
                "EGO_ACTION": "vehicle_state", "OBJECT": "scenario_object_atom",
                "TRAFFIC_PATTERN": "traffic_pattern", "EGO_X_ROAD": "ego_road_relation",
            }.get(item, item) for item in changed_dimensions
        }
        changed_keys.update(changed_driver_fields)
        dependent = sorted(changed_keys & dependencies)
        if dependent:
            return {
                "malfunction_id": synthesis_input.malfunction_id,
                "parent_scenario_id": synthesis_input.parent_scenario_id,
                "child_scenario_id": child.scenario_id,
                "status": "CAUSAL_REVALIDATION_REQUIRED",
                "changed_dimensions": changed_dimensions,
                "changed_fields": changed_driver_fields,
                "dependent_fields": dependent,
                "causal_reuse_basis": "",
                "reason": "EXPLICIT_CAUSAL_OR_IDENTITY_DEPENDENCY",
                "provider_required": True,
            }
        if changed_driver_fields and metadata.get("driver_configuration_noninterference") is not True:
            return {
                "malfunction_id": synthesis_input.malfunction_id,
                "parent_scenario_id": synthesis_input.parent_scenario_id,
                "child_scenario_id": child.scenario_id,
                "status": "CAUSAL_REVALIDATION_REQUIRED",
                "changed_dimensions": changed_dimensions,
                "changed_fields": changed_driver_fields,
                "causal_reuse_basis": "",
                "reason": "DRIVER_CONFIGURATION_NONINTERFERENCE_UNPROVEN",
                "provider_required": True,
            }
        return {
            "malfunction_id": synthesis_input.malfunction_id,
            "parent_scenario_id": synthesis_input.parent_scenario_id,
            "child_scenario_id": child.scenario_id,
            "status": "CAUSAL_REUSE_PROVEN",
            "changed_dimensions": changed_dimensions,
            "changed_fields": changed_driver_fields,
            "causal_reuse_basis": str(metadata["child_subset_refinement"]),
            "reason": "EXPLICIT_NONINTERFERENCE",
            "provider_required": False,
        }

    @staticmethod
    def _rescope_parent_scenario_facts(
        child: ScenarioCandidate, *, malfunction_id: str,
    ) -> ScenarioCandidate:
        """Carry source-valid parent analytical settings into one isolated child."""
        scope = {
            "malfunction_id": malfunction_id,
            "scenario_id": child.scenario_id,
            "parent_scenario_id": child.source_scenario_id,
        }
        provenance = deepcopy(child.fact_provenance)
        for field in (
            "object_type", "object_position", "relative_distance_m",
            "object_speed_kph", "road_user_type", "collision_type",
        ):
            metadata = provenance.get(field)
            if not isinstance(metadata, dict):
                continue
            if str(metadata.get("origin", metadata.get("provenance", ""))).upper() != "SCENARIO_DEFINED":
                continue
            if field not in child.facts:
                continue
            metadata["analysis_assumption_scope"] = scope
            metadata["applicable_scope"] = scope
            metadata["parent_scenario_id"] = child.source_scenario_id
            metadata["inherited_as_child_subset"] = True
            provenance[field] = metadata
        return replace(child, fact_provenance=provenance)

    def _child_state(
        self, *, parent: HARAState, target_run_id: str,
        children: list[ScenarioCandidate], child_assessments: list[dict[str, Any]],
        risk_results: list[RiskAssessment], pending: list[dict[str, Any]],
    ) -> HARAState:
        state = HARAState.read_committed(parent.to_dict())
        state.run_id = target_run_id
        state.stage = WorkflowStage.BLOCKED if pending else WorkflowStage.QUALITY_GATE
        state.scenarios = children
        state.item_definition = deepcopy(parent.item_definition)
        state.item_definition["scenario_assessments"] = child_assessments
        state.risk_results = risk_results
        state.safety_goals = []
        state.pending_reviews = pending
        state.audit_trail = [{
            "event": "scenario_synthesis_child_run_materialized",
            "source_run_id": parent.run_id,
            "target_run_id": target_run_id,
            "child_count": len(children),
            "provider_calls_for_upstream_regeneration": 0,
        }]
        state.errors = []
        return state

    @staticmethod
    def _pending_risk(
        index: int, child: ScenarioCandidate, malfunction_id: str,
        hazardous_event: str, reason: str,
    ) -> RiskAssessment:
        def pending(axis: str) -> EvidenceValue[str]:
            return EvidenceValue(
                value="", status=ReviewStatus.PENDING, sources=[],
                review_reason=f"{axis}: {reason}",
            )
        return RiskAssessment(
            assessment_id=f"RA-SYNTH-{index:04d}-{malfunction_id}-{child.scenario_id}",
            scenario_id=child.scenario_id,
            severity=pending("Severity"), exposure=pending("Exposure"),
            controllability=pending("Controllability"), asil=pending("ASIL"),
            malfunction_id=malfunction_id, hazardous_event=hazardous_event,
        )

    @staticmethod
    def _candidate_artifact(inputs: list[Any], preparation: dict[str, Any]) -> dict[str, Any]:
        return {
            "artifact_version": "scenario-synthesis-candidates-v1",
            "bounds": {
                "shortlist_policy": "coverage-aware adaptive family shortlist",
                "primary_budget": ScenarioShortlistPolicy.PRIMARY_BUDGET,
                "secondary_budget": ScenarioShortlistPolicy.SECONDARY_BUDGET,
                "default_budget": ScenarioShortlistPolicy.DEFAULT_BUDGET,
                "max_adaptive_budget": ScenarioShortlistPolicy.MAX_ADAPTIVE_BUDGET,
            },
            "semantic_ranking_inputs_exclude": ["e_z", "e_f", "E_total", "S", "C", "ASIL"],
            "groups": [{
                "semantic_group_id": item.semantic_group_id,
                "malfunction_id": item.malfunction_id,
                "parent_scenario_id": item.parent_scenario_id,
                "hazardous_event_id": item.hazardous_event_id,
                "structured_semantic_query": item.structured_semantic_query,
                "dimension_applicability": {
                    value.dimension: value.applicability.to_dict()
                    for value in item.dimension_candidate_sets
                },
                "coverage_plan": item.coverage_plan.to_dict(),
                "fm_scenario_template": item.fm_scenario_template,
                "candidate_sets": [value.to_dict() for value in item.dimension_candidate_sets],
            } for item in inputs],
            "summary": preparation["stats"],
        }

    @staticmethod
    def _audit_markdown(audit: dict[str, Any]) -> str:
        summary = audit["summary"]
        before = summary["dimension_resolution_before"]
        after = summary.get("dimension_resolution_after", {})
        lines = [
            "# Scenario Synthesis Audit", "",
            f"- Source run: `{audit['source_run_id']}`",
            f"- Child run: `{audit['run_id']}`",
            f"- Eligible parent HE: {summary['eligible_parent_he']}",
            f"- Unique semantic groups: {summary['unique_semantic_synthesis_groups']}",
            f"- Analytical children generated: {summary.get('analytical_children_generated', 0)}",
            f"- Method-valid: {summary.get('method_valid', 0)}",
            f"- Pending synthesis: {summary.get('pending_synthesis', 0)}",
            f"- Dimension shortlist truncated groups: {summary['dimension_shortlist_truncated_groups']}",
            f"- Combination beam truncated groups: {summary['combination_beam_truncated_groups']}",
            "", "## Dimension resolution", "",
            "| Dimension | Before | After |", "|---|---:|---:|",
        ]
        for dimension in before:
            lines.append(f"| {dimension} | {before[dimension]} | {after.get(dimension, 0)} |")
        lines.extend([
            "", "## Governance", "",
            "- Semantic ranking does not inspect Exposure or S/E/C values.",
            "- Empty ODD/semantic matches remain `PENDING_NO_COMPATIBLE_ATOM`; no full-catalog fallback is used.",
            "- Parent scenarios and accepted parent artifacts remain immutable.",
            "- Per-SC E domain decision: `KEEP_CURRENT_DOMAIN_RESOLUTION`.",
        ])
        return "\n".join(lines) + "\n"

    def run(
        self, *, source_run_id: str, target_run_id: str,
        smoke_count: int = 5, run_full: bool = False, max_workers: int = 4,
        smoke_identities: list[dict[str, Any]] | None = None,
        resume_provider_trace: bool = False,
        output_path: str | Path | None = None,
        baseline_path: str | Path = "method_assets/fusa_baseline_v1/manifest.yaml",
        report_template_path: str | Path = "references/HARA_Template_AI_20260327.xlsx",
    ) -> dict[str, Any]:
        if target_run_id == source_run_id:
            raise ValueError("Scenario synthesis must use a distinct child run ID")
        parent_inventory_before = self._parent_inventory(source_run_id)
        parent = CheckpointRepository(self.run_dir).load(source_run_id)
        inputs, scenarios, malfunctions, parent_assessments, preparation = self._prepare(parent)
        inputs = self._apply_smoke_plan_overrides(inputs, smoke_identities)
        review_dir = self.review_root / target_run_id
        provider_trace_path = review_dir / "scenario_synthesis_provider_trace.json"
        candidate_payload = self._candidate_artifact(inputs, preparation)
        _write_json(review_dir / "scenario_synthesis_candidates.json", candidate_payload)
        selector_auditor = ScenarioSelectorQualityAudit()
        selector_quality = selector_auditor.build(inputs)
        _write_json(review_dir / "scenario_selector_quality_audit.json", selector_quality)
        _write_text(
            review_dir / "scenario_selector_quality_audit.md",
            selector_auditor.markdown(selector_quality),
        )

        provider_ready = [item for item in inputs if self._provider_ready(item)]
        deterministic_resolved = [
            item for item in inputs
            if all(
                value.locked_atom_ids
                or value.applicability.status.value == "NOT_APPLICABLE"
                for value in item.dimension_candidate_sets
            )
        ]
        preparation["stats"].update({
            "raw_eligible_records": len(inputs),
            "deterministically_resolved_groups": len(deterministic_resolved),
            "provider_required_groups": len(provider_ready),
            "provider_ineligible_groups": len(inputs) - len(provider_ready),
        })
        selections: dict[str, tuple[tuple[ScenarioSynthesisAssessment, ...], dict[str, Any]]] = {}
        effective_smoke_count = (
            len(smoke_identities) if smoke_identities is not None else smoke_count
        )
        smoke_inputs = self._smoke_groups(
            provider_ready, effective_smoke_count, smoke_identities,
        )
        if resume_provider_trace:
            selections = self._recover_completed_smoke(
                trace_path=provider_trace_path,
                audit_path=review_dir / "scenario_synthesis_audit.json",
                smoke_inputs=smoke_inputs,
                source_run_id=source_run_id,
                target_run_id=target_run_id,
                parent_inventory=parent_inventory_before,
            )
        if smoke_inputs:
            selections = self._run_provider_groups(
                inputs=smoke_inputs, max_workers=1, progress_path=provider_trace_path,
                phase="smoke", existing=selections,
            )
        smoke_traces = [selections[item.semantic_group_id][1] for item in smoke_inputs]
        smoke_passed, smoke_failures = self._smoke_passed(smoke_traces, len(smoke_inputs))
        full_started = bool(run_full and smoke_passed)
        if full_started:
            selections = self._run_provider_groups(
                inputs=provider_ready, max_workers=max_workers,
                progress_path=provider_trace_path, phase="full", existing=selections,
            )

        all_traces = [value[1] for _, value in sorted(selections.items())]
        successful_calls = [
            call for trace in all_traces for call in trace.get("calls", [])
            if call.get("schema_status") == "PASS"
        ]
        provider_payload = {
            "artifact_version": "scenario-synthesis-provider-trace-v1",
            "configured_model": str(getattr(getattr(self.client, "config", None), "model", "")),
            "resolved_models": sorted({
                str(item.get("resolved_model", "")) for item in successful_calls
                if item.get("resolved_model")
            }),
            "thinking": str(getattr(getattr(self.client, "config", None), "scenario_thinking", "")),
            "smoke": {
                "requested": effective_smoke_count, "executed": len(smoke_inputs),
                "passed": smoke_passed, "failures": smoke_failures,
                "semantic_group_ids": [item.semantic_group_id for item in smoke_inputs],
                "stable_identities": [{
                    "malfunction_id": item.malfunction_id,
                    "parent_scenario_id": item.parent_scenario_id,
                    "hazardous_event_id": item.hazardous_event_id,
                    "requested_variant_count": item.coverage_plan.desired_variant_count,
                } for item in smoke_inputs],
                "strata": self._smoke_strata(smoke_inputs),
                "selection_records": [{
                    "semantic_group_id": item.semantic_group_id,
                    "variants": [
                        assessment.to_dict()
                        for assessment in selections[item.semantic_group_id][0]
                    ],
                } for item in smoke_inputs if item.semantic_group_id in selections],
            },
            "full_synthesis_started": full_started,
            "full_synthesis_groups": len(provider_ready) if full_started else 0,
            "groups": all_traces,
            "summary": {
                "provider_calls": sum(self._trace_call_count(item) for item in all_traces),
                "smoke_cache_hits_in_full": len(smoke_inputs) if full_started else 0,
                "repairs": sum(int(item.get("repairs", 0)) for item in all_traces),
                "method_compound_canonicalizations": sum(
                    int(item.get("method_compound_canonicalizations", 0))
                    for item in all_traces
                ),
                "failures": sum(item.get("status") != "PASS" for item in all_traces),
            },
        }
        _write_json(provider_trace_path, provider_payload)

        children: list[ScenarioCandidate] = []
        instantiations = []
        child_contexts: dict[str, tuple[Any, dict[str, Any]]] = {}
        typed = parent.item_definition.get("typed", {})
        typed = typed if isinstance(typed, dict) else {}
        driver_risk_facts = typed.get("risk_facts", [])
        driver_risk_facts = driver_risk_facts if isinstance(driver_risk_facts, list) else []
        materialization_inputs = (
            inputs if full_started else smoke_inputs if smoke_passed else []
        )
        if materialization_inputs:
            for synthesis_input in materialization_inputs:
                selection = selections.get(synthesis_input.semantic_group_id)
                if selection is None:
                    continue
                assessments, trace = selection
                parent_scenario = scenarios[synthesis_input.parent_scenario_id]
                for assessment in assessments:
                    child, instantiation = self.synthesis.materialize(
                        synthesis_input=synthesis_input, assessment=assessment,
                        parent=parent_scenario,
                        provider_evidence={
                            "semantic_group_id": synthesis_input.semantic_group_id,
                            "request_ids": [
                                call.get("request_id", "") for call in trace.get("calls", [])
                                if call.get("request_id")
                            ],
                            "resolved_model": next((
                                call.get("resolved_model", "") for call in reversed(trace.get("calls", []))
                                if call.get("resolved_model")
                            ), ""),
                            "prompt_version": SCENARIO_SYNTHESIS_PROMPT_VERSION,
                            "rationale": assessment.semantic_rationale,
                            "context_refs": list(assessment.context_refs),
                        },
                    )
                    child = self._rescope_parent_scenario_facts(
                        child, malfunction_id=synthesis_input.malfunction_id,
                    )
                    for driver_child in self.driver_brancher.expand(
                        child, driver_risk_facts,
                    ):
                        children.append(driver_child)
                        record = replace(
                            instantiation, scenario_id=driver_child.scenario_id,
                        ).to_dict()
                        branch = driver_child.analysis_instance.get(
                            "driver_configuration_branch"
                        )
                        if branch:
                            record["driver_configuration_branch"] = deepcopy(branch)
                        instantiations.append(record)
                        child_contexts[driver_child.scenario_id] = (
                            synthesis_input,
                            parent_assessments[(
                                synthesis_input.malfunction_id,
                                synthesis_input.parent_scenario_id,
                            )],
                        )

        after = Counter()
        for child in children:
            for dimension, binding in child.facts.get("method_scenario_dimensions", {}).items():
                if binding.get("resolution_status") == "RESOLVED":
                    after[dimension] += 1
        pending_synthesis = (
            len(inputs) - len({
                item.semantic_group_id for item, _ in child_contexts.values()
            }) if full_started else len(inputs)
        )
        preparation["stats"].update({
            "analytical_children_generated": len(children),
            "method_valid": len(children),
            "method_invalid": 0,
            "pending_synthesis": pending_synthesis,
            "dimension_resolution_after": {
                dimension: after[dimension] for dimension in self.synthesis.dimensions
            },
        })

        causal_records = []
        for child in children:
            synthesis_input, parent_assessment = child_contexts[child.scenario_id]
            causal_records.append(self._causal_delta(
                synthesis_input=synthesis_input,
                parent_assessment=parent_assessment, child=child,
            ))
        causal_counts = Counter(item["status"] for item in causal_records)
        causal_payload = {
            "artifact_version": "scenario-synthesis-causal-delta-queue-v1",
            "run_id": target_run_id,
            "records": causal_records,
            "summary": {
                "total": len(causal_records),
                "deterministic_reuse": causal_counts["CAUSAL_REUSE_PROVEN"],
                "provider_revalidation": causal_counts["CAUSAL_REVALIDATION_REQUIRED"],
                "causal_gap": causal_counts["CAUSAL_GAP"],
                "source_conflict": causal_counts["SOURCE_CONFLICT"],
            },
        }
        _write_json(review_dir / "causal_delta_queue.json", causal_payload)

        exposure_records = []
        causal_by_child = {item["child_scenario_id"]: item for item in causal_records}
        e_counts = Counter()
        readiness_counts = Counter()
        for child in children:
            synthesis_input, _ = child_contexts[child.scenario_id]
            scenario = {
                **child.facts,
                "component_category": synthesis_input.malfunction.get("component_category", ""),
            }
            readiness = self.exposure.assess(scenario)
            readiness_counts[readiness["status"]] += 1
            causal_status = causal_by_child[child.scenario_id]["status"]
            executed = causal_status == "CAUSAL_REUSE_PROVEN" and readiness["status"] in {
                "READY_COMPLETE", "READY_METHOD_IRRELEVANT_GAPS",
            }
            result = readiness["baseline_exposure"] if executed else {}
            value = str(result.get("value", "")) if executed else ""
            e_counts[value if value in {"E0", "E1", "E2", "E3", "E4"} else "Pending"] += 1
            exposure_records.append({
                "malfunction_id": synthesis_input.malfunction_id,
                "scenario_id": child.scenario_id,
                "causal_status": causal_status,
                "readiness": readiness,
                "execution_status": "FINALIZED" if executed else "NOT_REACHED_CAUSAL_OR_INPUT_GATE",
                "result": result,
            })

        physics_records = []
        for child in children:
            synthesis_input, _ = child_contexts[child.scenario_id]
            physics_records.append(self.physics.instantiate(
                scenario=child, malfunction=synthesis_input.malfunction,
                causal_status=causal_by_child[child.scenario_id]["status"],
            ))
        physics_payload = {
            "artifact_version": "analytical-physics-inputs-v1",
            "run_id": target_run_id,
            "records": physics_records,
            "summary": {
                "authority_distributions": self.physics.authority_distributions(physics_records),
                "relative_speed_derived": sum(
                    any(item.get("field") == "relative_speed_kph" for item in record.get("derived", []))
                    for record in physics_records
                ),
                "ttc_derived": sum(
                    any(item.get("field") == "ttc_s" for item in record.get("derived", []))
                    for record in physics_records
                ),
            },
        }
        _write_json(review_dir / "analytical_physics_inputs.json", physics_payload)
        if smoke_passed and not full_started:
            _write_json(review_dir / "supported_smoke_children.json", {
                "artifact_version": "supported-r4-smoke-children-v1",
                "run_id": target_run_id,
                "full_synthesis_started": False,
                "children": [item.to_dict() for item in children],
                "instantiations": instantiations,
                "causal_delta": causal_payload,
                "physics": physics_payload,
            })
        assumption_pack = self.physics.assumption_pack(
            physics_records, {item.scenario_id: item for item in children}, malfunctions,
        )
        assumption_pack["run_id"] = target_run_id
        _write_json(review_dir / "engineering_assumption_pack.json", assumption_pack)

        child_assessments = []
        pending_reviews = []
        risk_results = []
        for index, child in enumerate(children, start=1):
            synthesis_input, parent_assessment = child_contexts[child.scenario_id]
            causal = causal_by_child[child.scenario_id]
            reason = (
                "Analytical child contains new Scenario semantics; causal dependency metadata "
                "does not prove noninterference. Differential causal validation is required."
                if causal["status"] == "CAUSAL_REVALIDATION_REQUIRED" else
                "Analytical child is pending downstream physical/risk evaluation."
            )
            assessment = {
                "malfunction_id": synthesis_input.malfunction_id,
                "scenario_id": child.scenario_id,
                "hazardous_event_id": synthesis_input.hazardous_event_id,
                "physically_feasible": False, "functionally_relevant": False,
                "causally_relevant": False, "risk_dimensions_changed": [],
                "rationale": reason,
                "hazardous_event": parent_assessment.get("hazardous_event", ""),
                "potential_harm": "", "status": "PENDING", "confidence": 0.0,
                "breakpoint": "", "causal_chain": {}, "risk_dimension_changes": [],
                "evidence_contract_version": "scenario-causal-assessment-v4",
                "final_retain": False,
                "risk_eligibility_status": "PENDING_FEASIBILITY",
                "risk_eligibility_reason_codes": ["CAUSAL_REVALIDATION_REQUIRED"],
                "scenario_synthesis_status": "METHOD_VALID",
                "causal_delta_status": causal["status"],
            }
            child_assessments.append(assessment)
            pending_reviews.append({
                "field": "scenario_causal_validation",
                "malfunction_id": synthesis_input.malfunction_id,
                "scenario_id": child.scenario_id, "reason": reason,
            })
            risk_results.append(self._pending_risk(
                index, child, synthesis_input.malfunction_id,
                str(parent_assessment.get("hazardous_event", "")), reason,
            ))

        child_state = self._child_state(
            parent=parent, target_run_id=target_run_id, children=children,
            child_assessments=child_assessments, risk_results=risk_results,
            pending=pending_reviews,
        )
        checkpoint_path = None
        risk_trace_path = review_dir / "risk_execution_trace.json"
        excel_path = None
        if full_started:
            checkpoint_path = CheckpointRepository(self.run_dir).save(child_state)
            writer = ReviewArtifactWriter(target_run_id, self.review_root)
            for function in child_state.functions:
                writer.record_function(function)
            for malfunction in child_state.malfunctions:
                writer.record_malfunction(malfunction)
            for child in children:
                synthesis_input, _ = child_contexts[child.scenario_id]
                writer.record_scenario_candidate(
                    child, generated_for_malfunction_ids=(synthesis_input.malfunction_id,),
                )
            for assessment in child_assessments:
                malfunction = malfunctions.get(assessment["malfunction_id"], {})
                writer.record_scenario_feasibility(
                    assessment, function_id=str(malfunction.get("function_id", "")),
                    guideword=str(malfunction.get("guideword", "")),
                )
            trace = RiskExecutionTraceService(self.method).project(
                run_id=target_run_id, assessments=child_assessments,
                candidates=children, committed=True, risks=risk_results,
            )
            trace["scenario_synthesis"] = {
                "causal_delta_summary": causal_payload["summary"],
                "exposure_readiness": dict(sorted(readiness_counts.items())),
                "exposure_distribution": {
                    key: e_counts[key] for key in ("E0", "E1", "E2", "E3", "E4", "Pending")
                },
                "physics": physics_payload["summary"],
            }
            _write_json(risk_trace_path, trace)
            writer.write_summary(child_state, status="BLOCKED" if pending_reviews else "COMPLETED")
            if output_path is not None:
                excel_path = OfflineReportRebuilder().rebuild(
                    checkpoint_path=checkpoint_path,
                    method_baseline_path=baseline_path,
                    report_style_template_path=report_template_path,
                    output_path=output_path, review_root=self.review_root,
                )

        audit_payload = {
            "artifact_version": "scenario-synthesis-audit-v1",
            "run_id": target_run_id, "source_run_id": source_run_id,
            "full_started": full_started,
            "smoke": provider_payload["smoke"],
            "provider": provider_payload["summary"],
            "summary": {
                **preparation["stats"],
                "causal_delta": causal_payload["summary"],
                "exposure_readiness": dict(sorted(readiness_counts.items())),
                "exposure_distribution": {
                    key: e_counts[key] for key in ("E0", "E1", "E2", "E3", "E4", "Pending")
                },
                "physics": physics_payload["summary"],
                "engineering_assumptions_requiring_approval": assumption_pack["summary"]["fields_requiring_approval"],
            },
            "e_domain_decision": {
                "decision": "KEEP_CURRENT_DOMAIN_RESOLUTION",
                "reason": (
                    "Original per-SC resolver was free-form LLM inference with FM fallback; "
                    "no source-governed per-SC domain rule was present."
                ),
            },
            "parent_artifact_hashes_before": parent_inventory_before,
        }
        parent_inventory_after = self._parent_inventory(source_run_id)
        audit_payload["parent_artifact_hashes_after"] = parent_inventory_after
        audit_payload["parent_artifacts_mutated"] = parent_inventory_before != parent_inventory_after
        _write_json(review_dir / "scenario_synthesis_audit.json", audit_payload)
        _write_text(review_dir / "scenario_synthesis_audit.md", self._audit_markdown(audit_payload))
        selector_quality = selector_auditor.build(
            inputs,
            assessments_by_group={
                group_id: value[0] for group_id, value in selections.items()
            },
            children=children,
            mode=(
                "REALIZED_SELECTION" if full_started else
                "REALIZED_SUPPORTED_SMOKE" if smoke_passed else
                "OFFLINE_CANDIDATE_PLAN"
            ),
        )
        _write_json(review_dir / "scenario_selector_quality_audit.json", selector_quality)
        _write_text(
            review_dir / "scenario_selector_quality_audit.md",
            selector_auditor.markdown(selector_quality),
        )

        result = {
            "source_run_id": source_run_id, "target_run_id": target_run_id,
            "smoke_passed": smoke_passed, "smoke_failures": smoke_failures,
            "full_started": full_started,
            "summary": audit_payload["summary"],
            "configured_model": provider_payload["configured_model"],
            "resolved_models": provider_payload["resolved_models"],
            "thinking": provider_payload["thinking"],
            "provider": provider_payload["summary"],
            "parent_artifacts_mutated": audit_payload["parent_artifacts_mutated"],
            "checkpoint": str(checkpoint_path) if checkpoint_path else "",
            "risk_trace": str(risk_trace_path) if full_started else "",
            "excel": str(excel_path) if excel_path else "",
            "review_dir": str(review_dir),
        }
        return result


__all__ = ["ScenarioSynthesisRunner"]
