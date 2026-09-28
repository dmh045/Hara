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

from hara_agent.contracts import MethodContract
from hara_agent.infrastructure.llm.provider_budget import ProviderAttemptBudgetExceeded
from hara_agent.models import (
    EvidenceValue, ItemDefinitionFacts, MalfunctionCandidate, ReviewStatus,
    RiskAssessment, ScenarioCandidate, ScenarioFeasibilityAssessment, SourceRef,
)
from hara_agent.services.analysis import (
    AnalyticalPhysicsInstantiationService, ExposureInputReadinessService,
    RiskExecutionTraceService,
)
from hara_agent.services.semantic import (
    ScenarioFeasibilityAgent, build_project_evidence_registry,
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


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _malfunction(value: dict[str, Any]) -> MalfunctionCandidate:
    payload = dict(value)
    payload["sources"] = [SourceRef(**item) for item in payload.get("sources", [])]
    payload["status"] = ReviewStatus(payload.get("status", ReviewStatus.PENDING.value))
    return MalfunctionCandidate(**payload)


class ScenarioCausalRevalidationRunner:
    """Revalidate only synthesized child semantics in a new immutable run."""

    def __init__(
        self, *, method: MethodContract, client: Any,
        run_dir: str | Path = "runtime/agent",
        review_root: str | Path = "runtime/review",
    ):
        self.method = method
        self.client = client
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.review_root = Path(review_root).expanduser().resolve()
        self.physics = AnalyticalPhysicsInstantiationService(method)
        self.exposure = ExposureInputReadinessService(method)

    def _source_inventory(self, source_run_id: str) -> dict[str, str]:
        paths = [self.run_dir / f"{source_run_id}.checkpoint.json"]
        review_dir = self.review_root / source_run_id
        if review_dir.is_dir():
            paths.extend(sorted(path for path in review_dir.rglob("*") if path.is_file()))
        return {str(path): _sha256(path) for path in paths if path.is_file()}

    @staticmethod
    def _recover_completed(
        provider_path: Path, *, source_run_id: str, target_run_id: str,
        candidates_by_malfunction: dict[str, list[ScenarioCandidate]],
    ) -> dict[str, tuple[list[ScenarioFeasibilityAssessment], dict[str, Any]]]:
        """Recover only complete, previously validated malfunction results.

        The provider trace is atomically replaced after each completed
        malfunction.  Each audit carries the final parsed assessment for every
        child, including repaired items, so an interrupted child run can resume
        without treating a partial batch as valid or calling the Provider again
        for already validated children.
        """
        if not provider_path.is_file():
            return {}
        payload = json.loads(provider_path.read_text(encoding="utf-8"))
        if (
            payload.get("artifact_version")
            != "scenario-causal-revalidation-provider-trace-v1"
            or payload.get("source_run_id") != source_run_id
            or payload.get("run_id") != target_run_id
            or int(payload.get("target_malfunctions", -1))
            != len(candidates_by_malfunction)
        ):
            raise ValueError("Existing causal resume trace is not bound to this run")
        completed: dict[
            str, tuple[list[ScenarioFeasibilityAssessment], dict[str, Any]]
        ] = {}
        for audit in payload.get("audits", []):
            if not isinstance(audit, dict):
                raise ValueError("Causal resume trace contains a malformed audit")
            malfunction_id = str(audit.get("malfunction_id", ""))
            candidates = candidates_by_malfunction.get(malfunction_id)
            if candidates is None or malfunction_id in completed:
                raise ValueError("Causal resume trace contains an unknown or duplicate malfunction")
            parsed_by_id: dict[str, ScenarioFeasibilityAssessment] = {}
            for entry in audit.get("item_salvage_audit", []):
                if not isinstance(entry, dict):
                    continue
                parsed = entry.get("parsed_assessment")
                if not isinstance(parsed, dict):
                    continue
                assessment = ScenarioFeasibilityAssessment.from_dict(parsed)
                parsed_by_id[assessment.scenario_id] = assessment
            expected_ids = {item.scenario_id for item in candidates}
            if set(parsed_by_id) != expected_ids:
                raise ValueError(
                    f"Causal resume audit is incomplete for {malfunction_id}: "
                    f"expected={len(expected_ids)} recovered={len(parsed_by_id)}"
                )
            completed[malfunction_id] = (
                [parsed_by_id[item.scenario_id] for item in candidates], audit,
            )
        if int(payload.get("completed_malfunctions", -1)) != len(completed):
            raise ValueError("Causal resume trace completed count is inconsistent")
        return completed

    def _causal_projection(self, scenario: ScenarioCandidate) -> ScenarioCandidate:
        """Remove audit-heavy binding detail while preserving its typed facts."""
        facts = deepcopy(scenario.facts)
        bindings = facts.pop("method_scenario_dimensions", {})
        provenance = deepcopy(scenario.fact_provenance)
        provenance.pop("method_scenario_dimensions", None)
        scope = {
            "malfunction_id": scenario.analysis_instance.get("malfunction_id", ""),
            "scenario_id": scenario.scenario_id,
            "parent_scenario_id": scenario.source_scenario_id,
        }
        fact_keys = {
            "WHERE": "operating_scenario",
            "ROAD": "road_surface_conditions",
            "EGO_ACTION": "vehicle_state",
            "EGO_DYNAMICS": "ego_dynamics",
            "OBJECT": "scenario_object_atom",
            "TRAFFIC_PATTERN": "traffic_pattern",
            "EGO_X_ROAD": "ego_road_relation",
        }
        for dimension, fact_key in fact_keys.items():
            binding = bindings.get(dimension, {}) if isinstance(bindings, dict) else {}
            if not isinstance(binding, dict) or binding.get("resolution_status") != "RESOLVED":
                continue
            facts.setdefault(fact_key, str(binding.get("method_value", "")))
            atom = binding.get("atom_provenance", {})
            atom = atom if isinstance(atom, dict) else {}
            provenance[fact_key] = {
                "provenance": "SCENARIO_DEFINED",
                "origin": "SCENARIO_DEFINED",
                "approval": "FINALIZED",
                "source_refs": [{
                    "source_type": "method_contract",
                    "source_id": str(self.method.metadata.get("method_source_hash", "")),
                    "location": (
                        f"{atom.get('source_asset', '')}:{atom.get('source_rule', '')}"
                    ),
                    "excerpt": str(binding.get("method_value", "")),
                }],
                "applicable_scope": scope,
                "selection_basis": "VALIDATED_ANALYTICAL_SCENARIO_ATOM",
            }
        compact_instance = {
            key: deepcopy(scenario.analysis_instance.get(key))
            for key in (
                "parent_scenario_id", "malfunction_id", "hazardous_event_id",
                "selected_atoms", "validation_status", "synthesis_version",
            )
            if key in scenario.analysis_instance
        }
        return replace(
            scenario, facts=facts, fact_provenance=provenance,
            context_resolution={}, exposure_context=[],
            analysis_instance=compact_instance,
        )

    @staticmethod
    def _pending_risk(
        index: int, assessment: ScenarioFeasibilityAssessment,
    ) -> RiskAssessment:
        def pending(axis: str) -> EvidenceValue[str]:
            return EvidenceValue(
                value="", status=ReviewStatus.PENDING, sources=[],
                review_reason=f"{axis}: deterministic downstream scoring not yet executed",
            )
        return RiskAssessment(
            assessment_id=(
                f"RA-CAUSAL-{index:04d}-{assessment.malfunction_id}-"
                f"{assessment.scenario_id}"
            ),
            scenario_id=assessment.scenario_id,
            severity=pending("Severity"), exposure=pending("Exposure"),
            controllability=pending("Controllability"), asil=pending("ASIL"),
            malfunction_id=assessment.malfunction_id,
            hazardous_event=assessment.hazardous_event,
        )

    def run(
        self, *, source_run_id: str, target_run_id: str, max_workers: int = 4,
    ) -> dict[str, Any]:
        if source_run_id == target_run_id:
            raise ValueError("Causal revalidation requires a distinct child run ID")
        repository = CheckpointRepository(self.run_dir)
        target_checkpoint = repository.path_for(target_run_id)
        target_review = self.review_root / target_run_id
        provider_path = target_review / "causal_revalidation_provider_trace.json"
        if target_checkpoint.exists():
            raise ValueError("Causal target checkpoint already exists; use a new child run ID")
        if target_review.exists() and not provider_path.is_file():
            raise ValueError(
                "Causal target artifacts exist without a resumable provider trace; "
                "use a new child run ID"
            )

        source_inventory_before = self._source_inventory(source_run_id)
        source = repository.load(source_run_id)
        if not source.scenarios:
            raise ValueError("Synthesis source contains no analytical child scenarios")
        if any(
            item.analysis_instance.get("validation_status") != "VALIDATED"
            or not item.analysis_instance.get("malfunction_id")
            for item in source.scenarios
        ):
            raise ValueError("Every causal candidate must be a method-valid synthesis child")

        requested_queue_path = self.review_root / source_run_id / "causal_delta_queue.json"
        requested_queue = json.loads(requested_queue_path.read_text(encoding="utf-8"))
        required_ids = {
            str(item.get("child_scenario_id", ""))
            for item in requested_queue.get("records", [])
            if item.get("status") == "CAUSAL_REVALIDATION_REQUIRED"
        }
        candidate_ids = {item.scenario_id for item in source.scenarios}
        if required_ids != candidate_ids:
            raise ValueError("Causal queue does not exactly cover the synthesis children")

        malfunction_by_id = {
            item.malfunction_id: item for item in map(_malfunction, source.malfunctions)
        }
        candidates_by_malfunction: dict[str, list[ScenarioCandidate]] = defaultdict(list)
        for scenario in source.scenarios:
            candidates_by_malfunction[
                str(scenario.analysis_instance["malfunction_id"])
            ].append(self._causal_projection(scenario))
        unknown = set(candidates_by_malfunction) - set(malfunction_by_id)
        if unknown:
            raise ValueError(f"Synthesis children reference unknown malfunctions: {sorted(unknown)}")

        typed = source.item_definition.get("typed", {})
        project_registry = build_project_evidence_registry(
            ItemDefinitionFacts.from_dict(typed), self.method,
        )
        # Synthesis children carry a compact causal projection, so batching up
        # to eight independent children stays within the Provider context and
        # avoids one call per child without sharing evidence between items.
        agent = ScenarioFeasibilityAgent(
            self.client, batch_max_chars=60000, batch_max_items=8,
        )
        completed = self._recover_completed(
            provider_path, source_run_id=source_run_id,
            target_run_id=target_run_id,
            candidates_by_malfunction=candidates_by_malfunction,
        )
        ordered_ids = sorted(candidates_by_malfunction)
        failed_malfunctions: dict[str, dict[str, str]] = {}
        if completed:
            print(
                f"[HARA][P5-D] causal resume completed={len(completed)}/"
                f"{len(ordered_ids)}",
                flush=True,
            )
        with ThreadPoolExecutor(max_workers=max(1, max_workers)) as executor:
            futures = {
                executor.submit(
                    agent.assess, malfunction_by_id[malfunction_id],
                    candidates_by_malfunction[malfunction_id], project_registry,
                ): malfunction_id
                for malfunction_id in ordered_ids
                if malfunction_id not in completed
            }
            for future in as_completed(futures):
                malfunction_id = futures[future]
                try:
                    assessments, audit = future.result()
                except ProviderAttemptBudgetExceeded:
                    # The shared Provider cap ends this child run immediately.
                    # Do not convert it into a deferred semantic assessment.
                    raise
                except Exception as exc:
                    failed_malfunctions[malfunction_id] = {
                        "error_type": type(exc).__name__,
                        "error": str(exc)[:1200],
                    }
                    print(
                        f"[HARA][P5-D] causal deferred malfunction={malfunction_id} "
                        f"error={type(exc).__name__}",
                        flush=True,
                    )
                    continue
                completed[malfunction_id] = (assessments, audit)
                print(
                    f"[HARA][P5-D] causal {len(completed)}/{len(ordered_ids)} "
                    f"malfunction={malfunction_id} assessments={len(assessments)}",
                    flush=True,
                )
                _write_json(provider_path, {
                    "artifact_version": "scenario-causal-revalidation-provider-trace-v1",
                    "source_run_id": source_run_id,
                    "run_id": target_run_id,
                    "completed_malfunctions": len(completed),
                    "target_malfunctions": len(ordered_ids),
                    "audits": [completed[key][1] for key in sorted(completed)],
                    "deferred_malfunctions": failed_malfunctions,
                })

        if failed_malfunctions:
            _write_json(provider_path, {
                "artifact_version": "scenario-causal-revalidation-provider-trace-v1",
                "source_run_id": source_run_id,
                "run_id": target_run_id,
                "completed_malfunctions": len(completed),
                "target_malfunctions": len(ordered_ids),
                "audits": [completed[key][1] for key in sorted(completed)],
                "deferred_malfunctions": failed_malfunctions,
            })
            raise RuntimeError(
                "Causal Provider work was interrupted for "
                f"{len(failed_malfunctions)} malfunction(s); rerun the same "
                "source/target command to resume validated work"
            )

        assessments = [
            assessment
            for malfunction_id in ordered_ids
            for assessment in completed[malfunction_id][0]
        ]
        assessment_by_id = {item.scenario_id: item for item in assessments}
        if set(assessment_by_id) != candidate_ids or len(assessment_by_id) != len(assessments):
            raise ValueError("Provider causal results do not exactly cover child scenarios")

        assessment_payloads = []
        causal_records = []
        status_counts = Counter()
        for scenario in source.scenarios:
            assessment = assessment_by_id[scenario.scenario_id]
            causal_status = "CAUSAL_REVALIDATED" if assessment.retain else "CAUSAL_GAP"
            status_counts[causal_status] += 1
            payload = assessment.to_dict()
            payload.update({
                "hazardous_event_id": scenario.analysis_instance.get("hazardous_event_id", ""),
                "scenario_synthesis_status": "METHOD_VALID",
                "causal_delta_status": causal_status,
            })
            assessment_payloads.append(payload)
            causal_records.append({
                "malfunction_id": assessment.malfunction_id,
                "parent_scenario_id": scenario.source_scenario_id,
                "child_scenario_id": scenario.scenario_id,
                "status": causal_status,
                "provider_required": True,
                "provider_audit_ref": f"malfunction_id={assessment.malfunction_id}",
                "breakpoint": assessment.breakpoint,
                "causal_validation_status": (
                    assessment.causal_assessment.status.value
                    if assessment.causal_assessment is not None else "MISSING"
                ),
                "reason": assessment.rationale,
            })
        causal_payload = {
            "artifact_version": "scenario-synthesis-causal-delta-queue-v2",
            "source_run_id": source_run_id, "run_id": target_run_id,
            "records": causal_records,
            "summary": {
                "total": len(causal_records),
                "deterministic_reuse": 0,
                "provider_revalidation": len(causal_records),
                "causal_revalidated": status_counts["CAUSAL_REVALIDATED"],
                "causal_gap": status_counts["CAUSAL_GAP"],
                "source_conflict": 0,
            },
        }
        _write_json(target_review / "causal_delta_queue.json", causal_payload)

        malfunctions_raw = {
            str(item.get("malfunction_id", "")): item for item in source.malfunctions
        }
        physics_records = [
            self.physics.instantiate(
                scenario=scenario,
                malfunction=malfunctions_raw[str(scenario.analysis_instance["malfunction_id"])],
                causal_status=causal_records[index]["status"],
            )
            for index, scenario in enumerate(source.scenarios)
        ]
        physics_payload = {
            "artifact_version": "analytical-physics-inputs-v1",
            "source_run_id": source_run_id, "run_id": target_run_id,
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
        _write_json(target_review / "analytical_physics_inputs.json", physics_payload)
        assumption_pack = self.physics.assumption_pack(
            physics_records, {item.scenario_id: item for item in source.scenarios},
            malfunctions_raw,
        )
        assumption_pack.update(source_run_id=source_run_id, run_id=target_run_id)
        _write_json(target_review / "engineering_assumption_pack.json", assumption_pack)

        exposure_records = []
        readiness_counts = Counter()
        for scenario in source.scenarios:
            malfunction_id = str(scenario.analysis_instance["malfunction_id"])
            readiness = self.exposure.assess({
                **scenario.facts,
                "component_category": malfunctions_raw[malfunction_id].get(
                    "component_category", ""
                ),
            })
            readiness_counts[readiness["status"]] += 1
            exposure_records.append({
                "malfunction_id": malfunction_id,
                "scenario_id": scenario.scenario_id,
                "causal_status": causal_records[len(exposure_records)]["status"],
                "readiness": readiness,
            })
        exposure_payload = {
            "artifact_version": "scenario-synthesis-exposure-readiness-v1",
            "source_run_id": source_run_id, "run_id": target_run_id,
            "records": exposure_records,
            "summary": dict(sorted(readiness_counts.items())),
        }
        _write_json(target_review / "exposure_input_readiness.json", exposure_payload)

        state = HARAState.read_committed(source.to_dict())
        state.run_id = target_run_id
        state.item_definition = deepcopy(source.item_definition)
        state.item_definition["scenario_assessments"] = assessment_payloads
        eligible = [item for item in assessments if item.retain]
        state.risk_results = [
            self._pending_risk(index, assessment)
            for index, assessment in enumerate(eligible, start=1)
        ]
        state.safety_goals = []
        state.pending_reviews = []
        state.stage = WorkflowStage.SCORING if eligible else WorkflowStage.BLOCKED
        state.audit_trail = [*source.audit_trail, {
            "event": "scenario_synthesis_causal_revalidation_completed",
            "source_run_id": source_run_id,
            "target_run_id": target_run_id,
            "provider_calls_for_upstream_regeneration": 0,
            "causal_summary": causal_payload["summary"],
        }]
        checkpoint_path = repository.save(state)

        writer = ReviewArtifactWriter(target_run_id, self.review_root)
        for function in state.functions:
            writer.record_function(function)
        for malfunction in state.malfunctions:
            writer.record_malfunction(malfunction)
        for scenario in state.scenarios:
            writer.record_scenario_candidate(
                scenario,
                generated_for_malfunction_ids=(
                    str(scenario.analysis_instance["malfunction_id"]),
                ),
            )
        for payload in assessment_payloads:
            writer.record_scenario_feasibility(payload)
        trace = RiskExecutionTraceService(self.method).project(
            run_id=target_run_id, assessments=assessment_payloads,
            candidates=state.scenarios, committed=True, risks=state.risk_results,
        )
        trace["scenario_synthesis"] = {
            "causal_delta_summary": causal_payload["summary"],
            "exposure_readiness": exposure_payload["summary"],
            "physics": physics_payload["summary"],
        }
        _write_json(target_review / "risk_execution_trace.json", trace)
        writer.write_summary(state, status="CAUSAL_REVALIDATED")

        audits = [completed[key][1] for key in ordered_ids]
        resolved_models = sorted({
            str(model) for audit in audits for model in audit.get("models", []) if model
        })
        source_inventory_after = self._source_inventory(source_run_id)
        result = {
            "source_run_id": source_run_id, "target_run_id": target_run_id,
            "checkpoint": str(checkpoint_path),
            "causal": causal_payload["summary"],
            "provider": {
                "configured_model": str(getattr(getattr(self.client, "config", None), "model", "")),
                "resolved_models": resolved_models,
                "thinking": str(getattr(getattr(self.client, "config", None), "scenario_thinking", "")),
                "calls": sum(int(audit.get("provider_calls_total", 0)) for audit in audits),
                "repairs": sum(int(audit.get("item_repair_calls", 0)) for audit in audits),
                "failures": sum(int(audit.get("single_item_failures", 0)) for audit in audits),
            },
            "exposure_readiness": exposure_payload["summary"],
            "physics": physics_payload["summary"],
            "engineering_assumptions": assumption_pack["summary"],
            "source_artifacts_mutated": source_inventory_before != source_inventory_after,
            "review_dir": str(target_review),
        }
        _write_json(target_review / "causal_revalidation_summary.json", result)
        return result


__all__ = ["ScenarioCausalRevalidationRunner"]
