"""Recover a bounded parent HE assessment through the production causal agent.

Historical FM analytical children are never promoted into parent assessments.
The parent Scenario candidates are regenerated from committed ProjectFacts and
the active MethodContract before the existing ScenarioFeasibilityAgent is used.
"""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any

from hara_agent.contracts import MethodContract
from hara_agent.models import (
    FunctionDefinition, ItemDefinitionFacts, MalfunctionCandidate,
    ReviewStatus, SourceRef, evaluate_risk_eligibility,
)
from hara_agent.services.analysis import (
    HazardousEventRiskContextService, MethodScenarioCandidateService,
    ProjectFactResolver,
)
from hara_agent.services.semantic import (
    ScenarioFeasibilityAgent, build_project_evidence_registry,
)
from hara_agent.workflow.checkpoints import CheckpointRepository
from hara_agent.workflow.state import HARAState, WorkflowStage


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
        temporary.unlink(missing_ok=True)


def _function(value: dict[str, Any]) -> FunctionDefinition:
    payload = dict(value)
    payload["sources"] = [SourceRef(**item) for item in payload.get("sources", [])]
    payload["status"] = ReviewStatus(payload.get("status", ReviewStatus.PENDING.value))
    return FunctionDefinition(**payload)


def _malfunction(value: dict[str, Any]) -> MalfunctionCandidate:
    payload = dict(value)
    payload["sources"] = [SourceRef(**item) for item in payload.get("sources", [])]
    payload["status"] = ReviewStatus(payload.get("status", ReviewStatus.PENDING.value))
    return MalfunctionCandidate(**payload)


def _project_only_typed(value: dict[str, Any]) -> dict[str, Any]:
    """Exclude old child interpretation facts from parent causal evidence."""
    typed = deepcopy(value)
    dropped = {
        str(item.get("fact_id", ""))
        for item in typed.get("risk_facts", [])
        if isinstance(item, dict)
        and str(item.get("produced_by", "")).startswith("scenario-risk-facts-")
    }
    typed["risk_facts"] = [
        item for item in typed.get("risk_facts", [])
        if not isinstance(item, dict) or str(item.get("fact_id", "")) not in dropped
    ]
    typed["method_risk_fact_bindings"] = [
        item for item in typed.get("method_risk_fact_bindings", [])
        if not isinstance(item, dict)
        or str(item.get("source_fact_id", "")) not in dropped
    ]
    return typed


class ParentHERecoveryRunner:
    """One bounded parent-only causal batch, committed as a child checkpoint."""

    def __init__(
        self, *, method: MethodContract, client: Any | None,
        run_dir: str | Path = "runtime/agent",
        review_root: str | Path = "runtime/review",
    ):
        self.method = method
        self.client = client
        self.run_dir = Path(run_dir).expanduser().resolve()
        self.review_root = Path(review_root).expanduser().resolve()

    def prepare(
        self, *, source_run_id: str, historical_review_run_id: str,
        malfunction_id: str, historical_parent_scenario_ids: list[str],
        operating_mode: str,
    ) -> dict[str, Any]:
        if len(historical_parent_scenario_ids) != len(set(historical_parent_scenario_ids)):
            raise ValueError("Historical parent Scenario IDs must be unique")
        if not historical_parent_scenario_ids:
            raise ValueError("At least one historical parent Scenario ID is required")
        repository = CheckpointRepository(self.run_dir)
        source_path = repository.path_for(source_run_id)
        source = repository.load(source_run_id)
        current_hash = str(self.method.metadata.get("method_source_hash", ""))
        if source.method_contract.get("method_source_hash") != current_hash:
            raise ValueError("Parent recovery source MethodContract differs from active Method")
        malfunction_values = [
            item for item in source.malfunctions
            if item.get("malfunction_id") == malfunction_id
        ]
        if len(malfunction_values) != 1:
            raise ValueError("Parent recovery requires one committed malfunction")
        malfunction = _malfunction(malfunction_values[0])
        if malfunction.status is not ReviewStatus.FINALIZED:
            raise ValueError("Parent recovery malfunction must be FINALIZED")
        typed = _project_only_typed(source.item_definition.get("typed", {}))
        project_facts = ItemDefinitionFacts.from_dict(typed)
        speed = ProjectFactResolver().resolve_speed_context(
            project_facts, operating_mode, allow_aggregate_fallback=False,
        )
        candidates, _ = MethodScenarioCandidateService(self.method).generate(
            project_facts=project_facts,
            operating_mode=speed.operating_mode,
            speed_resolution=speed,
            functions=[_function(item) for item in source.functions],
        )
        review_path = (
            self.review_root / historical_review_run_id / "scenario_candidates.jsonl"
        )
        if not review_path.is_file():
            raise FileNotFoundError(f"Historical parent review artifact is absent: {review_path}")
        old = {
            str(item.get("scenario_id", "")): item
            for line in review_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
            for item in [json.loads(line)]
            if item.get("scenario_id") in historical_parent_scenario_ids
        }
        if set(old) != set(historical_parent_scenario_ids):
            raise ValueError("Historical review does not cover the requested parent IDs")
        child_groups: dict[str, list[str]] = {key: [] for key in old}
        for scenario in source.scenarios:
            if scenario.source_scenario_id in child_groups:
                if scenario.analysis_instance.get("malfunction_id") != malfunction_id:
                    raise ValueError("Historical child malfunction identity conflicts")
                child_groups[scenario.source_scenario_id].append(scenario.scenario_id)
        if any(not children for children in child_groups.values()):
            raise ValueError("Historical parent lacks committed child lineage")
        mappings = []
        selected = []
        for old_id in historical_parent_scenario_ids:
            historical = old[old_id]
            if malfunction_id not in historical.get("generated_for_malfunction_ids", []):
                raise ValueError("Historical parent was not generated for this malfunction")
            matches = [
                item for item in candidates
                if item.operating_scenario == historical.get("operating_scenario")
                and item.operating_mode == historical.get("operating_mode")
            ]
            if len(matches) != 1:
                raise ValueError("Current parent Scenario has no unique source-context match")
            current = matches[0]
            old_facts = historical.get("facts", {})
            old_bindings = old_facts.get("method_scenario_dimensions", {})
            new_bindings = current.facts.get("method_scenario_dimensions", {})
            if (
                not isinstance(old_bindings, dict)
                or set(old_bindings) != set(new_bindings)
                or any(
                    old_bindings[key].get("project_value")
                    != new_bindings[key].get("project_value")
                    for key in new_bindings
                )
                or old_facts.get("ego_speed_constraint")
                != current.facts.get("ego_speed_constraint")
            ):
                raise ValueError("Historical/current parent ProjectFacts differ")
            selected.append(current)
            mappings.append({
                "historical_parent_scenario_id": old_id,
                "current_parent_scenario_id": current.scenario_id,
                "operating_scenario": current.operating_scenario,
                "historical_child_scenario_ids": sorted(child_groups[old_id]),
                "project_value_bindings_match": True,
                "ego_speed_constraint_matches": True,
            })
        if len({item.scenario_id for item in selected}) != len(selected):
            raise ValueError("Multiple old parent groups map to one current parent")
        return {
            "source": source, "source_path": source_path,
            "historical_review_path": review_path,
            "malfunction": malfunction, "project_facts": project_facts,
            "typed": typed, "candidates": selected, "mappings": mappings,
        }

    def run(
        self, *, source_run_id: str, historical_review_run_id: str,
        target_run_id: str, malfunction_id: str,
        historical_parent_scenario_ids: list[str], operating_mode: str,
    ) -> dict[str, Any]:
        if self.client is None:
            raise ValueError("Parent recovery requires a Provider client")
        if target_run_id in {source_run_id, historical_review_run_id}:
            raise ValueError("Parent recovery requires a new run ID")
        repository = CheckpointRepository(self.run_dir)
        target_path = repository.path_for(target_run_id)
        target_review = self.review_root / target_run_id
        if target_path.exists() or target_review.exists():
            raise ValueError("Parent recovery target already exists")
        prepared = self.prepare(
            source_run_id=source_run_id,
            historical_review_run_id=historical_review_run_id,
            malfunction_id=malfunction_id,
            historical_parent_scenario_ids=historical_parent_scenario_ids,
            operating_mode=operating_mode,
        )
        before = {
            str(path): _sha256(path)
            for path in (prepared["source_path"], prepared["historical_review_path"])
        }
        assessments, agent_audit = ScenarioFeasibilityAgent(
            self.client, batch_max_chars=60000, batch_max_items=8,
        ).assess(
            prepared["malfunction"], prepared["candidates"],
            build_project_evidence_registry(prepared["project_facts"], self.method),
        )
        expected_ids = {item.scenario_id for item in prepared["candidates"]}
        if (
            len(assessments) != len(expected_ids)
            or {item.scenario_id for item in assessments} != expected_ids
            or any(item.malfunction_id != malfunction_id for item in assessments)
        ):
            raise ValueError("Parent causal assessments do not exactly cover the batch")
        by_id = {item.scenario_id: item for item in prepared["candidates"]}
        committed = []
        retained_ids = set()
        for assessment in assessments:
            payload = assessment.to_dict()
            if evaluate_risk_eligibility(assessment).eligible:
                causal = payload.get("causal_assessment", {})
                chain = causal.get("causal_chain", []) if isinstance(causal, dict) else []
                if not chain or not causal.get("status") == "VALIDATED":
                    raise ValueError("Retained parent lacks validated typed causal identity")
                candidate = by_id[assessment.scenario_id]
                context = HazardousEventRiskContextService(self.method).build(
                    malfunction_id=malfunction_id,
                    scenario_id=candidate.scenario_id,
                    hazard_node_id=str(chain[-1]),
                    scenario={
                        **candidate.facts,
                        "_fact_provenance": candidate.fact_provenance,
                    },
                )
                payload["hazardous_event_id"] = context.hazardous_event_id
                payload["identity_basis"] = "VALIDATED_TYPED_CAUSAL_HAZARD_NODE"
                retained_ids.add(candidate.scenario_id)
            else:
                payload["hazardous_event_id"] = ""
            committed.append(payload)
        source = prepared["source"]
        child = HARAState.read_committed(source.to_dict())
        child.run_id = target_run_id
        child.stage = WorkflowStage.SCENARIOS if retained_ids else WorkflowStage.BLOCKED
        child.item_definition = deepcopy(source.item_definition)
        child.item_definition["typed"] = prepared["typed"]
        child.item_definition["scenario_assessments"] = committed
        child.item_definition.pop("scenario_assessment_batches", None)
        child.item_definition.pop("scenario_risk_fact_batches", None)
        child.scenarios = [
            item for item in prepared["candidates"]
            if item.scenario_id in retained_ids
        ]
        child.risk_results = []
        child.safety_goals = []
        child.pending_reviews = []
        child.errors = []
        child.audit_trail = [{
            "event": "parent_he_recovery_child_run_materialized",
            "source_run_id": source_run_id,
            "historical_review_run_id": historical_review_run_id,
            "target_run_id": target_run_id,
            "malfunction_id": malfunction_id,
            "parent_context_mappings": prepared["mappings"],
            "provider_calls_for_upstream_regeneration": 0,
        }]
        after = {path: _sha256(Path(path)) for path in before}
        if before != after:
            raise RuntimeError("Parent recovery modified historical source evidence")
        checkpoint_path = repository.save(child)
        audit = {
            "artifact_version": "parent-he-recovery-v1",
            "run_id": target_run_id,
            "source_run_id": source_run_id,
            "historical_review_run_id": historical_review_run_id,
            "method_source_hash": self.method.metadata.get("method_source_hash", ""),
            "malfunction_id": malfunction_id,
            "historical_child_count": sum(
                len(item["historical_child_scenario_ids"])
                for item in prepared["mappings"]
            ),
            "parent_context_mappings": prepared["mappings"],
            "assessments": committed,
            "retained_parent_count": len(retained_ids),
            "causal_gap_count": len(committed) - len(retained_ids),
            "provider_audit": agent_audit,
            "source_sha256_before_after": before,
            "checkpoint": str(checkpoint_path),
        }
        _write_json(target_review / "parent_he_recovery.json", audit)
        return audit
