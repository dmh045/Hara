"""Offline risk-stage execution from an immutable committed checkpoint."""

from copy import deepcopy
import hashlib
import json
from pathlib import Path
from time import perf_counter
from uuid import uuid4

from hara_agent.method_sources import MethodSourceResolver
from hara_agent.method_sources.yaml_validation import validate_manifest
from hara_agent.models import ItemDefinitionFacts, evaluate_risk_eligibility_payload
from hara_agent.services.analysis.risk_services import RiskScoringServices
from hara_agent.services.analysis.scenario_physics import source_is_accepted_for
from hara_agent.services.reporting.canonical_renderer import HARAReportWorkbookRenderer
from hara_agent.services.reporting.projection import HARAReportProjectionService
from hara_agent.services.reporting.report_schema import load_report_schema
from hara_agent.services.semantic.scenario_evidence import SCENARIO_ASSESSMENT_CONTRACT_VERSION
from .checkpoints import CheckpointRepository
from .nodes.scoring import score_structured_scenarios, _validated_causal_assessment
from .review_artifacts import ReviewArtifactWriter
from .state import HARAState, WorkflowStage


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def value_counts(state: HARAState) -> dict[str, int]:
    return {field: sum(bool(getattr(risk, field).value) for risk in state.risk_results)
            for field in ("severity", "exposure", "controllability", "asil")}


def clean_risk_stage(state: HARAState) -> None:
    """Invalidate only results owned by scoring and its downstream stages."""
    state.risk_results = []
    state.safety_goals = []
    state.pending_reviews = [item for item in state.pending_reviews if not (
        item.get("field") in {"severity", "exposure", "controllability", "asil", "safety_goal", "safe_state", "ftti"}
        or "assessment_id" in item or "safety_goal_id" in item
    )]
    for item in state.item_definition["scenario_assessments"]:
        if not evaluate_risk_eligibility_payload(item).eligible:
            continue
        item["potential_harm"] = ""
        causal = item.get("causal_assessment")
        if causal:
            causal["potential_harm"] = ""
            graph = causal["causal_graph"]
            harm_ids = {node["node_id"] for node in graph["nodes"] if node["node_type"] == "HARM"}
            graph["nodes"] = [node for node in graph["nodes"] if node["node_id"] not in harm_ids]
            graph["edges"] = [edge for edge in graph["edges"] if edge["target"] not in harm_ids]
            causal["causal_chain"] = [node for node in causal["causal_chain"] if node not in harm_ids]
            causal["evidence_bindings"] = [binding for binding in causal["evidence_bindings"]
                                           if binding["edge_id"] != "H_TO_HARM"]
            if harm_ids:
                causal["provenance"] = list(dict.fromkeys(
                    ref for binding in causal["evidence_bindings"] for ref in binding["evidence_refs"]))
    state.audit_trail = [item for item in state.audit_trail if item.get("event") not in {
        "structured_risk_scoring_completed", "safety_goals_generated", "quality_gate_evaluated", "report_rendered",
        "safety_goals_aggregated", "quality_gate_blocked", "draft_excel_report_rendered",
    }]
    state.stage = WorkflowStage.SCORING


def apply_supplement(state: HARAState, payload: dict, checkpoint_hash: str) -> list[dict]:
    """Only reassert unchanged facts; changed conditions await differential validation.

    A source citation is necessary but is not proof that an old causal result
    applies to a newly introduced operating condition. No such result is copied.
    """
    if payload["source_run_id"] != state.run_id or payload["source_checkpoint_sha256"] != checkpoint_hash:
        raise ValueError("Risk supplement is not bound to this source checkpoint")
    decisions = []
    by_id = {item.scenario_id: item for item in state.scenarios}
    pairs = {(item["malfunction_id"], item["scenario_id"])
             for item in state.item_definition["scenario_assessments"]}
    for entry in payload.get("entries", []):
        field = entry["field"]
        scopes = payload["scopes"][entry["scope_id"]]
        if not scopes:
            raise ValueError("Supplement scope cannot be empty")
        for scope in scopes:
            pair = (scope["malfunction_id"], scope["scenario_id"])
            if pair not in pairs:
                raise ValueError(f"Supplement scope not present in source run: {pair}")
        if entry["value"] is None:
            decisions.append({"field": field, "scope_id": entry["scope_id"],
                              "status": "MISSING_INPUT", "association_count": len(scopes)})
            continue
        for scope in scopes:
            pair = (scope["malfunction_id"], scope["scenario_id"])
            decision = {"field": field, **scope}
            metadata = entry["provenance"]
            if not source_is_accepted_for(metadata, malfunction_id=pair[0], scenario_id=pair[1]):
                raise ValueError(f"Supplement lacks accepted scoped provenance: {pair}/{field}")
            candidate = by_id[pair[1]]
            if (field in candidate.facts and candidate.facts[field] == entry["value"]
                    and candidate.fact_provenance.get(field) == metadata):
                decisions.append({**decision, "status": "UNCHANGED_EXISTING_FACT"})
            else:
                # Persist the proposed input and parent link, never a cloned HE.
                decisions.append({**decision, "status": "PENDING_DIFFERENTIAL_VALIDATION",
                                  "value": entry["value"], "unit": entry["unit"],
                                  "provenance": metadata,
                                  "reason": "New or changed conditions have no deterministic causal-reuse proof."})
    exclusions = payload.get("excluded_risk_facts", [])
    facts = {item["fact_id"]: item for item in state.item_definition["typed"]["risk_facts"]}
    for exclusion in exclusions:
        fact_id = exclusion["fact_id"]
        if fact_id not in facts or not exclusion.get("reason") or not exclusion.get("source_locator"):
            raise ValueError("Risk fact exclusion requires an existing ID, reason and source locator")
        # Demotion affects only the target run and cannot manufacture a value.
        facts[fact_id]["approval"] = "PENDING"
        decisions.append({"fact_id": fact_id, "status": "SOURCE_INCOMPATIBLE", **exclusion})
    return decisions


class OfflineRiskRescorer:
    def run(self, *, source_run_id: str, target_run_id: str,
            baseline: Path, report_template: Path, output: Path,
            run_dir: Path = Path("runtime/agent"), review_root: Path = Path("runtime/review"),
            checkpoint: Path | None = None, supplement: Path | None = None,
            source_baseline: Path | None = None) -> dict:
        started = perf_counter()
        repository = CheckpointRepository(run_dir)
        source_path = (checkpoint or repository.path_for(source_run_id)).resolve()
        target_path = repository.path_for(target_run_id)
        target_review = review_root.resolve() / target_run_id
        output = output.resolve()
        if source_run_id == target_run_id or target_path.exists() or target_review.exists() or output.exists():
            raise ValueError("Use a new target run and output; source and existing artifacts are immutable")
        raw = json.loads(source_path.read_text(encoding="utf-8"))
        if raw["run_id"] != source_run_id:
            raise ValueError("Checkpoint run_id differs from requested source run")
        if (not raw.get("scenario_contract_version")
                or raw.get("scenario_assessment_contract_version") != SCENARIO_ASSESSMENT_CONTRACT_VERSION):
            raise ValueError("Source causal/scenario contracts are incompatible; differential validation required")
        for key in ("functions", "guideword_assessments", "malfunctions", "scenarios", "risk_results"):
            if not raw.get(key):
                raise ValueError(f"Canonical checkpoint lacks {key}: {source_path}")
        source = HARAState.read_committed(raw)
        ItemDefinitionFacts.from_dict(source.item_definition["typed"])
        if any(not item.semantic_fingerprint or not item.fact_provenance for item in source.scenarios):
            raise ValueError("Canonical scenarios require fingerprints and fact provenance")
        if any(item.scenario_contract_version != source.scenario_contract_version for item in source.scenarios):
            raise ValueError("Source scenario generation versions are inconsistent")
        assessments = source.item_definition["scenario_assessments"]
        pairs = [(item["malfunction_id"], item["scenario_id"]) for item in assessments]
        if len(pairs) != len(set(pairs)):
            raise ValueError("Duplicate source MF/Scenario assessments")
        eligible = {pair for pair, item in zip(pairs, assessments)
                    if evaluate_risk_eligibility_payload(item).eligible}
        original_risks = {(risk.malfunction_id, risk.scenario_id): risk for risk in source.risk_results}
        if len(original_risks) != len(source.risk_results) or set(original_risks) != eligible:
            raise ValueError("Source risks do not exactly cover the committed eligible queue")
        for item in assessments:
            if (item["malfunction_id"], item["scenario_id"]) in eligible:
                causal = _validated_causal_assessment(item)
                if any(not binding.source_refs for binding in causal.evidence_bindings):
                    raise ValueError("Source validated causal assessment lacks source references")
        resolution = MethodSourceResolver().resolve(
            template_path=None, baseline_manifest_path=baseline, report_template_path=report_template)
        method = resolution.method
        method_transition = None
        if source.method_contract["method_source_hash"] != method.metadata["method_source_hash"]:
            if source_baseline is None:
                raise ValueError("Source method hash changed; source baseline proof required")
            old_resolution = MethodSourceResolver().resolve(
                template_path=None, baseline_manifest_path=source_baseline,
                report_template_path=report_template,
            )
            old_manifest, old_assets, old_hashes = validate_manifest(source_baseline.resolve())
            new_manifest, new_assets, new_hashes = validate_manifest(baseline.resolve())
            policy_path = new_manifest["normalized_sources"]["project_analysis_policy"]
            old_policy = old_assets["project_analysis_policy"]
            new_policy = new_assets["project_analysis_policy"]
            expected_policy = deepcopy(old_policy)
            expected_policy["controllability"] = {
                "unknown_override_policy": "SKIP_TO_TTC",
                "basis": "POSITIVE_OVERRIDE_ONLY_THEN_TTC_FALLBACK",
                "status": "CONFIRMED_FOR_CURRENT_PROJECT",
            }
            expected_manifest = deepcopy(old_manifest)
            expected_manifest["asset_hashes"][policy_path] = new_hashes[policy_path]
            if not all((
                old_resolution.method.metadata["method_source_hash"]
                == source.method_contract["method_source_hash"],
                old_resolution.method.contract_version == method.contract_version,
                old_resolution.method.compiler_version == method.compiler_version,
                new_manifest == expected_manifest,
                set(old_hashes) == set(new_hashes),
                all(old_hashes[key] == new_hashes[key]
                    for key in old_hashes if key != policy_path),
                old_policy.get("controllability") in (None, {}),
                new_policy == expected_policy,
            )):
                raise ValueError("Method transition is not limited to current-project C routing")
            method_transition = {
                "type": "CURRENT_PROJECT_C_UNKNOWN_OVERRIDE_ROUTING_ONLY",
                "source_method_hash": old_resolution.method.metadata["method_source_hash"],
                "target_method_hash": method.metadata["method_source_hash"],
                "source_baseline": str(source_baseline.resolve()),
                "source_baseline_sha256": file_hash(source_baseline.resolve()),
                "unchanged_asset_count": len(old_hashes) - 1,
                "changed_asset": policy_path,
                "source_policy_sha256": old_hashes[policy_path],
                "target_policy_sha256": new_hashes[policy_path],
            }
        if (source.method_contract["contract_version"] != method.contract_version
                or source.method_contract["compiler_version"] != method.compiler_version):
            raise ValueError("Source method version/hash incompatible; stage dependency proof is required")
        source_files = [source_path, Path(source.item_definition["source_path"]).resolve(), report_template.resolve()]
        source_files.extend(Path(event["output_path"]).resolve() for event in source.audit_trail
                            if event.get("event") == "draft_excel_report_rendered")
        source_review = review_root.resolve() / source_run_id
        source_files.extend(path for path in source_review.rglob("*") if path.is_file())
        hashes = {str(path): file_hash(path) for path in source_files}
        state = deepcopy(source)
        supplement_payload = json.loads(supplement.read_text(encoding="utf-8")) if supplement else {}
        supplement_decisions = apply_supplement(
            state, supplement_payload, hashes[str(source_path)],
        ) if supplement else []
        clean_risk_stage(state)
        state.run_id = target_run_id
        if method_transition:
            state.method_contract["method_source_hash"] = method.metadata["method_source_hash"]
            state.method_contract["template_hash"] = method.metadata["method_source_hash"]
        execution_id = f"risk-{uuid4()}"
        state.record("offline_risk_rescoring_started", source_run_id=source_run_id,
                     source_checkpoint=str(source_path), source_checkpoint_sha256=hashes[str(source_path)],
                     risk_execution_id=execution_id, supplement_decisions=supplement_decisions,
                     method_transition=method_transition)
        writer = ReviewArtifactWriter(target_run_id, review_root)
        for item in state.functions:
            writer.record_function(item)
        for item in state.guideword_assessments:
            writer.record_guideword_assessment(item)
        for item in state.malfunctions:
            writer.record_malfunction(item)
        for item in state.scenarios:
            writer.record_scenario_candidate(item)
        services = RiskScoringServices.from_method(method)
        state = score_structured_scenarios(state, services.scoring, services.asil, services.binding, writer)
        new_pairs = {(risk.malfunction_id, risk.scenario_id) for risk in state.risk_results}
        if new_pairs != eligible:
            raise ValueError("Offline scoring changed the original eligible queue")
        trace_path = target_review / "risk_execution_trace.json"
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        trace.update(source_run_id=source_run_id, risk_execution_id=execution_id,
                     source_checkpoint_sha256=hashes[str(source_path)])
        risks = {(risk.malfunction_id, risk.scenario_id): risk for risk in state.risk_results}
        for row in trace["assessments"]:
            pair = (row["malfunction_id"], row["scenario_id"])
            row["source_run_id"] = source_run_id
            row["risk_execution_id"] = execution_id
            row["parent_assessment_id"] = original_risks[pair].assessment_id if pair in eligible else ""
            row["disposition"] = ("RECOMPUTED" if all(getattr(risks[pair], f).value for f in ("severity", "exposure", "controllability", "asil"))
                                  else "RECOMPUTED_PENDING_INPUT_OR_METHOD") if pair in eligible else "HISTORICAL_EXCLUSION"
        write_json(trace_path, trace)
        # Checkpoint holds the full authoritative typed records. Review artifacts
        # use the same source associations, not reconstructed Excel text.
        for item in state.item_definition["scenario_assessments"]:
            writer.record_scenario_feasibility(item)
        writer.write_summary(state)
        state.record("offline_risk_rescoring_completed", source_run_id=source_run_id,
                     risk_execution_id=execution_id, safety_goals_status="NOT_GENERATED_OFFLINE",
                     ftti_status="NOT_EVALUATED")
        state.stage = WorkflowStage.QUALITY_GATE
        repository.save(state)
        summary = {
            "source_run_id": source_run_id, "run_id": target_run_id,
            "risk_execution_id": execution_id, "source_checkpoint": str(source_path),
            "causal_reuse_basis": {
                "assessment_contract": source.scenario_assessment_contract_version,
                "preserved_generation_contract": source.scenario_contract_version,
                "scenario_fingerprints": {item.scenario_id: item.semantic_fingerprint for item in source.scenarios},
                "scenario_facts_and_semantics_unchanged": state.scenarios == source.scenarios,
                "method_source_unchanged": method_transition is None,
                "method_transition": method_transition,
            },
            "target_checkpoint": str(target_path),
            "reused": {"functions": len(state.functions), "guideword_assessments": len(state.guideword_assessments),
                       "malfunctions": len(state.malfunctions), "scenarios": len(state.scenarios),
                       "scenario_assessments": len(assessments)},
            "eligible_queue": len(eligible), "historical_exclusions": len(pairs) - len(eligible),
            "scenario_feasibility_count": len(pairs), "scenario_feasible_count": len(eligible),
            "scorer_calls": sum(row["risk_scoring_invoked"] for row in trace["assessments"]),
            "values_before": value_counts(source), "values_after": value_counts(state),
            "supplement_decisions": supplement_decisions, "provider_calls": 0,
            "supplement_path": str(supplement.resolve()) if supplement else "",
            "supplement_sha256": file_hash(supplement) if supplement else "",
            "analytical_options_pending_validation": len(supplement_payload.get("analytical_options_pending_validation", [])),
            "source_file_hashes": hashes, "source_files_unchanged": False,
            "output": str(output), "safety_goals_status": "NOT_GENERATED_OFFLINE", "ftti_status": "NOT_EVALUATED",
        }
        summary["scenario_eligibility_summary"] = trace["scenario_eligibility_summary"]
        schema = load_report_schema()
        view = HARAReportProjectionService(schema).project(
            state, method, risk_trace=trace, run_summary=summary,
            style_template_hash=resolution.report_template_hash,
            risk_trace_reference=f"{trace_path}#risk_execution_id={execution_id}",
        )
        HARAReportWorkbookRenderer().render(view, report_template, output, schema)
        if any(file_hash(Path(path)) != digest for path, digest in hashes.items()):
            raise ValueError("Source file changed during offline execution")
        summary["source_files_unchanged"] = True
        summary["elapsed_seconds"] = round(perf_counter() - started, 3)
        summary["business_scoring_complete"] = all(
            count == len(eligible) for count in summary["values_after"].values())
        write_json(target_review / "rescore_summary.json", summary)
        return summary
