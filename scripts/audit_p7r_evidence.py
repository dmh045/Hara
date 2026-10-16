"""Read-only P7R evidence reconciliation; no Provider and no input injection."""
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from hara_agent.method_sources import MethodSourceResolver
from hara_agent.models import ItemDefinitionFacts, MalfunctionCandidate, ScenarioCandidate, SourceRef, ReviewStatus
from hara_agent.services.analysis import MethodContractASILService
from hara_agent.services.semantic import ScenarioFeasibilityAgent, build_project_evidence_registry


ROOT = Path("output/p7r_closure")


def read(path):
    return json.loads(path.read_text())


def replay_review_samples(output: Path):
    """Reparse saved responses, rescore original production candidates, render.

    No client is constructed and no physical/scoring input is patched here.
    The fresh generation is produced separately by benchmark_p6_offline.py.
    """
    from hara_agent.workflow import HARAState, ReviewArtifactWriter
    from hara_agent.workflow.checkpoints import CheckpointRepository
    from hara_agent.workflow.nodes.scoring import score_structured_scenarios
    from hara_agent.services.analysis import StructuredRiskScoringService, MethodRiskFactBindingService
    from hara_agent.services.reporting.offline_rebuild import OfflineReportRebuilder
    baseline = Path("method_assets/fusa_baseline_v1/manifest.yaml")
    template = Path("references/HARA_Template_AI_20260327.xlsx")
    method = MethodSourceResolver().resolve(template_path=None, baseline_manifest_path=baseline,
                                            report_template_path=template).method
    generated = {row["scenario_id"]: row["source_candidate"]
                 for row in read(output / "generation/candidate_previews.json")}
    summaries = []
    all_complete_ids = set()
    for live, plan_name, label in (("live_001", "live_plan_001.json", "14_Source_Samples"),
                                    ("live_003", "live_plan_002.json", "2_Validator_Samples")):
        plan = read(ROOT / plan_name)
        directory = output / live
        if directory.exists():
            raise ValueError(f"Replay output already exists: {directory}")
        directory.mkdir(parents=True)
        latest = {}
        responses = ROOT / live / "provider_responses.jsonl"
        for line in responses.read_text().splitlines():
            record = json.loads(line)
            payload = record["response"]["data"]
            for item in payload.get("assessments", [payload] if "scenario_id" in payload else []):
                latest[(record["metadata"]["malfunction_id"], item["scenario_id"])] = item
        state = HARAState.read_committed(read(ROOT / live / "causal_state.json"))
        state.run_id = "p7r-review-replay-20261016-" + live[-3:]
        state.item_definition["scenario_assessments"] = []
        state.risk_results = []
        registry = build_project_evidence_registry(ItemDefinitionFacts.from_dict(plan["project"]), method)
        by_id = {candidate.scenario_id: candidate for candidate in state.scenarios}
        parsed = []
        source_matches = []
        for case in plan["cases"]:
            raw = case["malfunction"]
            mf = MalfunctionCandidate(**{**raw, "sources": [SourceRef(**s) for s in raw["sources"]],
                                         "status": ReviewStatus(raw["status"])})
            for original in case["candidates"]:
                sid = original["scenario_id"]
                candidate = by_id[sid]
                fresh = generated[sid]
                assert fresh["facts"] == original["facts"]
                assert fresh["fact_provenance"] == original["fact_provenance"]
                source_matches.append(sid)
                assessment = ScenarioFeasibilityAgent._parse(mf, latest[(mf.malfunction_id, sid)],
                    scenario=candidate, project_registry=registry)
                parsed.append(assessment.to_dict())
        state.item_definition["scenario_assessments"] = parsed
        state.record("p7r_independent_review_saved_response_replay", provider_calls=0,
                     source_responses=str(responses), source_sha256=hashlib.sha256(responses.read_bytes()).hexdigest(),
                     current_validator=ScenarioFeasibilityAgent.PROMPT_VERSION,
                     note="Saved v18/v20 outputs; no new live v21 validation; Safety Goal stage not run.")
        writer = ReviewArtifactWriter(state.run_id, directory / "review")
        for function in state.functions:
            writer.record_function(function)
        for mf in state.malfunctions:
            writer.record_malfunction(mf)
        for candidate in state.scenarios:
            writer.record_scenario_candidate(candidate, generated_for_malfunction_ids=[candidate.analysis_instance["malfunction_id"]])
        for assessment in parsed:
            writer.record_scenario_feasibility(assessment)
        state = score_structured_scenarios(state, StructuredRiskScoringService(method), MethodContractASILService(method),
            risk_fact_binding=MethodRiskFactBindingService(method), review_artifact_writer=writer)
        writer.write_summary(state)
        checkpoint = CheckpointRepository(directory / "checkpoints").save(state)
        report = OfflineReportRebuilder().rebuild(checkpoint_path=checkpoint, method_baseline_path=baseline,
            report_style_template_path=template, output_path=directory / f"HARA_P7R_Review_{label}_20261016.xlsx",
            review_root=directory / "review", audit_output_dir=directory / "report_audit")
        complete = [risk for risk in state.risk_results if all(getattr(risk, field).status is ReviewStatus.FINALIZED
                    for field in ("severity", "exposure", "controllability", "asil"))]
        all_complete_ids.update((r.malfunction_id, r.scenario_id) for r in complete)
        summary = {"run_id": state.run_id, "source_live_run": live, "new_provider_calls": 0,
            "source_candidate_matches": len(source_matches), "parsed": len(parsed),
            "retained": sum(a["causally_relevant"] for a in parsed), "risk_records": len(state.risk_results),
            "full_SEC_ASIL": len(complete), "ratings": dict(Counter(
                "/".join(getattr(r, key).value for key in ("severity", "exposure", "controllability", "asil")) for r in complete)),
            "report": str(report), "checkpoint": str(checkpoint)}
        (directory / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
        summaries.append(summary)
    result = {"runs": summaries, "unique_complete_source_identities_across_replays": len(all_complete_ids),
              "new_provider_calls": 0, "live_v21_test": "NOT_RUN"}
    (output / "sample_replay_summary.json").write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False))


def run():
    method = MethodSourceResolver().resolve(template_path=None,
        baseline_manifest_path=Path("method_assets/fusa_baseline_v1/manifest.yaml"),
        report_template_path=Path("references/HARA_Template_AI_20260327.xlsx")).method
    total, live = 0, []
    for name in ("live_001", "live_002", "live_003"):
        directory = ROOT / name
        events = [json.loads(line) for line in (directory / "provider_attempts.jsonl").read_text().splitlines()]
        attempts = sum(row["event"] == "attempt_started" for row in events)
        total += attempts
        summary = read(directory / "summary.json")
        summary["actual_returned_models"] = sorted({row["model"] for row in events
            if row["event"] == "attempt_finished" and row.get("model")})
        assert attempts == summary["provider_attempts"]
        checkpoint = next((directory / "checkpoints").glob("*.json"))
        state = read(checkpoint)
        verified = []
        for risk in state["risk_results"]:
            for branch in risk.get("branch_evaluations", []):
                if branch["ASIL"] and all(branch.get(key) for key in ("S", "E", "C")):
                    assert MethodContractASILService(method).determine(branch["S"], branch["E"], branch["C"]) == branch["ASIL"]
                    verified.append({key: branch[key] for key in ("branch_id", "S", "E", "C", "ASIL", "relative_speed_kph", "ttc_s")})
        summary["matrix_checked_branches"] = verified
        summary["source_paths"] = [{"malfunction_id": m["malfunction_id"], "description": m["description"]} for m in state["malfunctions"]]
        live.append(summary)
    assert total <= 20

    # Reparse the exact latest returned items under the final, scoped validator.
    # A repair replaces only its matching item; no payload fields are edited.
    plan = read(ROOT / "live_plan_002.json")
    case = plan["cases"][0]
    raw = case["malfunction"]
    mf = MalfunctionCandidate(**{**raw, "sources": [SourceRef(**s) for s in raw["sources"]], "status": ReviewStatus(raw["status"])})
    registry = build_project_evidence_registry(ItemDefinitionFacts.from_dict(plan["project"]), method)
    latest = {}
    for line in (ROOT / "live_003/provider_responses.jsonl").read_text().splitlines():
        payload = json.loads(line)["response"]["data"]
        for item in payload.get("assessments", [payload] if "scenario_id" in payload else []):
            latest[item["scenario_id"]] = item
    parsed = []
    for raw in case["candidates"]:
        candidate = ScenarioCandidate(**{**raw, "sources": [SourceRef(**s) for s in raw["sources"]], "status": ReviewStatus(raw["status"])})
        assessment = ScenarioFeasibilityAgent._parse(mf, latest[candidate.scenario_id], scenario=candidate, project_registry=registry)
        parsed.append({"scenario_id": candidate.scenario_id, "retain": assessment.retain,
            "production": candidate.analysis_instance["physical_input_production"],
            "causal": assessment.to_dict()})
        assert assessment.retain

    fixed = read(ROOT / "fixed_004/fixed_population_summary.json")
    generation = read(ROOT / "generation_004/summary.json")
    trace = read(ROOT / "fixed_004/review/p7r-offline-20261016-004/risk_execution_trace.json")
    rows = [row for row in trace["assessments"] if row.get("risk_scoring_invoked")]
    physical_pending = Counter(row.get("physical_input_production", {}).get("pending_reason", "") for row in rows)
    c_pending = Counter(row.get("controllability", {}).get("input_diagnostic", {}).get("reason", "") for row in rows)
    counts = generation["candidate_preview"]
    result = {"provider_budget": {"authorized": 20, "used": total, "remaining": 20-total},
        "method_hash": method.metadata["method_source_hash"], "compiler": method.compiler_version,
        "prompt_version": ScenarioFeasibilityAgent.PROMPT_VERSION, "live_runs": live,
        "final_validator_replay": parsed, "fixed_population": fixed, "fresh_generation": generation,
        "fixed_pending": {"S": dict(physical_pending), "C": dict(c_pending)},
        "optimization": {"exact_candidate_coalescing": counts["coalesced_candidates"],
            "causal_result_reuse": sum(r["reused_count"] for r in live),
            "shared_request_judgements": "paired requests retain separate judgements; not counted as reuse",
            "restored_response_cache_hits": sum(r["response_checkpoint_hits"] for r in live),
            "display_grouping": "existing malfunction index summarizes each child; no risk-equivalence claim"},
        "remaining_real_source_gap": "MF-F02-002 lacks a sourced post-fault acceleration/time or speed envelope; normal performance limits are not fault limits",
        "ready_for_unbounded_full_provider": False,
        "bounded_followup_condition": "New run and explicit budget; source-supported mechanisms can run. Fault-motion source gap and current zero causal reuse remain visible."}
    output = ROOT / "acceptance_evidence.json"
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"output": str(output), "provider_used": total, "final_reparsed": len(parsed),
                      "fixed_pending": result["fixed_pending"]}, ensure_ascii=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-replay-output", type=Path)
    args = parser.parse_args()
    if args.review_replay_output:
        replay_review_samples(args.review_replay_output)
    else:
        run()
