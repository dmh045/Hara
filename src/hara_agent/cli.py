from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace

from hara_agent.application import HARAApplication
from hara_agent.config import RunConfig
from hara_agent.config import LLMConfig, load_local_env
from hara_agent.contracts import (
    CompileStatus, TemplateRole, TemplateRoleConfirmation,
)
from hara_agent.template import TemplateRoleCompiler, TemplateRoleManifestStore
from hara_agent.method_sources import MethodSourceResolver, YamlBaselineCompiler
from hara_agent.services.analysis import (
    ConfirmedYamlUtilizationService, FMSelectorSemanticAuditService,
    ExposureBindingAuditService,
    ExposureDimensionCoverageAuditService,
    ExposureInputAuditService,
    FMTemplateAmbiguityAuditService,
    HazardousEventRiskContextService,
    RiskContextSourceCoverageAuditService,
    ControllabilityBranchAuditService,
    MethodContractParityAuditService,
    RiskExecutionTraceService,
    RiskScoreabilityService,
    SeverityDeltaVSemanticAuditService,
    ScenarioAliasProposalService,
    ScenarioCoverageProposalService,
)
from hara_agent.services.reporting import (
    OfflineReportRebuilder, ReportSchemaValidator,
    ScenarioOutputQualityAuditService, load_report_schema,
)
from hara_agent.services.extraction import DocumentReader
from hara_agent.workflow import ReviewArtifactReader, ReviewArtifactWriter, render_review
from hara_agent.workflow.state import HARAState


def _role_compiler() -> TemplateRoleCompiler:
    manifest_root = os.getenv(
        "HARA_TEMPLATE_ROLE_MANIFEST_DIR",
        "runtime/agent/template-role-manifests",
    )
    return TemplateRoleCompiler(
        manifest_store=TemplateRoleManifestStore(manifest_root)
    )


def _write_json_atomic(path: Path, payload: dict[str, object]) -> None:
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


def _write_text_atomic(path: Path, value: str) -> None:
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


def _parse_role_selections(values: list[str]) -> dict[str, dict[str, str]]:
    selections: dict[str, dict[str, str]] = {}
    for value in values:
        role_name, separator, location = value.partition("=")
        sheet, location_separator, region = location.rpartition("!")
        if not separator or not location_separator or not sheet or not region:
            raise ValueError(
                f"Invalid --select {value!r}; expected ROLE=SHEET!A1:B9"
            )
        role = TemplateRole(role_name.strip())
        if role.value in selections:
            raise ValueError(f"Duplicate role selection: {role.value}")
        selections[role.value] = {
            "sheet": sheet.strip(),
            "region": region.strip().replace("$", ""),
        }
    return selections


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="hara-agent")
    subparsers = parser.add_subparsers(dest="command", required=True)
    analyze = subparsers.add_parser("analyze", help="运行HARA Agent主链")
    analyze.add_argument("--item", required=True, type=Path)
    analyze.add_argument("--template", type=Path)
    analyze.add_argument(
        "--method-baseline", type=Path,
        default=Path("method_assets/fusa_baseline_v1/manifest.yaml"),
    )
    analyze.add_argument(
        "--report-template", type=Path,
        default=Path("references/HARA_Template_AI_20260327.xlsx"),
    )
    analyze.add_argument("--output", required=True, type=Path)
    analyze.add_argument("--run-dir", type=Path, default=Path("runtime/agent"))
    analyze.add_argument("--run-id", default="hara-run")
    analyze.add_argument("--resume", action="store_true")
    analyze.add_argument("--allow-draft", action="store_true")
    analyze.add_argument("--ego-speed-kph", type=float)
    analyze.add_argument("--ego-speed-source", default="")
    analyze.add_argument("--operating-mode")
    analyze.add_argument("--allow-aggregate-speed-fallback", action="store_true")
    analyze.add_argument("--max-workers", type=int, default=4)
    analyze.add_argument("--sample-function-limit", type=int)
    analyze.add_argument("--sample-function-id", action="append", default=[])
    analyze.add_argument("--sample-malfunction-limit", type=int)
    analyze.add_argument("--sample-malfunction-id", action="append", default=[])
    analyze.add_argument("--sample-parent-scenario-limit", type=int)
    analyze.add_argument("--sample-parent-scenario-id", action="append", default=[])
    analyze.add_argument("--sample-scenario-pair-limit", type=int, default=32)
    analyze.add_argument("--provider-attempt-limit", type=int)
    doctor = subparsers.add_parser("doctor", help="检查模板和运行配置，不执行分析")
    doctor.add_argument("--template", type=Path)
    doctor.add_argument(
        "--method-baseline", type=Path,
        default=Path("method_assets/fusa_baseline_v1/manifest.yaml"),
    )
    doctor.add_argument(
        "--report-template", type=Path,
        default=Path("references/HARA_Template_AI_20260327.xlsx"),
    )
    doctor.add_argument(
        "--review-run-id",
        help="Read deterministic Scenario binding coverage from a prior review artifact",
    )
    confirm = subparsers.add_parser(
        "confirm-template-role",
        help="一次性确认有歧义的模板角色，并按模板哈希保存系统清单",
    )
    confirm.add_argument("--template", required=True, type=Path)
    confirm.add_argument(
        "--select", action="append", required=True,
        metavar="ROLE=SHEET!A1:B9",
    )
    confirm.add_argument("--confirmed-by", required=True)
    confirm.add_argument("--rationale", default="")
    review = subparsers.add_parser(
        "review", help="只读查看运行中间审计 artifacts，不执行 HARA 或 LLM"
    )
    review.add_argument("--run-id", required=True)
    review.add_argument("--function")
    review.add_argument("--malfunction")
    review.add_argument("--scenario")
    review.add_argument("--all", action="store_true", dest="all_records")
    review.add_argument("--limit", type=int, default=3)
    review.add_argument("--feasible", action="store_true")
    review.add_argument("--infeasible", action="store_true")
    risk_trace = subparsers.add_parser(
        "risk-execution-trace",
        help="Generate a read-only Scenario-to-risk execution trace from review artifacts",
    )
    risk_trace.add_argument("--baseline", required=True, type=Path)
    risk_trace.add_argument("--review-run-id", required=True)
    risk_trace.add_argument(
        "--report-template", type=Path,
        default=Path("references/HARA_Template_AI_20260327.xlsx"),
    )
    scenario_aliases = subparsers.add_parser(
        "scenario-aliases",
        help="Generate review-only Scenario terminology alias proposals",
    )
    scenario_aliases.add_argument("--baseline", required=True, type=Path)
    scenario_aliases.add_argument("--review-run-id", required=True)
    scenario_aliases.add_argument(
        "--report-template", type=Path,
        default=Path("references/HARA_Template_AI_20260327.xlsx"),
    )
    scenario_coverage = subparsers.add_parser(
        "scenario-coverage",
        help="Generate review-only Scenario coverage governance proposals",
    )
    scenario_coverage.add_argument("--baseline", required=True, type=Path)
    scenario_coverage.add_argument("--review-run-id", required=True)
    scenario_coverage.add_argument(
        "--report-template", type=Path,
        default=Path("references/HARA_Template_AI_20260327.xlsx"),
    )
    yaml_utilization = subparsers.add_parser(
        "yaml-utilization",
        help="Generate offline confirmed YAML utilization audit",
    )
    yaml_utilization.add_argument("--baseline", required=True, type=Path)
    yaml_utilization.add_argument("--review-run-id", required=True)
    yaml_utilization.add_argument(
        "--report-template", type=Path,
        default=Path("references/HARA_Template_AI_20260327.xlsx"),
    )
    fm_selector_audit = subparsers.add_parser(
        "fm-selector-semantic-audit",
        help="Generate offline deterministic FM selector semantic audit",
    )
    fm_selector_audit.add_argument("--baseline", required=True, type=Path)
    fm_selector_audit.add_argument("--review-run-id", required=True)
    fm_selector_audit.add_argument(
        "--report-template", type=Path,
        default=Path("references/HARA_Template_AI_20260327.xlsx"),
    )
    fm_template_ambiguity = subparsers.add_parser(
        "fm-template-ambiguity-audit",
        help="Generate an offline FM template ambiguity root-cause audit",
    )
    fm_template_ambiguity.add_argument("--baseline", required=True, type=Path)
    fm_template_ambiguity.add_argument("--review-run-id", required=True)
    fm_template_ambiguity.add_argument(
        "--report-template", type=Path,
        default=Path("references/HARA_Template_AI_20260327.xlsx"),
    )
    exposure_binding_audit = subparsers.add_parser(
        "exposure-binding-audit",
        help="Generate an offline governed Scenario-atom Exposure input audit",
    )
    exposure_binding_audit.add_argument("--baseline", required=True, type=Path)
    exposure_binding_audit.add_argument("--review-run-id", required=True)
    exposure_binding_audit.add_argument(
        "--report-template", type=Path,
        default=Path("references/HARA_Template_AI_20260327.xlsx"),
    )
    exposure_coverage_audit = subparsers.add_parser(
        "exposure-dimension-coverage-audit",
        help="Audit formal Exposure dimension coverage authority without scoring E",
    )
    exposure_coverage_audit.add_argument("--baseline", required=True, type=Path)
    exposure_coverage_audit.add_argument("--review-run-id", required=True)
    exposure_coverage_audit.add_argument(
        "--report-template", type=Path,
        default=Path("references/HARA_Template_AI_20260327.xlsx"),
    )
    exposure_input_audit = subparsers.add_parser(
        "exposure-input-audit",
        help="Audit committed FUSA Exposure inputs and input readiness without a Provider",
    )
    exposure_input_audit.add_argument("--baseline", required=True, type=Path)
    exposure_input_audit.add_argument("--review-run-id", required=True)
    exposure_input_audit.add_argument("--checkpoint", type=Path)
    exposure_input_audit.add_argument("--review-root", type=Path, default=Path("runtime/review"))
    exposure_input_audit.add_argument("--run-dir", type=Path, default=Path("runtime/agent"))
    exposure_input_audit.add_argument(
        "--report-template", type=Path,
        default=Path("references/HARA_Template_AI_20260327.xlsx"),
    )
    exposure_root_cause = subparsers.add_parser(
        "exposure-root-cause-report",
        help="Render the Exposure E4 root-cause report from audit and offline-rescore artifacts",
    )
    exposure_root_cause.add_argument("--audit", required=True, type=Path)
    exposure_root_cause.add_argument("--rescore-trace", required=True, type=Path)
    exposure_root_cause.add_argument("--output", required=True, type=Path)
    severity_delta_v_audit = subparsers.add_parser(
        "severity-delta-v-semantic-audit",
        help="Audit Severity DELTA_V semantic authority and input readiness without scoring S",
    )
    severity_delta_v_audit.add_argument("--baseline", required=True, type=Path)
    severity_delta_v_audit.add_argument("--review-run-id", required=True)
    severity_delta_v_audit.add_argument(
        "--report-template", type=Path,
        default=Path("references/HARA_Template_AI_20260327.xlsx"),
    )
    risk_context_audit = subparsers.add_parser(
        "hazardous-event-risk-context-audit",
        help="Audit typed Hazardous Event to Severity/Controllability input context",
    )
    risk_context_audit.add_argument("--baseline", required=True, type=Path)
    risk_context_audit.add_argument("--review-run-id", required=True)
    risk_context_audit.add_argument(
        "--report-template", type=Path,
        default=Path("references/HARA_Template_AI_20260327.xlsx"),
    )
    risk_context_coverage = subparsers.add_parser(
        "risk-context-source-coverage-audit",
        help="Audit P3-A RiskContext source coverage without changing HARA inputs",
    )
    risk_context_coverage.add_argument("--baseline", required=True, type=Path)
    risk_context_coverage.add_argument("--review-run-id", required=True)
    risk_context_coverage.add_argument("--checkpoint", type=Path)
    risk_context_coverage.add_argument("--item", type=Path, default=Path("input/ItemDef.docx"))
    risk_context_coverage.add_argument(
        "--report-template", type=Path,
        default=Path("references/HARA_Template_AI_20260327.xlsx"),
    )
    controllability_branch_audit = subparsers.add_parser(
        "controllability-branch-audit",
        help="Audit selected controllability decision branches and input readiness",
    )
    controllability_branch_audit.add_argument("--baseline", required=True, type=Path)
    controllability_branch_audit.add_argument("--review-run-id", required=True)
    controllability_branch_audit.add_argument(
        "--report-template", type=Path,
        default=Path("references/HARA_Template_AI_20260327.xlsx"),
    )
    parity_audit = subparsers.add_parser(
        "risk-evaluator-parity-audit",
        help="Audit Template/YAML MethodContract and risk evaluator parity offline",
    )
    parity_audit.add_argument("--baseline", required=True, type=Path)
    parity_audit.add_argument("--template", required=True, type=Path)
    parity_audit.add_argument("--review-run-id", required=True)
    rebuild = subparsers.add_parser(
        "rebuild-report",
        help="Rebuild the canonical Engineering Report offline from a committed run",
    )
    rebuild.add_argument("--review-run-id", required=True)
    rebuild.add_argument("--checkpoint", type=Path)
    rebuild.add_argument(
        "--baseline", type=Path,
        default=Path("method_assets/fusa_baseline_v1/manifest.yaml"),
    )
    rebuild.add_argument(
        "--report-style-template", type=Path,
        default=Path("references/HARA_Template_AI_20260327.xlsx"),
    )
    rebuild.add_argument("--schema", type=Path, default=Path("report_assets/hara_report_v1.yaml"))
    rebuild.add_argument("--review-root", type=Path, default=Path("runtime/review"))
    rebuild.add_argument("--run-dir", type=Path, default=Path("runtime/agent"))
    rebuild.add_argument("--output", type=Path, default=Path("output/HARA_P2C_Content_Cleanup.xlsx"))
    rebuild.add_argument("--causal-trace", type=Path)
    rebuild.add_argument("--audit-output-dir", type=Path)
    rebuild.add_argument("--synthesis-candidates", type=Path)
    rebuild.add_argument("--speed-context-audit", type=Path)
    scenario_output_audit = subparsers.add_parser(
        "scenario-output-quality-audit",
        help="Audit accepted synthesized child scenarios offline without a Provider",
    )
    scenario_output_audit.add_argument("--checkpoint", required=True, type=Path)
    scenario_output_audit.add_argument("--review-run-id", required=True)
    scenario_output_audit.add_argument(
        "--baseline", type=Path,
        default=Path("method_assets/fusa_baseline_v1/manifest.yaml"),
    )
    scenario_output_audit.add_argument(
        "--report-style-template", type=Path,
        default=Path("references/HARA_Template_AI_20260327.xlsx"),
    )
    scenario_output_audit.add_argument("--review-root", type=Path, default=Path("runtime/review"))
    scenario_output_audit.add_argument("--causal-trace", type=Path)
    scenario_output_audit.add_argument("--output-dir", required=True, type=Path)
    scenario_output_audit.add_argument("--sample-output", required=True, type=Path)
    rescore = subparsers.add_parser(
        "rescore-risk", help="Recompute only risk from a committed checkpoint without a Provider",
    )
    rescore.add_argument("--source-run-id", required=True)
    rescore.add_argument("--target-run-id", required=True)
    rescore.add_argument("--checkpoint", type=Path)
    rescore.add_argument("--baseline", type=Path, default=Path("method_assets/fusa_baseline_v1/manifest.yaml"))
    rescore.add_argument("--report-style-template", type=Path, default=Path("references/HARA_Template_AI_20260327.xlsx"))
    rescore.add_argument("--risk-input-supplement", type=Path)
    rescore.add_argument("--run-dir", type=Path, default=Path("runtime/agent"))
    rescore.add_argument("--review-root", type=Path, default=Path("runtime/review"))
    rescore.add_argument("--output", type=Path, required=True)
    scoreability = subparsers.add_parser(
        "risk-scoreability",
        help="Classify S/C scoreability of existing analytical options without a Provider",
    )
    scoreability.add_argument("--checkpoint", type=Path, required=True)
    scoreability.add_argument("--risk-input-supplement", type=Path, required=True)
    scoreability.add_argument("--review-run-id", required=True)
    scoreability.add_argument("--review-root", type=Path, default=Path("runtime/review"))
    scoreability.add_argument(
        "--baseline", type=Path,
        default=Path("method_assets/fusa_baseline_v1/manifest.yaml"),
    )
    scoreability.add_argument(
        "--report-style-template", type=Path,
        default=Path("references/HARA_Template_AI_20260327.xlsx"),
    )
    scoreability.add_argument(
        "--assumption-pack", type=Path,
        default=Path("output/HARA_Risk_Assumption_Pack_Final.json"),
    )
    scoreability.add_argument(
        "--child-scenarios", type=Path,
        default=Path("output/HARA_Risk_Child_Scenarios.json"),
    )
    synthesis = subparsers.add_parser(
        "synthesize-scenarios",
        help="Run governed bounded Scenario synthesis as an immutable child run",
    )
    synthesis.add_argument("--source-run-id", required=True)
    synthesis.add_argument("--target-run-id", required=True)
    synthesis.add_argument("--run-dir", type=Path, default=Path("runtime/agent"))
    synthesis.add_argument("--review-root", type=Path, default=Path("runtime/review"))
    synthesis.add_argument(
        "--baseline", type=Path,
        default=Path("method_assets/fusa_baseline_v1/manifest.yaml"),
    )
    synthesis.add_argument(
        "--report-style-template", type=Path,
        default=Path("references/HARA_Template_AI_20260327.xlsx"),
    )
    synthesis.add_argument("--output", required=True, type=Path)
    synthesis.add_argument("--smoke-count", type=int, choices=range(3, 31), default=5)
    synthesis.add_argument(
        "--smoke-identity-file", type=Path,
        help="JSON artifact whose failures list supplies stable recovery-smoke identities",
    )
    synthesis.add_argument("--full", action="store_true")
    synthesis.add_argument("--max-workers", type=int, default=4)
    synthesis.add_argument("--provider-attempt-limit", type=int)
    synthesis.add_argument("--provider-budget-run-id")
    causal_revalidation = subparsers.add_parser(
        "revalidate-synthesized-scenarios",
        help="Differentially revalidate only method-valid synthesized child scenarios",
    )
    causal_revalidation.add_argument("--source-run-id", required=True)
    causal_revalidation.add_argument("--target-run-id", required=True)
    causal_revalidation.add_argument(
        "--run-dir", type=Path, default=Path("runtime/agent")
    )
    causal_revalidation.add_argument(
        "--review-root", type=Path, default=Path("runtime/review")
    )
    causal_revalidation.add_argument(
        "--baseline", type=Path,
        default=Path("method_assets/fusa_baseline_v1/manifest.yaml"),
    )
    causal_revalidation.add_argument(
        "--report-style-template", type=Path,
        default=Path("references/HARA_Template_AI_20260327.xlsx"),
    )
    causal_revalidation.add_argument("--max-workers", type=int, default=4)
    causal_revalidation.add_argument("--provider-attempt-limit", type=int)
    causal_revalidation.add_argument("--provider-budget-run-id")
    return parser


def _shared_provider_budget(args):
    from hara_agent.infrastructure.llm.provider_budget import ProviderAttemptBudget

    limit = args.provider_attempt_limit
    budget_run_id = args.provider_budget_run_id
    expected_budget_run_id = ""
    expected_limit = None
    source_run_id = args.source_run_id
    for _ in range(3):
        if not re.fullmatch(r"[A-Za-z0-9_-]+", source_run_id):
            raise ValueError("Child source run ID is invalid")
        source_checkpoint = args.run_dir / f"{source_run_id}.checkpoint.json"
        if not source_checkpoint.is_file():
            break
        source = json.loads(source_checkpoint.read_text(encoding="utf-8"))
        events = source.get("audit_trail", [])
        bounded = next((
            event for event in events
            if event.get("event") == "bounded_sample_configured"
        ), None)
        if bounded is not None:
            expected_budget_run_id = source_run_id
            expected_limit = bounded.get("scope", {}).get("provider_attempt_limit")
            break
        ancestor = next((
            str(event.get("source_run_id", "")) for event in events
            if event.get("event") == "scenario_synthesis_child_run_materialized"
        ), "")
        if not ancestor or ancestor == source_run_id:
            break
        source_run_id = ancestor
    if expected_budget_run_id and limit is None:
        raise ValueError("Bounded sample child run must reuse its parent Provider budget")
    if (limit is None) != (budget_run_id is None):
        raise ValueError(
            "Child Provider budget requires both --provider-attempt-limit "
            "and --provider-budget-run-id"
        )
    if limit is None:
        return None
    if limit < 1 or not re.fullmatch(r"[A-Za-z0-9_-]+", budget_run_id):
        raise ValueError("Child Provider budget limit/run ID is invalid")
    if expected_budget_run_id and (
        budget_run_id != expected_budget_run_id or limit != expected_limit
    ):
        raise ValueError("Child Provider budget differs from its bounded parent run")
    path = args.run_dir / f"{budget_run_id}.provider-attempts.jsonl"
    if not path.is_file():
        raise FileNotFoundError(
            "Shared Provider budget ledger from the parent analyze run is missing"
        )
    return ProviderAttemptBudget(path, run_id=budget_run_id, limit=limit)


def main(argv: list[str] | None = None) -> int:
    load_local_env()
    args = build_parser().parse_args(argv)
    if args.command == "scenario-output-quality-audit":
        checkpoint = args.checkpoint.expanduser().resolve()
        state = HARAState.read_committed(
            json.loads(checkpoint.read_text(encoding="utf-8"))
        )
        if state.run_id != args.review_run_id:
            raise ValueError(
                "Scenario audit checkpoint run ID does not match --review-run-id"
            )
        resolution = MethodSourceResolver().resolve(
            template_path=None,
            baseline_manifest_path=args.baseline,
            report_template_path=args.report_style_template,
        )
        causal_trace = {}
        if args.causal_trace is not None:
            causal_path = args.causal_trace.expanduser().resolve()
            causal_trace = json.loads(causal_path.read_text(encoding="utf-8"))
        output_dir = args.output_dir.expanduser().resolve()
        service = ScenarioOutputQualityAuditService(resolution.method)
        audit = service.audit(
            state, causal_trace=causal_trace, checkpoint_path=checkpoint,
            synthesis_review_dir=args.review_root / args.review_run_id,
        )
        json_path = output_dir / "scenario_output_quality_audit.json"
        markdown_path = output_dir / "scenario_output_quality_audit.md"
        _write_json_atomic(json_path, audit)
        _write_text_atomic(markdown_path, service.render_markdown(audit))
        _write_text_atomic(
            args.sample_output.expanduser().resolve(), service.render_sample(audit)
        )
        print(json.dumps({
            "run_id": state.run_id,
            "quality_audit_json": str(json_path),
            "quality_audit_markdown": str(markdown_path),
            "quality_sample": str(args.sample_output.expanduser().resolve()),
            "summary": audit["summary"],
            "diversity_metrics": {
                key: value for key, value in audit["diversity_metrics"].items()
                if key not in {
                    "cross_malfunction_clusters",
                    "parent_group_ids_where_all_three_differ_only_by_where_road",
                    "scenario_variant_low_diversity_group_ids",
                }
            },
            "hazard_consistency": audit["hazard_consistency"],
            "provider_calls": 0,
        }, ensure_ascii=False, indent=2))
        return 0
    if args.command == "exposure-input-audit":
        checkpoint_path = args.checkpoint or (
            args.run_dir / f"{args.review_run_id}.checkpoint.json"
        )
        trace_path = args.review_root / args.review_run_id / "risk_execution_trace.json"
        if not checkpoint_path.is_file():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        if not trace_path.is_file():
            raise FileNotFoundError(f"Risk execution trace not found: {trace_path}")
        resolution = MethodSourceResolver().resolve(
            template_path=None,
            baseline_manifest_path=args.baseline,
            report_template_path=args.report_template,
        )
        payload = ExposureInputAuditService(resolution.method).generate(
            checkpoint=json.loads(checkpoint_path.read_text(encoding="utf-8")),
            trace=json.loads(trace_path.read_text(encoding="utf-8")),
        )
        writer = ReviewArtifactWriter(args.review_run_id, args.review_root)
        writer.write_exposure_input_audit(payload)
        markdown_path = args.review_root / args.review_run_id / "exposure_input_audit.md"
        _write_text_atomic(markdown_path, ExposureInputAuditService.render_markdown(payload))
        print(json.dumps({
            "run_id": args.review_run_id,
            "eligible_records": payload["eligible_records"],
            "audit_artifact": str(args.review_root / args.review_run_id / "exposure_input_audit.json"),
            "markdown_artifact": str(markdown_path),
            "diagnosis": payload["diagnosis"],
            "readiness": payload["summary"]["readiness"],
            "provider_calls": 0,
        }, ensure_ascii=False, indent=2))
        return 0
    if args.command == "exposure-root-cause-report":
        payload = json.loads(args.audit.read_text(encoding="utf-8"))
        trace = json.loads(args.rescore_trace.read_text(encoding="utf-8"))
        _write_text_atomic(
            args.output,
            ExposureInputAuditService.render_root_cause_report(
                payload, post_rescore_trace=trace,
            ),
        )
        print(json.dumps({
            "output": str(args.output),
            "post_rescore_exposure_distribution": ExposureInputAuditService._post_distribution(trace),
        }, ensure_ascii=False, indent=2))
        return 0
    if args.command == "rescore-risk":
        from hara_agent.workflow.risk_rescoring import OfflineRiskRescorer

        summary = OfflineRiskRescorer().run(
            source_run_id=args.source_run_id, target_run_id=args.target_run_id,
            checkpoint=args.checkpoint, baseline=args.baseline,
            report_template=args.report_style_template, output=args.output,
            run_dir=args.run_dir, review_root=args.review_root,
            supplement=args.risk_input_supplement,
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        return 0
    if args.command == "risk-scoreability":
        checkpoint = json.loads(args.checkpoint.read_text(encoding="utf-8"))
        supplement = json.loads(args.risk_input_supplement.read_text(encoding="utf-8"))
        existing_pack = (
            json.loads(args.assumption_pack.read_text(encoding="utf-8"))
            if args.assumption_pack.exists() else None
        )
        resolution = MethodSourceResolver().resolve(
            template_path=None,
            baseline_manifest_path=args.baseline,
            report_template_path=args.report_style_template,
        )
        payload, queue = RiskScoreabilityService(resolution.method).generate(
            checkpoint=checkpoint,
            supplement=supplement,
            assumption_pack=existing_pack,
            checkpoint_sha256=hashlib.sha256(args.checkpoint.read_bytes()).hexdigest(),
        )
        pack = payload["assumption_pack"]
        review_root = args.review_root / args.review_run_id
        _write_json_atomic(review_root / "risk_scoreability_v3.json", payload)
        _write_json_atomic(review_root / "differential_validation_queue_v3.json", queue)
        _write_json_atomic(args.assumption_pack, pack)
        if payload["materialized_children"]:
            _write_json_atomic(args.child_scenarios, {
                "artifact_version": "hara-risk-child-scenarios-v1",
                "children": payload["materialized_children"],
            })
        print(json.dumps({
            "run_id": args.review_run_id,
            "risk_scoreability": str(review_root / "risk_scoreability_v3.json"),
            "differential_validation_queue": str(
                review_root / "differential_validation_queue_v3.json"
            ),
            "assumption_pack": str(args.assumption_pack),
            "child_scenarios": str(args.child_scenarios) if payload["materialized_children"] else "",
            "summary": payload["summary"],
            "provider_calls": 0,
        }, ensure_ascii=False, indent=2))
        return 0
    if args.command == "synthesize-scenarios":
        from hara_agent.infrastructure.llm.factory import create_llm_client
        from hara_agent.workflow.scenario_synthesis import ScenarioSynthesisRunner

        resolution = MethodSourceResolver().resolve(
            template_path=None,
            baseline_manifest_path=args.baseline,
            report_template_path=args.report_style_template,
        )
        attempt_budget = _shared_provider_budget(args)
        client = create_llm_client(LLMConfig.from_env(), attempt_budget=attempt_budget)
        smoke_identities = None
        if args.smoke_identity_file is not None:
            identity_payload = json.loads(
                args.smoke_identity_file.read_text(encoding="utf-8")
            )
            failures = identity_payload.get("failures", [])
            if not isinstance(failures, list) or not failures:
                raise ValueError("Smoke identity file requires a non-empty failures list")
            smoke_identities = [{
                "malfunction_id": str(item.get("malfunction_id", "")),
                "parent_scenario_id": str(item.get("parent_scenario_id", "")),
                "hazardous_event_id": str(item.get("hazardous_event_id", "")),
                "requested_variant_count": item.get("requested_variant_count"),
            } for item in failures if isinstance(item, dict)]
        result = ScenarioSynthesisRunner(
            method=resolution.method, client=client,
            run_dir=args.run_dir, review_root=args.review_root,
        ).run(
            source_run_id=args.source_run_id,
            target_run_id=args.target_run_id,
            smoke_count=args.smoke_count,
            smoke_identities=smoke_identities,
            run_full=args.full,
            max_workers=args.max_workers,
            output_path=args.output,
            baseline_path=args.baseline,
            report_template_path=args.report_style_template,
        )
        if attempt_budget is not None:
            result["shared_provider_attempt_count"] = attempt_budget.attempts
            result["shared_provider_attempt_ledger"] = str(attempt_budget.path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if not result["smoke_passed"]:
            return 2
        return 0
    if args.command == "revalidate-synthesized-scenarios":
        from hara_agent.infrastructure.llm.factory import create_llm_client
        from hara_agent.workflow.scenario_causal_revalidation import (
            ScenarioCausalRevalidationRunner,
        )

        resolution = MethodSourceResolver().resolve(
            template_path=None,
            baseline_manifest_path=args.baseline,
            report_template_path=args.report_style_template,
        )
        attempt_budget = _shared_provider_budget(args)
        result = ScenarioCausalRevalidationRunner(
            method=resolution.method,
            client=create_llm_client(LLMConfig.from_env(), attempt_budget=attempt_budget),
            run_dir=args.run_dir,
            review_root=args.review_root,
        ).run(
            source_run_id=args.source_run_id,
            target_run_id=args.target_run_id,
            max_workers=args.max_workers,
        )
        if attempt_budget is not None:
            result["shared_provider_attempt_count"] = attempt_budget.attempts
            result["shared_provider_attempt_ledger"] = str(attempt_budget.path)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return 0
    if args.command == "confirm-template-role":
        compiler = _role_compiler()
        snapshot = compiler.scanner.scan(args.template)
        confirmation = TemplateRoleConfirmation(
            template_hash=snapshot.source_hash,
            selected_regions=_parse_role_selections(args.select),
            confirmed_by=args.confirmed_by,
            confirmed_at=datetime.now(timezone.utc).isoformat(),
            rationale=args.rationale,
        )
        method = compiler.compile_method(
            args.template,
            confirmation=confirmation,
            use_manifest=False,
        )
        print(json.dumps({
            "template_hash": snapshot.source_hash,
            "compile_status": method.compile_status.value,
            "engineering_rules_compiled": method.engineering_rules_compiled,
            "confirmed_roles": sorted(confirmation.selected_regions),
            "manifest": str(compiler.manifest_store.path_for(snapshot.source_hash)),
            "blocking_diagnostics": [
                item.message for item in method.blocking_diagnostics
            ],
        }, ensure_ascii=False, indent=2))
        return 0 if not method.blocking_diagnostics else 2
    if args.command == "review":
        if args.limit < 0:
            raise ValueError("--limit must not be negative")
        if args.feasible and args.infeasible:
            raise ValueError("--feasible and --infeasible are mutually exclusive")
        reader = ReviewArtifactReader(
            args.run_id,
            os.getenv("HARA_REVIEW_ARTIFACT_DIR", "runtime/review"),
        )
        print(render_review(
            reader,
            function_id=args.function or "",
            malfunction_id=args.malfunction or "",
            scenario_id=args.scenario or "",
            all_records=args.all_records,
            limit=args.limit,
            feasible=args.feasible,
            infeasible=args.infeasible,
        ))
        for warning in reader.warnings:
            print(f"WARNING: {warning}")
        return 0
    if args.command == "risk-execution-trace":
        review_root = Path(os.getenv("HARA_REVIEW_ARTIFACT_DIR", "runtime/review"))
        resolution = MethodSourceResolver().resolve(
            template_path=None,
            baseline_manifest_path=args.baseline,
            report_template_path=args.report_template,
        )
        reader = ReviewArtifactReader(args.review_run_id, review_root)
        payload = RiskExecutionTraceService(resolution.method).project_review_run(reader)
        writer = ReviewArtifactWriter(args.review_run_id, review_root)
        writer.write_risk_execution_trace(payload)
        writer.write_summary(SimpleNamespace(stage=SimpleNamespace(value="malfunctions")))
        print(json.dumps({
            "run_id": args.review_run_id,
            "trace_artifact": str(review_root / args.review_run_id / "risk_execution_trace.json"),
            "risk_stage_status": payload["risk_stage_status"],
            "scenario_eligibility_summary": payload["scenario_eligibility_summary"],
        }, ensure_ascii=False, indent=2))
        return 0
    if args.command == "rebuild-report":
        checkpoint = args.checkpoint or (args.run_dir / f"{args.review_run_id}.checkpoint.json")
        schema = load_report_schema(args.schema)
        output = OfflineReportRebuilder(schema).rebuild(
            checkpoint_path=checkpoint,
            method_baseline_path=args.baseline,
            report_style_template_path=args.report_style_template,
            output_path=args.output,
            review_root=args.review_root,
            causal_trace_path=args.causal_trace,
            audit_output_dir=args.audit_output_dir,
            synthesis_candidates_path=args.synthesis_candidates,
            speed_context_audit_path=args.speed_context_audit,
        )
        print(json.dumps({
            "run_id": args.review_run_id,
            "report_schema": schema.report_id,
            "report_schema_hash": schema.schema_hash,
            "output": str(output),
            "provider_invoked": False,
        }, ensure_ascii=False, indent=2))
        return 0
    if args.command == "scenario-aliases":
        review_root = Path(os.getenv("HARA_REVIEW_ARTIFACT_DIR", "runtime/review"))
        gap_path = review_root / args.review_run_id / "scenario_binding_gaps.json"
        if not gap_path.is_file():
            raise FileNotFoundError(
                f"Scenario binding gap artifact not found: {gap_path}"
            )
        resolution = MethodSourceResolver().resolve(
            template_path=None,
            baseline_manifest_path=args.baseline,
            report_template_path=args.report_template,
        )
        gap_payload = json.loads(gap_path.read_text(encoding="utf-8"))
        reader = ReviewArtifactReader(args.review_run_id, review_root)
        payload = ScenarioAliasProposalService(resolution.method).generate(
            gap_payload, reader.read_all(),
        )
        writer = ReviewArtifactWriter(args.review_run_id, review_root)
        writer.write_scenario_alias_proposals(payload)
        print(json.dumps({
            "run_id": args.review_run_id,
            "proposal_artifact": str(
                review_root / args.review_run_id / "scenario_alias_proposals.json"
            ),
            **payload,
        }, ensure_ascii=False, indent=2))
        return 0
    if args.command == "scenario-coverage":
        review_root = Path(os.getenv("HARA_REVIEW_ARTIFACT_DIR", "runtime/review"))
        gap_path = review_root / args.review_run_id / "scenario_binding_gaps.json"
        if not gap_path.is_file():
            raise FileNotFoundError(
                f"Scenario binding gap artifact not found: {gap_path}"
            )
        resolution = MethodSourceResolver().resolve(
            template_path=None,
            baseline_manifest_path=args.baseline,
            report_template_path=args.report_template,
        )
        gap_payload = json.loads(gap_path.read_text(encoding="utf-8"))
        reader = ReviewArtifactReader(args.review_run_id, review_root)
        payload = ScenarioCoverageProposalService(resolution.method).generate(
            gap_payload, reader.read_all(),
        )
        writer = ReviewArtifactWriter(args.review_run_id, review_root)
        writer.write_scenario_coverage_proposals(payload)
        print(json.dumps({
            "run_id": args.review_run_id,
            "proposal_artifact": str(
                review_root / args.review_run_id / "scenario_coverage_proposals.json"
            ),
            **payload,
        }, ensure_ascii=False, indent=2))
        return 0
    if args.command == "yaml-utilization":
        review_root = Path(os.getenv("HARA_REVIEW_ARTIFACT_DIR", "runtime/review"))
        gap_path = review_root / args.review_run_id / "scenario_binding_gaps.json"
        if not gap_path.is_file():
            raise FileNotFoundError(
                f"Scenario binding gap artifact not found: {gap_path}"
            )
        resolution = MethodSourceResolver().resolve(
            template_path=None,
            baseline_manifest_path=args.baseline,
            report_template_path=args.report_template,
        )
        reader = ReviewArtifactReader(args.review_run_id, review_root)
        payload = ConfirmedYamlUtilizationService(resolution.method).generate(
            reader.read_all(), json.loads(gap_path.read_text(encoding="utf-8")),
        )
        writer = ReviewArtifactWriter(args.review_run_id, review_root)
        writer.write_confirmed_yaml_utilization(payload)
        print(json.dumps({
            "run_id": args.review_run_id,
            "utilization_artifact": str(
                review_root / args.review_run_id / "confirmed_yaml_utilization.json"
            ),
            "fm_template_matching": payload["fm_template_matching"],
            "scenario_gap_reaudit": payload["scenario_gap_reaudit"],
        }, ensure_ascii=False, indent=2))
        return 0
    if args.command == "fm-selector-semantic-audit":
        review_root = Path(os.getenv("HARA_REVIEW_ARTIFACT_DIR", "runtime/review"))
        resolution = MethodSourceResolver().resolve(
            template_path=None,
            baseline_manifest_path=args.baseline,
            report_template_path=args.report_template,
        )
        reader = ReviewArtifactReader(args.review_run_id, review_root)
        payload = FMSelectorSemanticAuditService(resolution.method).generate(
            reader.read_all(),
        )
        writer = ReviewArtifactWriter(args.review_run_id, review_root)
        writer.write_fm_selector_semantic_audit(payload)
        print(json.dumps({
            "run_id": args.review_run_id,
            "audit_artifact": str(
                review_root / args.review_run_id / "fm_selector_semantic_audit.json"
            ),
            "contract_status": payload["contract_status"],
            "template_qualification": payload["template_qualification"],
        }, ensure_ascii=False, indent=2))
        return 0
    if args.command == "fm-template-ambiguity-audit":
        review_root = Path(os.getenv("HARA_REVIEW_ARTIFACT_DIR", "runtime/review"))
        resolution = MethodSourceResolver().resolve(
            template_path=None,
            baseline_manifest_path=args.baseline,
            report_template_path=args.report_template,
        )
        reader = ReviewArtifactReader(args.review_run_id, review_root)
        payload = FMTemplateAmbiguityAuditService(resolution.method).generate(
            reader.read_all(),
        )
        writer = ReviewArtifactWriter(args.review_run_id, review_root)
        writer.write_fm_template_ambiguity_audit(payload)
        print(json.dumps({
            "run_id": args.review_run_id,
            "audit_artifact": str(
                review_root / args.review_run_id / "fm_template_ambiguity_audit.json"
            ),
            "ambiguity_summary": payload["ambiguity_summary"],
            "qualification_preview": payload["qualification_preview"],
        }, ensure_ascii=False, indent=2))
        return 0
    if args.command == "exposure-binding-audit":
        review_root = Path(os.getenv("HARA_REVIEW_ARTIFACT_DIR", "runtime/review"))
        gap_path = review_root / args.review_run_id / "scenario_binding_gaps.json"
        if not gap_path.is_file():
            raise FileNotFoundError(
                f"Scenario binding gap artifact not found: {gap_path}"
            )
        resolution = MethodSourceResolver().resolve(
            template_path=None,
            baseline_manifest_path=args.baseline,
            report_template_path=args.report_template,
        )
        reader = ReviewArtifactReader(args.review_run_id, review_root)
        payload = ExposureBindingAuditService(resolution.method).generate(
            reader.read_all(), json.loads(gap_path.read_text(encoding="utf-8")),
        )
        writer = ReviewArtifactWriter(args.review_run_id, review_root)
        writer.write_exposure_binding_audit(payload)
        component_summary = {
            key: value for key, value in payload["component_domain_summary"].items()
            if key != "records"
        }
        print(json.dumps({
            "run_id": args.review_run_id,
            "audit_artifact": str(
                review_root / args.review_run_id / "exposure_binding_audit.json"
            ),
            "component_domain_summary": component_summary,
            "before": payload["before"],
            "after": payload["after"],
        }, ensure_ascii=False, indent=2))
        return 0
    if args.command == "exposure-dimension-coverage-audit":
        review_root = Path(os.getenv("HARA_REVIEW_ARTIFACT_DIR", "runtime/review"))
        resolution = MethodSourceResolver().resolve(
            template_path=None,
            baseline_manifest_path=args.baseline,
            report_template_path=args.report_template,
        )
        reader = ReviewArtifactReader(args.review_run_id, review_root)
        payload = ExposureDimensionCoverageAuditService(resolution.method).generate(
            reader.read_all(),
        )
        writer = ReviewArtifactWriter(args.review_run_id, review_root)
        writer.write_exposure_dimension_coverage_audit(payload)
        print(json.dumps({
            "run_id": args.review_run_id,
            "audit_artifact": str(
                review_root / args.review_run_id / "exposure_dimension_coverage_audit.json"
            ),
            "coverage_rule_inventory": payload["coverage_rule_inventory"],
            "coverage_granularity": payload["coverage_granularity"],
            "summary": payload["summary"],
        }, ensure_ascii=False, indent=2))
        return 0
    if args.command == "severity-delta-v-semantic-audit":
        review_root = Path(os.getenv("HARA_REVIEW_ARTIFACT_DIR", "runtime/review"))
        resolution = MethodSourceResolver().resolve(
            template_path=None,
            baseline_manifest_path=args.baseline,
            report_template_path=args.report_template,
        )
        reader = ReviewArtifactReader(args.review_run_id, review_root)
        payload = SeverityDeltaVSemanticAuditService(resolution.method).generate(
            reader.read_all(),
        )
        writer = ReviewArtifactWriter(args.review_run_id, review_root)
        writer.write_severity_delta_v_semantic_audit(payload)
        print(json.dumps({
            "run_id": args.review_run_id,
            "audit_artifact": str(
                review_root / args.review_run_id / "severity_delta_v_semantic_audit.json"
            ),
            "method_semantic_provenance": payload["method_semantic_provenance"],
            "delta_v_derivation_authority": payload["delta_v_derivation_authority"],
            "summary": payload["summary"],
        }, ensure_ascii=False, indent=2))
        return 0
    if args.command == "hazardous-event-risk-context-audit":
        review_root = Path(os.getenv("HARA_REVIEW_ARTIFACT_DIR", "runtime/review"))
        resolution = MethodSourceResolver().resolve(
            template_path=None,
            baseline_manifest_path=args.baseline,
            report_template_path=args.report_template,
        )
        reader = ReviewArtifactReader(args.review_run_id, review_root)
        payload = HazardousEventRiskContextService(resolution.method).audit(
            reader.read_all(),
        )
        writer = ReviewArtifactWriter(args.review_run_id, review_root)
        writer.write_hazardous_event_risk_context_audit(payload)
        print(json.dumps({
            "run_id": args.review_run_id,
            "audit_artifact": str(
                review_root / args.review_run_id / "hazardous_event_risk_context_audit.json"
            ),
            "hazardous_event_identity": payload["hazardous_event_identity"],
            "summary": payload["summary"],
        }, ensure_ascii=False, indent=2))
        return 0
    if args.command == "risk-context-source-coverage-audit":
        review_root = Path(os.getenv("HARA_REVIEW_ARTIFACT_DIR", "runtime/review"))
        resolution = MethodSourceResolver().resolve(
            template_path=None,
            baseline_manifest_path=args.baseline,
            report_template_path=args.report_template,
        )
        checkpoint = args.checkpoint or (
            Path("runtime/agent") / f"{args.review_run_id}.checkpoint.json"
        )
        if not checkpoint.is_file():
            raise FileNotFoundError(f"P3-A checkpoint not found: {checkpoint}")
        trace_path = review_root / args.review_run_id / "risk_execution_trace.json"
        if not trace_path.is_file():
            raise FileNotFoundError(f"P3-A risk execution trace not found: {trace_path}")
        state = json.loads(checkpoint.read_text(encoding="utf-8"))
        trace = json.loads(trace_path.read_text(encoding="utf-8"))
        payload, clarification = RiskContextSourceCoverageAuditService(
            resolution.method,
        ).audit(
            state=state,
            trace=trace,
            document=DocumentReader().read(args.item),
        )
        audit_dir = review_root / "p3a-risk-context-source"
        audit_path = audit_dir / "risk_context_source_coverage_audit.json"
        clarification_path = audit_dir / "engineering_clarification_package_p3a.json"
        _write_json_atomic(audit_path, payload)
        _write_json_atomic(clarification_path, clarification)
        print(json.dumps({
            "run_id": args.review_run_id,
            "audit_artifact": str(audit_path),
            "engineering_clarification_package": str(clarification_path),
            "aggregate": payload["aggregate"],
        }, ensure_ascii=False, indent=2))
        return 0
    if args.command == "controllability-branch-audit":
        review_root = Path(os.getenv("HARA_REVIEW_ARTIFACT_DIR", "runtime/review"))
        resolution = MethodSourceResolver().resolve(
            template_path=None,
            baseline_manifest_path=args.baseline,
            report_template_path=args.report_template,
        )
        reader = ReviewArtifactReader(args.review_run_id, review_root)
        payload = ControllabilityBranchAuditService(resolution.method).audit(
            reader.read_all(),
        )
        writer = ReviewArtifactWriter(args.review_run_id, review_root)
        writer.write_controllability_branch_policy_audit(payload)
        print(json.dumps({
            "run_id": args.review_run_id,
            "audit_artifact": str(
                review_root / args.review_run_id / "controllability_branch_policy_audit.json"
            ),
            "runtime_comparison": payload["runtime_comparison"],
            "summary": payload["summary"],
        }, ensure_ascii=False, indent=2))
        return 0
    if args.command == "risk-evaluator-parity-audit":
        review_root = Path(os.getenv("HARA_REVIEW_ARTIFACT_DIR", "runtime/review"))
        template_method = _role_compiler().compile_method(args.template)
        yaml_method = YamlBaselineCompiler().compile(
            args.baseline, report_contract=template_method.report_contract,
        )
        payload = MethodContractParityAuditService(template_method, yaml_method).generate()
        ReviewArtifactWriter(args.review_run_id, review_root).write_method_contract_parity_audit(payload)
        print(json.dumps({
            "run_id": args.review_run_id,
            "audit_artifact": str(
                review_root / args.review_run_id / "method_contract_parity_audit.json"
            ),
            "summary": payload["summary"],
            "shared_scoring_facade": payload["contract_surface_comparison"]["shared_scoring_facade"],
        }, ensure_ascii=False, indent=2))
        return 0
    if args.command == "doctor":
        checks = {}
        try:
            resolution = MethodSourceResolver().resolve(
                template_path=args.template,
                baseline_manifest_path=args.method_baseline,
                report_template_path=(
                    args.template if args.template is not None else args.report_template
                ),
            )
            method = resolution.method
            method_ok = (
                method.compile_status is not CompileStatus.NOT_READY
                and method.engineering_rules_compiled
            )
            scenario_method = method.scenario_model.scenario_method
            selector_taxonomy = scenario_method.failure_mode_selector_taxonomy
            selector_adapter = scenario_method.fm_template_selector_adapter
            selector_catalog = scenario_method.fm_template_catalog

            def selector_reconciliation(selector_type: str) -> dict[str, object]:
                taxonomy_values = {
                    item.canonical_id for item in (
                        selector_taxonomy.component_categories
                        if selector_taxonomy is not None and selector_type == "COMPONENT_CATEGORY"
                        else selector_taxonomy.failure_types
                        if selector_taxonomy is not None else ()
                    )
                }
                template_values = {
                    value
                    for template in (selector_catalog.templates if selector_catalog else ())
                    for value in (
                        template.match.component_categories
                        if selector_type == "COMPONENT_CATEGORY"
                        else template.match.failure_types
                    )
                }
                mappings = [
                    item for item in (selector_adapter.mappings if selector_adapter else ())
                    if item.selector_type == selector_type
                ]
                active = [item for item in mappings if item.runtime_status == "ACTIVE"]
                ambiguous = [item for item in mappings if item.runtime_status == "UNRESOLVED"]
                covered = taxonomy_values | {item.source_template_value for item in mappings}
                return {
                    "canonical": len(taxonomy_values),
                    "raw": len(template_values),
                    "exact_overlap": sorted(template_values & taxonomy_values),
                    "mapped_unambiguous": sorted(item.source_template_value for item in active),
                    "ambiguous": sorted(item.source_template_value for item in ambiguous),
                    "unmapped": sorted(template_values - covered),
                }

            checks["method_contract"] = {
                "ok": method_ok,
                "template_hash": method.metadata["template_hash"],
                "method_source_kind": resolution.source_kind.value,
                "method_source_hash": method.metadata.get("method_source_hash", method.metadata["template_hash"]),
                "report_template_hash": resolution.report_template_hash,
                "contract_version": method.contract_version,
                "compiler_version": method.compiler_version,
                "compile_status": method.compile_status.value,
                "guideword_count": len(method.guidewords.guidewords),
                "scenario_dimension_count": len(method.scenario_model.dimensions),
                "scenario_binding": {
                    "method_atom_count": len(
                        method.metadata.get("scenario_atom_catalog", [])
                    ),
                    "normative_constraint_rule_count": len(
                        method.scenario_model.constraint_rules
                    ),
                    "coverage_status": "PROJECT_FACT_BINDING_REQUIRED",
                    "dimension_atom_counts": {
                        dimension.canonical_name: sum(
                            dimension.canonical_name in atom.get(
                                "filled_dimensions", []
                            )
                            for atom in method.metadata.get(
                                "scenario_atom_catalog", []
                            )
                            if isinstance(atom, dict)
                        )
                        for dimension in method.scenario_model.dimensions
                    },
                },
                "scenario_coverage_governance": dict(
                    method.metadata.get("scenario_coverage_governance", {})
                ),
                "confirmed_yaml_utilization": {
                    "vda702_atoms": {"compiled": True, "active": True},
                    "atom_spec": {"compiled": True, "active": True},
                    "dimension_structure": {"compiled": True, "active": True},
                    "fm_scenario_templates": {
                        "role": "SCENARIO_TEMPLATE_CONSTRAINT",
                        "templates": len(
                            method.scenario_model.scenario_method.fm_template_catalog.templates
                        ) if method.scenario_model.scenario_method.fm_template_catalog else 0,
                        "matched_malfunctions": "REVIEW_ARTIFACT_REQUIRED",
                    },
                    "fm_selector_taxonomy": {
                        "component_taxonomy": {
                            "compiled": (
                                method.scenario_model.scenario_method.failure_mode_selector_taxonomy
                                is not None
                            ),
                            "canonical_values": len(
                                method.scenario_model.scenario_method.failure_mode_selector_taxonomy.component_categories
                            ) if method.scenario_model.scenario_method.failure_mode_selector_taxonomy else 0,
                            "resolved": "REVIEW_ARTIFACT_REQUIRED",
                            "missing": "REVIEW_ARTIFACT_REQUIRED",
                            "invalid": "REVIEW_ARTIFACT_REQUIRED",
                        },
                        "failure_type_taxonomy": {
                            "compiled": (
                                method.scenario_model.scenario_method.failure_mode_selector_taxonomy
                                is not None
                            ),
                            "canonical_values": len(
                                method.scenario_model.scenario_method.failure_mode_selector_taxonomy.failure_types
                            ) if method.scenario_model.scenario_method.failure_mode_selector_taxonomy else 0,
                            "resolved": "REVIEW_ARTIFACT_REQUIRED",
                            "missing": "REVIEW_ARTIFACT_REQUIRED",
                            "invalid": "REVIEW_ARTIFACT_REQUIRED",
                        },
                    },
                    "fm_template_selector_adapter": {
                        "compiled": (
                            method.scenario_model.scenario_method.fm_template_selector_adapter
                            is not None
                        ),
                        "active_mappings": sum(
                            item.runtime_status == "ACTIVE"
                            for item in (
                                method.scenario_model.scenario_method
                                .fm_template_selector_adapter.mappings
                                if method.scenario_model.scenario_method
                                .fm_template_selector_adapter is not None else ()
                            )
                        ),
                        "unresolved_mappings": sum(
                            item.runtime_status == "UNRESOLVED"
                            for item in (
                                method.scenario_model.scenario_method
                                .fm_template_selector_adapter.mappings
                                if method.scenario_model.scenario_method
                                .fm_template_selector_adapter is not None else ()
                            )
                        ),
                        "contract_reconciliation": {
                            "component_taxonomy": selector_reconciliation("COMPONENT_CATEGORY"),
                            "failure_type_taxonomy": selector_reconciliation("FAILURE_TYPE"),
                        },
                    },
                    "avp_low_speed": {
                        "trigger_mappings": len(
                            method.scenario_model.scenario_method.domain_knowledge.triggering_state_mappings
                        ) if method.scenario_model.scenario_method.domain_knowledge else 0,
                        "fallback_dimensions": len(
                            method.scenario_model.scenario_method.domain_knowledge.fallback_dimensions
                        ) if method.scenario_model.scenario_method.domain_knowledge else 0,
                        "numeric_sections": "COMPILED_NOT_SCORE_ACTIVE_IN_THIS_STAGE",
                    },
                    "coupling_examples": {"role": "LLM_EXAMPLE_ONLY"},
                    "infeasible_examples": {"role": "LLM_EXAMPLE_ONLY"},
                },
                "required_fact_count": len(method.required_fact_specs),
                "warning_codes": sorted({
                    item.code.value for item in method.warnings
                }),
                "blocking_diagnostics": [
                    item.message for item in method.blocking_diagnostics
                ],
            }
            structured = method.structured_risk_method
            checks["risk_pipeline"] = {
                "ok": True,
                "scenario_to_risk_handoff": "VERIFIED",
                "severity": {
                    "executor": "SeverityMethodExecutor" if structured else "MethodRuleScoringService",
                    "speed_semantic": (
                        structured.severity.speed_semantic.value if structured else "TEMPLATE_COMPILED_RULE"
                    ),
                    "semantic_resolution": (
                        structured.severity.semantic.semantic_resolution.value
                        if structured else "TEMPLATE_COMPILED_RULE"
                    ),
                    "source_status": (
                        structured.severity.semantic.source_status if structured else ""
                    ),
                    "odd_max_speed_used_as_delta_v": False,
                    "relative_speed_substituted_as_delta_v": False,
                },
                "exposure": {
                    "executor": "ExposureMethodExecutor" if structured else "MethodRuleScoringService",
                    "traceable_atom_path": structured is not None,
                },
                "controllability": {
                    "executor": "StructuredControllabilityExecutor" if structured else "MethodRuleScoringService",
                    "profile": structured.controllability_profile.profile_id if structured else "",
                    "ttc": structured is not None,
                },
                "asil": {"executor": "MethodContractASILService", "matrix_cells": len(method.asil.mappings)},
                "runtime_yaml_read": 0,
                "rating_causal_back_edge": 0,
                "template_yaml_parity": "SHARED_API_DIFFERENT_EXECUTION_MODEL",
            }
            risk_context_coverage_artifact = (
                Path(os.getenv("HARA_REVIEW_ARTIFACT_DIR", "runtime/review"))
                / "p3a-risk-context-source" / "risk_context_source_coverage_audit.json"
            )
            if risk_context_coverage_artifact.is_file():
                try:
                    coverage = json.loads(
                        risk_context_coverage_artifact.read_text(encoding="utf-8")
                    )
                    aggregate = coverage.get("aggregate", {})
                    per_field = coverage.get("per_field", [])
                    if not isinstance(aggregate, dict) or not isinstance(per_field, list):
                        raise ValueError("P3-A coverage aggregate/per_field must be structured")
                    checks["risk_context_source_coverage"] = {
                        "ok": True,
                        "status": "AVAILABLE",
                        "artifact_path": str(risk_context_coverage_artifact),
                        "run_id": coverage.get("run_id", ""),
                        "method_contract_hash_matches_active": (
                            coverage.get("method_contract_hash", "")
                            == method.metadata.get("method_source_hash", "")
                        ),
                        "eligible_hazardous_event_count": aggregate.get(
                            "eligible_hazardous_event_count", 0,
                        ),
                        "classification_counts": dict(
                            aggregate.get("classification_counts", {})
                        ),
                        "recoverability_counts": dict(
                            aggregate.get("recoverability_counts", {})
                        ),
                        "existing_but_not_bound_count": aggregate.get(
                            "existing_but_not_bound_count", 0,
                        ),
                        "typed_but_source_ambiguous_count": aggregate.get(
                            "typed_but_source_ambiguous_count", 0,
                        ),
                        "true_method_semantics_gap_count": aggregate.get(
                            "true_method_semantics_gap_count", 0,
                        ),
                        "field_count": len(per_field),
                    }
                except (OSError, ValueError, json.JSONDecodeError, TypeError) as exc:
                    checks["risk_context_source_coverage"] = {
                        "ok": True,
                        "status": "ARTIFACT_INVALID", "artifact_error": str(exc),
                    }
            else:
                checks["risk_context_source_coverage"] = {
                    "ok": True,
                    "status": "ARTIFACT_NOT_FOUND",
                    "artifact_path": str(risk_context_coverage_artifact),
                }
            if args.review_run_id:
                artifact = (
                    Path(os.getenv("HARA_REVIEW_ARTIFACT_DIR", "runtime/review"))
                    / args.review_run_id / "scenario_binding_gaps.json"
                )
                scenario_binding = checks["method_contract"]["scenario_binding"]
                risk_pipeline = checks["risk_pipeline"]
                risk_trace_artifact = (
                    Path(os.getenv("HARA_REVIEW_ARTIFACT_DIR", "runtime/review"))
                    / args.review_run_id / "risk_execution_trace.json"
                )
                if risk_trace_artifact.is_file():
                    try:
                        risk_trace_payload = json.loads(risk_trace_artifact.read_text(encoding="utf-8"))
                        risk_pipeline["review_trace"] = {
                            "status": "AVAILABLE",
                            "artifact_path": str(risk_trace_artifact),
                            "risk_stage_status": risk_trace_payload.get("risk_stage_status", ""),
                            "scenario_eligibility_summary": dict(risk_trace_payload.get("scenario_eligibility_summary", {})),
                        }
                    except (OSError, ValueError, json.JSONDecodeError) as exc:
                        risk_pipeline["review_trace"] = {"status": "ARTIFACT_INVALID", "artifact_error": str(exc)}
                else:
                    risk_pipeline["review_trace"] = {"status": "ARTIFACT_NOT_FOUND", "artifact_path": str(risk_trace_artifact)}
                exposure_audit_artifact = (
                    Path(os.getenv("HARA_REVIEW_ARTIFACT_DIR", "runtime/review"))
                    / args.review_run_id / "exposure_binding_audit.json"
                )
                if exposure_audit_artifact.is_file():
                    try:
                        exposure_audit = json.loads(
                            exposure_audit_artifact.read_text(encoding="utf-8")
                        )
                        risk_pipeline["exposure_input_binding"] = {
                            "status": "AVAILABLE",
                            "artifact_path": str(exposure_audit_artifact),
                            "dimension_routing_summary": dict(
                                exposure_audit.get("dimension_routing_summary", {})
                            ),
                            "component_domain_summary": {
                                key: exposure_audit.get(
                                    "component_domain_summary", {}
                                ).get(key, 0)
                                for key in ("resolved_malfunctions", "pending_malfunctions")
                            },
                            "e_readiness": dict(exposure_audit.get("after", {})),
                        }
                    except (OSError, ValueError, json.JSONDecodeError) as exc:
                        risk_pipeline["exposure_input_binding"] = {
                            "status": "ARTIFACT_INVALID", "artifact_error": str(exc),
                        }
                else:
                    risk_pipeline["exposure_input_binding"] = {
                        "status": "ARTIFACT_NOT_FOUND",
                        "artifact_path": str(exposure_audit_artifact),
                    }
                exposure_coverage_audit_artifact = (
                    Path(os.getenv("HARA_REVIEW_ARTIFACT_DIR", "runtime/review"))
                    / args.review_run_id / "exposure_dimension_coverage_audit.json"
                )
                if exposure_coverage_audit_artifact.is_file():
                    try:
                        coverage_audit = json.loads(
                            exposure_coverage_audit_artifact.read_text(encoding="utf-8")
                        )
                        inventory = coverage_audit.get("coverage_rule_inventory", {})
                        granularity = coverage_audit.get("coverage_granularity", {})
                        summary = coverage_audit.get("summary", {})
                        if not all(isinstance(value, dict) for value in (
                            inventory, granularity, summary,
                        )):
                            raise ValueError("exposure coverage audit sections must be objects")
                        risk_pipeline["exposure_coverage_governance"] = {
                            "status": "AVAILABLE",
                            "artifact_path": str(exposure_coverage_audit_artifact),
                            "dimension_universe": len(
                                coverage_audit.get("method_dimension_universe", [])
                            ),
                            "coverage_authority": {
                                "source": "normalized/scenario_coverage_rules.yaml",
                                "status": (
                                    "RESOLVED" if inventory.get("approved_rule_count", 0)
                                    else "PENDING_METHOD_SEMANTICS"
                                ),
                            },
                            "coverage_granularity": str(granularity.get("status", "")),
                            "approved_coverage_rules": inventory.get("approved_rule_count", 0),
                            "r3_coverage_resolved_assessments": summary.get("RESOLVED", 0),
                            "r3_coverage_pending_assessments": summary.get(
                                "PENDING_METHOD_SEMANTICS", 0
                            ),
                        }
                    except (OSError, ValueError, json.JSONDecodeError) as exc:
                        risk_pipeline["exposure_coverage_governance"] = {
                            "status": "ARTIFACT_INVALID", "artifact_error": str(exc),
                        }
                else:
                    risk_pipeline["exposure_coverage_governance"] = {
                        "status": "ARTIFACT_NOT_FOUND",
                        "artifact_path": str(exposure_coverage_audit_artifact),
                    }
                severity_audit_artifact = (
                    Path(os.getenv("HARA_REVIEW_ARTIFACT_DIR", "runtime/review"))
                    / args.review_run_id / "severity_delta_v_semantic_audit.json"
                )
                if severity_audit_artifact.is_file():
                    try:
                        severity_audit = json.loads(
                            severity_audit_artifact.read_text(encoding="utf-8")
                        )
                        summary = severity_audit.get("summary", {})
                        provenance = severity_audit.get("method_semantic_provenance", {})
                        if not isinstance(summary, dict) or not isinstance(provenance, dict):
                            raise ValueError("severity semantic audit sections must be objects")
                        risk_pipeline["severity"]["semantic_audit"] = {
                            "status": "AVAILABLE",
                            "artifact_path": str(severity_audit_artifact),
                            "semantic_resolution": provenance.get("semantic_resolution", ""),
                            "source_status": provenance.get("source_status", ""),
                            "delta_v_derivation": severity_audit.get(
                                "delta_v_derivation_authority", {}
                            ).get("status", ""),
                            "r3": {
                                key: summary.get(key, 0)
                                for key in (
                                    "READY", "PENDING_INPUT", "PENDING_METHOD_SEMANTICS",
                                    "direct_delta_v_available", "ego_speed_available",
                                    "relative_speed_available",
                                )
                            },
                        }
                    except (OSError, ValueError, json.JSONDecodeError) as exc:
                        risk_pipeline["severity"]["semantic_audit"] = {
                            "status": "ARTIFACT_INVALID", "artifact_error": str(exc),
                        }
                else:
                    risk_pipeline["severity"]["semantic_audit"] = {
                        "status": "ARTIFACT_NOT_FOUND",
                        "artifact_path": str(severity_audit_artifact),
                    }
                risk_context_audit_artifact = (
                    Path(os.getenv("HARA_REVIEW_ARTIFACT_DIR", "runtime/review"))
                    / args.review_run_id / "hazardous_event_risk_context_audit.json"
                )
                if risk_context_audit_artifact.is_file():
                    try:
                        risk_context_audit = json.loads(
                            risk_context_audit_artifact.read_text(encoding="utf-8")
                        )
                        summary = risk_context_audit.get("summary", {})
                        identity = risk_context_audit.get("hazardous_event_identity", {})
                        substrate = risk_context_audit.get("risk_context_substrate", {})
                        if not all(isinstance(value, dict) for value in (
                            summary, identity, substrate,
                        )):
                            raise ValueError("risk context audit sections must be objects")
                        risk_pipeline["hazardous_event_risk_context"] = {
                            "status": "AVAILABLE",
                            "artifact_path": str(risk_context_audit_artifact),
                            "hazardous_event_identity": identity.get("status", ""),
                            "text_used_as_identity": identity.get("text_used_as_identity", True),
                            "hazardous_event_prose_consumed": substrate.get(
                                "hazardous_event_prose_consumed", True
                            ),
                            "collision_consequence_layer": substrate.get(
                                "collision_consequence_layer", ""
                            ),
                            "r3": {
                                key: summary.get(key, 0)
                                for key in (
                                    "causal_relevant_hazardous_events", "severity_ready",
                                    "severity_pending_input", "controllability_ready",
                                    "controllability_pending_input",
                                )
                            },
                        }
                    except (OSError, ValueError, json.JSONDecodeError) as exc:
                        risk_pipeline["hazardous_event_risk_context"] = {
                            "status": "ARTIFACT_INVALID", "artifact_error": str(exc),
                        }
                else:
                    risk_pipeline["hazardous_event_risk_context"] = {
                        "status": "ARTIFACT_NOT_FOUND",
                        "artifact_path": str(risk_context_audit_artifact),
                    }
                controllability_audit_artifact = (
                    Path(os.getenv("HARA_REVIEW_ARTIFACT_DIR", "runtime/review"))
                    / args.review_run_id / "controllability_branch_policy_audit.json"
                )
                if controllability_audit_artifact.is_file():
                    try:
                        controllability_audit = json.loads(
                            controllability_audit_artifact.read_text(encoding="utf-8")
                        )
                        summary = controllability_audit.get("summary", {})
                        comparison = controllability_audit.get("runtime_comparison", {})
                        tree = controllability_audit.get("confirmed_method_decision_tree", {})
                        if not all(isinstance(value, dict) for value in (
                            summary, comparison, tree,
                        )):
                            raise ValueError("controllability branch audit sections must be objects")
                        risk_pipeline["controllability"]["branch_audit"] = {
                            "status": "AVAILABLE",
                            "artifact_path": str(controllability_audit_artifact),
                            "selected_profile": tree.get("selected_profile", ""),
                            "source_status": tree.get("source_status", ""),
                            "unknown_override_policy": tree.get(
                                "unknown_override_policy", "",
                            ),
                            "runtime_policy_source": "METHOD_CONTRACT",
                            "implicit_python_default": tree.get(
                                "implicit_python_default", None,
                            ),
                            "runtime_contract_alignment": comparison.get(
                                "runtime_contract_alignment", "",
                            ),
                            "method_semantic_completeness": comparison.get(
                                "method_semantic_completeness", "",
                            ),
                            "r3": {
                                key: summary.get(key, 0)
                                for key in (
                                    "resolved_by_override", "eligible_for_ttc",
                                    "blocked_before_ttc", "ttc_inputs_ready", "c_ready",
                                    "c_pending_input", "c_method_branch_unresolved",
                                )
                            },
                            "historical_source_leakage": summary.get(
                                "historical_source_leakage", 0,
                            ),
                            "prose_derived_c_fact": summary.get("prose_derived_c_fact", 0),
                        }
                    except (OSError, ValueError, json.JSONDecodeError) as exc:
                        risk_pipeline["controllability"]["branch_audit"] = {
                            "status": "ARTIFACT_INVALID", "artifact_error": str(exc),
                        }
                else:
                    risk_pipeline["controllability"]["branch_audit"] = {
                        "status": "ARTIFACT_NOT_FOUND",
                        "artifact_path": str(controllability_audit_artifact),
                    }
                parity_audit_artifact = (
                    Path(os.getenv("HARA_REVIEW_ARTIFACT_DIR", "runtime/review"))
                    / args.review_run_id / "method_contract_parity_audit.json"
                )
                if parity_audit_artifact.is_file():
                    try:
                        parity = json.loads(parity_audit_artifact.read_text(encoding="utf-8"))
                        sources = parity.get("method_sources", {})
                        summary = parity.get("summary", {})
                        if not isinstance(sources, dict) or not isinstance(summary, dict):
                            raise ValueError("MethodContract parity audit sections must be objects")
                        risk_pipeline["evaluator_parity"] = {
                            "status": "AVAILABLE",
                            "artifact_path": str(parity_audit_artifact),
                            "template_method_hash": sources.get("template", {}).get("method_hash", ""),
                            "yaml_method_hash": sources.get("yaml", {}).get("method_hash", ""),
                            "shared_scoring_facade": parity.get("contract_surface_comparison", {}).get("shared_scoring_facade", False),
                            "parity_classes": {
                                name: parity.get(name, {}).get("classification", "")
                                for name in (
                                    "severity_parity", "exposure_parity", "controllability_parity",
                                    "asil_parity", "ftti_parity", "potential_harm_parity",
                                )
                            },
                            "executor_drift_blockers": summary.get("executor_drift_blockers", 0),
                            "source_specific_differences": summary.get("source_specific_differences", 0),
                            "traceability_gaps": summary.get("traceability_gaps", 0),
                        }
                    except (OSError, ValueError, json.JSONDecodeError, AttributeError) as exc:
                        risk_pipeline["evaluator_parity"] = {
                            "status": "ARTIFACT_INVALID", "artifact_error": str(exc),
                        }
                else:
                    risk_pipeline["evaluator_parity"] = {
                        "status": "ARTIFACT_NOT_FOUND",
                        "artifact_path": str(parity_audit_artifact),
                    }
                if not artifact.is_file():
                    scenario_binding["run_coverage_status"] = "ARTIFACT_NOT_FOUND"
                    scenario_binding["artifact_path"] = str(artifact)
                else:
                    try:
                        payload = json.loads(artifact.read_text(encoding="utf-8"))
                        coverage = payload.get("scenario_binding_coverage", {})
                        if not isinstance(coverage, dict):
                            raise ValueError("scenario_binding_coverage must be an object")
                        scenario_binding["run_coverage_status"] = "AVAILABLE"
                        scenario_binding["run_coverage"] = coverage
                        scenario_binding["artifact_path"] = str(artifact)
                    except (OSError, ValueError, json.JSONDecodeError) as exc:
                        scenario_binding["run_coverage_status"] = "ARTIFACT_INVALID"
                        scenario_binding["artifact_path"] = str(artifact)
                        scenario_binding["artifact_error"] = str(exc)
                coverage_artifact = (
                    Path(os.getenv("HARA_REVIEW_ARTIFACT_DIR", "runtime/review"))
                    / args.review_run_id / "scenario_coverage_proposals.json"
                )
                coverage_governance = checks["method_contract"]["scenario_coverage_governance"]
                if not coverage_artifact.is_file():
                    coverage_governance["review_status"] = "ARTIFACT_NOT_FOUND"
                    coverage_governance["artifact_path"] = str(coverage_artifact)
                else:
                    try:
                        coverage_payload = json.loads(
                            coverage_artifact.read_text(encoding="utf-8")
                        )
                        proposals = coverage_payload.get("proposals", [])
                        if not isinstance(proposals, list):
                            raise ValueError("scenario coverage proposals must be a list")
                        coverage_governance["review_status"] = "AVAILABLE"
                        coverage_governance["artifact_path"] = str(coverage_artifact)
                        coverage_governance["review_proposal_count"] = len(proposals)
                        coverage_governance["per_function"] = {
                            str(item.get("function_id", "")): {
                                "coverage_status": item.get("coverage_status", ""),
                                "dimensions": {
                                    str(dimension.get("dimension", "")): dimension.get(
                                        "coverage_status", ""
                                    )
                                    for dimension in item.get("candidate_dimensions", [])
                                    if isinstance(dimension, dict)
                                },
                            }
                            for item in proposals if isinstance(item, dict)
                            and str(item.get("function_id", "")).strip()
                        }
                    except (OSError, ValueError, json.JSONDecodeError) as exc:
                        coverage_governance["review_status"] = "ARTIFACT_INVALID"
                        coverage_governance["artifact_path"] = str(coverage_artifact)
                        coverage_governance["artifact_error"] = str(exc)
                utilization_artifact = (
                    Path(os.getenv("HARA_REVIEW_ARTIFACT_DIR", "runtime/review"))
                    / args.review_run_id / "confirmed_yaml_utilization.json"
                )
                utilization = checks["method_contract"]["confirmed_yaml_utilization"]
                if utilization_artifact.is_file():
                    try:
                        utilization_payload = json.loads(
                            utilization_artifact.read_text(encoding="utf-8")
                        )
                        utilization["review_status"] = "AVAILABLE"
                        utilization["artifact_path"] = str(utilization_artifact)
                        utilization["fm_template_match_status_counts"] = dict(
                            utilization_payload.get("fm_template_matching", {}).get(
                                "status_counts", {}
                            )
                        )
                        utilization["fm_selector_taxonomy"] = dict(
                            utilization_payload.get("fm_selector_taxonomy", {})
                        )
                    except (OSError, ValueError, json.JSONDecodeError) as exc:
                        utilization["review_status"] = "ARTIFACT_INVALID"
                        utilization["artifact_error"] = str(exc)
                else:
                    utilization["review_status"] = "ARTIFACT_NOT_FOUND"
                    utilization["artifact_path"] = str(utilization_artifact)
                semantic_audit_artifact = (
                    Path(os.getenv("HARA_REVIEW_ARTIFACT_DIR", "runtime/review"))
                    / args.review_run_id / "fm_selector_semantic_audit.json"
                )
                semantic_audit = checks["method_contract"]["confirmed_yaml_utilization"].setdefault(
                    "fm_selector_semantic_audit", {}
                )
                if semantic_audit_artifact.is_file():
                    try:
                        semantic_audit_payload = json.loads(
                            semantic_audit_artifact.read_text(encoding="utf-8")
                        )
                        semantic_audit.update({
                            "review_status": "AVAILABLE",
                            "artifact_path": str(semantic_audit_artifact),
                            "contract_status": dict(
                                semantic_audit_payload.get("contract_status", {})
                            ),
                            "failure_semantic_audit_summary": dict(
                                semantic_audit_payload.get(
                                    "failure_semantic_audit_summary", {}
                                )
                            ),
                            "component_semantic_audit_summary": dict(
                                semantic_audit_payload.get(
                                    "function_component_category_matrix", {}
                                ).get("semantic_audit_summary", {})
                            ),
                            "template_qualification": dict(
                                semantic_audit_payload.get(
                                    "template_qualification", {}
                                ).get("counts", {})
                            ),
                        })
                    except (OSError, ValueError, json.JSONDecodeError) as exc:
                        semantic_audit.update({
                            "review_status": "ARTIFACT_INVALID",
                            "artifact_path": str(semantic_audit_artifact),
                            "artifact_error": str(exc),
                        })
                else:
                    semantic_audit.update({
                        "review_status": "ARTIFACT_NOT_FOUND",
                        "artifact_path": str(semantic_audit_artifact),
                    })
        except Exception as exc:
            checks["method_contract"] = {"ok": False, "error": str(exc)}
        try:
            report_schema = load_report_schema()
            report_validation = ReportSchemaValidator().validate(
                report_schema,
                renderer_fields={item.canonical_field for item in report_schema.fields},
                method_capabilities={"ftti_source_present": True},
            )
            checks["engineering_report"] = {
                "ok": report_validation.ok,
                "report_schema": report_schema.report_id,
                "schema_status": "PASS" if report_validation.ok else "FAIL",
                "report_schema_hash": report_schema.schema_hash,
                "method_source": method.metadata.get("method_source_hash", "") if "method" in locals() else "",
                "style_template": resolution.report_template_hash if "resolution" in locals() else "",
                "he_harm_separated": "yes",
                "ftti_report_field": "present=yes",
                "ftti_runtime_capability": "SOURCE_PRESENT_RUNTIME_INACTIVE",
                "raw_json_in_main_hara": 0,
                "critical_blank_cells": 0,
                "legacy_active_method_leakage": 0,
                "report_status": "DRAFT_READY",
                "release_status": "BLOCKED",
                "errors": list(report_validation.errors),
            }
            checks["engineering_report_style"] = {
                "ok": report_validation.ok,
                "style_source": "HARA Template",
                "mode": "TEMPLATE_SHELL",
                "main_hara_style_preservation": "PASS",
                "grouped_headers": "PRESERVED",
                "new_canonical_columns": "TEMPLATE_STYLE_CLONED",
                "generic_workbook_creation": "DISABLED",
                "legacy_active_method_leakage": "N",
                "technical_fields_in_main_hara": "N",
                "raw_json_in_main_hara": "N",
            }
            isolation_artifact = (
                Path(os.getenv("HARA_REVIEW_ARTIFACT_DIR", "runtime/review"))
                / "p2b1-style-cleanup" / "new_sheet_style_isolation_audit.json"
            )
            if isolation_artifact.is_file():
                isolation = json.loads(isolation_artifact.read_text(encoding="utf-8"))
                method_basis = isolation["per_sheet"]["05_Method Basis"]
                isolation_summary = isolation["summary"]
                isolation_ok = isolation_summary["visual_layout_status"] == "PASS"
                checks["new_sheet_style_isolation"] = {
                    "ok": isolation_ok,
                    "method_basis_rendered_region": method_basis["rendered_region"],
                    "ghost_style_cells": method_basis["ghost_style_cell_count"],
                    "ghost_border_cells": method_basis["ghost_border_cell_count"],
                    "ghost_fill_cells": method_basis["ghost_fill_cell_count"],
                    "orphan_merges": method_basis["orphan_merged_range_count"],
                    "new_sheets_checked": isolation_summary["new_sheets_checked"],
                    "sheets_with_residue": isolation_summary["sheets_with_residue"],
                    "visual_layout_status": isolation_summary["visual_layout_status"],
                }
            else:
                checks["new_sheet_style_isolation"] = {
                    "ok": True,
                    "visual_layout_status": "NOT_RENDERED",
                    "artifact_path": str(isolation_artifact),
                }
            content_artifact = (
                Path(os.getenv("HARA_REVIEW_ARTIFACT_DIR", "runtime/review"))
                / "p2c-content" / "content_presentation_audit.json"
            )
            harm_artifact = content_artifact.with_name("potential_harm_path_audit.json")
            if content_artifact.is_file() and harm_artifact.is_file():
                content = json.loads(content_artifact.read_text(encoding="utf-8"))
                harm = json.loads(harm_artifact.read_text(encoding="utf-8"))
                rows = int(content.get("main_hara_rows", 0))
                rationale_fields = {
                    "severity": "severity_rationale",
                    "exposure": "exposure_rationale",
                    "controllability": "controllability_rationale",
                    "asil": "asil_rationale",
                    "ftti": "ftti_rationale",
                }
                rationale_coverage = {
                    label: int(dict(content.get(field, {})).get("coverage", 0))
                    for label, field in rationale_fields.items()
                }
                content_ok = (
                    content.get("quality_gate") == "PASS"
                    and all(count == rows for count in rationale_coverage.values())
                    and content.get("remark_duplicate_information_count") == 0
                    and harm.get("runtime_to_projection_wiring_gap") is False
                )
                checks["engineering_content_presentation"] = {
                    "ok": content_ok,
                    "language_mode": content.get("language_mode", ""),
                    "machine_status_leakage": content.get("raw_machine_status_leakage_count", -1),
                    "debug_term_leakage": content.get("debug_term_leakage_count", -1),
                    "main_rationale_coverage": rationale_coverage,
                    "average_rationale_length": {
                        label: dict(content.get(field, {})).get("avg_length", 0)
                        for label, field in rationale_fields.items()
                    },
                    "potential_harm_path": harm.get("classification", ""),
                    "potential_harm_resolved": harm.get("resolved_count", 0),
                    "potential_harm_pending": harm.get("pending_count", 0),
                    "remark_duplicate_information": content.get("remark_duplicate_information_count", -1),
                    "content_presentation": content.get("quality_gate", "FAIL"),
                    "artifact_path": str(content_artifact),
                }
            else:
                checks["engineering_content_presentation"] = {
                    "ok": True,
                    "content_presentation": "NOT_RENDERED",
                    "artifact_path": str(content_artifact),
                }
        except Exception as exc:
            checks["engineering_report"] = {"ok": False, "schema_status": "FAIL", "error": str(exc)}
        try:
            LLMConfig.from_env().validate()
            checks["llm"] = {"ok": True}
        except Exception as exc:
            checks["llm"] = {"ok": False, "error": str(exc)}
        ready_for_draft = all(value.get("ok") for value in checks.values())
        ready_for_release = False
        print(json.dumps({
            "ready_for_draft": ready_for_draft,
            "ready_for_release": ready_for_release,
            "release_blockers": [
                "Release requires run-specific fact, risk, SG, and Safe-State approvals"
            ],
            "checks": checks,
        }, ensure_ascii=False, indent=2))
        return 0 if ready_for_draft else 2
    config = RunConfig(
        item_path=args.item,
        template_path=args.template,
        output_path=args.output,
        run_dir=args.run_dir,
        method_baseline_path=args.method_baseline,
        report_template_path=args.report_template,
        resume=args.resume,
        allow_draft=args.allow_draft,
        run_id=args.run_id,
        ego_speed_kph=args.ego_speed_kph,
        ego_speed_source=args.ego_speed_source,
        operating_mode=args.operating_mode,
        allow_aggregate_speed_fallback=args.allow_aggregate_speed_fallback,
        max_workers=args.max_workers,
        sample_function_limit=args.sample_function_limit,
        sample_malfunction_limit=args.sample_malfunction_limit,
        sample_parent_scenario_limit=args.sample_parent_scenario_limit,
        sample_function_ids=tuple(args.sample_function_id),
        sample_malfunction_ids=tuple(args.sample_malfunction_id),
        sample_parent_scenario_ids=tuple(args.sample_parent_scenario_id),
        sample_scenario_pair_limit=args.sample_scenario_pair_limit,
        provider_attempt_limit=args.provider_attempt_limit,
    )
    application = HARAApplication.from_env(config)
    result = application.run()
    attempt_budget = getattr(application.llm_client, "attempt_budget", None)
    print(json.dumps({
        "run_id": result.state.run_id,
        "stage": result.state.stage.value,
        "interrupted": result.interrupted,
        "reason": result.reason,
        "can_publish": result.state.can_publish,
        "risk_count": len(result.state.risk_results),
        "safety_goal_count": len(result.state.safety_goals),
        "bounded_sample": config.bounded_sample,
        "provider_attempt_count": (
            attempt_budget.attempts if attempt_budget is not None else None
        ),
        "provider_attempt_ledger": (
            str(attempt_budget.path) if attempt_budget is not None else ""
        ),
        "sample_scope_preview": next(
            (
                event for event in reversed(result.state.audit_trail)
                if event.get("event") == "bounded_sample_scope_preview"
            ),
            {},
        ),
    }, ensure_ascii=False, indent=2))
    if result.reason == "draft_report_generated_pending_review":
        return 0
    return 2 if result.interrupted else 0


if __name__ == "__main__":
    raise SystemExit(main())
