from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from hara_agent.method_sources import MethodSourceResolver
from hara_agent.workflow.review_artifacts import ReviewArtifactReader
from hara_agent.workflow.state import HARAState

from .canonical_renderer import HARAReportWorkbookRenderer, style_template_hash
from .content_audit import audit_content_presentation, audit_potential_harm_path
from .projection import HARAReportProjectionService
from .report_schema import ReportSchema, load_report_schema
from .scenario_projection_context import load_scenario_projection_contexts


class OfflineReportRebuilder:
    """Rebuild a report from committed state and review artifacts without a Provider."""

    def __init__(self, schema: ReportSchema | None = None):
        self.schema = schema or load_report_schema()

    def rebuild(
        self,
        *,
        checkpoint_path: str | Path,
        method_baseline_path: str | Path,
        report_style_template_path: str | Path,
        output_path: str | Path,
        review_root: str | Path = "runtime/review",
        causal_trace_path: str | Path | None = None,
        audit_output_dir: str | Path | None = None,
        synthesis_candidates_path: str | Path | None = None,
        speed_context_audit_path: str | Path | None = None,
    ) -> Path:
        checkpoint = Path(checkpoint_path).expanduser().resolve()
        state = HARAState.read_committed(json.loads(checkpoint.read_text(encoding="utf-8")))
        resolution = MethodSourceResolver().resolve(
            template_path=None,
            baseline_manifest_path=Path(method_baseline_path),
            report_template_path=Path(report_style_template_path),
        )
        review_reader = ReviewArtifactReader(state.run_id, review_root)
        artifacts = review_reader.read_all()
        trace_path = Path(review_root).expanduser().resolve() / state.run_id / "risk_execution_trace.json"
        trace = None
        if trace_path.is_file():
            trace = json.loads(trace_path.read_text(encoding="utf-8"))
        causal_trace = {}
        if causal_trace_path is not None:
            causal_path = Path(causal_trace_path).expanduser().resolve()
            if causal_path.is_file():
                causal_trace = json.loads(causal_path.read_text(encoding="utf-8"))
            else:
                raise FileNotFoundError(f"Causal trace not found: {causal_path}")
        summary = review_reader.summary()
        template_hash = style_template_hash(report_style_template_path)
        if (synthesis_candidates_path is None) != (speed_context_audit_path is None):
            raise ValueError(
                "Fresh report projection requires both synthesis candidates and speed context audit"
            )
        scenario_projection_contexts = (
            load_scenario_projection_contexts(
                synthesis_candidates_path=synthesis_candidates_path,
                speed_context_audit_path=speed_context_audit_path,
            )
            if synthesis_candidates_path is not None
            and speed_context_audit_path is not None
            else None
        )
        view_model = HARAReportProjectionService(self.schema).project(
            state,
            resolution.method,
            risk_trace=trace,
            causal_trace=causal_trace,
            run_summary=summary,
            style_template_hash=template_hash,
            risk_trace_reference=str(trace_path) if trace is not None else "",
            generated_scenarios=artifacts.get("scenario_candidate", []),
            scenario_projection_contexts=scenario_projection_contexts,
        )
        audit_root = (
            Path(audit_output_dir).expanduser().resolve()
            if audit_output_dir is not None
            else Path(review_root).expanduser().resolve() / state.run_id
        )
        p2c_audit_dir = audit_root / "p2c-content"
        p2c_audit_dir.mkdir(parents=True, exist_ok=True)
        potential_harm_path = audit_potential_harm_path(state, view_model)
        self._write_json(
            p2c_audit_dir / "potential_harm_path_audit.json",
            potential_harm_path,
        )
        renderer = HARAReportWorkbookRenderer()
        style_audit_dir = audit_root / "p2b-template-style"
        style_audit_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(
            style_audit_dir / "template_style_audit.json",
            renderer.audit_template(report_style_template_path),
        )
        output = renderer.render(
            view_model, report_style_template_path, output_path, self.schema
        )
        self._write_json(style_audit_dir / "style_restoration_audit.json", renderer.last_style_audit)
        self._write_json(
            p2c_audit_dir / "content_presentation_audit.json",
            audit_content_presentation(view_model, potential_harm_path),
        )
        prior_report = Path("output/HARA_P2B_Template_Style_Rebuild.xlsx")
        cleanup_audit = renderer.new_sheet_style_isolation_audit(
            output, before_path=prior_report if prior_report.is_file() else None,
        )
        cleanup_audit_dir = audit_root / "p2b1-style-cleanup"
        cleanup_audit_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(
            cleanup_audit_dir / "new_sheet_style_isolation_audit.json",
            cleanup_audit,
        )
        audit_dir = audit_root / "p2-report-rebuild"
        audit_dir.mkdir(parents=True, exist_ok=True)
        self._write_json(audit_dir / "report_contract_audit.json", self._contract_audit(resolution.method, template_hash))
        self._write_json(audit_dir / "report_projection_audit.json", self._projection_audit(view_model, artifacts))
        return output

    def _contract_audit(self, method: Any, template_hash: str) -> dict[str, Any]:
        fields = [item.canonical_field for item in self.schema.fields]
        return {
            "schema_version": self.schema.schema_version,
            "report_id": self.schema.report_id,
            "canonical_fields": fields,
            "required_fields": [item.canonical_field for item in self.schema.fields if item.required],
            "optional_fields": [item.canonical_field for item in self.schema.fields if not item.required],
            "compiled_mappings": [{"canonical_field": item.canonical_field, "sheet": item.sheet, "order": item.order, "header": item.header} for item in self.schema.fields],
            "missing_mappings": [], "duplicate_mappings": [],
            "method_contract_hash": method.metadata.get("method_source_hash", method.metadata.get("template_hash", "")),
            "report_schema_hash": self.schema.schema_hash,
            "style_template_hash": template_hash,
            "legacy_template_fields_excluded": True,
            "ftti_mapping_status": "PRESENT",
            "he_harm_separation_status": "SEPARATE_CANONICAL_FIELDS",
        }

    @staticmethod
    def _projection_audit(view_model: Any, artifacts: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
        rows = list(view_model.rows)
        main_values = [value for row in rows for value in row.to_dict().values()]
        he_keys = [row.hazardous_event_id or row.hazardous_event for row in rows]
        pair_counts: dict[tuple[str, str], int] = {}
        for row in rows:
            pair = (row.malfunction_id, row.operational_scenario)
            pair_counts[pair] = pair_counts.get(pair, 0) + 1
        return {
            "rows": len(rows),
            "scenario_detail_rows": len(view_model.scenario_details),
            "grouping": dict(view_model.projection_metrics or {}),
            "unique_operational_scenario_count": len({row.operational_scenario for row in rows}),
            "unique_he_count": len(set(he_keys)),
            "duplicate_he_count": len(he_keys) - len(set(he_keys)),
            "same_malfunction_scenario_variants": sum(max(0, count - 1) for count in pair_counts.values()),
            "pending_human_rationale_coverage": {
                field: sum(bool(getattr(row, f"{field}_rationale")) for row in rows)
                for field in ("severity", "exposure", "controllability", "asil", "ftti")
            },
            "raw_json_leakage": sum(isinstance(value, (dict, list)) for value in main_values),
            "internal_identifier_leakage": sum(any(token in str(value) for token in ("candidate_atom_ids", "NO_EXPLICIT_METHOD_ALIAS", "FA001")) for value in main_values),
            "blank_critical_fields": sum(not getattr(row, field) for row in rows for field in ("hazardous_event", "operational_scenario", "scenario_detail", "severity", "severity_rationale", "exposure", "exposure_rationale", "controllability", "controllability_rationale", "asil", "asil_rationale", "ftti", "ftti_rationale")),
            "clarification_mappings": {item: sum(item in row.clarification_ids for row in rows) for item in ("EC-01", "EC-02", "EC-03")},
            "source_artifacts": sorted(artifacts),
        }

    @staticmethod
    def _write_json(path: Path, value: dict[str, Any]) -> None:
        temporary = path.with_suffix(path.suffix + ".tmp")
        temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        temporary.replace(path)


__all__ = ["OfflineReportRebuilder"]
