from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

from hara_agent.models import EvidenceValue, ReviewStatus, RiskAssessment, ScenarioCandidate
from hara_agent.services.reporting import (
    HARAReportProjectionService, HARAReportWorkbookRenderer, ReportSchemaValidator,
    audit_content_presentation, audit_potential_harm_path, load_report_schema,
)
from hara_agent.services.reporting.view_model import (
    AuditReferenceView, HARAReportRowView, HARAReportViewModel, MethodBasisView,
    SummaryView,
)
from hara_agent.workflow.state import HARAState


ROOT = Path(__file__).resolve().parents[2]


def test_canonical_schema_compiles_with_independent_hash():
    schema = load_report_schema(ROOT / "report_assets" / "hara_report_v1.yaml")
    assert schema.report_id == "hara_report_v1"
    assert {item.canonical_field for item in schema.fields} >= {
        "hazardous_event", "potential_harm", "ftti", "ftti_rationale",
    }
    assert schema.field("hazardous_event").order != schema.field("potential_harm").order
    visible = schema.presentation["main_hara"]["visible_fields"]
    assert "ftti" in visible
    assert "hazardous_event_id" not in visible


def test_schema_capability_check_catches_missing_ftti_mapping():
    schema = load_report_schema(ROOT / "report_assets" / "hara_report_v1.yaml")
    changed = replace(schema, fields=tuple(item for item in schema.fields if item.canonical_field != "ftti"))
    result = ReportSchemaValidator().validate(changed, method_capabilities={"ftti_source_present": True})
    assert not result.ok
    assert any("REPORT_FIELD_MISSING: FTTI" in error for error in result.errors)


def test_projection_separates_harm_and_exposes_human_pending_reasons():
    schema = load_report_schema(ROOT / "report_assets" / "hara_report_v1.yaml")
    pending = lambda reason: EvidenceValue(None, ReviewStatus.PENDING, review_reason=reason)
    risk = RiskAssessment(
        assessment_id="RA-1", scenario_id="SCN-1",
        severity=pending("MISSING_RELATIVE_SPEED"),
        exposure=pending("EXPOSURE_DIMENSION_COVERAGE"),
        controllability=pending("METHOD_BRANCH_UNRESOLVED"),
        asil=pending("missing=S,E,C"), malfunction_id="MF-1",
        hazardous_event="Vehicle enters an unsafe path", potential_harm="",
    )
    state = HARAState(run_id="projection-test", functions=[{"function_id": "F-1", "name": "Function", "output": "Output"}], malfunctions=[{"malfunction_id": "MF-1", "function_id": "F-1", "guideword": "Loss", "description": "Output is lost", "vehicle_level_hazard": "Vehicle control is affected"}], scenarios=[ScenarioCandidate("SCN-1", "Parking area", "ignored", "ignored", facts={"operating_mode": "active"})], risk_results=[risk])
    method = SimpleNamespace(metadata={"method_source_hash": "method-hash"}, guidewords=SimpleNamespace(guidewords=["Loss"]))
    row = HARAReportProjectionService(schema).project(state, method).rows[0]
    assert row.hazardous_event == "Vehicle enters an unsafe path。"
    assert row.potential_harm == "待S评定完成后确定"
    assert row.severity_rationale == "缺少该危险事件的实际相对速度，S 暂不评定。"
    assert row.exposure_rationale == "暴露度场景维度覆盖规则未定义，E 暂不评定。"
    assert row.controllability_rationale == "未知分支策略未定义，C 暂不评定。"
    assert row.asil_rationale == "S/E/C 未全部确定，ASIL 暂不评定。"
    assert row.ftti == "Pending"
    assert row.ftti_rationale == "当前运行未启用 FTTI 计算。"
    assert row.remark == "S 未计算；E 未计算；C 未计算；ASIL 未计算"
    assert row.clarification_ids == "EC-03; EC-01; EC-02"
    assert "ignored" not in row.operational_scenario


def test_template_shell_renderer_preserves_grouped_hara_layout():
    schema = load_report_schema(ROOT / "report_assets" / "hara_report_v1.yaml")
    row = HARAReportRowView(
        hara_id="HARA-001", malfunction_id="MF-001", scenario_id="SCN-001",
        hazardous_event_id="HE-001", function_id="F-001", function_name="Drive",
        function_output="Vehicle motion", guideword="Loss", malfunction="Loss of drive",
        hazard="Loss of control", operational_scenario="Vehicle is moving",
        scenario_detail="Urban road at 20 km/h", hazardous_event="Vehicle departs lane",
        potential_harm="Pending（上游风险评定未完成）", severity="Pending",
        severity_rationale="缺少该危险事件的实际相对速度，S 暂不评定。",
        exposure="Pending", exposure_rationale="Exposure 场景维度覆盖规则未定义，E 暂不评定。",
        controllability="Pending", controllability_rationale="UNKNOWN 分支策略未定义，C 暂不评定。",
        asil="Pending", asil_rationale="S/E/C 未全部确定，ASIL 暂不评定。",
        ftti="Pending", ftti_rationale="当前运行未启用 FTTI 计算。",
        sg_id="", safety_goal="", safe_state="", assessment_status="PENDING",
        clarification_ids="EC-01; EC-02", remark="Draft — not for release.",
    )
    summary = SummaryView(
        run_id="style-test", method_source="yaml", method_hash="method-hash",
        report_schema="hara_report_v1", report_schema_version=schema.schema_version,
        report_schema_hash=schema.schema_hash, style_template_hash="style-hash",
        report_status="DRAFT", release_status="BLOCKED", function_count=1,
        guideword_assessment_count=1, malfunction_count=1, scenario_count=1,
        eligible_hazardous_event_count=1, severity_finalized=0, severity_pending=1,
        exposure_finalized=0, exposure_pending=1, controllability_finalized=0,
        controllability_pending=1, asil_finalized=0, asil_pending=1,
        clarification_ids="EC-01; EC-02",
    )
    view_model = HARAReportViewModel(
        rows=(row,), summary=summary,
        method_basis=MethodBasisView("yaml", "Loss", "pending", "pending", "pending", "pending", "inactive"),
        safety_goals=(),
        audit_references=(AuditReferenceView("style-test", "method-hash", schema.schema_hash, "style-hash", "HARA-001", "HE-001", "SCN-001", "trace.json", "EC-01; EC-02", "PENDING"),),
        schema_hash=schema.schema_hash, method_contract_hash="method-hash", style_template_hash="style-hash",
    )
    output = ROOT / "runtime" / f".test-template-shell-{uuid4().hex}.xlsx"
    renderer = HARAReportWorkbookRenderer()
    renderer.render(view_model, ROOT / "references" / "HARA_Template_AI_20260327.xlsx", output, schema)

    from openpyxl import load_workbook

    workbook = load_workbook(output, data_only=False)
    try:
        hara = workbook["04_HARA"]
        assert hara["I6"].value == "Vehicle departs lane"
        assert hara["J6"].value == "Pending（上游风险评定未完成）"
        assert hara["S6"].value == "Pending"
        assert hara["T6"].value == "当前运行未启用 FTTI 计算。"
        assert "I4:J4" in {str(item) for item in hara.merged_cells.ranges}
        assert "S4:T4" in {str(item) for item in hara.merged_cells.ranges}
        assert hara["I4"].value == "Hazard / Harm Analysis"
        assert hara.freeze_panes == "A6"
        assert "Malfunction ID" not in [cell.value for cell in hara[5]]
        assert "Malfunction ID" in [cell.value for cell in workbook["99_Audit"][4]]
        assert workbook["AI-process"].sheet_state == "hidden"
        assert workbook["0_Document Information"].sheet_state == "hidden"
        assert workbook["06_Safety Goal"]["A4"].value.startswith("No Safety Goal generated")
    finally:
        workbook.close()
        output.unlink(missing_ok=True)
    assert renderer.last_style_audit["group_structures_preserved"] is True
    assert renderer.last_style_audit["technical_fields_moved_from_main"] is True
    assert renderer.last_style_audit["data_style_reuse"]["rate"] == 1.0
    isolation = renderer.last_new_sheet_style_audit
    assert isolation["05_Method Basis"]["rendered_region"] == "B2:H5"
    assert isolation["summary"]["visual_layout_status"] == "PASS"
    for metrics in isolation["per_sheet"].values():
        assert metrics["ghost_style_cell_count"] == 0
        assert metrics["ghost_border_cell_count"] == 0
        assert metrics["ghost_fill_cell_count"] == 0
        assert metrics["orphan_merged_range_count"] == 0
