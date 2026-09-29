from dataclasses import replace
from types import SimpleNamespace

from hara_agent.models import EvidenceValue, ReviewStatus, RiskAssessment, ScenarioCandidate
from hara_agent.services.reporting import (
    EngineeringReportTextMapper, HARAReportProjectionService,
    audit_content_presentation, audit_potential_harm_path, load_report_schema,
)
from hara_agent.workflow.state import HARAState


def _pending(reason: str) -> EvidenceValue:
    return EvidenceValue(None, ReviewStatus.PENDING, review_reason=reason)


def _method() -> SimpleNamespace:
    return SimpleNamespace(
        metadata={"method_source_hash": "method-hash"},
        guidewords=SimpleNamespace(guidewords=["No/Loss"]),
    )


def _state(*, potential_harm: str = "") -> HARAState:
    risk = RiskAssessment(
        assessment_id="RA-1", scenario_id="SCN-1",
        severity=_pending("MISSING_RELATIVE_SPEED"),
        exposure=_pending("EXPOSURE_DIMENSION_COVERAGE"),
        controllability=_pending("METHOD_BRANCH_UNRESOLVED"),
        asil=_pending("missing=S,E,C"), malfunction_id="MF-1",
        hazardous_event="车辆出现危险状态", potential_harm=potential_harm,
    )
    return HARAState(
        run_id="content-audit",
        functions=[{"function_id": "F-1", "name": "功能", "output": "输出"}],
        malfunctions=[{
            "malfunction_id": "MF-1", "function_id": "F-1", "guideword": "No/Loss",
            "description": "失效", "vehicle_level_hazard": "危害",
        }],
        scenarios=[ScenarioCandidate(
            "SCN-1", "室外停车场", "", "", operating_mode="active",
            facts={
                "operating_mode": "active",
                "ego_speed_constraint": {"speed_min_kph": 0, "speed_max_kph": 20},
                "method_scenario_dimensions": {
                    "ROAD": {"unresolved_reason": "AMBIGUOUS_BINDING"},
                    "TRAFFIC_PATTERN": {"unresolved_reason": "NO_ITEM_FACT"},
                    "OBJECT": {"unresolved_reason": "NO_ITEM_FACT"},
                    "EGO_X_ROAD": {"unresolved_reason": "NO_ITEM_FACT"},
                },
            },
        )],
        risk_results=[risk],
        audit_trail=[{
            "event": "structured_risk_scoring_completed",
            "risk_calculation_inputs": [{
                "malfunction_id": "MF-1", "scenario_id": "SCN-1",
                "potential_harm": {
                    "potential_harm": "", "status": "PENDING_METHOD_SEMANTICS",
                    "reason": "Severity has not been finalized.",
                },
            }],
        }],
    )


def test_text_mapper_keeps_scenario_facts_and_hides_dimension_reason_codes():
    mapper = EngineeringReportTextMapper()
    scenario = _state().scenarios[0]
    operational, detail = mapper.scenario(scenario)
    assert operational == "室外停车场，AVP处于激活状态，适用车速范围为0–20 km/h。"
    assert detail == "适用车速范围：0–20 km/h；分析变体：受控分析场景。"
    assert "交通参与者信息未提供" not in detail
    assert "AMBIGUOUS_BINDING" not in detail
    assert "NO_ITEM_FACT" not in detail


def test_text_mapper_projects_resolved_synthesized_dimensions_separately():
    scenario = ScenarioCandidate(
        "SCN-CHILD", "legacy location", "", "", operating_mode="active",
        facts={
            "method_scenario_dimensions": {
                "WHERE": {"resolution_status": "RESOLVED", "method_value": "SO010 | Garage"},
                "ROAD": {"resolution_status": "RESOLVED", "method_value": "FB005 | Normal friction"},
                "EGO_ACTION": {"resolution_status": "RESOLVED", "method_value": "FV010 | Parking in/out"},
                "EGO_X_ROAD": {"resolution_status": "RESOLVED", "method_value": "PH012 | Slope 5-8%"},
                "TRAFFIC_PATTERN": {"resolution_status": "PENDING", "unresolved_reason": "NO_ITEM_FACT"},
                "EGO_DYNAMICS": {"resolution_status": "RESOLVED", "method_value": "FA001 | Low speed"},
                "OBJECT": {"resolution_status": "RESOLVED", "method_value": "CN_other_road_users | Pedestrian"},
            },
        },
    )
    operational, detail = EngineeringReportTextMapper().scenario(scenario)
    assert operational == "项目运行区域内，AVP处于激活状态。"
    assert detail == "分析变体：受控分析场景。"
    for raw in ("Garage", "Parking in/out", "Low speed", "Pedestrian", "EGO_X_ROAD", "FA001"):
        assert raw not in operational + detail


def test_potential_harm_path_is_upstream_pending_not_a_projection_gap():
    state = _state()
    view = HARAReportProjectionService(load_report_schema()).project(state, _method())
    path = audit_potential_harm_path(state, view)
    content = audit_content_presentation(view, path)

    assert path["resolver_invocation_count"] == 1
    assert path["classification"] == "UPSTREAM_RISK_NOT_READY"
    assert path["runtime_to_projection_wiring_gap"] is False
    assert view.rows[0].potential_harm == "待S评定完成后确定"
    assert content["quality_gate"] == "PASS"
    assert content["raw_machine_status_leakage_count"] == 0
    assert content["remark_duplicate_information_count"] == 0


def test_potential_harm_audit_distinguishes_partial_and_complete_resolution():
    state = _state()
    state.risk_results.append(replace(
        state.risk_results[0], assessment_id="RA-2", scenario_id="SCN-2",
        severity=EvidenceValue("S1", ReviewStatus.FINALIZED, rule_version="method-hash"),
        potential_harm="轻中度伤害",
    ))
    state.audit_trail[0]["risk_calculation_inputs"].append({
        "malfunction_id": "MF-1", "scenario_id": "SCN-2",
        "potential_harm": {"potential_harm": "轻中度伤害", "status": "FINALIZED"},
    })
    view = SimpleNamespace(
        rows=[
            SimpleNamespace(scenario_id="SCN-1", potential_harm="待S评定完成后确定"),
            SimpleNamespace(scenario_id="SCN-2", potential_harm="轻中度伤害"),
        ],
        scenario_details=[],
    )
    partial = audit_potential_harm_path(state, view)
    assert partial["classification"] == "PARTIAL_UPSTREAM_RISK_PENDING"
    assert partial["runtime_to_projection_wiring_gap"] is False

    state.risk_results[0] = replace(
        state.risk_results[0], severity=EvidenceValue(
            "S1", ReviewStatus.FINALIZED, rule_version="method-hash"
        ),
        potential_harm="轻中度伤害",
    )
    state.audit_trail[0]["risk_calculation_inputs"][0]["potential_harm"] = {
        "potential_harm": "轻中度伤害", "status": "FINALIZED",
    }
    view.rows[0].potential_harm = "轻中度伤害"
    assert audit_potential_harm_path(state, view)["classification"] == "RESOLVED"


def test_projection_preserves_a_resolved_potential_harm_without_recomputing_it():
    state = _state(potential_harm="人员受伤")
    view = HARAReportProjectionService(load_report_schema()).project(state, _method())
    assert view.rows[0].potential_harm == "人员受伤"


def test_projection_derives_synthesis_causal_and_risk_statuses():
    state = _state()
    state.scenarios[0].analysis_instance = {
        "malfunction_id": "MF-1", "validation_status": "VALIDATED",
    }
    service = HARAReportProjectionService(load_report_schema())
    synthesis = service.project(state, _method())
    assert synthesis.rows[0].assessment_status == (
        "METHOD_VALID — CAUSAL_REVALIDATION_REQUIRED"
    )
    assert "RISK SCORING NOT YET EXECUTED" in synthesis.summary.report_status

    parsed = {
        "malfunction_id": "MF-1", "scenario_id": "SCN-1",
        "final_retain": True,
    }
    causal = service.project(state, _method(), causal_trace={
        "audits": [{"item_salvage_audit": [{"parsed_assessment": parsed}]}],
    })
    assert causal.rows[0].assessment_status == "METHOD_VALID — CAUSAL_REVALIDATED"

    scored = service.project(state, _method(), risk_trace={
        "assessments": [{
            "malfunction_id": "MF-1", "scenario_id": "SCN-1",
            "risk_scoring_invoked": True,
            **{
                field: {"status": "PENDING_INPUT", "result": None}
                for field in ("severity", "exposure", "controllability", "asil")
            },
        }],
    })
    assert scored.rows[0].assessment_status == "ELIGIBLE — RISK SCORING INVOKED"


def test_projection_does_not_publish_stale_values_when_trace_says_scoring_not_invoked():
    state = _state()
    for field, value in (
        ("severity", "S2"), ("exposure", "E3"),
        ("controllability", "C2"), ("asil", "B"),
    ):
        setattr(
            state.risk_results[0], field,
            EvidenceValue(value, ReviewStatus.FINALIZED, rule_version="method-v1"),
        )
    view = HARAReportProjectionService(load_report_schema()).project(
        state, _method(), risk_trace={"assessments": [{
            "malfunction_id": "MF-1", "scenario_id": "SCN-1",
            "risk_scoring_invoked": False,
        }]},
    )

    assert view.rows[0].severity == "Pending"
    assert view.rows[0].asil == "Pending"
    assert "暂不展示旧值" in view.rows[0].severity_rationale
    assert view.rows[0].remark == "S 未计算；E 未计算；C 未计算；ASIL 未计算"
    assert view.rows[0].assessment_status == "PENDING — RISK SCORING NOT EXECUTED"
    assert view.projection_metrics["score_status_counts"]["severity"]["calculated"] == 0
    assert "RISK SCORING NOT YET EXECUTED" in view.summary.report_status


def test_exposure_rationale_is_trace_derived_for_finalized_evidence():
    rationale = EngineeringReportTextMapper().exposure_rationale(
        EvidenceValue("E3", ReviewStatus.FINALIZED, rule_version="test"),
        {
            "requested_domain": "Z", "domain": "Z", "result": "E3",
            "atom_bindings": [
                {
                    "atom_id": "FA001", "dimension": ["EGO_DYNAMICS"],
                    "E_class": "E4", "used": True,
                },
                {
                    "atom_id": "PH005", "dimension": ["EGO_ACTION"],
                    "E_class": "E3", "used": True,
                },
            ],
            "dependency_coupling": {"policy_branch": "e3_e4_mix"},
        },
    )
    assert "请求 Z 域" in rationale
    assert "自车动态=E4" in rationale
    assert "自车动作=E3" in rationale
    assert "FA001" not in rationale
    assert "PH005" not in rationale
    assert "FUSA v1" in rationale
    assert "最终 E3" in rationale
    assert rationale != "已按当前方法完成 E 评定。"
