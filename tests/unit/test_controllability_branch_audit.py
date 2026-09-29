from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from hara_agent.contracts import UnknownOverridePolicy
from hara_agent.method_sources import YamlBaselineCompiler
from hara_agent.services.analysis import ControllabilityBranchAuditService
from hara_agent.template import TemplateRoleCompiler
from hara_agent.workflow import ReviewArtifactReader


ROOT = Path(__file__).resolve().parents[2]


def _service() -> ControllabilityBranchAuditService:
    report = TemplateRoleCompiler().compile_method(
        ROOT / "references/HARA_Template_AI_20260327.xlsx"
    ).report_contract
    method = YamlBaselineCompiler().compile(
        ROOT / "method_assets/fusa_baseline_v1/manifest.yaml",
        report_contract=report,
    )
    return ControllabilityBranchAuditService(method)


def _context(*, driver=False, remote=False, avoidance=False, ttc=4.5):
    def fact(value):
        return {
            "status": "AVAILABLE", "value": value,
            "source_type": "DIRECT_SCENARIO_FACT", "source_ref": "test:fact",
        }

    return {
        "driver_in_vehicle": fact(driver),
        "remote_intervention_available": fact(remote),
        "other_road_user_avoidance_possible": fact(avoidance),
        "ttc_s": fact(ttc),
    }


def _with_policy(policy: UnknownOverridePolicy) -> ControllabilityBranchAuditService:
    service = _service()
    service.structured = replace(
        service.structured,
        controllability_branch_policy=replace(
            service.structured.controllability_branch_policy,
            unknown_override_policy=policy,
        ),
    )
    return service


def test_confirmed_positive_override_directly_resolves_c3_without_ttc():
    result = _service().readiness(_context())
    assert result["status"] == "READY"
    assert result["decision_stage"] == "OVERRIDE"
    assert result["matched_rule_id"] == "driver_outside_no_intervention"
    assert result["result"] == "C3"


def test_proven_nonmatch_of_overrides_allows_ttc_branch():
    result = _service().readiness(_context(driver=True, remote=False, avoidance=False))
    assert result["status"] == "READY"
    assert result["decision_stage"] == "TTC"
    assert result["required_inputs"] == [
        "relative_distance_m", "relative_speed_kph", "ttc_s",
    ]


def test_current_project_unknown_override_fact_routes_to_ttc_without_false_fact():
    context = _context()
    context["driver_in_vehicle"] = {
        "status": "UNAVAILABLE", "value": None,
        "source_type": "UNAVAILABLE", "source_ref": "",
    }
    result = _service().readiness(context)
    assert result["status"] == "READY"
    assert result["unknown_override_policy"] == "SKIP_TO_TTC"
    assert result["decision_stage"] == "TTC"
    assert result["unknown_policy_action"] == "SKIP_TO_TTC"
    assert result["override_states"]["driver_in_vehicle"] == "UNKNOWN"
    assert context["driver_in_vehicle"]["value"] is None


def test_later_positive_override_wins_after_an_earlier_unknown_rule():
    context = _context(remote=True)
    context["driver_in_vehicle"] = {"status": "UNAVAILABLE", "value": None}
    result = _service().readiness(context)
    assert result["status"] == "READY"
    assert result["matched_rule_id"] == "external_evasion_available"


def test_block_and_skip_policies_are_distinct_fixture_behaviors():
    context = _context(driver=True, remote=False, avoidance=False)
    context["driver_in_vehicle"] = {"status": "UNAVAILABLE", "value": None}
    blocked = _with_policy(UnknownOverridePolicy.BLOCK_TTC).readiness(context)
    assert blocked["status"] == "PENDING_INPUT"
    assert blocked["unknown_policy_action"] == "BLOCK_TTC"
    assert blocked["decision_stage"] == "OVERRIDE"

    skipped = _with_policy(UnknownOverridePolicy.SKIP_TO_TTC).readiness(context)
    assert skipped["status"] == "READY"
    assert skipped["unknown_policy_action"] == "SKIP_TO_TTC"
    assert skipped["decision_stage"] == "TTC"


def test_conflicts_in_each_override_fact_never_select_a_controllability_branch():
    for field in (
        "driver_in_vehicle", "remote_intervention_available",
        "other_road_user_avoidance_possible",
    ):
        context = _context()
        context[field]["status"] = "CONFLICT"
        result = _service().readiness(context)
        assert result["status"] == "FACT_SOURCE_CONFLICT"


def test_ttc_branch_reports_missing_and_non_closing_inputs_without_speed_fallbacks():
    unavailable = _context(driver=True, remote=False, avoidance=False)
    unavailable["ttc_s"] = {"status": "UNAVAILABLE", "value": None}
    assert _service().readiness(unavailable)["missing_inputs"] == [
        "relative_distance_m", "relative_speed_kph",
    ]

    non_closing = _context(driver=True, remote=False, avoidance=False)
    non_closing["ttc_s"] = {"status": "TTC_NOT_CLOSING", "value": None}
    assert _service().readiness(non_closing)["missing_inputs"] == ["TTC_NOT_CLOSING"]


def test_selected_profile_thresholds_and_source_roles_are_auditable():
    payload = _service().audit({"scenario_candidate": [], "scenario_feasibility": []})
    tree = payload["confirmed_method_decision_tree"]
    assert tree["selected_profile"] == "iav_avp_v1"
    assert [(item["upper_ttc_s"], item["result"]) for item in tree["ttc_thresholds"]] == [
        (3.0, "C3"), (4.0, "C2"), (5.0, "C1"), (None, "C0"),
    ]
    assert tree["unknown_override_policy"] == "SKIP_TO_TTC"
    assert tree["policy_source_ref"]["source"] == "normalized/project_analysis_policy.yaml"
    assert tree["policy_absence_evidence"] is None
    assert tree["implicit_python_default"] is False
    assert payload["runtime_comparison"] == {
        "runtime_contract_alignment": "MATCH",
        "method_semantic_completeness": "COMPLETE",
    }
    assert payload["source_authority_hierarchy"][0]["selected"] is True
    assert payload["source_authority_hierarchy"][2]["runtime_consumer"] == "NONE"


def test_r3_projection_uses_current_project_ttc_route_without_scoring():
    records = ReviewArtifactReader(
        "hara-c9f-validation-r3", ROOT / "runtime/review",
    ).read_all()
    summary = _service().audit(records)["summary"]
    assert summary["causal_relevant_hazardous_events"] == 54
    assert summary["c_method_branch_unresolved"] == 0
    assert summary["eligible_for_ttc"] == 54
    assert summary["c_ready"] == 0
    assert summary["c_pending_input"] == 54


def test_r3_policy_fixture_projections_keep_method_and_input_gaps_distinct():
    records = ReviewArtifactReader(
        "hara-c9f-validation-r3", ROOT / "runtime/review",
    ).read_all()
    fixtures = _service().audit(records)["policy_fixtures"]
    assert fixtures["UNSPECIFIED"]["summary"]["c_method_branch_unresolved"] == 54
    assert fixtures["BLOCK_TTC"]["summary"]["c_pending_input"] == 54
    assert fixtures["BLOCK_TTC"]["summary"]["eligible_for_ttc"] == 0
    assert fixtures["SKIP_TO_TTC"]["summary"]["c_pending_input"] == 54
    assert fixtures["SKIP_TO_TTC"]["summary"]["eligible_for_ttc"] == 54
