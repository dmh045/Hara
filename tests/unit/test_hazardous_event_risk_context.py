from __future__ import annotations

from pathlib import Path

from hara_agent.contracts import RiskContextFactStatus
from hara_agent.method_sources import YamlBaselineCompiler
from hara_agent.services.analysis import HazardousEventRiskContextService
from hara_agent.services.analysis.scenario_physics import derive_scenario_physics
from hara_agent.models import ReviewStatus, ScenarioCandidate
from hara_agent.template import TemplateRoleCompiler
from hara_agent.workflow import ReviewArtifactReader


ROOT = Path(__file__).resolve().parents[2]


def _service() -> HazardousEventRiskContextService:
    report = TemplateRoleCompiler().compile_method(
        ROOT / "references/HARA_Template_AI_20260327.xlsx"
    ).report_contract
    method = YamlBaselineCompiler().compile(
        ROOT / "method_assets/fusa_baseline_v1/manifest.yaml",
        report_contract=report,
    )
    return HazardousEventRiskContextService(method)


def _scenario(**overrides):
    facts = {
        "road_user_type": "VEHICLE", "collision_type": "FRONTAL",
        "relative_speed_kph": 8.0, "relative_distance_m": 10.0,
        "ttc_s": 4.5, "driver_in_vehicle": False,
        "remote_intervention_available": False,
        "other_road_user_avoidance_possible": False,
        "hazardous_event": "prose must never be used for risk inputs",
    }
    provenance = {
        key: {
            "provenance": "PROJECT_INPUT", "approval": "FINALIZED",
            "source_refs": [{"source_id": "ItemDef.docx", "location": f"facts.{key}"}],
        }
        for key in facts if key != "hazardous_event"
    }
    facts.update(overrides)
    return {**facts, "_fact_provenance": provenance}


def test_context_uses_canonical_node_identity_and_source_grounded_facts_only():
    service = _service()
    context = service.build(
        malfunction_id="MF-1", scenario_id="SCN-1", hazard_node_id="SCN-1:MF-1:hazard",
        scenario=_scenario(),
    )

    assert context.hazardous_event_id == "HE::MF-1::SCN-1::SCN-1:MF-1:hazard"
    assert context.relative_speed_kph.status is RiskContextFactStatus.AVAILABLE
    assert context.relative_speed_kph.value == 8.0
    assert context.relative_speed_kph.source_provenance == "PROJECT_INPUT"
    assert service.severity_readiness(context) == {
        "status": "READY", "missing_reasons": [],
        "selected_speed_semantic": "RELATIVE_SPEED",
        "collision_type_requirement": "REQUIRED",
    }
    controllability = service.controllability_readiness(context)
    assert controllability["status"] == "READY"
    assert controllability["branch"] == "OVERRIDE"
    assert controllability["rule_id"] == "driver_outside_no_intervention"


def test_prose_never_backfills_missing_collision_or_road_user_context():
    service = _service()
    scenario = _scenario()
    for key in ("road_user_type", "collision_type"):
        scenario.pop(key)
        scenario["_fact_provenance"].pop(key)
    scenario["hazardous_event"] = "vehicle collides with a pedestrian from the side"
    context = service.build(
        malfunction_id="MF-1", scenario_id="SCN-1", hazard_node_id="H", scenario=scenario,
    )

    assert context.road_user_type.status is RiskContextFactStatus.UNAVAILABLE
    assert context.collision_type.status is RiskContextFactStatus.UNAVAILABLE
    readiness = service.severity_readiness(context)
    assert readiness["missing_reasons"] == ["MISSING_ROAD_USER_TYPE"]


def test_odd_speed_envelope_and_ego_speed_do_not_become_relative_speed():
    service = _service()
    scenario = _scenario()
    scenario.pop("relative_speed_kph")
    scenario["ego_speed_kph"] = 20.0
    scenario["ego_speed_constraint"] = {"min_kph": 0.0, "max_kph": 20.0}
    scenario["_fact_provenance"].pop("relative_speed_kph")
    scenario["_fact_provenance"]["ego_speed_kph"] = {
        "provenance": "SCENARIO_INPUT", "approval": "FINALIZED",
        "source_refs": [{"source_id": "ItemDef.docx", "location": "speed"}],
    }

    context = service.build(
        malfunction_id="MF-1", scenario_id="SCN-1", hazard_node_id="H", scenario=scenario,
    )
    assert context.relative_speed_kph.status is RiskContextFactStatus.UNAVAILABLE
    assert "MISSING_RELATIVE_SPEED" in service.severity_readiness(context)["missing_reasons"]


def test_vehicle_branch_reports_missing_collision_type_with_grounded_speed():
    service = _service()
    scenario = _scenario()
    scenario.pop("collision_type")
    scenario["_fact_provenance"].pop("collision_type")
    context = service.build(
        malfunction_id="MF-1", scenario_id="SCN-1", hazard_node_id="H", scenario=scenario,
    )
    assert service.severity_readiness(context)["missing_reasons"] == ["MISSING_COLLISION_TYPE"]


def test_vru_branch_does_not_require_collision_type():
    service = _service()
    scenario = _scenario(road_user_type="PEDESTRIAN")
    scenario.pop("collision_type")
    scenario["_fact_provenance"].pop("collision_type")
    context = service.build(
        malfunction_id="MF-1", scenario_id="SCN-1", hazard_node_id="H", scenario=scenario,
    )

    readiness = service.severity_readiness(context)
    assert readiness["status"] == "READY"
    assert readiness["collision_type_requirement"] == "NOT_REQUIRED_BY_METHOD"


def test_non_closing_relative_speed_is_not_reported_as_generic_ttc_missing():
    service = _service()
    scenario = _scenario(relative_speed_kph=0.0)
    scenario.pop("ttc_s")
    scenario["_fact_provenance"].pop("ttc_s")
    context = service.build(
        malfunction_id="MF-1", scenario_id="SCN-1", hazard_node_id="H", scenario=scenario,
    )

    assert context.ttc_s.status is RiskContextFactStatus.TTC_NOT_CLOSING
    readiness = service.controllability_readiness(context)
    assert readiness["status"] == "READY"  # approved override decides before TTC


def test_scoring_facts_drop_unavailable_risk_inputs_instead_of_falling_back():
    service = _service()
    scenario = _scenario()
    scenario.pop("relative_speed_kph")
    scenario["_fact_provenance"].pop("relative_speed_kph")
    scenario["ego_speed_constraint"] = {"speed_max_kph": 20.0}
    context = service.build(
        malfunction_id="MF-1", scenario_id="SCN-1", hazard_node_id="H", scenario=scenario,
    )
    values = service.scoring_facts(context, scenario)

    assert "relative_speed_kph" not in values
    assert values["ego_speed_constraint"]["speed_max_kph"] == 20.0


def test_relative_distance_m_keeps_finalized_provenance_for_deterministic_ttc():
    candidate = ScenarioCandidate(
        scenario_id="SCN-1", operating_scenario="parking", situational_description="",
        situational_detailing="", status=ReviewStatus.FINALIZED,
        facts={"relative_distance_m": 10.0, "closing_speed_kph": 8.0},
        fact_provenance={
            key: {
                "approval": "FINALIZED",
                "source_refs": [{
                    "source_type": "ITEM_DOCUMENT", "source_id": "ItemDef.docx",
                    "location": key,
                }],
            }
            for key in ("relative_distance_m", "closing_speed_kph")
        },
    )

    ttc = next(item for item in derive_scenario_physics(candidate) if item.evidence_ref == "DERIVED.ttc_s")
    assert ttc.approval_status is ReviewStatus.FINALIZED
    assert ttc.metadata["inputs"] == ["SCN.relative_distance_m", "SCN.closing_speed_kph"]


def test_audit_materializes_the_same_source_grounded_ttc_as_runtime_scoring():
    records = {
        "scenario_candidate": [{
            "scenario_id": "SCN-1", "operating_scenario": "parking",
            "situational_description": "", "situational_detailing": "",
            "status": "FINALIZED",
            "facts": {"relative_distance_m": 10.0, "closing_speed_kph": 8.0},
            "fact_provenance": {
                key: {
                    "approval": "FINALIZED",
                    "source_refs": [{
                        "source_type": "ITEM_DOCUMENT", "source_id": "ItemDef.docx",
                        "location": key,
                    }],
                }
                for key in ("relative_distance_m", "closing_speed_kph")
            },
        }],
        "scenario_feasibility": [{
            "malfunction_id": "MF-1", "scenario_id": "SCN-1", "status": "FINALIZED",
            "physically_feasible": True, "functionally_relevant": True,
            "causally_relevant": True,
            "causal_assessment": {"causal_chain": ["M", "B", "I", "H"]},
        }],
    }

    context = _service().audit(records)["r3_risk_context"][0]["risk_context"]
    assert context["ttc_s"]["value"] == 4.5
    assert context["ttc_s"]["source_type"] == "DERIVED_PHYSICS"


def test_conflicting_project_and_scenario_risk_facts_fail_closed():
    try:
        _service().validate_source_conflicts(
            {"relative_speed_kph": 5.0}, {"relative_speed_kph": 8.0},
        )
    except ValueError as exc:
        assert str(exc).startswith("FACT_SOURCE_CONFLICT:")
    else:
        raise AssertionError("conflicting risk facts must fail closed")


def test_method_bound_project_fact_is_not_misclassified_as_physics():
    scenario = _scenario()
    scenario["_fact_provenance"]["relative_speed_kph"] = {
        "provenance": "DERIVED", "approval": "FINALIZED",
        "source_binding_kind": "METHOD_RISK_FACT_BINDING",
        "method_contract_hash": "method-hash",
        "source_refs": [{"source_id": "ItemDef.docx", "location": "risk_fact"}],
    }
    context = _service().build(
        malfunction_id="MF-1", scenario_id="SCN-1", hazard_node_id="H", scenario=scenario,
    )
    assert context.relative_speed_kph.source_type.value == "DIRECT_PROJECT_FACT"


def test_r3_audit_projects_stable_ids_and_pending_structured_inputs_without_prose_parse():
    records = ReviewArtifactReader(
        "hara-c9f-validation-r3", ROOT / "runtime/review",
    ).read_all()
    payload = _service().audit(records)

    assert payload["runtime_yaml_read"] == 0
    assert payload["hazardous_event_identity"]["text_used_as_identity"] is False
    assert payload["risk_context_substrate"]["hazardous_event_prose_consumed"] is False
    assert payload["summary"]["causal_relevant_hazardous_events"] == 54
    assert payload["summary"]["severity_ready"] == 0
    assert payload["summary"]["severity_pending_input"] == 54
    assert all(item["hazardous_event_id"] for item in payload["r3_risk_context"])
