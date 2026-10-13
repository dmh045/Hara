from __future__ import annotations

from pathlib import Path

from hara_agent.method_sources import YamlBaselineCompiler
from hara_agent.services.analysis import ExposureBindingAuditService
from hara_agent.template import TemplateRoleCompiler


ROOT = Path(__file__).resolve().parents[2]


def _method():
    report = TemplateRoleCompiler().compile_method(
        ROOT / "references/HARA_Template_AI_20260327.xlsx"
    ).report_contract
    return YamlBaselineCompiler().compile(
        ROOT / "method_assets/fusa_baseline_v1/manifest.yaml",
        report_contract=report,
    )


def _candidate() -> dict:
    return {
        "scenario_id": "SC-1",
        "method_source_hash": "method-hash",
        "generated_for_malfunction_ids": ["MF-1"],
        "facts": {"scenario_atom_ids": []},
        "context_resolution": {"dimension_bindings": {
            "EGO_DYNAMICS": {
                "project_value": "0..20 km/h",
                "resolution_status": "PENDING",
                "unresolved_reason": "SPEED_CONSTRAINT_NOT_CONCRETE",
                "speed_constraint": {"min_kph": 0, "max_kph": 20, "unit": "km/h"},
            },
        }},
    }


def test_audit_keeps_range_ambiguity_without_changing_current_candidate():
    payload = ExposureBindingAuditService(_method()).generate({
        "scenario_candidate": [_candidate()],
        "malfunction": [{"malfunction_id": "MF-1", "component_category": "computing"}],
    }, {"scenario_binding_coverage": {"dimensions": {
        "ROAD": {"context_values": [{
            "source_value": "clear", "status": "PENDING",
            "reason": "NO_EXPLICIT_METHOD_ALIAS", "candidate_method_atoms": [],
        }]},
    }}})

    speed = next(item for item in payload["scenario_term_inventory"] if item["scenario_id"] == "SC-1")
    assert speed["range_resolution_preview"]["reason"] == "MULTI_ATOM_RANGE_AMBIGUITY"
    assert speed["range_resolution_preview"]["atom_id"] == ""
    assert payload["scenario_readiness"][0]["scenario_atom_ids"] == []
    assert payload["scenario_readiness"][0]["after_range_preview_atom_ids"] == []
    assert payload["scenario_readiness"][0]["coverage_status"] == "PENDING"
    assert payload["scenario_readiness"][0]["exposure_input_status"] == "NOT_EVALUABLE"
    assert payload["component_domain_summary"]["resolved_malfunctions"] == 1
    assert payload["runtime_yaml_read"] == 0
    road = next(item for item in payload["scenario_term_inventory"] if item["source_term"] == "clear")
    assert road["audit_classification"] == "PENDING_NO_METHOD_ATOM"


def test_audit_never_defaults_an_unknown_component_to_a_domain():
    payload = ExposureBindingAuditService(_method()).generate({
        "scenario_candidate": [_candidate()],
        "malfunction": [{"malfunction_id": "MF-1", "component_category": "unknown"}],
    }, {})

    assert payload["component_domain_summary"]["pending_malfunctions"] == 1
    assert payload["scenario_readiness"][0]["component_domain_readiness"] == "PENDING_COMPONENT_DOMAIN"
    assert payload["scenario_readiness"][0]["exposure_input_status"] == "NOT_EVALUABLE"


def test_where_terms_distinguish_location_from_slot_geometry_context():
    payload = ExposureBindingAuditService(_method()).generate({}, {
        "scenario_binding_coverage": {"dimensions": {"WHERE": {"context_values": [
            {"source_value": "室外停车场", "status": "PENDING", "reason": "NO_EXPLICIT_METHOD_ALIAS", "candidate_method_atoms": ["FO010"]},
            {"source_value": "垂直车位", "status": "PENDING", "reason": "NO_EXPLICIT_METHOD_ALIAS", "candidate_method_atoms": ["FO010"]},
        ]}}},
    })

    location, slot = payload["scenario_term_inventory"]
    assert location["audit_classification"] == "PENDING_PROJECT_BINDING"
    assert location["candidate_method_dimensions"] == ["WHERE"]
    assert slot["audit_classification"] == "PROJECT_CONTEXT_NO_METHOD_DIMENSION"
    assert slot["candidate_method_dimensions"] == []
    assert slot["candidate_atoms"] == []
    assert payload["project_binding_summary"] == {
        "PENDING_PROJECT_BINDING": 1,
        "PROJECT_CONTEXT_NO_METHOD_DIMENSION": 1,
    }


def test_coverage_resolved_reports_partial_when_a_required_binding_is_missing():
    candidate = _candidate()
    candidate["facts"]["scenario_atom_ids"] = ["FA001"]
    candidate["context_resolution"]["dimension_bindings"]["EGO_DYNAMICS"]["resolution_status"] = "RESOLVED"
    candidate["context_resolution"]["dimension_bindings"]["OBJECT"] = {
        "project_value": "pedestrian",
        "resolution_status": "PENDING",
        "unresolved_reason": "NO_EXPLICIT_METHOD_ALIAS",
    }
    payload = ExposureBindingAuditService(_method()).generate({
        "scenario_candidate": [candidate],
        "malfunction": [{"malfunction_id": "MF-1", "component_category": "computing"}],
    }, {"coverage_governance": {
        "status": "RESOLVED", "required_dimensions": ["EGO_DYNAMICS", "OBJECT"],
    }})

    readiness = payload["scenario_readiness"][0]
    assert readiness["coverage_status"] == "RESOLVED"
    assert readiness["exposure_input_status"] == "PARTIAL"


def test_coverage_resolved_reports_ready_when_all_required_atoms_are_bound():
    candidate = _candidate()
    candidate["facts"]["scenario_atom_ids"] = ["FA001"]
    candidate["context_resolution"]["dimension_bindings"]["EGO_DYNAMICS"]["resolution_status"] = "RESOLVED"
    payload = ExposureBindingAuditService(_method()).generate({
        "scenario_candidate": [candidate],
        "malfunction": [{"malfunction_id": "MF-1", "component_category": "computing"}],
    }, {"coverage_governance": {
        "status": "RESOLVED", "required_dimensions": ["EGO_DYNAMICS"],
    }})

    readiness = payload["scenario_readiness"][0]
    assert readiness["coverage_status"] == "RESOLVED"
    assert readiness["exposure_input_status"] == "READY"
