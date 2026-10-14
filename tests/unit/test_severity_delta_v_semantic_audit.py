from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from hara_agent.contracts import (
    SeverityInputAuthorityStatus, SeveritySemanticResolution, SpeedSemantic,
)
from hara_agent.method_sources import YamlBaselineCompiler
from hara_agent.services.analysis import SeverityDeltaVSemanticAuditService
from hara_agent.template import TemplateRoleCompiler
from hara_agent.workflow import ReviewArtifactReader


ROOT = Path(__file__).resolve().parents[2]


def _method():
    report = TemplateRoleCompiler().compile_method(
        ROOT / "references/HARA_Template_AI_20260327.xlsx"
    ).report_contract
    return YamlBaselineCompiler().compile(
        ROOT / "method_assets/fusa_baseline_v1/manifest.yaml",
        report_contract=report,
    )


def test_compiler_preserves_confirmed_relative_velocity_source_roles():
    semantic = _method().structured_risk_method.severity.semantic

    assert semantic.source_status == "CONFIRMED"
    assert semantic.source_named_semantic == "RELATIVE_VELOCITY"
    assert semantic.compiled_semantic is SpeedSemantic.RELATIVE_SPEED
    assert semantic.semantic_resolution is SeveritySemanticResolution.CONFIRMED_RELATIVE_VELOCITY
    assert [item.field_name for item in semantic.source_elements] == [
        "severity_primary_method", "s_by_relative_velocity",
        "s_by_relative_velocity.aeb_delta_v", "ΔV threshold annotations",
    ]
    assert semantic.source_elements[0].authority.value == "PRIMARY"
    assert semantic.source_elements[1].authority.value == "PRIMARY"
    assert semantic.source_elements[2].authority.value == "SUPPORTING"
    assert semantic.source_elements[3].authority.value == "NON_NORMATIVE"


def test_direct_source_grounded_relative_speed_is_ready():
    authority = SeverityDeltaVSemanticAuditService(_method()).input_authority({
        "facts": {"relative_speed_kph": 7.0},
        "fact_provenance": {"relative_speed_kph": {
            "provenance": "PROJECT_INPUT",
            "approval": "FINALIZED",
            "source_refs": [{"source_id": "ItemDef.docx", "location": "p12"}],
        }},
    })

    assert authority.status is SeverityInputAuthorityStatus.METHOD_CONFIRMED_INPUT_DIRECT
    assert authority.value == 7.0


def test_other_speed_facts_are_not_substituted_for_relative_speed():
    service = SeverityDeltaVSemanticAuditService(_method())
    for key in ("ego_speed_kph", "impact_speed_kph", "delta_v_kph"):
        authority = service.input_authority({"facts": {key: 20.0}})
        assert authority.value is None
        assert authority.status is SeverityInputAuthorityStatus.INPUT_MISSING

    odd_only = service.input_authority({
        "facts": {"ego_speed_constraint": {"speed_max_kph": 20.0}},
    })
    assert odd_only.value is None
    assert odd_only.status is SeverityInputAuthorityStatus.INPUT_MISSING


def test_confirmed_method_without_relative_speed_is_pending_input():
    payload = SeverityDeltaVSemanticAuditService(_method()).generate({
        "scenario_candidate": [{"scenario_id": "SC-1", "facts": {}}],
        "scenario_feasibility": [{
            "malfunction_id": "MF-1", "scenario_id": "SC-1", "status": "FINALIZED",
            "physically_feasible": True, "functionally_relevant": True,
            "causally_relevant": True, "hazardous_event": "event",
        }],
    })

    assert payload["r3_severity_readiness"][0]["severity_input_status"] == "PENDING_INPUT"


def test_relative_speed_without_provenance_is_not_ready():
    payload = SeverityDeltaVSemanticAuditService(_method()).generate({
        "scenario_candidate": [{"scenario_id": "SC-1", "facts": {"relative_speed_kph": 8.0}}],
        "scenario_feasibility": [{
            "malfunction_id": "MF-1", "scenario_id": "SC-1", "status": "FINALIZED",
            "physically_feasible": True, "functionally_relevant": True,
            "causally_relevant": True, "hazardous_event": "event",
        }],
    })

    record = payload["r3_severity_readiness"][0]
    assert record["severity_input_status"] == "PENDING_INPUT"
    assert record["available"]["relative_speed"] is True
    assert record["available"]["delta_v"] is False


def test_r3_audit_has_no_speed_substitution_or_delta_v_derivation():
    records = ReviewArtifactReader(
        "hara-c9f-validation-r3", ROOT / "runtime/review",
    ).read_all()
    payload = SeverityDeltaVSemanticAuditService(_method()).generate(records)

    assert payload["runtime_yaml_read"] == 0
    assert payload["scenario_physics_audit"]["derives_delta_v"] is False
    assert payload["summary"]["causal_relevant_hazardous_events"] == 54
    assert payload["summary"]["direct_delta_v_available"] == 0
    assert payload["summary"]["PENDING_METHOD_SEMANTICS"] == 0
    assert payload["summary"]["PENDING_INPUT"] == 54
    assert payload["collision_context_requirements"]["non_collision_branch_present"] is False


def test_approved_source_conflict_prevents_any_severity_input_selection():
    method = _method()
    severity = method.structured_risk_method.severity
    semantic = replace(
        severity.semantic,
        compiled_semantic=SpeedSemantic.UNRESOLVED,
        semantic_resolution=SeveritySemanticResolution.APPROVED_SOURCE_INTERNAL_CONFLICT,
    )
    conflicted = replace(
        method,
        structured_risk_method=replace(
            method.structured_risk_method,
            severity=replace(severity, speed_semantic=SpeedSemantic.UNRESOLVED, semantic=semantic),
        ),
    )

    authority = SeverityDeltaVSemanticAuditService(conflicted).input_authority({
        "facts": {"relative_speed_kph": 8.0, "delta_v_kph": 8.0},
    })

    assert authority.status is SeverityInputAuthorityStatus.METHOD_SEMANTIC_AMBIGUITY
