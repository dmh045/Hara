from __future__ import annotations

import copy
from pathlib import Path

import pytest

from hara_agent.method_sources import YamlBaselineCompiler
from hara_agent.services.analysis import RiskContextSourceCoverageAuditService
from hara_agent.services.extraction.document_reader import DocumentArtifact, DocumentBlock
from hara_agent.template import TemplateRoleCompiler


ROOT = Path(__file__).resolve().parents[2]
FIELDS = (
    "road_user_type", "collision_type", "ego_speed_kph", "object_speed_kph",
    "relative_speed_kph", "impact_speed_kph", "relative_distance_m", "ttc_s",
    "driver_in_vehicle", "remote_intervention_available",
    "other_road_user_avoidance_possible", "direct_control_available",
    "vehicle_stability", "emergency_braking_available", "function_type",
    "has_remote_app",
)


@pytest.fixture(scope="module")
def service() -> RiskContextSourceCoverageAuditService:
    report = TemplateRoleCompiler().compile_method(
        ROOT / "references/HARA_Template_AI_20260327.xlsx",
    ).report_contract
    method = YamlBaselineCompiler().compile(
        ROOT / "method_assets/fusa_baseline_v1/manifest.yaml",
        report_contract=report,
    )
    return RiskContextSourceCoverageAuditService(method)


def _document() -> DocumentArtifact:
    return DocumentArtifact(
        source_path=Path("ItemDef.docx"), source_id="ItemDef.docx", blocks=[
            DocumentBlock("P-0001", "paragraph", "paragraph[1]", "驾驶员干预且车速≤20km/h"),
            DocumentBlock("P-0002", "paragraph", "paragraph[2]", "前向碰撞的事件构型"),
            DocumentBlock("T-001-R-0001", "table_row", "table[1].row[1]", "障碍物距离1m~4m"),
        ],
    )


def _context(*, relative_available: bool = False) -> dict[str, dict[str, object]]:
    result = {
        field: {
            "status": "UNAVAILABLE", "value": None, "source_type": "UNAVAILABLE",
            "source_ref": "", "reason": "MISSING_STRUCTURED_FACT",
        }
        for field in FIELDS
    }
    if relative_available:
        result["relative_speed_kph"] = {
            "status": "AVAILABLE", "value": 7.0,
            "source_type": "DIRECT_SCENARIO_FACT", "source_ref": "ItemDef.docx:paragraph[1]",
            "reason": "SOURCE_GROUNDED_STRUCTURED_FACT",
        }
    return result


def _state(*, direct_relative_speed: bool = False) -> dict[str, object]:
    facts: dict[str, object] = {
        "ego_speed_constraint": {"speed_min_kph": 0, "speed_max_kph": 20, "unit": "km/h"},
    }
    provenance: dict[str, object] = {}
    if direct_relative_speed:
        facts["relative_speed_kph"] = 7.0
        provenance["relative_speed_kph"] = {
            "approval": "FINALIZED", "provenance": "SCENARIO_INPUT",
            "source_refs": [{"source_id": "ItemDef.docx", "location": "paragraph[1]"}],
        }
    return {
        "run_id": "p3a-fixture",
        "item_definition": {"typed": {
            "speed_envelopes": [{"speed_max_kph": 20, "status": "FINALIZED"}],
            "risk_facts": [{
                "fact_id": "RF-1", "parameter": "COLLISION_TYPE", "value": "REAR_END",
                "approval": "FINALIZED",
                "context": {"malfunction_id": "MF-1", "scenario_id": "SCN-1"},
                "source_refs": [{"source_id": "ItemDef.docx", "location": "paragraph[2]"}],
            }],
        }},
        "scenarios": [{"scenario_id": "SCN-1", "facts": facts, "fact_provenance": provenance}],
        "audit_trail": [{
            "event": "structured_risk_scoring_completed",
            "risk_fact_binding_audits": [{
                "malfunction_id": "MF-1", "scenario_id": "SCN-1",
                "bound_fact_types": [], "missing_fact_types": ["EXPOSURE"],
                "automatic_binding_count": 1, "explicit_binding_count": 0,
            }],
        }],
    }


def _trace(*, relative_available: bool = False) -> dict[str, object]:
    return {"assessments": [{
        "malfunction_id": "MF-1", "scenario_id": "SCN-1", "hazardous_event_id": "HE-1",
        "risk_scoring_invoked": True,
        "hazardous_event_risk_context": _context(relative_available=relative_available),
    }]}


def _field(payload: dict[str, object], name: str) -> dict[str, object]:
    return next(item for item in payload["per_field"] if item["field"] == name)


def test_existing_finalized_collision_fact_is_separated_from_true_project_gaps(service):
    state = _state()
    original = copy.deepcopy(state)
    payload, clarification = service.audit(
        state=state, trace=_trace(), document=_document(),
    )

    collision = _field(payload, "collision_type")
    assert collision["classification_counts"] == {"SOURCE_PRESENT_BUT_NOT_BOUND": 1}
    assert collision["examples"][0]["secondary_classifications"] == ["RUNTIME_BINDING_DEFECT"]
    assert clarification["ec03_split"]["existing_but_not_bound"][0]["project_risk_fact_ids"] == ["RF-1"]
    assert state == original
    assert payload["audit_scope"]["provider_calls"] == 0
    assert payload["audit_scope"]["risk_context_writeback"] is False


def test_speed_envelope_is_not_promoted_to_relative_speed_or_ttc(service):
    payload, _ = service.audit(state=_state(), trace=_trace(), document=_document())

    relative = _field(payload, "relative_speed_kph")
    assert relative["classification_counts"] == {"TRUE_PROJECT_FACT_GAP": 1}
    assert relative["source_coverage"]["C_scenario_fields"]["direct_canonical_field_count"] == 0
    ttc = _field(payload, "ttc_s")
    assert ttc["classification_counts"] == {"TRUE_PROJECT_FACT_GAP": 1}
    assert ttc["source_coverage"]["G_deterministic_derivation"]["status"].startswith("PRESENT_AND_WIRED")


def test_generic_itemdef_citation_is_not_accepted_as_collision_evidence(service):
    state = _state()
    state["item_definition"]["typed"]["risk_facts"][0]["source_refs"][0]["location"] = "paragraph[1]"
    payload, clarification = service.audit(
        state=state, trace=_trace(), document=_document(),
    )

    collision = _field(payload, "collision_type")
    assert collision["classification_counts"] == {"SOURCE_PRESENT_BUT_AMBIGUOUS": 1}
    assert clarification["ec03_split"]["existing_but_not_bound"] == []
    assert clarification["rejected_typed_source_records"][0]["project_risk_fact_source_grounding"] == (
        "CITED_ITEM_BLOCK_DOES_NOT_SUPPORT_FIELD_SEMANTICS"
    )


def test_direct_finalized_scenario_fact_is_reported_as_bound(service):
    payload, _ = service.audit(
        state=_state(direct_relative_speed=True),
        trace=_trace(relative_available=True), document=_document(),
    )

    relative = _field(payload, "relative_speed_kph")
    assert relative["classification_counts"] == {"SOURCE_PRESENT_BOUND": 1}
    assert relative["recoverability_counts"] == {"AUTO_RECOVERABLE": 1}
