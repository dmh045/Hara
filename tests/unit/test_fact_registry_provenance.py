import pytest

from hara_agent.models import (
    EvidenceKind, FactProvenance, MalfunctionCandidate, ReviewStatus,
    ScenarioCandidate, SourceRef,
)
from hara_agent.services.semantic.scenario_evidence import build_fact_registry
from hara_agent.services.semantic.scenario_batching import build_scenario_user_prompt


def _malfunction() -> MalfunctionCandidate:
    return MalfunctionCandidate(
        "MF-1", "FUN-1", "loss", "function lost", "output lost",
        "vehicle movement unavailable", ["function lost", "output lost"],
    )


def test_scenario_fact_inherits_project_provenance_approval_and_source():
    source = SourceRef("docx", "ItemDef.docx", "table:parking-speed")
    scenario = ScenarioCandidate(
        "SCN-1", "parking", "parking", "low speed parking",
        facts={"ego_speed_kph": 5},
        fact_provenance={
            "ego_speed_kph": {
                "provenance": "PROJECT_INPUT",
                "approval": "PENDING",
                "source_refs": [source],
                "fallback_used": False,
                "resolution_source": "mode_specific_project_fact",
            },
        },
    )
    record = build_fact_registry(_malfunction(), scenario).resolve_record(
        "SCN.ego_speed_kph"
    )
    assert record is not None
    assert record.kind is EvidenceKind.DIRECT_FACT
    assert record.provenance is FactProvenance.PROJECT_INPUT
    assert record.approval_status is ReviewStatus.PENDING
    assert record.source_refs == (source,)
    assert record.metadata["fallback_used"] is False


def test_removed_legacy_provenance_is_rejected():
    source = SourceRef("domain_profile", "avp.yaml", "risk_candidates")
    scenario = ScenarioCandidate(
        "SCN-LEGACY", "parking", "parking", "candidate",
        facts={"relative_distance": "5 m"},
        fact_provenance={
            "relative_distance": {
                "provenance": "LEGACY_MIGRATION",
                "approval": "PENDING",
                "source_refs": [source],
            },
        },
    )
    with pytest.raises(ValueError, match="LEGACY_MIGRATION"):
        build_fact_registry(_malfunction(), scenario)


def test_ttc_is_derived_physics_with_canonical_inputs():
    scenario = ScenarioCandidate(
        "SCN-TTC", "parking", "parking", "closing target",
        facts={"relative_distance": "10 m", "closing_speed_kph": 18},
    )
    record = build_fact_registry(_malfunction(), scenario).resolve_record("DERIVED.ttc_s")
    assert record is not None
    assert record.value == 2.0
    assert record.kind is EvidenceKind.DERIVED_PHYSICS
    assert record.provenance is FactProvenance.DERIVED
    assert record.metadata == {
        "derivation_type": "TTC",
        "inputs": ["SCN.relative_distance", "SCN.closing_speed_kph"],
        "formula_identity": "relative_distance_m / (closing_speed_kph / 3.6)",
    }


def test_internal_bindings_and_blank_values_are_not_executable_scenario_facts():
    scenario = ScenarioCandidate(
        "SCN-INTERNAL", "parking", "parking", "candidate",
        facts={
            "ego_speed_kph": 5,
            "road_surface_conditions": "",
            "method_scenario_dimensions": {
                "ROAD_SURFACE": {"binding_status": "MISSING"},
            },
        },
    )

    registry = build_fact_registry(_malfunction(), scenario)
    prompt = build_scenario_user_prompt(_malfunction(), [scenario])

    assert registry.resolve_record("SCN.ego_speed_kph") is not None
    assert registry.resolve_record("SCN.road_surface_conditions") is None
    assert registry.resolve_record("SCN.method_scenario_dimensions") is None
    assert '"ego_speed_kph":5' in prompt
    assert "road_surface_conditions" not in prompt
    assert "method_scenario_dimensions" not in prompt
