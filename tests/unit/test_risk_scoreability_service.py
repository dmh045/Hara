from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pytest

from hara_agent.contracts import (
    CausalBreakpoint, CausalEdge, CausalGraph, CausalNode, CausalNodeType,
    CausalRelation, EvidenceBinding, ScenarioCausalAssessment,
)
from hara_agent.cli import main
from hara_agent.method_sources import YamlBaselineCompiler
from hara_agent.models import EvidenceKind, ReviewStatus
from hara_agent.services.analysis import RiskScoreabilityService
from hara_agent.template import TemplateRoleCompiler


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def service() -> RiskScoreabilityService:
    report = TemplateRoleCompiler().compile_method(
        ROOT / "references/HARA_Template_AI_20260327.xlsx"
    ).report_contract
    method = YamlBaselineCompiler().compile(
        ROOT / "method_assets/fusa_baseline_v1/manifest.yaml",
        report_contract=report,
    )
    return RiskScoreabilityService(method)


def _accepted() -> dict:
    return {
        "provenance": "PROJECT_INPUT",
        "approval": "FINALIZED",
        "source_refs": [{
            "source_type": "ITEM_DOCUMENT", "source_id": "ItemDef.docx",
            "location": "risk-input",
        }],
    }


def _validated_analysis_setting(*, scenario_id: str) -> dict:
    return {
        "provenance": "SCENARIO_DEFINED",
        "origin": "SCENARIO_DEFINED",
        "approval": "FINALIZED",
        "validation_status": "VALIDATED",
        "source_refs": [{
            "source_type": "ENGINEERING_ANALYSIS_SETTING",
            "source_id": "ENG-SETTING-1",
            "location": "ASSUMPTION-1:ego_speed_kph",
        }],
        "analysis_assumption_origin": "SCENARIO_DEFINED",
        "analysis_assumption_scope": {
            "malfunction_id": "MF-1",
            "scenario_id": scenario_id,
        },
    }


def _noninterference() -> dict:
    return {
        "complete": True,
        "causal_evidence_fields": [],
        "physical_feasibility_fields": [],
        "scenario_identity_fields": [],
        "child_subset_refinement": "Validated analytical child is within the parent ODD envelope.",
    }


def _causal() -> dict:
    nodes = tuple(CausalNode(identity, kind, identity) for identity, kind in (
        ("M", CausalNodeType.MALFUNCTION), ("B", CausalNodeType.SYSTEM_BEHAVIOR_CHANGE),
        ("I", CausalNodeType.OPERATIONAL_CONSEQUENCE), ("H", CausalNodeType.HAZARD),
    ))
    edges = tuple(
        CausalEdge(edge_id, source, target, CausalRelation.CAUSES, edge_id, (edge_id,))
        for edge_id, source, target in (("M_B", "M", "B"), ("B_I", "B", "I"), ("I_H", "I", "H"))
    )
    return ScenarioCausalAssessment(
        "SC-1", CausalGraph(nodes, edges), ("M", "B", "I", "H"), CausalBreakpoint.NONE,
        tuple(EvidenceBinding(edge.edge_id, edge.evidence_refs, EvidenceKind.DIRECT_FACT,
                              status=ReviewStatus.FINALIZED) for edge in edges),
        (), (), "collision", review_status=ReviewStatus.FINALIZED,
    ).to_dict()


def _checkpoint() -> dict:
    return {
        "scenarios": [{
            "scenario_id": "SC-1",
            "facts": {"ego_speed_constraint": {"speed_max_kph": 20}},
            "fact_provenance": {},
        }],
        "malfunctions": [{
            "malfunction_id": "MF-1", "functional_effect": "lost braking",
        }],
        "item_definition": {"scenario_assessments": [{
            "malfunction_id": "MF-1", "scenario_id": "SC-1",
            "status": "FINALIZED", "physically_feasible": True,
            "functionally_relevant": True, "causally_relevant": True,
            "final_retain": True, "hazardous_event": "collision",
            "causal_assessment": _causal(),
        }]},
    }


def _supplement() -> dict:
    return {"source_run_id": "run-1", "analytical_options_pending_validation": [{
        "malfunction_id": "MF-1", "parent_scenario_id": "SC-1",
        "analysis_instance": {
            "source_template_id": "TEMPLATE-1", "source_option_id": "TEMPLATE-1:OPTION:1",
            "source_option_values": {
                "obj_type": "pedestrian", "obj_distance_m": 2.0,
                "obj_v_kph": 0.0, "collision_type": "side",
            },
        },
    }]}


def test_scoreability_never_promotes_odd_range_to_an_ego_point_speed(service):
    payload, queue = service.generate(
        checkpoint=_checkpoint(), supplement=_supplement(),
    )

    row = payload["records"][0]
    assert row["S"]["ego_speed"]["status"] == "MISSING"
    assert row["motion"]["status"] == "MISSING"
    assert "EGO_POINT_SPEED_ENGINEERING_ASSUMPTION_REQUIRED" in row["S"]["blockers"]
    assert row["C"]["override_state"] == "UNKNOWN"
    assert row["C"]["ttc_ready"] is False
    assert payload["provider_calls"] == queue["provider_calls"] == 0
    assert queue["total_options"] == queue["deduplicated_semantic_groups"] == 1
    assert queue["engineering_blocked_groups"] == 1
    pack = service.minimal_assumption_pack(
        checkpoint_sha256="checkpoint-hash",
        supplement=_supplement(), records=payload["records"],
    )
    assert pack["summary"]["groups"] == 1
    assert pack["summary"]["affected_options"] == 1
    assert {item["field"] for item in pack["groups"][0]["required_fields"]} == {
        "ego_speed_kph", "driver_in_vehicle", "remote_intervention_available",
        "other_road_user_avoidance_possible",
    }


def test_motion_derives_only_with_explicit_longitudinal_project_facts(service):
    facts = {
        "ego_speed_kph": 7.0,
        "ego_longitudinal_direction": "FORWARD",
        "object_longitudinal_direction": "STATIONARY",
        "_fact_provenance": {
            "ego_speed_kph": _accepted(),
            "ego_longitudinal_direction": _accepted(),
            "object_longitudinal_direction": _accepted(),
        },
    }
    collision = service.vocabulary.resolve(field="collision_type", raw_value="front")
    derived = service._motion_readiness(
        facts=facts, parent_scenario_id="SC-1", malfunction_id="MF-1",
        object_speed_kph=0.0, collision=collision,
    )
    assert derived["status"] == "DERIVED_PHYSICS"
    assert derived["value"] == 7.0
    assert derived["derivation_rule_id"]

    facts["_fact_provenance"].pop("ego_longitudinal_direction")
    missing_direction = service._motion_readiness(
        facts=facts, parent_scenario_id="SC-1", malfunction_id="MF-1",
        object_speed_kph=0.0, collision=collision,
    )
    assert missing_direction["reason"] == "LONGITUDINAL_DIRECTION_MISSING"

    side = service.vocabulary.resolve(field="collision_type", raw_value="side")
    no_lateral_shortcut = service._motion_readiness(
        facts={**facts, "_fact_provenance": {**facts["_fact_provenance"], "ego_longitudinal_direction": _accepted()}},
        parent_scenario_id="SC-1", malfunction_id="MF-1",
        object_speed_kph=0.0, collision=side,
    )
    assert no_lateral_shortcut["reason"] == "LATERAL_OR_UNMAPPED_COLLISION_PHYSICS_UNRESOLVED"


def test_motion_accepts_scoped_validated_engineering_analysis_setting(service):
    facts = {
        "ego_speed_kph": 7.0,
        "ego_longitudinal_direction": "FORWARD",
        "object_longitudinal_direction": "STATIONARY",
        "_fact_provenance": {
            "ego_speed_kph": _validated_analysis_setting(scenario_id="SCN-ANALYTICAL-1"),
            "ego_longitudinal_direction": _validated_analysis_setting(scenario_id="SCN-ANALYTICAL-1"),
            "object_longitudinal_direction": _validated_analysis_setting(scenario_id="SCN-ANALYTICAL-1"),
        },
    }
    collision = service.vocabulary.resolve(field="collision_type", raw_value="front")
    result = service._motion_readiness(
        facts=facts, parent_scenario_id="SCN-ANALYTICAL-1", malfunction_id="MF-1",
        object_speed_kph=0.0, collision=collision,
    )
    assert result["status"] == "DERIVED_PHYSICS"
    assert result["ego"]["status"] == "SCENARIO_DEFINED"
    assert result["ego"]["provenance"]["origin"] == "SCENARIO_DEFINED"


def test_controllability_uses_exact_overrides_before_ttc(service):
    facts = {
        "driver_in_vehicle": False,
        "remote_intervention_available": False,
        "other_road_user_avoidance_possible": False,
        "_fact_provenance": {
            "driver_in_vehicle": _accepted(),
            "remote_intervention_available": _accepted(),
            "other_road_user_avoidance_possible": _accepted(),
        },
    }
    result = service._control_readiness(
        facts=facts, parent_scenario_id="SC-1", malfunction_id="MF-1",
        values={"obj_distance_m": 2.0}, motion={"status": "MISSING"},
    )
    assert result["override_finalized"] is True
    assert result["ttc_ready"] is False

    ttc_facts = {
        "driver_in_vehicle": True,
        "emergency_braking_available": True,
        "remote_intervention_available": False,
        "other_road_user_avoidance_possible": False,
        "ttc_s": 2.0,
        "_fact_provenance": {
            "driver_in_vehicle": _accepted(),
            "emergency_braking_available": _accepted(),
            "remote_intervention_available": _accepted(),
            "other_road_user_avoidance_possible": _accepted(),
            "ttc_s": _accepted(),
        },
    }
    ttc = service._control_readiness(
        facts=ttc_facts, parent_scenario_id="SC-1", malfunction_id="MF-1",
        values={"obj_distance_m": 2.0}, motion={"status": "MISSING"},
    )
    assert ttc["override_state"] == "NO_MATCH"
    assert ttc["ttc_ready"] is True

    unknown = service._control_readiness(
        facts={"_fact_provenance": {}}, parent_scenario_id="SC-1", malfunction_id="MF-1",
        values={"obj_distance_m": 2.0}, motion={"status": "MISSING"},
    )
    assert unknown["override_state"] == "UNKNOWN"


def test_delta_classifier_requires_noninterference_and_prerequisites(service):
    missing_metadata = service.classify_delta(
        assessment={"causal_assessment": {"evidence_bindings": []}},
        parent_facts={}, child_values={"object_speed_kph": 0.0},
        parent_scenario_id="SC-1", malfunction_id="MF-1",
    )
    assert missing_metadata["classification"] == "UNCLASSIFIED"
    assert missing_metadata["dependency_check"] == "DEPENDENCY_METADATA_INCOMPLETE"

    risk_only = service.classify_delta(
        assessment={"dependency_metadata": _noninterference()},
        parent_facts={}, child_values={"object_speed_kph": 0.0},
        parent_scenario_id="SC-1", malfunction_id="MF-1",
    )
    assert risk_only["classification"] == "DETERMINISTIC_CAUSAL_REUSE"
    assert risk_only["causal_revalidation_required"] is False

    dependency = service.classify_delta(
        assessment={"dependency_metadata": {
            **_noninterference(), "causal_evidence_fields": ["object_speed_kph"],
        }},
        parent_facts={}, child_values={"object_speed_kph": 0.0},
        parent_scenario_id="SC-1", malfunction_id="MF-1",
    )
    assert dependency["classification"] == "CAUSAL_RELEVANT_CHANGE"

    conflict = service.classify_delta(
        assessment={"dependency_metadata": _noninterference()},
        parent_facts={"object_speed_kph": 5.0}, child_values={"object_speed_kph": 0.0},
        parent_scenario_id="SC-1", malfunction_id="MF-1",
    )
    assert conflict["classification"] == "SOURCE_CONFLICT"

    subset = service.classify_delta(
        assessment={"dependency_metadata": _noninterference()},
        parent_facts={"ego_speed_constraint": {"min_kph": 0.0, "max_kph": 20.0}},
        child_values={"ego_speed_kph": 7.0},
        parent_scenario_id="SC-1", malfunction_id="MF-1",
    )
    assert subset["classification"] == "DETERMINISTIC_CAUSAL_REUSE"


def test_valid_pack_materializes_child_without_mutating_parent(service):
    checkpoint = _checkpoint()
    payload, _ = service.generate(checkpoint=checkpoint, supplement=_supplement())
    pack = payload["assumption_pack"]
    assert len(pack["groups"]) == 1
    for field in pack["groups"][0]["fields"]:
        field.update({
            "value": {
                "ego_speed_kph": 7.0,
                "driver_in_vehicle": False,
                "remote_intervention_available": False,
                "other_road_user_avoidance_possible": False,
            }[field["field"]],
            "source_type": "ENGINEERING_ANALYSIS_SETTING",
            "source_id": "ENG-SETTING-42",
            "approval": "FINALIZED",
            "validation_status": "VALIDATED",
        })
    payload, queue = service.generate(
        checkpoint=checkpoint, supplement=_supplement(), assumption_pack=pack,
    )
    assert payload["assumption_validation"]["status"] == "VALID"
    assert len(payload["materialized_children"]) == 1
    child = payload["materialized_children"][0]
    assert child["scenario_id"].startswith("SCN-ANALYTICAL-")
    assert child["facts"]["ego_speed_kph"] == 7.0
    assert child["fact_provenance"]["ego_speed_kph"]["origin"] == "SCENARIO_DEFINED"
    assert child["fact_provenance"]["object_speed_kph"]["source_refs"][0]["source_type"] == "METHOD_CONTRACT"
    assert "ego_speed_kph" not in checkpoint["scenarios"][0]["facts"]
    assert queue["provider_calls"] == 0


def test_invalid_assumption_values_fail_closed(service):
    checkpoint = _checkpoint()
    payload, _ = service.generate(checkpoint=checkpoint, supplement=_supplement())
    pack = payload["assumption_pack"]
    fields = {item["field"]: item for item in pack["groups"][0]["fields"]}
    fields["ego_speed_kph"].update({
        "value": 99.0, "source_type": "ENGINEERING_ANALYSIS_SETTING",
        "source_id": "ENG-SETTING-42", "approval": "FINALIZED", "validation_status": "VALIDATED",
    })
    fields["driver_in_vehicle"].update({
        "value": "false", "source_type": "ENGINEERING_ANALYSIS_SETTING",
        "source_id": "ENG-SETTING-42", "approval": "FINALIZED", "validation_status": "VALIDATED",
    })
    payload, _ = service.generate(
        checkpoint=checkpoint, supplement=_supplement(), assumption_pack=pack,
    )
    codes = {item["code"] for item in payload["assumption_validation"]["errors"]}
    assert {"EGO_SPEED_OUTSIDE_ENVELOPE", "CONTROL_INPUT_NOT_BOOLEAN"}.issubset(codes)
    assert payload["materialized_children"] == []


def test_assumption_scope_splits_unsafe_function_sharing_and_rejects_cross_scope(service):
    checkpoint = _checkpoint()
    payload, _ = service.generate(checkpoint=checkpoint, supplement=_supplement())
    records = payload["records"]
    assert len(service._assumption_groups(records)) == 1
    unsafe = deepcopy(records[0])
    unsafe["function_id"] = "FN-OTHER"
    assert len(service._assumption_groups([records[0], unsafe])) == 2

    pack = payload["assumption_pack"]
    pack["groups"][0]["scope"]["function_id"] = "FN-OTHER"
    payload, _ = service.generate(
        checkpoint=checkpoint, supplement=_supplement(), assumption_pack=pack,
    )
    codes = {item["code"] for item in payload["assumption_validation"]["errors"]}
    assert "ASSUMPTION_SCOPE_MISMATCH" in codes


def test_risk_scoreability_cli_writes_v3_artifacts_and_final_pack(tmp_path):
    checkpoint = tmp_path / "checkpoint.json"
    supplement = tmp_path / "supplement.json"
    assumption_pack = tmp_path / "HARA_Risk_Assumption_Pack_Final.json"
    review_root = tmp_path / "review"
    checkpoint.write_text(json.dumps(_checkpoint()), encoding="utf-8")
    supplement.write_text(json.dumps(_supplement()), encoding="utf-8")

    result = main([
        "risk-scoreability",
        "--checkpoint", str(checkpoint),
        "--risk-input-supplement", str(supplement),
        "--review-run-id", "scoreability-v3-test",
        "--review-root", str(review_root),
        "--assumption-pack", str(assumption_pack),
        "--baseline", str(ROOT / "method_assets/fusa_baseline_v1/manifest.yaml"),
        "--report-style-template", str(ROOT / "references/HARA_Template_AI_20260327.xlsx"),
    ])
    assert result == 0
    assert assumption_pack.is_file()
    assert (review_root / "scoreability-v3-test" / "risk_scoreability_v3.json").is_file()
    assert (review_root / "scoreability-v3-test" / "differential_validation_queue_v3.json").is_file()
