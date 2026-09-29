from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hara_agent.contracts import (
    ScenarioBindingAuthority, ScenarioDimensionApplicability,
)
from hara_agent.evaluation.scenario_selector import independent_evidence_tier
from hara_agent.method_sources import YamlBaselineCompiler
from hara_agent.models import ReviewStatus, ScenarioCandidate
from hara_agent.services.analysis import ConstrainedScenarioSynthesisService
from hara_agent.services.analysis.hazardous_event_risk_context_service import (
    HazardousEventRiskContextService,
)
from hara_agent.services.analysis.risk_calculation_input_service import (
    RiskCalculationInputService,
)
from hara_agent.services.analysis.risk_scoreability_service import RiskScoreabilityService
from hara_agent.services.analysis.analytical_physics_instantiation_service import (
    AnalyticalPhysicsInstantiationService,
)
from hara_agent.services.analysis.scenario_physics import (
    closing_relative_speed_kph, time_to_collision_s,
)
from hara_agent.services.analysis.scenario_selection_quality import (
    ScenarioCandidateRanker, ScenarioShortlistPolicy,
)
from hara_agent.template import TemplateRoleCompiler


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def method():
    report = TemplateRoleCompiler().compile_method(
        ROOT / "references/HARA_Template_AI_20260327.xlsx"
    ).report_contract
    return YamlBaselineCompiler().compile(
        ROOT / "method_assets/fusa_baseline_v1/manifest.yaml",
        report_contract=report,
    )


def _candidate(atom_id: str, family: str, tier: int, *, template=False, authority=None):
    return SimpleNamespace(
        atom_id=atom_id, semantic_family=family,
        ranking_scores={"source_evidence_tier": float(tier), "final_rank_score": 1.0},
        template_relationship="SOURCE_TEMPLATE_COMPATIBLE" if template else "NONE",
        binding_authority=authority or ScenarioBindingAuthority.ANALYTICAL_SELECTION.value,
        label=atom_id, method_semantics={}, speed_range_kph=None,
    )


def _required():
    return SimpleNamespace(status=ScenarioDimensionApplicability.REQUIRED)


def test_bm25_and_mechanism_overlap_never_promote_authority_tier():
    ranker = ScenarioCandidateRanker()
    scores, _ = ranker.score(
        dimension="OBJECT",
        atom={"atom_id": "LEX", "label": "opaque loss command", "filled_dimensions": ["OBJECT"]},
        query={
            "query_tokens": ["opaque", "loss", "command"],
            "causal_tokens": ["opaque"], "failure_type": "loss",
            "guideword": "loss", "object_categories": ["OBJECT_STATIC"],
            "structured_source_categories": [],
        },
        fm_template={}, corpus_labels=["opaque loss command"], odd_passed=False,
    )
    assert scores["lexical_score"] > 0
    assert scores["causal_score"] > 0
    assert scores["mechanism_score"] > 0
    assert scores["source_evidence_tier"] == 0


def test_evaluation_evidence_tiers_match_production_authority_hierarchy():
    assert independent_evidence_tier(["EXACT_BINDING"]) == 4
    assert independent_evidence_tier(["EXACT_STRUCTURED_SOURCE_MATCH"]) == 3
    assert independent_evidence_tier(["FIELD_CORRECT_CATEGORY_MATCH"]) == 2
    assert independent_evidence_tier(["ODD_CONTEXT_COMPATIBILITY"]) == 1
    assert independent_evidence_tier(["BM25_MATCH"]) == 0


def test_structured_source_tier_outranks_lexical_only():
    ranker = ScenarioCandidateRanker()
    structured, _ = ranker.score(
        dimension="OBJECT",
        atom={
            "atom_id": "STRUCTURED", "label": "vehicle object",
            "filled_dimensions": ["OBJECT"],
            "physical_semantics": {"object": {"type": "vehicle"}},
        },
        query={
            "object_categories": ["OBJECT_VEHICLE"],
            "structured_source_categories": ["OBJECT_VEHICLE"],
        }, fm_template={}, corpus_labels=["vehicle object"], odd_passed=False,
    )
    lexical, _ = ranker.score(
        dimension="OBJECT",
        atom={"atom_id": "LEXICAL", "label": "opaque collision", "filled_dimensions": ["OBJECT"]},
        query={"query_tokens": ["opaque", "collision"], "object_categories": ["OBJECT_VEHICLE"]},
        fm_template={}, corpus_labels=["opaque collision"], odd_passed=False,
    )
    assert structured["source_evidence_tier"] == 3
    assert lexical["source_evidence_tier"] == 0
    assert ranker.rank_key("STRUCTURED", structured) < ranker.rank_key("LEXICAL", lexical)


def test_fm_and_authoritative_candidates_cannot_be_cut():
    rows = [_candidate(f"WEAK-{index}", f"F-{index}", 1) for index in range(20)]
    fm = _candidate("FM", "FM", 4, template=True)
    exact = _candidate(
        "EXACT", "EXACT", 4,
        authority=ScenarioBindingAuthority.EXACT_METHOD_MAPPING.value,
    )
    selected, _, _ = ScenarioShortlistPolicy.select(
        dimension="OBJECT", ranked=(*rows, fm, exact), applicability=_required(),
        exact=False, primary_dimensions=(), secondary_dimensions=("OBJECT",), query={},
    )
    assert {fm.atom_id, exact.atom_id} <= {item.atom_id for item in selected}


def test_family_dedup_preserves_coverage_and_limits_intensity_concentration():
    rows = [
        *(_candidate(f"DECEL-{index}", "DECEL", 1) for index in range(10)),
        _candidate("REVERSE", "REVERSE", 2),
        _candidate("HOLD", "HOLD", 2),
    ]
    selected, _, _ = ScenarioShortlistPolicy.select(
        dimension="EGO_ACTION", ranked=rows, applicability=_required(), exact=False,
        primary_dimensions=("EGO_ACTION",), secondary_dimensions=(), query={},
    )
    counts = {}
    for item in selected:
        counts[item.semantic_family] = counts.get(item.semantic_family, 0) + 1
    assert {"DECEL", "REVERSE", "HOLD"} <= set(counts)
    assert counts["DECEL"] <= 2


def test_adaptive_budget_expands_only_for_high_authority_and_secondary_is_smaller():
    ordinary = [_candidate(f"O-{index}", f"F-{index}", 1) for index in range(20)]
    primary, primary_budget, _ = ScenarioShortlistPolicy.select(
        dimension="EGO_ACTION", ranked=ordinary, applicability=_required(), exact=False,
        primary_dimensions=("EGO_ACTION",), secondary_dimensions=(), query={},
    )
    secondary, secondary_budget, _ = ScenarioShortlistPolicy.select(
        dimension="ROAD", ranked=ordinary, applicability=_required(), exact=False,
        primary_dimensions=(), secondary_dimensions=("ROAD",), query={},
    )
    authoritative = [_candidate(f"A-{index}", f"A-{index}", 3) for index in range(15)]
    expanded, expanded_budget, _ = ScenarioShortlistPolicy.select(
        dimension="EGO_ACTION", ranked=authoritative, applicability=_required(), exact=False,
        primary_dimensions=("EGO_ACTION",), secondary_dimensions=(), query={},
    )
    assert len(primary) == primary_budget == ScenarioShortlistPolicy.PRIMARY_BUDGET
    assert len(secondary) == secondary_budget == ScenarioShortlistPolicy.SECONDARY_BUDGET
    assert len(expanded) == expanded_budget == 15


def test_shared_physics_primitives_match_instantiation_derivations():
    assert closing_relative_speed_kph(
        20.0, 10.0, ego_direction="FORWARD",
        object_direction="REVERSE", collision_type="FRONT",
    ) == 30.0
    assert time_to_collision_s(30.0, 30.0) == 3.6
    assert closing_relative_speed_kph(
        20.0, 0.0, ego_direction="", object_direction="STATIONARY",
        collision_type="FRONT",
    ) == 20.0
    assert time_to_collision_s(30.0, 0.0) is None
    assert "closing_relative_speed_kph" in AnalyticalPhysicsInstantiationService.instantiate.__code__.co_names


def test_canonical_risk_readiness_has_one_owner():
    assert callable(HazardousEventRiskContextService.severity_readiness)
    assert callable(HazardousEventRiskContextService.controllability_readiness)
    assert not hasattr(RiskCalculationInputService, "severity_readiness")
    assert not hasattr(RiskCalculationInputService, "controllability_readiness")
    assert not hasattr(RiskScoreabilityService, "severity_readiness")
    assert not hasattr(RiskScoreabilityService, "controllability_readiness")


def _build_gap(service, *, malfunction_id: str, hazard: str, object_type=""):
    parent = ScenarioCandidate(
        "SCN-GAP", "parking garage", "parking", "parking",
        operating_mode="active", facts={
            "ego_speed_constraint": {"min_kph": 0.0, "max_kph": 20.0},
            **({"object_type": object_type} if object_type else {}),
        }, status=ReviewStatus.FINALIZED,
    )
    malfunction = {
        "malfunction_id": malfunction_id, "function_id": "F-GAP", "guideword": "loss",
        "description": hazard, "functional_effect": hazard,
        "vehicle_level_hazard": hazard, "component_category": "control",
        "failure_type": "loss",
    }
    return service.build_input(
        malfunction=malfunction, parent=parent,
        assessment={
            "hazardous_event_id": "HE-GAP", "hazardous_event": hazard,
            "causal_assessment": {"status": "VALIDATED", "causal_chain": [hazard]},
        },
        project_context={
            "odd_locations": ["parking garage"], "odd_road_types": ["parking garage"],
            "speed_min_kph": 0.0, "speed_max_kph": 20.0,
        },
    )


def test_static_object_gap_retains_unknown_candidates_but_stays_fail_closed(method):
    synthesis_input = _build_gap(
        ConstrainedScenarioSynthesisService(method), malfunction_id="MF-STATIC",
        hazard="The vehicle can collide with a static obstacle.",
        object_type="static_obstacle",
    )
    objects = next(item for item in synthesis_input.dimension_candidate_sets if item.dimension == "OBJECT")
    assert objects.applicability.status.value == "NOT_APPLICABLE"
    assert objects.generation_status == "NOT_APPLICABLE"
    assert objects.candidates == ()


def test_abort_action_gap_is_explicit_and_stays_fail_closed(method):
    synthesis_input = _build_gap(
        ConstrainedScenarioSynthesisService(method), malfunction_id="MF-ABORT",
        hazard="The active maneuver cannot be cancelled and driver takeover is blocked.",
    )
    actions = next(item for item in synthesis_input.dimension_candidate_sets if item.dimension == "EGO_ACTION")
    assert synthesis_input.structured_semantic_query["action_categories"] == ["ACTION_ABORT"]
    assert actions.generation_status == "METHOD_GAP"
    assert actions.candidates
    assert all(item.semantic_compatibility.value == "UNKNOWN" for item in actions.candidates)
