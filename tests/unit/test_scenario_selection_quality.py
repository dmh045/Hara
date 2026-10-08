from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from hara_agent.method_sources import YamlBaselineCompiler
from hara_agent.contracts import (
    ScenarioBindingAuthority, ScenarioBindingDecision,
    ScenarioDimensionApplicability,
)
from hara_agent.models import ReviewStatus, ScenarioCandidate
from hara_agent.services.analysis import ConstrainedScenarioSynthesisService
from hara_agent.services.analysis.scenario_selection_quality import (
    ScenarioCandidateRanker, ScenarioCoveragePlanner,
    ScenarioDimensionApplicabilityService, ScenarioSemanticQueryBuilder,
    ScenarioRefinementEvidencePolicy,
)
from hara_agent.evaluation.scenario_selector import top_k_recall_audit
from hara_agent.services.semantic.scenario_synthesis_agent import (
    BoundedScenarioSynthesisAgent,
)
from hara_agent.services.reporting import ScenarioSelectorQualityAudit
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


def _parent(
    *, object_type: str = "pedestrian", collision_type: str = "front",
    operating_scenario: str = "parking garage", description: str = "parking maneuver",
    dynamic_authority: str = "EXACT_PROJECT_FACT", source_template: bool = False,
) -> ScenarioCandidate:
    binding = {
        "resolution_status": "RESOLVED",
        "binding_status": "EXACT",
        "atom_id": "FA001",
        "canonical_atom_id": "FA001",
        "project_value": "0..20 km/h",
        "binding_authority": dynamic_authority,
        "speed_constraint": {
            "min_kph": 0.0, "max_kph": 20.0,
            "resolved_by": dynamic_authority,
        },
    }
    instance = {}
    if source_template:
        instance = {
            "source_template_id": "FM_TEMPLATE_001",
            "source_option_id": "FM_TEMPLATE_001:OPTION:1",
        }
    return ScenarioCandidate(
        scenario_id="SCN-Q", operating_scenario=operating_scenario,
        situational_description=description, situational_detailing=description,
        operating_mode="active",
        facts={
            "ego_speed_constraint": {"min_kph": 0.0, "max_kph": 20.0},
            "object_type": object_type, "collision_type": collision_type,
            "object_position": "rear" if collision_type == "rear" else "front",
            "method_scenario_dimensions": {"EGO_DYNAMICS": binding},
            "scenario_atom_ids": ["FA001"],
        },
        analysis_instance=instance,
        semantic_fingerprint="quality-fixture", status=ReviewStatus.FINALIZED,
    )


def _malfunction(hazard: str, *, description: str = "parking control failure") -> dict:
    return {
        "malfunction_id": "MF-Q", "function_id": "F-Q",
        "guideword": "No/Loss", "description": description,
        "functional_effect": hazard, "vehicle_level_hazard": hazard,
        "component_category": "actuator_longitudinal", "failure_type": "loss",
    }


def _assessment(hazard: str) -> dict:
    return {
        "malfunction_id": "MF-Q", "scenario_id": "SCN-Q",
        "hazardous_event_id": "HE-Q", "hazardous_event": hazard,
        "causal_assessment": {
            "status": "VALIDATED", "hazardous_event": hazard,
            "causal_chain": ["malfunction", "vehicle behavior", hazard],
        },
    }


def _project(location: str = "parking garage", road: str = "normal road surface") -> dict:
    return {
        "odd_locations": [location], "odd_road_types": [location],
        "odd_weather_conditions": ["clear"], "odd_road_surfaces": [road],
        "speed_min_kph": 0.0, "speed_max_kph": 20.0,
    }


def _build(
    service, hazard: str, *, parent=None, project=None, description=None,
    function=None,
):
    parent = parent or _parent()
    return service.build_input(
        malfunction=_malfunction(hazard, description=description or "parking control failure"),
        parent=parent, assessment=_assessment(hazard),
        project_context=project or _project(),
        function=function or {},
    )


def _sets(synthesis_input):
    return {item.dimension: item for item in synthesis_input.dimension_candidate_sets}


@pytest.mark.parametrize("signature_count", [1, 2, 3])
def test_coverage_variant_count_uses_distinct_engineering_support(signature_count):
    candidates = []
    score_names = (
        "template_score", "mechanism_score", "action_score", "object_score",
        "traffic_relation_score", "causal_score",
    )
    for index in range(signature_count):
        scores = {name: 0.0 for name in score_names}
        scores[score_names[index]] = 1.0
        candidates.append(SimpleNamespace(
            ranking_scores=scores, method_semantics={}, speed_range_kph=None,
            template_relationship="NONE",
        ))
    # A duplicate evidence signature must not inflate the requested count.
    candidates.append(SimpleNamespace(
        ranking_scores=dict(candidates[0].ranking_scores), method_semantics={},
        speed_range_kph=None, template_relationship="NONE",
    ))
    candidate_set = SimpleNamespace(
        dimension="EGO_ACTION", candidates=tuple(candidates),
        applicability=SimpleNamespace(status=ScenarioDimensionApplicability.REQUIRED),
        binding_decision=SimpleNamespace(parent_atom_id="", refinable=True),
    )
    plan = ScenarioCoveragePlanner.plan(
        query={"action_categories": ["ACTION_STOP"], "source_refs": ["HE.hazardous_event"]},
        candidate_sets=(candidate_set,),
    )
    assert plan.desired_variant_count == signature_count
    assert plan.variant_intents[0]["supported_primary_signature_counts"] == {
        "EGO_ACTION": signature_count,
    }


def test_rear_end_uses_source_defined_following_compounds_without_duplicate_traffic_gap(method):
    service = ConstrainedScenarioSynthesisService(method)
    hazard = "Unexpected braking causes a rear-end collision with the following rear vehicle."
    synthesis_input = _build(
        service, hazard,
        parent=_parent(object_type="passenger_car", collision_type="rear"),
    )
    sets = _sets(synthesis_input)
    assert sets["TRAFFIC_PATTERN"].applicability.status.value == "NOT_APPLICABLE"
    assert sets["TRAFFIC_PATTERN"].generation_status == "NOT_APPLICABLE"
    assert sets["TRAFFIC_PATTERN"].candidates == ()
    assert sets["EGO_ACTION"].candidates[0].atom_id in {"PU015", "PU016"}
    assert sets["OBJECT"].candidates[0].atom_id in {"PU015", "PU016"}
    represented = synthesis_input.structured_semantic_query[
        "traffic_relations_represented_elsewhere"
    ]
    assert {item["atom_id"] for item in represented["TRAFFIC_FOLLOWING"]} == {
        "PU015", "PU016",
    }


def test_oncoming_relation_prioritizes_source_compatible_action_atom(method):
    service = ConstrainedScenarioSynthesisService(method)
    hazard = "Turning across oncoming traffic can cause a head-on collision."
    synthesis_input = _build(
        service, hazard,
        parent=_parent(
            object_type="passenger_car", operating_scenario="motorway",
            description="turning on motorway",
        ),
        project=_project("motorway", "normal road surface"),
    )
    sets = _sets(synthesis_input)
    assert sets["TRAFFIC_PATTERN"].applicability.status.value == "NOT_APPLICABLE"
    assert sets["EGO_ACTION"].candidates[0].atom_id == "FA042"
    assert sets["EGO_ACTION"].candidates[0].ranking_scores[
        "traffic_relation_score"
    ] == 1.0


def test_stationary_obstacle_does_not_force_traffic_relation(method):
    service = ConstrainedScenarioSynthesisService(method)
    synthesis_input = _build(
        service, "Stationary vehicle may contact a static obstacle.",
        parent=_parent(object_type="static_obstacle"),
    )
    sets = _sets(synthesis_input)
    assert sets["OBJECT"].applicability.status.value == "NOT_APPLICABLE"
    assert sets["OBJECT"].generation_status == "NOT_APPLICABLE"
    assert sets["OBJECT"].candidates == ()
    assert sets["TRAFFIC_PATTERN"].applicability.status.value == "NOT_APPLICABLE"
    assert sets["TRAFFIC_PATTERN"].candidates == ()


def _speed_envelope(mode, minimum, maximum, excerpt, *, condition=""):
    return {
        "operating_mode": mode,
        "speed_min_kph": minimum,
        "speed_max_kph": maximum,
        "condition": condition,
        "unit": "km/h",
        "status": "FINALIZED",
        "sources": [{
            "source_type": "item_definition", "source_id": "ItemDef.docx",
            "location": f"speed.{mode}", "excerpt": excerpt,
        }],
    }


def test_contextual_parking_and_control_speed_envelopes_are_handed_to_selection(method):
    service = ConstrainedScenarioSynthesisService(method)
    project = _project()
    project["speed_envelopes"] = [
        _speed_envelope("Active", 0.0, 20.0, "Active speed 0-20km/h"),
        _speed_envelope("parking", 0.0, 5.0, "泊车时最高车速≤5km/h"),
        _speed_envelope("control", 0.0, 7.0, "车辆控制速度范围0-7km/h"),
    ]
    parking = _build(
        service, "Vehicle may collide during parking.", project=project,
        description="parking maneuver failure",
        function={"odd_constraints": ["泊车时最高车速≤5km/h"]},
    )
    control = _build(
        service, "Unintended acceleration during vehicle control.", project=project,
        description="unintended acceleration",
        function={"operational_context": "control"},
    )

    assert parking.contextual_speed == {
        **parking.contextual_speed,
        "status": "RESOLVED", "selected_context": "PARKING",
        "min_kph": 0.0, "max_kph": 5.0, "display_kind": "UPPER_BOUND",
    }
    assert parking.contextual_speed["match_basis"] == "FUNCTION.odd_constraints[0]"
    parking_dynamics = _sets(parking)["EGO_DYNAMICS"]
    assert parking_dynamics.binding_decision.project_speed_envelope_kph == (0.0, 5.0)
    assert control.contextual_speed["status"] == "RESOLVED"
    assert control.contextual_speed["selected_context"] == "CONTROL"
    assert (control.contextual_speed["min_kph"], control.contextual_speed["max_kph"]) == (0.0, 7.0)
    assert control.contextual_speed["display_kind"] == "RANGE"
    control_dynamics = _sets(control)["EGO_DYNAMICS"]
    assert control_dynamics.binding_decision.project_speed_envelope_kph == (0.0, 7.0)


@pytest.mark.parametrize(
    ("context", "description", "hazard", "maximum", "display_kind"),
    (
        ("cruise", "cruise function failure", "Vehicle departs its path while cruising.", 15.0, "UPPER_BOUND"),
        ("entry", "entry activation failure", "Vehicle cannot enter the function safely.", 20.0, "UPPER_BOUND"),
        ("transition", "state transition failure", "Unsafe state transition occurs.", 12.0, "UPPER_BOUND"),
    ),
)
def test_other_contextual_speed_classes_are_consumable(
    method, context, description, hazard, maximum, display_kind,
):
    service = ConstrainedScenarioSynthesisService(method)
    project = _project()
    project["speed_envelopes"] = [
        _speed_envelope("Active", 0.0, 20.0, "Active speed 0-20km/h"),
        _speed_envelope(
            context, 0.0, maximum,
            f"{context} maximum speed ≤{maximum:g}km/h",
        ),
    ]

    synthesis_input = _build(
        service, hazard, project=project, description=description,
        function={"operational_context": context},
    )

    assert synthesis_input.contextual_speed["status"] == "RESOLVED"
    assert synthesis_input.contextual_speed["selected_context"] == context.upper()
    assert synthesis_input.contextual_speed["max_kph"] == maximum
    assert synthesis_input.contextual_speed["display_kind"] == display_kind


def test_conflicting_search_speed_sources_are_preserved_as_conflict(method):
    service = ConstrainedScenarioSynthesisService(method)
    project = _project()
    project["speed_envelopes"] = [
        _speed_envelope("Active", 0.0, 20.0, "Active speed 0-20km/h"),
        _speed_envelope("search", 0.0, 24.0, "搜索阶段最高车速≤24km/h"),
        _speed_envelope("search", 0.0, 30.0, "搜索阶段速度范围0-30km/h"),
    ]
    synthesis_input = _build(
        service, "Vehicle searches for a parking space.", project=project,
        description="parking-space search failure",
        function={"operational_context": "search"},
    )

    assert synthesis_input.contextual_speed["status"] == "SOURCE_CONFLICT"
    assert synthesis_input.contextual_speed["selected_context"] == "SEARCH"
    assert synthesis_input.contextual_speed["source_expressions"] == [
        "不高于24 km/h", "0–30 km/h",
    ]


def test_parent_active_range_is_only_used_after_specific_contexts(method):
    service = ConstrainedScenarioSynthesisService(method)
    project = _project()
    project["speed_envelopes"] = [
        _speed_envelope("Active", 0.0, 20.0, "Active speed 0-20km/h"),
        _speed_envelope("parking", 0.0, 5.0, "泊车时最高车速≤5km/h"),
    ]

    synthesis_input = _build(
        service, "Vehicle may collide during parking.", project=project,
        function={"operational_context": "parking"},
    )

    assert synthesis_input.contextual_speed["selected_context"] == "PARKING"
    assert synthesis_input.contextual_speed["max_kph"] == 5.0
    assert synthesis_input.contextual_speed["source_expressions"] == ["不高于5 km/h"]


def test_condition_specific_active_envelope_is_not_a_generic_parent_fallback(method):
    service = ConstrainedScenarioSynthesisService(method)
    project = _project()
    project["speed_envelopes"] = [
        _speed_envelope(
            "Active", 0.0, 20.0, "Active transition speed <=20km/h",
            condition="transition_speed",
        ),
    ]

    synthesis_input = _build(
        service, "Vehicle may collide with an obstacle.", project=project,
        description="generic function failure",
    )

    assert synthesis_input.contextual_speed == {
        "status": "PARENT_FALLBACK",
        "classification": "PARENT_BROAD_RANGE_FALLBACK",
        "requested_contexts": [],
        "min_kph": 0.0,
        "max_kph": 20.0,
    }


def test_slope_holding_plan_treats_road_variation_as_primary(method):
    service = ConstrainedScenarioSynthesisService(method)
    hazard = "Loss of parking brake holding capability causes rollaway on a slope."
    synthesis_input = _build(
        service, hazard,
        parent=_parent(description="vehicle holding on slope"),
        project=_project(road="parking slope gradient up to 15%"),
    )
    plan = synthesis_input.coverage_plan
    assert plan.primary_variation_dimensions[:2] == ("ROAD", "EGO_X_ROAD")
    assert "ROAD" not in plan.prohibited_trivial_only_dimensions
    selections = (
        {"ROAD": ("FB001",), "EGO_X_ROAD": ("PH012",), "EGO_ACTION": ("PH002",)},
        {"ROAD": ("FB002",), "EGO_X_ROAD": ("PH013",), "EGO_ACTION": ("PH002",)},
    )
    assert service.diversity_validator.reasons(plan, selections) == []


def test_reverse_pedestrian_plan_rejects_environment_only_siblings(method):
    service = ConstrainedScenarioSynthesisService(method)
    hazard = "Vehicle reversing into a pedestrian during parking."
    synthesis_input = _build(
        service, hazard,
        parent=_parent(description="reversing parking maneuver"),
    )
    plan = synthesis_input.coverage_plan
    assert plan.primary_variation_dimensions[:2] == ("EGO_ACTION", "OBJECT")
    selections = (
        {"EGO_ACTION": ("FA033",), "OBJECT": ("PU004",), "WHERE": ("SO010",), "ROAD": ("FB005",)},
        {"EGO_ACTION": ("FA033",), "OBJECT": ("PU004",), "WHERE": ("SO020",), "ROAD": ("FB002",)},
        {"EGO_ACTION": ("FA033",), "OBJECT": ("PU004",), "WHERE": ("VD013",), "ROAD": ("FB003",)},
    )
    assert service.diversity_validator.reasons(plan, selections) == [
        "TRIVIAL_VARIANT_DIVERSITY"
    ]


def test_range_containment_parent_is_refinable(method):
    service = ConstrainedScenarioSynthesisService(method)
    synthesis_input = _build(
        service, "Vehicle is reversing during a parking maneuver.",
        parent=_parent(
            object_type="", description="reversing parking maneuver",
            dynamic_authority="RANGE_CONTAINMENT",
        ),
    )
    dynamics = _sets(synthesis_input)["EGO_DYNAMICS"]
    assert dynamics.binding_decision.authority.value == "RANGE_CONTAINMENT"
    assert dynamics.binding_decision.refinable is True
    assert dynamics.locked_atom_ids == ()
    assert "FA033" in {item.atom_id for item in dynamics.candidates}


def test_exact_authoritative_dynamic_remains_locked(method):
    service = ConstrainedScenarioSynthesisService(method)
    dynamics = _sets(_build(service, "Parking maneuver near a pedestrian."))["EGO_DYNAMICS"]
    assert dynamics.binding_decision.authority.value == "EXACT_PROJECT_FACT"
    assert dynamics.binding_decision.refinable is False
    assert dynamics.locked_atom_ids == ("FA001",)


def test_unrepresentable_crossing_relation_is_method_gap_not_catalog_fallback(method):
    service = ConstrainedScenarioSynthesisService(method)
    traffic = _sets(_build(
        service, "Crossing traffic enters the parking path.",
        parent=_parent(object_type="passenger_car"),
    ))["TRAFFIC_PATTERN"]
    assert traffic.applicability.status.value == "REQUIRED"
    assert traffic.generation_status == "METHOD_GAP"
    assert traffic.candidates
    assert all(item.semantic_compatibility.value == "UNKNOWN" for item in traffic.candidates)
    assert traffic.hard_filtered_pool_size >= len(traffic.candidates)


def _query(
    *, malfunction_text="", hazard_text="", parent=None, facts=None,
    project=None,
):
    parent = parent or ScenarioCandidate(
        "SCN-FIELD", "parking garage", "", "", operating_mode="active",
        facts=dict(facts or {}),
    )
    return ScenarioSemanticQueryBuilder.build(
        malfunction={
            "description": malfunction_text, "functional_effect": "",
            "vehicle_level_hazard": hazard_text,
        },
        parent=parent,
        assessment={
            "hazardous_event": hazard_text,
            "causal_assessment": {"causal_chain": []},
        },
        project_context=project or {"odd_locations": ["parking garage"]},
        fm_template={},
    )


def test_parking_location_alone_does_not_prove_action_park():
    query = _query(malfunction_text="control output unavailable")
    assert query["location_categories"] == ["LOCATION_PARKING"]
    assert query["project_location_categories"] == ["LOCATION_PARKING"]
    assert query["parent_location_categories"] == ["LOCATION_PARKING"]
    assert "ACTION_PARK" not in query["action_categories"]


def test_odd_slope_capability_does_not_prove_active_road_mechanism():
    query = ScenarioSemanticQueryBuilder.build(
        malfunction={
            "description": "control output unavailable",
            "functional_effect": "command is not applied",
            "vehicle_level_hazard": "vehicle response is unavailable",
        },
        parent=ScenarioCandidate(
            "SCN-FIELD", "parking garage", "", "", operating_mode="active",
            facts={},
        ),
        assessment={
            "hazardous_event": "vehicle response is unavailable",
            "causal_assessment": {"causal_chain": []},
        },
        project_context={
            "odd_locations": ["parking garage"],
            "odd_road_surfaces": ["supports a 15% slope gradient"],
        },
        fm_template={},
    )
    assert query["road_relations"] == []
    assert query["odd_road_categories"] == ["ROAD_SLOPE"]


def test_ego_vehicle_mention_alone_does_not_prove_object_vehicle():
    query = _query(hazard_text="The ego vehicle departs from its intended path.")
    assert query["object_categories"] == []


def test_passenger_car_object_does_not_prove_occupant():
    query = _query(
        hazard_text="Collision with a passenger car.",
        facts={"object_type": "passenger_car"},
    )
    assert query["object_categories"] == ["OBJECT_VEHICLE"]


@pytest.mark.parametrize("object_type", ["occupant", "static_obstacle"])
def test_reverse_object_does_not_automatically_require_traffic(object_type):
    query = _query(
        hazard_text="The ego vehicle reverses unexpectedly.",
        facts={"object_type": object_type},
    )
    assert query["action_categories"] == ["ACTION_REVERSE"]
    assert query["traffic_relations"] == []


@pytest.mark.parametrize(
    ("relation", "category"),
    [
        ("following", "TRAFFIC_FOLLOWING"),
        ("oncoming", "TRAFFIC_ONCOMING"),
        ("crossing", "TRAFFIC_CROSSING"),
    ],
)
def test_explicit_structured_relation_requires_traffic_when_not_represented_elsewhere(
    relation, category,
):
    query = _query(facts={"traffic_relation": relation})
    assert query["traffic_relations"] == [category]
    binding = ScenarioBindingDecision(
        dimension="TRAFFIC_PATTERN",
        authority=ScenarioBindingAuthority.ANALYTICAL_SELECTION,
        refinable=True, source_refs=("PARENT.TRAFFIC_PATTERN",), basis="fixture",
    )
    decision = ScenarioDimensionApplicabilityService.assess(
        "TRAFFIC_PATTERN", query, binding,
    )
    assert decision.status is ScenarioDimensionApplicability.REQUIRED


def test_source_defined_compound_traffic_atom_survives_hard_filter(method):
    service = ConstrainedScenarioSynthesisService(method)
    parent = _parent(
        object_type="passenger_car", operating_scenario="motorway",
        description="turning on motorway",
    )
    query = _query(
        parent=parent, facts=parent.facts,
        hazard_text="Turning across oncoming traffic.",
        project=_project("motorway"),
    )
    binding = ScenarioBindingDecision(
        dimension="TRAFFIC_PATTERN",
        authority=ScenarioBindingAuthority.ANALYTICAL_SELECTION,
        refinable=True, source_refs=("HE.hazardous_event",), basis="fixture",
    )
    applicability = ScenarioDimensionApplicabilityService.assess(
        "TRAFFIC_PATTERN", query, binding,
    )
    atom = service.by_id["CN_oncoming_motorway"]
    passed, reason = service.hard_filter_decision(
        dimension="TRAFFIC_PATTERN", atom=atom, parent=parent,
        project_context=_project("motorway"), query=query,
        applicability=applicability, decision=binding, exact_locks={},
        fm_template={},
    )
    assert (passed, reason) == (True, "SUPPORTED:EXPLICIT_FIELD_CATEGORY_MATCH")


def test_structured_physical_semantics_override_conflicting_label_marker():
    sources = ScenarioCandidateRanker.atom_category_sources({
        "label": "Pedestrian beside the path",
        "physical_semantics": {"object": {"type": "vehicle"}},
    })
    assert "OBJECT_VEHICLE" in sources
    assert "OBJECT_PEDESTRIAN" not in sources
    assert sources["OBJECT_VEHICLE"] == "physical_semantics.object.type"


def test_bm25_cannot_resurrect_hard_incompatible_candidate(method):
    service = ConstrainedScenarioSynthesisService(method)
    parent = _parent(object_type="pedestrian")
    synthesis_input = _build(service, "Pedestrian may be struck near the vehicle.", parent=parent)
    candidate_set = _sets(synthesis_input)["OBJECT"]
    atom = service.by_id["PU015"]
    passed, reason = service.hard_filter_decision(
        dimension="OBJECT", atom=atom, parent=parent,
        project_context=_project(), query=synthesis_input.structured_semantic_query,
        applicability=candidate_set.applicability,
        decision=candidate_set.binding_decision, exact_locks={}, fm_template={},
    )
    assert passed is False
    assert reason == "CONTRADICTED:EXPLICIT_FIELD_CATEGORY_CONTRADICTION"


def test_fm_object_filter_uses_active_option_not_first_query_category(method):
    service = ConstrainedScenarioSynthesisService(method)
    parent = _parent(object_type="passenger_car")
    decision = ScenarioBindingDecision(
        dimension="OBJECT",
        authority=ScenarioBindingAuthority.ANALYTICAL_SELECTION,
        refinable=True, source_refs=("METHOD.fm_scenario_template",),
        basis="fixture",
    )
    applicability = SimpleNamespace(status=ScenarioDimensionApplicability.REQUIRED)
    passed, reason = service.hard_filter_decision(
        dimension="OBJECT", atom=service.by_id["PU015"], parent=parent,
        project_context=_project(),
        query={"object_categories": ["OBJECT_PEDESTRIAN", "OBJECT_VEHICLE"]},
        applicability=applicability, decision=decision, exact_locks={},
        fm_template={"active_option": {"obj_type": "passenger_car"}},
    )
    assert (passed, reason) == (True, "SUPPORTED:FM_TEMPLATE_OBJECT_MATCH")


def test_top_k_audit_flags_stronger_source_evidence_below_cutoff():
    result = top_k_recall_audit((
        {
            "atom_id": "WEAK", "final_rank_score": 9.0,
            "independent_evidence_tier": 1, "independent_evidence": [
                "FIELD_CORRECT_CATEGORY_MATCH"
            ],
        },
        {
            "atom_id": "STRONG", "final_rank_score": 8.0,
            "independent_evidence_tier": 3, "independent_evidence": [
                "EXACT_STRUCTURED_SOURCE_MATCH"
            ],
        },
    ), cap=1)
    assert result["top_k_recall_risk"] is True
    assert result["at_risk_atom_ids"] == ["STRONG"]
    assert result["source_strong_candidates_below_cutoff"] == 1


def test_refinement_evidence_policy_rejects_lexical_only_but_accepts_structured():
    decision = ScenarioBindingDecision(
        dimension="EGO_DYNAMICS",
        authority=ScenarioBindingAuthority.METHOD_TEMPLATE_INFERENCE,
        refinable=True, source_refs=("PARENT.scenario",), basis="fixture",
        parent_atom_id="FA001",
    )
    lexical = SimpleNamespace(
        atom_id="FA005", speed_range_kph=None, template_relationship="NONE",
        ranking_scores={"lexical_score": 4.0},
    )
    structured = SimpleNamespace(
        atom_id="FA005", speed_range_kph=None, template_relationship="NONE",
        ranking_scores={"structured_source_score": 1.0},
    )
    assert ScenarioRefinementEvidencePolicy.classify(lexical, decision)[0] == (
        "REFINEMENT_UNSUPPORTED"
    )
    assert ScenarioRefinementEvidencePolicy.classify(structured, decision)[0] == (
        "REFINEMENT_SUPPORTED"
    )


def test_refinement_evidence_policy_accepts_range_containment():
    decision = ScenarioBindingDecision(
        dimension="EGO_DYNAMICS",
        authority=ScenarioBindingAuthority.RANGE_CONTAINMENT,
        refinable=True, source_refs=("PROJECT.ODD",), basis="fixture",
        parent_atom_id="FA001",
    )
    candidate = SimpleNamespace(
        atom_id="FA005", speed_range_kph=(0.0, 15.0),
        template_relationship="NONE", ranking_scores={"lexical_score": 0.0},
    )
    assert ScenarioRefinementEvidencePolicy.classify(candidate, decision) == (
        "REFINEMENT_SUPPORTED", "RANGE_CONTAINMENT",
    )


def test_offline_provider_request_contains_governance_and_bounded_metadata(method):
    service = ConstrainedScenarioSynthesisService(method)
    synthesis_input = _build(
        service, "Parking brake failure may contact a pedestrian.",
        parent=_parent(source_template=True),
    )
    payload = BoundedScenarioSynthesisAgent(None, service)._user_payload(
        synthesis_input
    )
    assert payload["dimension_applicability"]
    assert payload["dimension_constraints"]
    assert payload["scenario_coverage_plan"]["desired_variant_count"] in {1, 2, 3}
    assert payload["fm_scenario_template"]["template_id"] == "FM_TEMPLATE_001"
    candidate = payload["logical_atom_candidates"][0]
    assert candidate["filled_dimensions"]
    evidence = next(iter(candidate["evidence_by_dimension"].values()))
    assert evidence["candidate_origin"]
    assert evidence["selection_reason"]
    assert set(evidence["ranking_scores"]) == {
        "template_score", "mechanism_score", "action_score", "object_score",
        "traffic_relation_score", "odd_score", "causal_score", "lexical_score",
        "category_context_score", "structured_source_score",
        "physical_semantics_score",
        "source_evidence_tier", "final_rank_score",
    }
    assert "E_total" not in str(payload)
    all_ids = {item["atom_id"] for item in payload["logical_atom_candidates"]}
    assert len(all_ids) < len(service.by_id)


def test_selector_quality_audit_reports_plan_metrics_without_provider(method):
    service = ConstrainedScenarioSynthesisService(method)
    rear = _build(
        service,
        "Rear-end collision with a following rear vehicle.",
        parent=_parent(object_type="passenger_car", collision_type="rear"),
    )
    static = _build(
        service, "Stationary vehicle may contact a static obstacle.",
        parent=_parent(object_type="static_obstacle"),
    )
    audit = ScenarioSelectorQualityAudit().build((rear, static))
    assert audit["provider_calls"] == 0
    assert audit["parent_groups"] == 2
    assert audit["child_count"] == 0
    assert audit["traffic_pattern"]["required"] == 0
    assert audit["traffic_pattern"]["required_but_missing"] == 0
    assert audit["traffic_pattern"]["not_applicable"] == 2
    assert audit["ranking"]["combination_beam_truncated_groups"] == 0


def test_selector_quality_counts_unrefined_parent_atom_as_integer(method):
    service = ConstrainedScenarioSynthesisService(method)
    parent = _build(
        service, "Stationary vehicle may contact a static obstacle.",
        parent=_parent(object_type="static_obstacle"),
    )
    child = ScenarioCandidate(
        "CHILD", "parking", "bounded child", "",
        facts={"method_scenario_dimensions": {
            "EGO_DYNAMICS": {
                "parent_method_atom": "Audi_decel_emergency",
                "child_refined_atom": "",
            },
        }},
    )
    audit = ScenarioSelectorQualityAudit().build((parent,), children=(child,))
    assert audit["ego_dynamics"]["broad_bucket_retained"] == 1
