"""P7R counterexamples reconstructed from the supplied independent review.

These are explicitly synthetic algorithm oracles, not source integration proof.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import json

import pytest

from test_p7_closure import method, candidate, source, scoring_view, branch
from hara_agent.services.analysis import StructuredRiskScoringService
from hara_agent.services.analysis.scenario_physics import derive_scenario_physics, causal_physics_conflicts


def test_r03_lower_severity_higher_asil_is_retained(method):
    view = scoring_view(method)
    car = branch("car", "VEHICLE", "FRONTAL", 7); car["ttc_s"] = .154286
    pedestrian = branch("pedestrian", "PEDESTRIAN", "FRONTAL", 7); pedestrian["ttc_s"] = 8
    view["collision_branches"] = [car, pedestrian]
    result = StructuredRiskScoringService(method).score(view, "impact")
    assert result["branch_summary"]["highest_known_asil"] == "B"
    rows = {row["branch_id"]: row for row in result["branch_evaluations"]}
    assert (rows["car"]["S"], rows["car"]["E"], rows["car"]["C"], rows["car"]["ASIL"]) == ("S1", "E4", "C3", "B")
    assert rows["pedestrian"]["ASIL"] == "QM"


def test_r04_distance_override_invalidates_parent_ttc(method):
    view = scoring_view(method)
    view.update(ttc_s=8, closing_speed_kph=7, relative_distance_m=15.555556)
    view["_fact_provenance"].update({k: source() for k in ("ttc_s", "closing_speed_kph", "relative_distance_m")})
    child = branch("near", "VEHICLE", "FRONTAL", 7)
    child["relative_distance_m"] = .3
    child["_fact_provenance"]["relative_distance_m"] = source()
    view["collision_branches"] = [child]
    result = StructuredRiskScoringService(method).score(view, "impact")
    assert result["controllability"]["controllability_score"] == "C3"


def test_r05_accelerating_from_rest_uses_same_trajectory_for_time_and_speed():
    sc = candidate(ego_speed_kph=0, object_speed_kph=0, object_impact_speed_kph=0,
        ego_longitudinal_direction="FORWARD", object_longitudinal_direction="STATIONARY",
        object_position="FRONT", collision_type="FRONTAL", relative_distance_m=2,
        travel_to_impact_m=2, longitudinal_acceleration_mps2=2,
        fault_motion_model="CONSTANT_ACCELERATION_TO_IMPACT_DISTANCE", impact_time_id="t1")
    values = {r.evidence_ref: r.value for r in derive_scenario_physics(sc)}
    assert values["DERIVED.ego_impact_speed_kph"] == pytest.approx(10.182338, abs=1e-6)
    assert values.get("DERIVED.ttc_s") == pytest.approx(1.414214, abs=1e-6)


def test_r06_normalization_and_json_restore_preserve_time_role():
    sc = candidate(ego_speed_kph=7, object_speed_kph=0)
    sc.fact_provenance["ego_speed_kph"]["analysis_value_semantic"] = "PRE_FAULT_OPERATING_SPEED"
    sizes = []
    for _ in range(5):
        records = {r.evidence_ref: r for r in derive_scenario_physics(sc)}
        assert "DERIVED.relative_speed_kph" not in records
        record = records["DERIVED.ego_speed_kph"]
        metadata = {**record.metadata, "approval": record.approval_status.value,
                    "provenance": record.provenance.value,
                    "source_refs": [vars(ref) for ref in record.source_refs]}
        sc.fact_provenance["ego_speed_kph"] = json.loads(json.dumps(metadata))
        sizes.append(len(json.dumps(metadata)))
    assert sizes[-1] == sizes[1]


def test_r07_reverse_front_collision_requires_interaction_revalidation():
    sc = candidate(ego_speed_kph=7, object_speed_kph=0, ego_longitudinal_direction="REVERSE",
        object_longitudinal_direction="STATIONARY", object_position="FRONT", collision_type="FRONTAL")
    payload = {"hazardous_event": "车辆持续倒车，与前方静止行人越来越接近并发生碰撞。"}
    assert causal_physics_conflicts(sc, payload)


@pytest.mark.parametrize("initial,accel,distance,horizon,status,time", [
    (3.6, 0, 2, None, "CONTACT", 2),
    (3.6, -2, 2, None, "STOPPED_BEFORE_CONTACT", None),
    (0, 0, 2, None, "NO_POSITIVE_CONTACT_ROOT", None),
    (0, 2, 2, 1, "CONTACT_OUTSIDE_MODEL_INTERVAL", None),
])
def test_contact_model_boundaries(initial, accel, distance, horizon, status, time):
    from hara_agent.services.analysis.scenario_physics import constant_acceleration_contact
    result = constant_acceleration_contact(initial, accel, distance, horizon)
    assert result['status'] == status
    assert result.get('time_s') == time


def test_equivalence_keeps_sources_slope_and_time_but_ignores_generated_identity():
    from hara_agent.services.analysis.scenario_equivalence import scenario_conditions
    a = candidate(scenario_atom_ids=['slope'], slope_percent=15, slope_direction='UP', impact_time_id='event-a')
    b = deepcopy(a); b.scenario_id = 'SC-B'; b.facts['impact_time_id'] = 'event-b'
    assert scenario_conditions(a) == scenario_conditions(b)
    b.facts['slope_direction'] = 'DOWN'
    assert scenario_conditions(a) != scenario_conditions(b)
    b = deepcopy(a); b.fact_provenance['slope_percent']['source_refs'][0]['location'] = 'conflicting-source'
    assert scenario_conditions(a) != scenario_conditions(b)


def test_branch_without_own_approved_ttc_cannot_borrow_parent_authority(method):
    view = scoring_view(method)
    child = branch("unapproved", "VEHICLE", "FRONTAL", 7)
    child["ttc_s"] = .154286
    child["_fact_provenance"]["ttc_s"] = {**source(), "approval": "PENDING"}
    view["collision_branches"] = [child]
    result = StructuredRiskScoringService(method).score(view, "impact")
    assert result["branch_evaluations"][0]["C"] == ""
    assert result["branch_summary"]["status"] == "PENDING"


def test_acceleration_model_does_not_contact_target_behind_motion():
    sc = candidate(ego_speed_kph=0, object_speed_kph=0, object_impact_speed_kph=0,
        ego_longitudinal_direction="REVERSE", object_longitudinal_direction="STATIONARY",
        object_position="FRONT", collision_type="FRONTAL", relative_distance_m=2,
        travel_to_impact_m=2, longitudinal_acceleration_mps2=2,
        fault_motion_model="CONSTANT_ACCELERATION_TO_IMPACT_DISTANCE", impact_time_id="t1")
    values = {r.evidence_ref: r.value for r in derive_scenario_physics(sc)}
    assert values["DERIVED.motion_model_outcome"]["status"] == "MODEL_GEOMETRY_NOT_APPLICABLE"
    assert "DERIVED.ego_impact_speed_kph" not in values
    assert "DERIVED.ttc_s" not in values
    assert causal_physics_conflicts(sc, {"hazardous_event": "与前方行人碰撞"})


def test_risk_fact_binding_and_json_restore_keep_time_semantics(method):
    from dataclasses import asdict
    from hara_agent.models import ItemDefinitionFacts, RiskFact, SourceRef, FactProvenance, ReviewStatus
    from hara_agent.models.project_facts import risk_fact_from_dict
    from hara_agent.services.analysis import MethodRiskFactBindingService
    fact = RiskFact("v0", "ego_speed_kph", 7, "km/h",
        source_refs=[SourceRef("human_confirmation", "synthetic", "initial")],
        provenance=FactProvenance.PROJECT_INPUT, approval=ReviewStatus.FINALIZED,
        physical_semantics={"analysis_value_semantic": "PRE_FAULT_OPERATING_SPEED", "time_role": "INITIAL"})
    restored = risk_fact_from_dict(json.loads(json.dumps(asdict(fact))))
    project = ItemDefinitionFacts(system_description="synthetic", item_boundary="synthetic", risk_facts=[restored], sources=fact.source_refs)
    binding = MethodRiskFactBindingService(method).bind(project, {})
    assert binding.provenance["ego_speed_kph"]["analysis_value_semantic"] == "PRE_FAULT_OPERATING_SPEED"


def test_producer_selects_model_only_from_explicit_post_fault_source(method):
    from hara_agent.services.analysis.analytical_physics_instantiation_service import AnalyticalPhysicsInstantiationService
    sc = candidate(ego_speed_kph=0, object_speed_kph=0, relative_distance_m=2,
        longitudinal_acceleration_mps2=2, ego_longitudinal_direction="FORWARD",
        object_longitudinal_direction="STATIONARY", object_position="front", collision_type="FRONTAL", road_user_type="VEHICLE")
    mf = dict(malfunction_id="MF", function_id="F", guideword="More", description="synthetic drive",
        functional_effect="synthetic", vehicle_level_hazard="impact", causal_chain=["fault", "effect"],
        component_category="actuator_motor", failure_type="excessive")
    producer = AnalyticalPhysicsInstantiationService(method)
    assert "fault_motion_model" not in producer.produce_inputs(sc, mf).facts
    sc.fact_provenance["longitudinal_acceleration_mps2"]["analysis_value_semantic"] = "POST_FAULT_CONSTANT_ACCELERATION"
    produced = producer.produce_inputs(sc, mf)
    assert produced.facts["fault_motion_model"] == "CONSTANT_ACCELERATION_TO_IMPACT_DISTANCE"
    assert produced.facts["relative_speed_kph"] == pytest.approx(10.182338, abs=1e-6)
    assert produced.facts["ttc_s"] == pytest.approx(1.414214, abs=1e-6)
    assert len(produced.facts["collision_branches"]) == 1
    from hara_agent.models import ItemDefinitionFacts, RiskFact, SourceRef, FactProvenance, ReviewStatus
    refs = [SourceRef("human_confirmation", "synthetic", "post-fault acceleration")]
    fact = RiskFact("accel", "longitudinal_acceleration_mps2", 2, "m/s²",
        context={"malfunction_id": "MF"}, source_refs=refs, provenance=FactProvenance.PROJECT_INPUT,
        approval=ReviewStatus.FINALIZED, physical_semantics={"analysis_value_semantic": "POST_FAULT_CONSTANT_ACCELERATION"})
    project = ItemDefinitionFacts(system_description="synthetic", item_boundary="synthetic", risk_facts=[fact], sources=refs)
    sc.facts.pop("longitudinal_acceleration_mps2")
    sc.fact_provenance.pop("longitudinal_acceleration_mps2")
    bound = producer.produce_inputs(sc, mf, project_facts=project)
    assert bound.facts["ttc_s"] == pytest.approx(1.414214, abs=1e-6)
    assert bound.fact_provenance["longitudinal_acceleration_mps2"]["source_binding_kind"] == "EXACT_SOURCE_PHYSICAL_QUANTITY"
    invalid = replace(fact, unit="m/s")
    project.risk_facts = [invalid]
    assert "fault_motion_model" not in producer.produce_inputs(sc, mf, project_facts=project).facts


def test_impact_speed_from_different_phases_is_not_combined():
    sc = candidate(ego_speed_kph=7, object_speed_kph=0, ego_impact_speed_kph=10,
        object_impact_speed_kph=0, impact_time_id="contact1")
    sc.fact_provenance["object_impact_speed_kph"]["impact_time_id"] = "contact2"
    assert "DERIVED.relative_speed_kph" not in {r.evidence_ref for r in derive_scenario_physics(sc)}


def test_branch_geometry_change_discards_obsolete_modeled_impact():
    from hara_agent.services.analysis.scenario_physics import branch_physics_view
    facts = dict(ego_speed_kph=0, object_speed_kph=0, ego_impact_speed_kph=10.182338,
        object_impact_speed_kph=0, relative_speed_kph=10.182338, ttc_s=1.414214,
        ego_longitudinal_direction="FORWARD", object_longitudinal_direction="STATIONARY",
        object_position="FRONT", collision_type="FRONTAL", relative_distance_m=2, travel_to_impact_m=2,
        longitudinal_acceleration_mps2=2, fault_motion_model="CONSTANT_ACCELERATION_TO_IMPACT_DISTANCE", impact_time_id="t1")
    parent = {**facts, "_fact_provenance": {key: source() for key in facts}}
    child = {"object_position": "REAR", "_fact_provenance": {"object_position": source()}}
    view = branch_physics_view(parent, child)
    assert "ego_impact_speed_kph" not in view
    assert "relative_speed_kph" not in view
    assert "ttc_s" not in view


def test_review_new_geometry_recomputes_parent_closing_and_ttc(method):
    from hara_agent.services.analysis.scenario_physics import branch_physics_view
    parent = scoring_view(method)
    parent.update(relative_distance_m=.3, closing_speed_kph=.135, ttc_s=8)
    parent["_fact_provenance"].update({k: source() for k in ("relative_distance_m", "closing_speed_kph")})
    child = branch("new-geometry", "VEHICLE", "FRONTAL", 7)
    child.update(ego_speed_kph=7, object_speed_kph=0, ego_longitudinal_direction="FORWARD",
                 object_longitudinal_direction="STATIONARY", object_position="FRONT")
    child["_fact_provenance"].update({k: source() for k in child if not k.startswith("_")})
    view = branch_physics_view(parent, child)
    assert view["closing_speed_kph"] == 7
    assert view["ttc_s"] == pytest.approx(.154286)
    parent["collision_branches"] = [child]
    row = StructuredRiskScoringService(method).score(parent, "impact")["branch_evaluations"][0]
    assert (row["S"], row["E"], row["C"], row["ASIL"]) == ("S1", "E4", "C3", "B")


def test_review_new_model_horizon_cannot_retain_finalized_impact(method):
    from hara_agent.services.analysis.scenario_physics import branch_physics_view
    parent = scoring_view(method)
    facts = dict(ego_speed_kph=0, object_speed_kph=0, object_impact_speed_kph=0,
        ego_impact_speed_kph=10.182338, relative_speed_kph=10.182338, ttc_s=1.414214,
        ego_longitudinal_direction="FORWARD", object_longitudinal_direction="STATIONARY",
        object_position="FRONT", collision_type="FRONTAL", road_user_type="VEHICLE",
        relative_distance_m=2, travel_to_impact_m=2, longitudinal_acceleration_mps2=2,
        fault_motion_model="CONSTANT_ACCELERATION_TO_IMPACT_DISTANCE", impact_time_id="t1")
    parent.update(facts)
    parent["_fact_provenance"].update({k: source() for k in facts})
    child = {"branch_id": "short-horizon", "mechanism_id": "IMPACT", "impact_time_id": "t1",
             "physically_possible": True, "model_valid_until_s": .1,
             "_fact_provenance": {"model_valid_until_s": source()}}
    view = branch_physics_view(parent, child)
    assert view["motion_model_outcome"]["status"] == "CONTACT_OUTSIDE_MODEL_INTERVAL"
    assert "ego_impact_speed_kph" not in view
    assert "relative_speed_kph" not in view
    assert "ttc_s" not in view
    parent["collision_branches"] = [child]
    result = StructuredRiskScoringService(method).score(parent, "impact")
    assert result["branch_summary"]["status"] == "PENDING"
    assert result["branch_evaluations"][0]["ASIL"] == ""


def test_review_noncontact_harm_is_not_external_target_approach():
    sc = candidate(ego_speed_kph=7, object_speed_kph=0, ego_longitudinal_direction="REVERSE",
        object_longitudinal_direction="STATIONARY", object_position="FRONT", collision_type="FRONTAL")
    assert causal_physics_conflicts(sc, {"hazardous_event":
        "车辆倒车远离前方静止目标；非预期急减速导致车内乘员前倾磕碰。"}) == []


def test_review_atomic_extraction_preserves_explicit_time_semantics():
    from hara_agent.models import ProjectFactOutputType, SourceRef
    from hara_agent.services.extraction import ProjectFactNormalizer, RequiredProjectFactSpec
    spec = RequiredProjectFactSpec("longitudinal_acceleration_mps2", ProjectFactOutputType.RISK_FACT,
        ("longitudinal_acceleration_mps2",), ("m/s²",), (), ("value",))
    raw = {"value": 2, "unit": "m/s²", "context": {},
           "physical_semantics": {"analysis_value_semantic": "POST_FAULT_CONSTANT_ACCELERATION"}}
    result = ProjectFactNormalizer()._atomic_fact(spec, raw,
        SourceRef("item_definition", "synthetic", "p1", "POST_FAULT_CONSTANT_ACCELERATION 2 m/s²"))
    assert result.physical_semantics == raw["physical_semantics"]


@pytest.mark.parametrize("change", ["delete", "source", "phase"])
def test_review_dependency_changes_invalidate_inherited_values(change):
    from hara_agent.services.analysis.scenario_physics import branch_physics_view
    facts = dict(ego_speed_kph=7, object_speed_kph=0, ego_longitudinal_direction="FORWARD",
        object_longitudinal_direction="STATIONARY", object_position="FRONT", collision_type="FRONTAL",
        relative_distance_m=.3, closing_speed_kph=7, relative_speed_kph=7, ttc_s=.154286)
    parent = {**facts, "_fact_provenance": {k: source() for k in facts}}
    if change == "delete":
        child = {"ego_speed_kph": None}
    else:
        metadata = {**source(), **({"approval": "PENDING"} if change == "source" else
                                    {"analysis_value_semantic": "PRE_FAULT_OPERATING_SPEED"})}
        child = {"_fact_provenance": {"ego_speed_kph": metadata}}
    view = branch_physics_view(parent, child)
    if change == "delete":
        assert "closing_speed_kph" not in view and "ttc_s" not in view
    elif change == "source":
        assert view["_fact_provenance"]["ttc_s"]["approval"] == "PENDING"
    else:
        assert "relative_speed_kph" not in view


@pytest.mark.parametrize("distance", [.2, .3])
def test_review_direct_branch_measurement_conflict_is_visible_and_not_final(method, distance):
    parent = scoring_view(method)
    parent.update(ego_speed_kph=7, object_speed_kph=0, relative_distance_m=.3,
        ego_longitudinal_direction="FORWARD", object_longitudinal_direction="STATIONARY", object_position="FRONT")
    parent["_fact_provenance"].update({k: source() for k in parent if not k.startswith("_")})
    child = branch("measurement", "VEHICLE", "FRONTAL", 7)
    child.update(relative_distance_m=distance, ttc_s=8)
    child["_fact_provenance"]["relative_distance_m"] = source()
    parent["collision_branches"] = [child]
    row = StructuredRiskScoringService(method).score(parent, "impact")["branch_evaluations"][0]
    assert row["ASIL"] == ""
    assert any(c["field"] == "ttc_s" and c["supplied"] == 8 for c in row["physical_input_conflicts"])


def test_review_structured_contact_is_scoped_to_target_and_phase():
    sc = candidate(ego_speed_kph=7, object_speed_kph=0, ego_longitudinal_direction="REVERSE",
        object_longitudinal_direction="STATIONARY", object_position="FRONT", collision_type="FRONTAL")
    interaction = {"target": "SCENARIO_OBJECT", "object_position": "FRONT",
                   "relation": "CONTACT", "assessment_reference": "CURRENT_STATE"}
    assert causal_physics_conflicts(sc, {"physical_interaction": interaction})
    for change in ({"target": "OTHER"}, {"assessment_reference": "OTHER"}, {"object_position": "REAR"}):
        assert causal_physics_conflicts(sc, {"physical_interaction": {**interaction, **change}}) == []
    from hara_agent.models import ScenarioFeasibilityAssessment
    assessment = ScenarioFeasibilityAssessment("MF", "SC", True, True, True, [], "synthetic",
        hazardous_event="scoped contact", physical_interaction=interaction)
    restored = ScenarioFeasibilityAssessment.from_dict(json.loads(json.dumps(assessment.to_dict())))
    assert causal_physics_conflicts(sc, restored.to_dict())


@pytest.mark.parametrize("text,model_expected", [
    ("longitudinal_acceleration_mps2=2 m/s²; POST_FAULT_CONSTANT_ACCELERATION", True),
    ("故障发生后，在接触前方目标前，车辆沿前进方向以恒定纵向加速度2m/s²运动。", True),
    ("正常功能的起步纵向加速度上限为2m/s²。", False),
])
def test_review_targeted_extraction_to_automatic_dynamic_model(method, text, model_expected):
    """Synthetic source response through production request, normalize, bind, producer."""
    from hara_agent.services.extraction import build_project_fact_spec_batches
    from hara_agent.services.semantic.targeted_project_fact_agent import TargetedProjectFactExtractionAgent
    from hara_agent.services.semantic.item_supplement_agent import RoutedDocumentBlocks
    from hara_agent.infrastructure.llm import LLMResponse
    from hara_agent.models import ItemDefinitionFacts
    from hara_agent.services.analysis.analytical_physics_instantiation_service import AnalyticalPhysicsInstantiationService
    specs = build_project_fact_spec_batches((), ())["physical_motion_inputs"]
    class Client:
        def complete_json(self, request):
            assert "physical_semantics" in request.system_prompt
            assert "longitudinal_acceleration_mps2" in request.user_prompt
            return LLMResponse({"results": [{"fact_type": spec.fact_type, "status": "FOUND",
                "value": 2, "unit": "m/s²", "context": {}, "source_block_id": "B1", "source_excerpt": text,
                "physical_semantics": {"analysis_value_semantic": "POST_FAULT_CONSTANT_ACCELERATION"}}
                if spec.fact_type == "longitudinal_acceleration_mps2" else
                {"fact_type": spec.fact_type, "status": "NOT_FOUND"} for spec in specs]}, "offline-fixture")
    route = RoutedDocumentBlocks("project_evidence", ["B1"], text,
        [{"block_id": "B1", "text": text, "location": "synthetic:p1"}])
    normalized, _ = TargetedProjectFactExtractionAgent(Client()).extract("physical_motion_inputs", specs, route, "synthetic")
    assert not normalized.failures and len(normalized.risk_facts) == 1
    from hara_agent.models.project_facts import risk_fact_from_dict
    restored = risk_fact_from_dict(json.loads(json.dumps(asdict(normalized.risk_facts[0]))))
    assert restored == normalized.risk_facts[0]
    project = ItemDefinitionFacts("synthetic", "synthetic", risk_facts=[restored],
                                  sources=normalized.risk_facts[0].source_refs)
    sc = candidate(ego_speed_kph=0, object_speed_kph=0, relative_distance_m=2,
        ego_longitudinal_direction="FORWARD", object_longitudinal_direction="STATIONARY",
        object_position="FRONT", collision_type="FRONTAL", road_user_type="VEHICLE")
    mf = dict(malfunction_id="MF", function_id="F", guideword="More", description="synthetic drive",
        functional_effect="synthetic", vehicle_level_hazard="impact", causal_chain=["fault", "effect"],
        component_category="actuator_motor", failure_type="excessive")
    result = AnalyticalPhysicsInstantiationService(method).produce_inputs(sc, mf, project_facts=project)
    if not model_expected:
        assert normalized.risk_facts[0].approval.value == "PENDING"
        assert "fault_motion_model" not in result.facts
        assert "collision_branches" not in result.facts
        return
    assert result.facts["fault_motion_model"] == "CONSTANT_ACCELERATION_TO_IMPACT_DISTANCE"
    assert result.facts["ttc_s"] == pytest.approx(1.414214)
    assert result.facts["relative_speed_kph"] == pytest.approx(10.182338)
    assert len(result.facts["collision_branches"]) == 1
    if text.startswith("故障"):
        from hara_agent.services.extraction.project_fact_normalization import POST_FAULT_ACCELERATION_ZH_RULE
        evidence = [{"field": "analysis_value_semantic", "canonical_value": "POST_FAULT_CONSTANT_ACCELERATION",
                     "rule_id": POST_FAULT_ACCELERATION_ZH_RULE, "source_excerpt": text}]
        assert restored.normalization_evidence == evidence
        assert result.fact_provenance["longitudinal_acceleration_mps2"]["normalization_evidence"] == evidence


@pytest.mark.parametrize("text,value,unit,finalized", [
    ("故障后，本车以恒定的纵向加速度为2m/s^2运动。", 2, "m/s²", True),
    ("自车在故障发生后以恒定纵向加速度 2.0 m/s2 运动。", 2, "m/s²", True),
    ("故障后，车辆以恒定纵向加速度-2m/s²运动。", -2, "m/s²", True),
    ("正常功能的起步纵向加速度上限为2m/s²。", 2, "m/s²", False),
    ("故障后，车辆纵向加速度上限为2m/s²。", 2, "m/s²", False),
    ("故障后，车辆并非以恒定纵向加速度2m/s²运动。", 2, "m/s²", False),
    ("假设故障后，车辆以恒定纵向加速度2m/s²运动。", 2, "m/s²", False),
    ("故障前，车辆以恒定纵向加速度2m/s²运动；故障后加速度未知。", 2, "m/s²", False),
    ("故障后，车辆以恒定纵向加速度2m/s²运动，但此数值未经确认。", 2, "m/s²", False),
    ("故障后，车辆以恒定纵向加速度2m/s²运动。故障后加速度不恒定。", 2, "m/s²", False),
    ("故障后，车辆以恒定纵向加速度2m/s²运动。", 3, "m/s²", False),
    ("故障后，车辆以恒定纵向加速度2m/s²运动。", 2, "km/h", False),
])
def test_explicit_chinese_phase_normalization_is_quantity_bound(text, value, unit, finalized):
    from hara_agent.models import SourceRef, ReviewStatus
    from hara_agent.services.extraction.project_fact_normalization import extracted_physical_semantics
    raw = {"fact_type": "longitudinal_acceleration_mps2", "value": value, "unit": unit,
           "physical_semantics": {"analysis_value_semantic": "POST_FAULT_CONSTANT_ACCELERATION"}}
    evidence = []
    semantics, approval = extracted_physical_semantics(raw,
        [SourceRef("item_definition", "synthetic", "p1", text)], normalization_evidence=evidence)
    assert semantics == raw["physical_semantics"]
    assert (approval is ReviewStatus.FINALIZED) is finalized
    assert bool(evidence) is finalized


def test_phase_excerpt_cannot_discard_source_qualification_or_conflict():
    from hara_agent.models import SourceRef, ReviewStatus
    from hara_agent.services.extraction.project_fact_normalization import extracted_physical_semantics
    statement = "故障后，车辆以恒定纵向加速度2m/s²运动。"
    raw = {"fact_type": "longitudinal_acceleration_mps2", "value": 2, "unit": "m/s²",
           "physical_semantics_source_excerpt": statement,
           "physical_semantics": {"analysis_value_semantic": "POST_FAULT_CONSTANT_ACCELERATION"}}
    for sources in (
        [SourceRef("item_definition", "synthetic", "p1", "假设" + statement)],
        [SourceRef("item_definition", "synthetic", "p1", statement),
         SourceRef("item_definition", "synthetic", "p2", "故障后，车辆以恒定纵向加速度3m/s²运动。")],
    ):
        evidence = []
        _, approval = extracted_physical_semantics(raw, sources, normalization_evidence=evidence)
        assert approval is ReviewStatus.PENDING
        assert evidence == []


def test_targeted_quote_cannot_cut_a_hypothesis_out_of_its_source():
    from hara_agent.models import ReviewStatus
    from hara_agent.services.extraction import ProjectFactNormalizer, build_project_fact_spec_batches
    specs = build_project_fact_spec_batches((), ())["physical_motion_inputs"]
    statement = "故障后，车辆以恒定纵向加速度2m/s²运动。"
    raw = [{"fact_type": spec.fact_type, "status": "FOUND", "value": 2, "unit": "m/s²", "context": {},
            "source_block_id": "B1", "source_excerpt": statement,
            "physical_semantics": {"analysis_value_semantic": "POST_FAULT_CONSTANT_ACCELERATION"}}
           if spec.fact_type == "longitudinal_acceleration_mps2" else
           {"fact_type": spec.fact_type, "status": "NOT_FOUND"} for spec in specs]
    normalized = ProjectFactNormalizer().normalize(specs, raw,
        [{"block_id": "B1", "location": "p1", "text": "假设" + statement}], "synthetic")
    assert not normalized.failures
    assert normalized.risk_facts[0].approval is ReviewStatus.PENDING
    assert normalized.risk_facts[0].source_refs[0].excerpt == "假设" + statement


def test_review_semantic_interpretation_never_promotes_a_normal_limit():
    from hara_agent.services.extraction.project_fact_normalization import extracted_physical_semantics
    from hara_agent.models import SourceRef, ReviewStatus
    refs = [SourceRef("item_definition", "synthetic", "p1", "Normal longitudinal acceleration limit 2 m/s²")]
    assert extracted_physical_semantics({}, refs) == ({}, ReviewStatus.FINALIZED)
    semantics = {"analysis_value_semantic": "POST_FAULT_CONSTANT_ACCELERATION"}
    assert extracted_physical_semantics({"physical_semantics": semantics}, refs) == (semantics, ReviewStatus.PENDING)
    with pytest.raises(ValueError, match="INVALID_PHYSICAL_SEMANTICS"):
        extracted_physical_semantics({"physical_semantics": {"approval": "FINALIZED"}}, refs)


def test_review_scenario_risk_extraction_keeps_time_semantics():
    from hara_agent.contracts import FactType, FactOrigin, RequiredFactSpec
    from hara_agent.models import SourceRef
    from hara_agent.services.semantic.scenario_risk_fact_agent import ScenarioRiskFactAgent
    key = ("MF", "SC", "VEHICLE_SPEED")
    spec = RequiredFactSpec(FactType.VEHICLE_SPEED, ("SEVERITY",), "km/h", (), "", FactOrigin.SCENARIO_FACT, (), ())
    raw = {"malfunction_id": "MF", "scenario_id": "SC", "fact_type": "VEHICLE_SPEED", "status": "FOUND",
           "value": 7, "unit": "km/h", "evidence_ids": ["E1"],
           "physical_semantics": {"analysis_value_semantic": "PRE_FAULT_OPERATING_SPEED"}}
    source_ref = SourceRef("item_definition", "synthetic", "p1", "PRE_FAULT_OPERATING_SPEED 7 km/h")
    facts, seen, _, failures = ScenarioRiskFactAgent(None)._parse_results(
        {"results": [raw]}, {key}, {"VEHICLE_SPEED": spec}, {"E1": source_ref}, {("MF", "SC"): {"E1"}})
    assert not failures and key in seen
    assert facts[0].physical_semantics == raw["physical_semantics"]
