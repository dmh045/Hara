"""P6 ordinary-analyze integration; all semantic responses are local fixtures."""
from copy import deepcopy
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from hara_agent.method_sources import MethodSourceResolver
from hara_agent.models import (
    FunctionDefinition, ItemDefinitionFacts, MalfunctionCandidate, ReviewStatus,
    ScenarioCandidate, ScenarioFeasibilityAssessment, SourceRef,
)
from hara_agent.services.analysis import (
    ConstrainedScenarioSynthesisService, ExposureInputReadinessService,
    MethodScenarioCandidateService, StructuredRiskScoringService,
)
from hara_agent.services.analysis.scenario_equivalence import coalesce_candidates, driver_reuse_proof
from hara_agent.services.semantic import ScenarioFeasibilityAgent
from hara_agent.workflow import HARAState, WorkflowStage, ReviewArtifactReader, ReviewArtifactWriter
from hara_agent.workflow.nodes.scenarios import _assess_instantiated_scenarios, assess_scenarios

ROOT = Path(__file__).resolve().parents[2]
SOURCE = SourceRef("item_definition", "test-item.docx", "p1", "active parking vehicle control at 0–7 km/h")


@pytest.fixture(scope="module")
def method():
    return MethodSourceResolver().resolve(
        template_path=None,
        baseline_manifest_path=ROOT / "method_assets/fusa_baseline_v1/manifest.yaml",
        report_template_path=ROOT / "references/HARA_Template_AI_20260327.xlsx",
    ).method


def malfunction():
    return MalfunctionCandidate(
        "MF-P6", "F-P6", "No/Loss", "驱动控制指令丢失", "车辆无法按需求驱动",
        "车辆运动控制丧失", ["驱动指令丢失", "车辆运动控制丧失"],
        component_category="actuator_propulsion", failure_type="loss",
        status=ReviewStatus.FINALIZED, sources=[SOURCE],
    )


def parent(identity="SCN-P6"):
    return ScenarioCandidate(
        identity, "室内停车场", "AVP低速泊车控车", "停车场泊入和泊出",
        operating_mode="parking vehicle control",
        facts={"ego_speed_constraint": {"min_kph": 0, "max_kph": 7},
               "vehicle_state": "active", "method_scenario_dimensions": {}, "scenario_atom_ids": []},
        fact_provenance={"vehicle_state": {"provenance": "PROJECT_INPUT", "approval": "FINALIZED",
                                            "source_refs": [asdict(SOURCE)]}},
        sources=[SOURCE], status=ReviewStatus.FINALIZED,
    )


def project():
    return ItemDefinitionFacts(
        "parking vehicle control", "AVP", operating_modes=["parking vehicle control"],
        odd_locations=["室内停车场", "室外停车场"], odd_road_types=["停车场道路", "停车位"],
        speed_min_kph=0, speed_max_kph=7, sources=[SOURCE], status=ReviewStatus.FINALIZED,
    )


def function():
    return FunctionDefinition("F-P6", "驱动控制", "驱动控制指令", sources=[SOURCE])


def causal_payload(candidate, *, driver_relevant=False):
    action_field = "ego_action" if "ego_action" in candidate.facts else "vehicle_state"
    dependencies = [action_field, "vehicle_state"] + (["driver_position"] if driver_relevant else [])
    return {
        "scenario_id": candidate.scenario_id,
        "physically_feasible": True, "functionally_relevant": True, "causally_relevant": True,
        "breakpoint": "NONE", "confidence": .9, "hazardous_event": "车辆运动控制丧失",
        "risk_dimension_changes": [],
        "causal_chain": {
            "m_to_b": {"claim": "指令丢失导致无法按需求驱动", "basis_type": "DIRECT_FACT",
                       "evidence_refs": ["MF.functional_effect"]},
            "b_to_i": {"claim": "控制丧失发生于激活状态", "basis_type": "DIRECT_FACT",
                       "evidence_refs": ["SCN." + action_field]},
            "i_to_h": {"claim": "激活时无法按需求控制车辆运动", "basis_type": "DIRECT_FACT",
                       "evidence_refs": ["MF.vehicle_level_hazard", "SCN.vehicle_state"]},
        },
        "dependency_metadata": {
            "complete": True, "causal_evidence_fields": dependencies,
            "physical_feasibility_fields": ["ego_speed_constraint"], "scenario_identity_fields": [],
            "noninterference_fields": [] if driver_relevant else ["driver_position", "allowed_driver_position", "driver_in_vehicle"],
            "child_subset_refinement": "The fixture's vehicle-control hazard exists for either allowed driver position.",
        },
    }


class MockCausalAgent:
    prompt_version = ScenarioFeasibilityAgent.PROMPT_VERSION
    assessment_contract_version = "p6-test"
    client = SimpleNamespace(config=SimpleNamespace(provider="mock", base_url="local", model="local"))

    def __init__(self, *, driver_relevant=False):
        self.seen = []
        self.driver_relevant = driver_relevant

    def assess(self, mf, candidates, project_registry=None):
        self.seen.extend(candidates)
        return [ScenarioFeasibilityAgent._parse(
            mf, causal_payload(candidate, driver_relevant=self.driver_relevant),
            scenario=candidate, project_registry=project_registry,
        ) for candidate in candidates], {"llm_calls": 1, "usage": {"batches": []}}


def test_a_b_c_parking_candidates_are_bounded_and_have_method_atoms(method):
    service = ConstrainedScenarioSynthesisService(method)
    children, audit = service.generate_candidates(
        malfunction=asdict(malfunction()), parent=parent(), project_context=asdict(project()), function=asdict(function()),
    )
    assert children
    selected = {atom for child in children for atom in child.facts["scenario_atom_ids"]}
    assert {"FA005", "FA033"} <= selected
    for child in children:
        assert child.facts["scenario_atom_ids"]
        assert child.facts["vehicle_state"] == "active"
        assert child.fact_provenance["vehicle_state"]["provenance"] == "PROJECT_INPUT"
        assert child.fact_provenance["ego_action"]["provenance"] == "SCENARIO_DEFINED"
        assert child.fact_provenance["operating_scenario"]["selection_basis"] == "VALIDATED_ANALYTICAL_SCENARIO_ATOM"
        bounds = child.facts["ego_speed_constraint"]
        assert 0 <= bounds["min_kph"] <= bounds["max_kph"] <= 7
        for atom_id in child.facts["scenario_atom_ids"]:
            assert service._speed_compatible(service.by_id[atom_id], (0, 7))
    high_speed = {atom_id for atom_id, atom in service.by_id.items() if not service._speed_compatible(atom, (0, 7))}
    assert high_speed
    assert not selected & high_speed
    assert len(children) == audit["coverage_plan"]["desired_variant_count"]
    assert audit["provider_calls"] == 0


def test_b_open_speed_boundary_is_not_a_valid_intersection():
    atom = {"speed_range_kph": [7, None], "physical_semantics": {"ego_dynamics": {"min_inclusive": False}}}
    assert not ConstrainedScenarioSynthesisService._speed_compatible(atom, (0, 7))
    assert ConstrainedScenarioSynthesisService._speed_compatible(atom, (0, 8))


def test_d_e_generated_raw_atom_identity_reaches_native_e_scoring(method):
    mf = replace(malfunction(), component_category="actuator_longitudinal", description="制动指令丢失", functional_effect="制动输出丢失")
    source_parent = parent()
    source_parent.facts["method_scenario_dimensions"]["EGO_ACTION"] = {
        "atom_id": "FV010", "canonical_atom_id": "PH005", "resolution_status": "RESOLVED",
        "binding_authority": "EXACT_METHOD_MAPPING",
    }
    children, _ = MethodScenarioCandidateService(method).generate_for_malfunction(
        mf, [source_parent], project_facts=project(), function=function(),
    )
    alias_children = [c for c in children if any(
        b.get("atom_id") and b.get("atom_id") != b.get("canonical_atom_id")
        for b in c.facts["method_scenario_dimensions"].values()
    )]
    assert alias_children
    for child in alias_children:
        facts = {**child.facts, "component_category": mf.component_category}
        readiness = ExposureInputReadinessService(method).assess(facts)
        assert readiness["status"].startswith("READY")
        from hara_agent.services.analysis import ExposureDimensionCoverageService
        facts["_exposure_dimension_coverage_decision"] = ExposureDimensionCoverageService(method).decide(
            assessment_key=f"{mf.malfunction_id}::{child.scenario_id}", function=function(), operating_mode=child.operating_mode,
        )
        scored = StructuredRiskScoringService(method).score(facts, "vehicle hazard")
        assert scored["exposure"]["exposure_score"] == readiness["baseline_exposure"]["value"]
        assert scored["exposure"]["engineering_status"] == "FINALIZED"


def test_odds_apply_to_location_words_in_every_atom_dimension(method):
    service = ConstrainedScenarioSynthesisService(method)
    mf = replace(malfunction(), vehicle_level_hazard="停车场车辆转向时与对向车辆碰撞")
    built = service.build_input(malfunction=asdict(mf), parent=parent(), assessment={},
                               project_context=asdict(project()))
    selected = {c.atom_id for group in built.dimension_candidate_sets for c in group.candidates}
    forbidden = {a["atom_id"] for a in service.catalog if any(
        word in a["label"].casefold() for word in ("motorway", "expressway", "highway")
    )}
    assert forbidden
    assert not selected & forbidden, sorted(selected & forbidden)
    assert "PU008" not in selected and "FA037" not in selected


def test_source_environment_and_geometry_survive_generation_and_coalescing(method):
    parents = [replace(parent(str(i)), operating_scenario=place,
                       facts={**parent().facts, "operating_scenario": place})
               for i, place in enumerate(["室外停车场", "室内停车场", "水平车位", "垂直车位", "室外停车场"])]
    children, audit = MethodScenarioCandidateService(method).generate_for_malfunction(
        malfunction(), parents, project_facts=project(), function=function(),
    )
    scopes = {c.facts["source_operating_conditions"]["operating_scenario"] for c in children}
    assert scopes == {"室外停车场", "室内停车场", "水平车位", "垂直车位"}
    for child in children:
        if child.facts["source_operating_conditions"]["operating_scenario"] == "室外停车场":
            assert "CN_parking_multistory" not in child.facts["scenario_atom_ids"]
    assert audit["atom_generation"]["coalesced_candidate_count"] > 0  # true duplicate outdoor input
    before = {}
    service = ConstrainedScenarioSynthesisService(method)
    for source in parents:
        generated, _ = service.generate_candidates(malfunction=asdict(malfunction()), parent=source,
                                                   project_context=asdict(project()))
        before.update({c.scenario_id: c.facts["source_operating_conditions"] for c in generated})
    _, mappings = coalesce_candidates([c for source in parents for c in service.generate_candidates(
        malfunction=asdict(malfunction()), parent=source, project_context=asdict(project()))[0]])
    assert all(len({str(before[sid]) for sid in group["scenario_ids"]}) == 1 for group in mappings)


def test_explicit_uphill_downhill_boundaries_are_kept_and_not_flattened(method):
    p = replace(project(), odd_road_surfaces=["停车场路面", "上坡15%坡度", "下坡15%坡度"])
    children, audit = ConstrainedScenarioSynthesisService(method).generate_candidates(
        malfunction=asdict(malfunction()), parent=parent(), project_context=asdict(p),
    )
    slopes = [child for child in children if child.facts.get("source_slope_pct") == 15]
    assert {c.facts["source_slope_direction"] for c in slopes} == {"上坡", "下坡"}
    assert len({c.scenario_id for c in children}) == len(children)
    assert all(c.facts["method_scenario_dimensions"]["ROAD"]["atom_id"] == "FB003" for c in slopes)
    assert all(c.facts["source_operating_conditions"]["road_surface_conditions"] in p.odd_road_surfaces for c in slopes)
    assert len(audit["source_contexts"]) == 3
    assert all('FV040' not in c.facts['method_scenario_dimensions']['EGO_X_ROAD'].get('candidate_atom_ids', []) for c in slopes)
    catalog = ConstrainedScenarioSynthesisService(method).by_id
    assert catalog['FV040']['physical_semantics']['slope']['range_pct'] == [2, 8]
    assert catalog['FV050']['physical_semantics']['slope']['pct_min'] == 8


def test_source_road_axis_is_bounded_without_crossing_every_location(method):
    p = replace(project(), odd_road_surfaces=["停车场路面", "上坡15%坡度", "下坡15%坡度"])
    parents = [replace(parent(str(i)), source_scenario_id="METHOD_SCENARIO_ONTOLOGY",
                       operating_scenario=place, facts={**parent().facts, "operating_scenario": place})
               for i, place in enumerate(["室外停车场", "室内停车场", "垂直车位"])]
    children, audit = MethodScenarioCandidateService(method).generate_for_malfunction(
        malfunction(), parents, project_facts=p, function=function(),
    )
    scopes = [c.facts["source_operating_conditions"] for c in children]
    assert {s["operating_scenario"] for s in scopes} == {p.operating_scenario for p in parents}
    assert {s.get("source_slope_direction") for s in scopes if s.get("source_slope_pct")} == {"上坡", "下坡"}
    assert {s["operating_scenario"] for s in scopes if s.get("source_slope_pct")} == {"室外停车场"}
    assert audit["atom_generation"]["source_boundary_policy"] == "ONE_SOURCE_ANCHOR_PER_PHYSICAL_OPTION"
    from hara_agent.services.semantic.scenario_batching import _scenario_json
    import json
    slope_child = next(c for c in children if c.facts.get("source_slope_pct"))
    prompt = json.loads(_scenario_json(malfunction(), slope_child, causal_evidence_budget=1))
    assert prompt["source_operating_conditions"] == slope_child.facts["source_operating_conditions"]
    assert prompt["selected_method_conditions"]["ROAD"] == slope_child.facts["road_surface_conditions"]


def test_road_type_scope_is_not_multiplied_into_where_locations(method):
    from hara_agent.services.analysis import ProjectFactResolver
    from hara_agent.models import SpeedEnvelope
    service = MethodScenarioCandidateService(method)
    p = replace(project(), odd_road_types=["停车场道路", "水平车位", "垂直车位", "斜向车位"],
                speed_envelopes=[SpeedEnvelope("parking vehicle control", 0, 7,
                                              sources=[SOURCE], status=ReviewStatus.FINALIZED)])
    speed = ProjectFactResolver().resolve_speed_context(p, "parking vehicle control")
    parents, _ = service.generate(project_facts=p, operating_mode="parking vehicle control",
                                  speed_resolution=speed, functions=[function()])
    assert {s.operating_scenario for s in parents} == set(p.odd_locations)
    assert all(s.facts["source_road_type_scope"] == p.odd_road_types for s in parents)
    children, _ = service.generate_for_malfunction(malfunction(), parents, project_facts=p, function=function())
    assert all(s.facts["source_operating_conditions"]["permitted_road_types"] == p.odd_road_types for s in children)
    # More markings in the allowed scope must not create extra WHERE worlds.
    expanded = replace(p, odd_road_types=[*p.odd_road_types, "全框型标线车位", "T字框型标线车位"])
    more_parents, _ = service.generate(project_facts=expanded, operating_mode="parking vehicle control",
                                       speed_resolution=speed, functions=[function()])
    more, _ = service.generate_for_malfunction(malfunction(), more_parents, project_facts=expanded, function=function())
    assert len(more) == len(children)


def test_source_axis_sampling_preserves_primary_variants_without_claiming_equivalence(method):
    service = ConstrainedScenarioSynthesisService(method)
    sources = [replace(parent(str(i)), operating_scenario=place,
                       facts={**parent().facts, "operating_scenario": place})
               for i, place in enumerate(project().odd_locations)]
    full, pool, audits = [], [], []
    for source in sources:
        args = dict(malfunction=asdict(malfunction()), parent=source, project_context=asdict(project()))
        full_children, audit = service.generate_candidates(**args)
        full.extend(full_children)
        pool.extend((child, audit) for child in full_children)
        audits.append(audit)
    bounded = MethodScenarioCandidateService._select_context_representatives(pool)
    assert [c.scenario_id for c in bounded] == [c.scenario_id for c in MethodScenarioCandidateService._select_context_representatives(list(reversed(deepcopy(pool))))]
    axes = set(audits[0]["coverage_plan"]["primary_variation_dimensions"])
    def primary_variants(children):
        return {tuple((d, b.get("atom_id")) for d, b in c.facts["method_scenario_dimensions"].items() if d in axes)
                for c in children}
    assert primary_variants(full) == primary_variants(bounded)
    assert {c.facts["source_operating_conditions"]["operating_scenario"] for c in bounded} == set(project().odd_locations)
    assert len(bounded) < len(full)
    deferred = [row for a in audits for row in a["deferred_context_variants"]]
    assert len(deferred) == len(full) - len(bounded)
    assert all(row["status"] == "COMBINATION_NOT_EVALUATED" and not row["risk_equivalence_proven"] for row in deferred)


def test_h_exact_conditions_coalesce_with_complete_identity_map():
    a = replace(parent("A"), facts={"scenario_atom_ids": ["A1"], "ego_longitudinal_direction": "FORWARD"})
    b = replace(a, scenario_id="B", situational_description="different display wording")
    reverse = replace(a, scenario_id="C", facts={**a.facts, "ego_longitudinal_direction": "REVERSE"})
    retained, mapping = coalesce_candidates([a, b, reverse])
    assert len(retained) == 2
    assert {identity for group in mapping for identity in group["scenario_ids"]} == {"A", "B", "C"}
    assert retained[0].analysis_instance["equivalent_candidate_ids"] == ["A", "B"]


def test_equivalence_does_not_promote_pending_source_facts():
    a = replace(parent("A"), facts={"scenario_atom_ids": ["A1"], "vehicle_state": "active"})
    b = deepcopy(replace(a, scenario_id="B"))
    b.fact_provenance["vehicle_state"]["approval"] = "PENDING"
    retained, _ = coalesce_candidates([a, b])
    assert len(retained) == 2


@pytest.mark.parametrize("driver_relevant,independent,reused", [(False, 1, 1), (True, 2, 0)])
def test_h_i_driver_causal_reuse_is_conditional_and_revalidates_child(driver_relevant, independent, reused):
    a = replace(parent("IN"), facts={**parent().facts, "driver_position": "in_driver_seat"},
                analysis_instance={"p6_generation": True})
    b = replace(a, scenario_id="OUT", facts={**a.facts, "driver_position": "outside_driver_seat"})
    agent = MockCausalAgent(driver_relevant=driver_relevant)
    results, audit = _assess_instantiated_scenarios(agent, malfunction(), [a, b], {}, None)
    assert len(agent.seen) == independent
    assert audit["causal_reuse"]["reused_count"] == reused
    assert {r.scenario_id for r in results} == {"IN", "OUT"}
    assert all(r.retain for r in results)
    assert results[1].causal_assessment.scenario_id == "OUT"
    assert ScenarioFeasibilityAssessment.from_dict(results[0].to_dict()).dependency_metadata == results[0].dependency_metadata


def test_driver_reuse_rejects_a_cited_driver_input_even_when_declared_irrelevant():
    a = replace(parent("A"), facts={**parent().facts, "driver_position": "in_driver_seat"})
    b = replace(a, scenario_id="B", facts={**a.facts, "driver_position": "outside_driver_seat"})
    payload = causal_payload(a)
    payload["causal_chain"]["b_to_i"]["evidence_refs"].append("SCN.driver_position")
    assert driver_reuse_proof(payload, a, b)["reason"] == "DRIVER_FACT_USED_BY_CAUSAL_CHAIN"


def test_causal_response_checkpoint_survives_mid_malfunction_failure(tmp_path):
    from hara_agent.infrastructure.llm import LLMResponse
    a, b = parent("A"), parent("B")
    class Client:
        config = SimpleNamespace(provider="mock", model="local", base_url="local")
        def __init__(self, responses):
            self.responses, self.calls = list(responses), 0
        def complete_json(self, request):
            self.calls += 1
            value = self.responses.pop(0)
            if isinstance(value, Exception):
                raise value
            return LLMResponse({"assessments": [causal_payload(value)]}, "local",
                               usage={"prompt_tokens": 12, "completion_tokens": 8})
    first = Client([a, RuntimeError("interrupted")])
    with pytest.raises(RuntimeError, match="interrupted"):
        ScenarioFeasibilityAgent(first, batch_max_items=1, response_checkpoint_dir=tmp_path).assess(malfunction(), [a, b])
    assert len(list(tmp_path.glob('*.json'))) == 1
    resumed = Client([b])
    assessments, audit = ScenarioFeasibilityAgent(resumed, batch_max_items=1, response_checkpoint_dir=tmp_path).assess(malfunction(), [a, b])
    assert [a.scenario_id for a in assessments] == ["A", "B"]
    assert resumed.calls == audit['actual_llm_calls'] == 1
    assert audit['response_checkpoint_hits'] == 1
    assert audit['prompt_tokens_total'] == 12
    assert audit['completion_tokens_total'] == 8


def test_response_checkpoint_invalidates_changed_prompt_or_model(tmp_path):
    from hara_agent.infrastructure.llm import LLMRequest, LLMResponse
    from hara_agent.infrastructure.llm.response_checkpoint import ResponseCheckpoint
    class Client:
        config = SimpleNamespace(provider="mock", base_url="local", model="first")
        calls = 0
        def complete_json(self, request):
            self.calls += 1
            return LLMResponse({"answer": request.user_prompt}, self.config.model)
    client, store = Client(), ResponseCheckpoint(tmp_path)
    request = LLMRequest('test', 'system', 'source A', 'schema', 'version')
    store.complete(client, request)
    assert store.complete(client, request).usage['response_checkpoint_hit']
    store.complete(client, replace(request, user_prompt='source B'))
    client.config.model = 'second'
    store.complete(client, request)
    assert client.calls == 3


def test_l_causal_reuse_does_not_reuse_c_or_asil_ratings(method):
    from hara_agent.services.analysis import ExposureDimensionCoverageService, MethodContractASILService
    base = {**parent().facts, "scenario_atom_ids": ["SO010", "PH005"],
            "relative_speed_kph": 4.0, "collision_type": "FRONTAL", "road_user_type": "VEHICLE",
            "component_category": "sensor_camera", "ttc_s": 4.1,
            "emergency_braking_available": True,
            "remote_intervention_available": False, "other_road_user_avoidance_possible": False}
    a = replace(parent("IN"), facts={**base, "driver_in_vehicle": True}, analysis_instance={"p6_generation": True})
    b = replace(a, scenario_id="OUT", facts={**base, "driver_in_vehicle": False})
    agent = MockCausalAgent()
    results, audit = _assess_instantiated_scenarios(agent, malfunction(), [a, b], {}, None)
    assert audit["causal_reuse"]["reused_count"] == 1
    scoring, asil = StructuredRiskScoringService(method), MethodContractASILService(method)
    ratings = []
    for candidate, causal in zip((a, b), results):
        facts = {**candidate.facts, "scenario_id": candidate.scenario_id, "malfunction_id": malfunction().malfunction_id,
                 "_fact_provenance": {"relative_speed_kph": {"provenance": "PROJECT_INPUT", "approval": "FINALIZED", "source_refs": [asdict(SOURCE)]}}}
        facts["_exposure_dimension_coverage_decision"] = ExposureDimensionCoverageService(method).decide(
            assessment_key=candidate.scenario_id, function=None, operating_mode=candidate.operating_mode,
        )
        scored = scoring.score(facts, causal.hazardous_event)
        sec = tuple(scored[field][key] for field, key in (
            ("severity", "severity_score"), ("exposure", "exposure_score"), ("controllability", "controllability_score")))
        assert sec[0] == "S1"
        ratings.append((*sec, asil.determine(*sec)))
    assert ratings[0][2] == "C1"
    assert ratings[1][2] == "C3"
    assert ratings[0][3] != ratings[1][3]


def test_ordinary_scenario_node_generates_atoms_before_causal_agent(method, tmp_path):
    mf = malfunction()
    state = HARAState("p6-mock-main", stage=WorkflowStage.MALFUNCTIONS,
                      item_definition={"typed": asdict(project())}, functions=[asdict(function())], malfunctions=[asdict(mf)])
    agent = MockCausalAgent()
    writer = ReviewArtifactWriter(state.run_id, tmp_path)
    result = assess_scenarios(state, agent, [mf], [parent(), parent("SECOND")], method=method, review_artifact_writer=writer)
    assert agent.seen
    assert all(c.facts["scenario_atom_ids"] for c in agent.seen)
    assert all(c.analysis_instance["p6_generation"] for c in agent.seen)
    assert result.item_definition["scenario_assessments"]
    assert all(a["causal_assessment"]["status"] == "VALIDATED" for a in result.item_definition["scenario_assessments"])
    records = ReviewArtifactReader(state.run_id, tmp_path).read_all()
    from hara_agent.workflow.review_artifacts import _expected_candidate_count
    assert _expected_candidate_count(records["scenario_candidate"], mf.malfunction_id) == len(agent.seen)
    assert len(records["scenario_candidate"]) > len(agent.seen)
    assert all(row["facts"]["ego_action"] for row in records["scenario_candidate"])
    assert all(row["fact_provenance"]["ego_action"]["origin"] == "SCENARIO_DEFINED" for row in records["scenario_candidate"])
    from hara_agent.services.analysis import MethodContractASILService
    from hara_agent.workflow.nodes import score_structured_scenarios
    scored = score_structured_scenarios(result, StructuredRiskScoringService(method), MethodContractASILService(method))
    assert len(scored.risk_results) == len(agent.seen)
    assert all(r.exposure.status is ReviewStatus.FINALIZED for r in scored.risk_results)


def test_specific_residential_garage_is_not_inferred_from_indoor_parking(method):
    service = ConstrainedScenarioSynthesisService(method)
    def pool(label):
        p = replace(parent(), operating_scenario=label, facts={**parent().facts, 'operating_scenario': label})
        built = service.build_input(malfunction=asdict(malfunction()), parent=p, assessment={}, project_context=asdict(project()))
        return {c.atom_id for d in built.dimension_candidate_sets for c in d.candidates}
    assert 'VD013' not in pool('室内停车场')
    assert 'VD013' in pool('Parking in garage with direct link to living area')


def test_person_on_vehicle_is_not_a_generic_external_pedestrian(method):
    from hara_agent.services.analysis.scenario_selection_quality import ScenarioSemanticCompatibilityClassifier
    from hara_agent.contracts.scenario_synthesis import SemanticCompatibility
    atom = {'atom_id': 'example-person', 'label': 'Persons on roof / hood', 'filled_dimensions': ['OBJECT']}
    query = {'object_categories': ['OBJECT_PEDESTRIAN'], 'object_relations': []}
    classify = ScenarioSemanticCompatibilityClassifier.classify
    assert classify(dimension='OBJECT', atom=atom, query=query, fm_template={}) == (
        SemanticCompatibility.CONTRADICTED, 'OBJECT_RELATION_UNPROVEN')
    query['object_relations'] = ['ON_VEHICLE']
    assert classify(dimension='OBJECT', atom=atom, query=query, fm_template={})[0] is SemanticCompatibility.SUPPORTED
    service = ConstrainedScenarioSynthesisService(method)
    p = replace(parent(), facts={**parent().facts, 'object_type': 'PEDESTRIAN', 'object_position': 'front'})
    built = service.build_input(malfunction=asdict(malfunction()), parent=p, assessment={}, project_context=asdict(project()))
    pool = {c.atom_id for d in built.dimension_candidate_sets for c in d.candidates}
    assert 'PU001' not in pool and 'CN_persons_in_vehicle' not in pool
    assert pool & {'PU002', 'PU004', 'PU005', 'CN_other_road_users', 'XP010'}


def test_joint_source_coverage_is_order_independent_and_retains_grade_directions(method):
    p = replace(project(), odd_road_surfaces=['上坡15%', '下坡15%'])
    parents = [replace(parent(str(i)), operating_scenario=place,
        facts={**parent().facts, 'operating_scenario': place})
        for i, place in enumerate(['停车场', '室内停车场', '室外停车场'])]
    service = MethodScenarioCandidateService(method)
    a, audit = service.generate_for_malfunction(malfunction(), parents, project_facts=p, function=function())
    b, _ = service.generate_for_malfunction(malfunction(), list(reversed(parents)), project_facts=p, function=function())
    assert [asdict(c) for c in a] == [asdict(c) for c in b]
    contexts = [c.facts['source_operating_conditions'] for c in a]
    assert {c['operating_scenario'] for c in contexts} == {'停车场', '室内停车场', '室外停车场'}
    assert {c.get('source_slope_direction') for c in contexts} >= {'上坡', '下坡'}
    assert audit['atom_generation']['coverage_policy'] == 'JOINT_PRIMARY_SOURCE_AND_ROAD_COVERAGE'
