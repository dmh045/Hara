"""Independent numeric and production-boundary regressions for P7."""
from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from hara_agent.method_sources import MethodSourceResolver
from hara_agent.models import ScenarioCandidate, ItemDefinitionFacts, RiskFact, SourceRef, ReviewStatus, FactProvenance
from hara_agent.services.analysis import MethodRiskFactBindingService, MethodContractASILService, StructuredRiskScoringService, ExposureDimensionCoverageService
from hara_agent.services.analysis.scenario_physics import derive_scenario_physics, causal_physics_conflicts, select_ego_speed_from_policy
from hara_agent.services.analysis.driver_configuration_service import DriverConfigurationBrancher
from hara_agent.services.analysis.hazardous_event_risk_context_service import HazardousEventRiskContextService

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def method():
    return MethodSourceResolver().resolve(template_path=None,
        baseline_manifest_path=ROOT / "method_assets/fusa_baseline_v1/manifest.yaml",
        report_template_path=ROOT / "references/HARA_Template_AI_20260327.xlsx").method


def source():
    return {"provenance": "PROJECT_INPUT", "approval": "FINALIZED",
            "source_refs": [{"source_type": "human_confirmation", "source_id": "golden", "location": "physical-case"}]}


def candidate(**facts):
    return ScenarioCandidate("SC", "parking", "parking", "parking", facts=facts,
        operating_mode="parking_vehicle_control", fact_provenance={key: source() for key in facts},
        analysis_instance={"malfunction_id": "MF"})


@pytest.mark.parametrize("ego,direction,position,closing,ttc", [
    (7, "REVERSE", "REAR", 7, .154286), (7, "REVERSE", "FRONT", 0, None),
    (7, "FORWARD", "FRONT", 7, .154286), (7, "FORWARD", "LEFT", None, None),
])
def test_ttc_geometry_golden(ego, direction, position, closing, ttc):
    records = derive_scenario_physics(candidate(ego_speed_kph=ego, object_speed_kph=0,
        ego_longitudinal_direction=direction, object_longitudinal_direction="STATIONARY",
        object_position=position, collision_type="FRONTAL", relative_distance_m=.3))
    values = {r.evidence_ref: r.value for r in records}
    assert values.get("DERIVED.closing_speed_kph") == closing
    assert values.get("DERIVED.ttc_s") == ttc


def test_direct_closing_survives_partial_geometry_and_static_relative_has_minimal_lineage():
    records = {r.evidence_ref: r for r in derive_scenario_physics(candidate(
        ego_speed_kph=7, object_speed_kph=0, object_longitudinal_direction="STATIONARY",
        closing_speed_kph=3.6, relative_distance_m=2))}
    assert records["DERIVED.ttc_s"].value == 2.0
    assert records["DERIVED.relative_speed_kph"].metadata["inputs"] == ["SCN.ego_speed_kph", "SCN.object_speed_kph"]


@pytest.mark.parametrize("bad", [True, -1, float("nan"), float("inf")])
def test_invalid_quantities_cannot_enter_physics(bad):
    values = {r.evidence_ref: r.value for r in derive_scenario_physics(candidate(
        relative_speed_kph=bad, closing_speed_kph=bad, relative_distance_m=bad))}
    assert not values


def test_binding_structured_specs_and_canonical_units(method):
    facts = ItemDefinitionFacts("system", "boundary", sources=[SourceRef("item", "golden", "p1")], risk_facts=[RiskFact(
        fact_id=f"RF-{name}", parameter=name, value=value, unit=unit,
        source_refs=[SourceRef("human_confirmation", "golden", name)],
        provenance=FactProvenance.HUMAN_CONFIRMATION, approval=ReviewStatus.FINALIZED,
    ) for name, value, unit in [("TTC", 4, "s"), ("RELATIVE_SPEED", 7, "km/h"),
                                ("EGO_SPEED", 7, "km/h"), ("COLLISION_TYPE", "FRONTAL", ""),
                                ("ROAD_USER_TYPE", "VEHICLE", "")]])
    result = MethodRiskFactBindingService(method).bind(facts, {})
    assert result.values["ttc_s"] == 4
    assert result.values["relative_speed_kph"] == 7
    assert result.values["ego_speed_kph"] == 7
    assert result.values["collision_type"] == "FRONTAL"
    assert result.values["road_user_type"] == "VEHICLE"
    facts.risk_facts[0] = replace(facts.risk_facts[0], unit="km/h")
    assert "ttc_s" not in MethodRiskFactBindingService(method).bind(facts, {}).values


def test_normal_speed_is_not_impact_and_stationary_initial_is_zero(method):
    sc = candidate(ego_speed_constraint={"min_kph": 0, "max_kph": 7})
    value, meta = select_ego_speed_from_policy(sc, policy=method.metadata["project_analysis_policy"], malfunction_id="MF")
    assert value == 7
    sc.facts.update(ego_speed_kph=value, object_speed_kph=0)
    sc.fact_provenance.update(ego_speed_kph=meta, object_speed_kph=source())
    assert "DERIVED.relative_speed_kph" not in {r.evidence_ref for r in derive_scenario_physics(sc)}
    sc.facts.pop("ego_speed_kph")
    sc.facts["method_scenario_dimensions"] = {"EGO_DYNAMICS": {"resolution_status": "RESOLVED",
        "atom_id": "standing", "method_semantics": {"ego_dynamics": {"motion_state": "STATIONARY"}}}}
    assert select_ego_speed_from_policy(sc, policy=method.metadata["project_analysis_policy"], malfunction_id="MF")[0] == 0


def test_same_impact_time_speeds_are_separate_from_normal_speed():
    sc = candidate(ego_speed_kph=7, object_speed_kph=0, ego_impact_speed_kph=12,
                   object_impact_speed_kph=0, impact_time_id="collision-1")
    sc.fact_provenance["ego_speed_kph"]["analysis_value_semantic"] = "PRE_FAULT_OPERATING_SPEED"
    assert next(r.value for r in derive_scenario_physics(sc) if r.evidence_ref == "DERIVED.relative_speed_kph") == 12
    sc.facts.pop("impact_time_id")
    assert "DERIVED.relative_speed_kph" not in {r.evidence_ref for r in derive_scenario_physics(sc)}


def test_driver_project_convention_does_not_escape_scope(method):
    brancher = DriverConfigurationBrancher(policy_id="P7", policy=method.metadata["project_analysis_policy"])
    sc = candidate(driver_position="outside_driver_seat")
    assert brancher.apply_location_convention(sc).facts["driver_in_vehicle"] is False
    other = replace(sc, operating_mode="motorway")
    assert "driver_in_vehicle" not in brancher.apply_location_convention(other).facts
    assert "driver_in_vehicle" not in DriverConfigurationBrancher().apply_location_convention(sc).facts


@pytest.mark.parametrize("approved", [{"controllability": "C0"}, {"exposure": "E0"}, {"severity": "S0"}])
def test_partial_qm_uses_governed_matrix_without_fabricating_scores(method, approved):
    value, sources = MethodContractASILService(method).partial_zero_decision(approved)
    assert value == "QM" and sources
    assert len(approved) == 1


def test_no_zero_no_shortcut(method):
    assert MethodContractASILService(method).partial_zero_decision({}) == ("", [])
    assert MethodContractASILService(method).partial_zero_decision({"controllability": "C3"}) == ("", [])


def test_physical_claim_conflicts_are_semantic_not_display_rewrites():
    sc = candidate(ego_speed_kph=7, object_speed_kph=0, ego_longitudinal_direction="REVERSE",
        object_longitudinal_direction="STATIONARY", object_position="REAR", collision_type="REAR_END")
    assert causal_physics_conflicts(sc, {"hazardous_event": "与前方行人相撞"}) == ["OBJECT_POSITION_CONTRADICTS_CAUSAL_CLAIM"]
    sc.facts["object_position"] = "FRONT"
    assert causal_physics_conflicts(sc, {"hazardous_event": "前方行人距离快速缩小"}) == ["NOT_CLOSING_CONTRADICTS_CAUSAL_CLAIM"]


def scoring_view(method):
    return {"scenario_id": "SC", "malfunction_id": "MF", "harm_mechanism_id": "IMPACT",
        "component_category": "sensor_camera", "scenario_atom_ids": ["SO010", "PH005"],
        "method_scenario_dimensions": {}, "ttc_s": 4.1,
        "_fact_provenance": {"collision_branches": source()},
        "_exposure_dimension_coverage_decision": ExposureDimensionCoverageService(method).decide(
            assessment_key="MF::SC", function=None, operating_mode="parking_vehicle_control")}


def branch(name, road_user, collision, speed):
    return {"branch_id": name, "mechanism_id": "IMPACT", "impact_time_id": "t1",
        "physically_possible": True, "road_user_type": road_user,
        "collision_type": collision, "relative_speed_kph": speed,
        "_fact_provenance": {"relative_speed_kph": source(), "relative_speed_range_kph": source()}}


@pytest.mark.parametrize("speed,expected,invariant", [(7, "S1", True), (10, "S2", False)])
def test_finite_configurations_use_real_executor(method, speed, expected, invariant):
    service = StructuredRiskScoringService(method)
    view = scoring_view(method)
    view["collision_branches"] = [branch(c, "VEHICLE", c, speed) for c in ("FRONTAL", "REAR_END", "SIDE")]
    scored = service.score(view, "established collision mechanism")
    assert scored["severity"]["severity_score"] == expected
    assert scored["severity"]["invariant"] is invariant
    assert len(scored["severity"]["branch_evaluations"]) == 3


def test_finite_objects_maximum_keeps_deciding_branch_c(method):
    view = scoring_view(method)
    car = branch("car", "VEHICLE", "FRONTAL", 7); car["ttc_s"] = 8
    pedestrian = branch("pedestrian", "PEDESTRIAN", "FRONTAL", 7); pedestrian["ttc_s"] = .154286
    view["collision_branches"] = [car, pedestrian]
    scored = StructuredRiskScoringService(method).score(view, "impact")
    assert scored["severity"]["severity_score"] == "S2"
    assert scored["controllability"]["controllability_score"] == "C3"
    assert scored["severity"]["deciding_branch_id"] == "pedestrian"


def test_empty_set_not_s0_and_ranges_visit_internal_boundaries(method):
    view = scoring_view(method); view["collision_branches"] = []
    assert StructuredRiskScoringService(method).score(view, "impact")["severity"]["severity_score"] == ""
    item = branch("range", "VEHICLE", "SIDE", 7); item["relative_speed_range_kph"] = [7, 12]
    view["collision_branches"] = [item]
    result = StructuredRiskScoringService(method).score(view, "impact")["severity"]
    assert result["severity_score"] == "S2"
    assert len(result["branch_evaluations"]) > 2


def test_ttc_diagnostics_distinguish_lateral_from_missing(method):
    service = HazardousEventRiskContextService(method)
    for facts, reason in [({"collision_type": "SIDE"}, "MODEL_NOT_APPLICABLE"), ({}, "MISSING_DISTANCE")]:
        context = service.build(malfunction_id="MF", scenario_id="SC", hazard_node_id="H", scenario=facts)
        assert context.ttc_s.reason == reason


@pytest.mark.parametrize('acceleration,distance,expected', [(2, 2, 12.356375), (-1, 1, 4.804165)])
def test_sourced_fault_motion_uses_reachable_impact_speed(acceleration, distance, expected):
    sc = candidate(ego_speed_kph=7, object_speed_kph=0, object_impact_speed_kph=0,
        longitudinal_acceleration_mps2=acceleration, travel_to_impact_m=distance,
        fault_motion_model='CONSTANT_ACCELERATION_TO_IMPACT_DISTANCE', impact_time_id='impact-1')
    sc.fact_provenance['ego_speed_kph']['analysis_value_semantic'] = 'PRE_FAULT_OPERATING_SPEED'
    records = {r.evidence_ref: r for r in derive_scenario_physics(sc)}
    assert records['DERIVED.ego_impact_speed_kph'].value == pytest.approx(expected, abs=1e-6)
    assert records['DERIVED.relative_speed_kph'].value == pytest.approx(expected, abs=1e-6)
    assert records['DERIVED.relative_speed_kph'].approval_status is ReviewStatus.FINALIZED
    sc.facts.pop('fault_motion_model')
    assert 'DERIVED.relative_speed_kph' not in {r.evidence_ref for r in derive_scenario_physics(sc)}


def test_unapproved_acceleration_remains_pending_and_stopping_short_is_not_s0():
    sc = candidate(ego_speed_kph=7, object_impact_speed_kph=0,
        longitudinal_acceleration_mps2=2, travel_to_impact_m=2,
        fault_motion_model='CONSTANT_ACCELERATION_TO_IMPACT_DISTANCE', impact_time_id='impact-1')
    sc.fact_provenance['longitudinal_acceleration_mps2']['approval'] = 'PENDING'
    assert next(r for r in derive_scenario_physics(sc) if r.evidence_ref == 'DERIVED.ego_impact_speed_kph').approval_status is ReviewStatus.PENDING
    sc.facts['longitudinal_acceleration_mps2'] = -10
    assert 'DERIVED.ego_impact_speed_kph' not in {r.evidence_ref for r in derive_scenario_physics(sc)}


def test_tram_and_overtaking_require_their_own_semantics():
    from hara_agent.services.analysis.scenario_selection_quality import ScenarioSemanticCompatibilityClassifier as Classifier
    tram = {'atom_id': 'source-tram', 'label': 'Tram on roadway'}
    result, reason = Classifier.classify(dimension='OBJECT', atom=tram,
        query={'object_categories': ['OBJECT_STATIC']}, fm_template={})
    assert result.value == 'CONTRADICTED'
    result, reason = Classifier.classify(dimension='EGO_ACTION',
        atom={'label': 'Overtaking in oncoming traffic'}, query={
            'action_categories': ['ACTION_PARK'], 'traffic_relations': ['TRAFFIC_ONCOMING']}, fm_template={})
    assert result.value == 'CONTRADICTED' and reason == 'EGO_ACTION_CONTRADICTION'
    from hara_agent.services.reporting.engineering_text_mapper import EngineeringReportTextMapper
    sc = candidate(scenario_object_atom='Tram on roadway')
    sc.analysis_instance['structured_semantic_query'] = {'object_categories': ['OBJECT_STATIC']}
    assert 'OBJECT_TRAM' in EngineeringReportTextMapper._categories(sc)
    assert 'OBJECT_STATIC' not in EngineeringReportTextMapper._categories(sc)


def test_response_cache_identity_contains_compiler_and_policy(tmp_path):
    from hara_agent.infrastructure.llm import LLMRequest, LLMResponse
    from hara_agent.services.semantic.scenario_agent import ScenarioFeasibilityAgent
    class Client:
        calls = 0
        def complete_json(self, request):
            self.calls += 1
            return LLMResponse({'ok': True}, 'fixture')
    client = Client()
    agent = ScenarioFeasibilityAgent(client, response_checkpoint_dir=tmp_path)
    request = LLMRequest('test', 'system', 'user', 'schema', 'prompt')
    agent.method_execution_identity = {'compiler': 'v1', 'policy': 'p1'}
    agent._complete_json(request)
    assert agent._complete_json(request).usage['response_checkpoint_hit'] is True
    agent.method_execution_identity = {'compiler': 'v2', 'policy': 'p1'}
    agent._complete_json(request)
    agent.method_execution_identity = {'compiler': 'v2', 'policy': 'p2'}
    agent._complete_json(request)
    assert client.calls == 3


def test_template_direction_is_compiled_from_maneuver_not_object_position(method):
    from hara_agent.services.analysis.scenario_physics import restore_template_ego_direction
    templates = method.scenario_model.scenario_method.fm_template_catalog.templates
    options = [option for template in templates for option in template.required_scenarios]
    assert all(o.ego_longitudinal_direction == 'FORWARD' for o in options if o.label.startswith('前进'))
    assert all(o.ego_longitudinal_direction == 'REVERSE' for o in options if o.label.startswith('倒车'))
    assert all(not o.ego_longitudinal_direction for o in options if o.label.startswith('加速撞前方'))
    sc = candidate(object_position='FRONT')
    assert restore_template_ego_direction(sc, templates) is None
    sc.fact_provenance['object_position'].update(selection_basis='EXACT_ACTIVE_OPTION', source_option_id='FM_TEMPLATE_001:OPTION:1')
    assert restore_template_ego_direction(sc, templates)[0] == 'FORWARD'


def test_static_object_e_exemption_does_not_disappear_when_traffic_is_present():
    from hara_agent.services.analysis.scenario_selection_quality import ScenarioDimensionApplicabilityService
    from types import SimpleNamespace
    result = ScenarioDimensionApplicabilityService().assess('OBJECT', {
        'object_categories':['OBJECT_STATIC'], 'traffic_relations':['TRAFFIC_ONCOMING']},
        SimpleNamespace(source_refs=('source',), parent_atom_id='', refinable=True))
    assert result.status.value == 'NOT_APPLICABLE'


def test_replayed_report_names_its_historical_scope():
    from hara_agent.workflow import HARAState
    from hara_agent.services.reporting.projection import HARAReportProjectionService
    state = HARAState('replay', audit_trail=[{'event':'p7_fixed_population_replay','source_run_id':'historical'}])
    scope = HARAReportProjectionService._run_scope(state, [], 0)
    assert scope['run_scope'] == 'OFFLINE FIXED-POPULATION REPLAY'
    assert '0 new Provider' in scope['scope_note']
    assert 'no new full causal validation' in scope['full_project_population']


@pytest.mark.parametrize('raw,state', [('tbd','UNKNOWN_TBD'),('-','SOURCE_DASH'),('None','SOURCE_NULL'),('','MISSING')])
def test_exposure_unknown_source_states_are_distinct(method, raw, state):
    from hara_agent.contracts import ExposureMethodDomain
    from hara_agent.services.analysis import ExposureMethodExecutor
    atom = replace(method.structured_risk_method.exposure.atoms[0], duration_level='', raw_duration_value=raw)
    details, levels = ExposureMethodExecutor._atom_values((atom.atom_id,), {atom.atom_id:atom}, ExposureMethodDomain.TIME)
    assert details[0]['value_state'] == state
    assert not details[0]['used'] and levels == []
