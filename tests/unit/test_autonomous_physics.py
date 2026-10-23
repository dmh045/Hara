"""Offline proposal transport + real production consumers; no Provider usage."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
import math

import pytest

from test_p7_closure import method, candidate, source
from hara_agent.contracts.physical_scenario_plan import PhysicalScenarioPlan
from hara_agent.infrastructure.llm.protocol import LLMResponse
from hara_agent.models import MalfunctionCandidate, ReviewStatus, SourceRef
from hara_agent.services.semantic.physical_scenario_agent import PhysicalScenarioAgent
from hara_agent.services.semantic.scenario_agent import ScenarioFeasibilityAgent
from hara_agent.services.analysis.physical_hypothesis_instantiation_service import PhysicalHypothesisInstantiationService
from hara_agent.services.analysis.scenario_physics import source_is_accepted_for, branch_physics_view
from hara_agent.services.analysis import StructuredRiskScoringService, ExposureDimensionCoverageService
from hara_agent.services.analysis.hazardous_event_risk_context_service import HazardousEventRiskContextService


def raw_plan(model="CONSTANT_ACCELERATION_TO_IMPACT_DISTANCE", initial=0, acceleration=2):
    if "LATERAL" in model and initial == 0:
        initial = 3.6
    parameters = [{"field": field, "unit": unit, "kind": "POINT", "value": value, "rationale": "offline analytical assumption"}
        for field, unit, value in [("ego_speed_kph", "km/h", initial), ("relative_distance_m", "m", 2),
                                   ("longitudinal_acceleration_mps2", "m/s2", acceleration)]]
    if model == "GRADE_ROLLAWAY_TO_IMPACT_DISTANCE":
        parameters = parameters[:2] + [dict(field="slope_percent", unit="%", kind="POINT", value=15, rationale="source grade context"),
            dict(field="residual_resistance_mps2", unit="m/s2", kind="POINT", value=0, rationale="conditional no residual holding")]
    if model == "CONSTANT_LATERAL_ACCELERATION_TO_CONTACT":
        parameters = parameters[:1] + [dict(field=f, unit=u, kind="POINT", value=v, rationale="conditional lateral trajectory")
            for f, u, v in [("lateral_clearance_m", "m", 1), ("lateral_speed_mps", "m/s", 0), ("lateral_acceleration_mps2", "m/s2", .5)]]
    lateral = "LATERAL" in model
    return {"hypotheses": [{"hypothesis_id": "h1", "model": model, "rationale": "offline hypothetical post-fault motion",
        **({"grade_frame": {"coordinate_frame": "BODY_FORWARD_POSITIVE", "slope_reference": "UNSPECIFIED",
            "heading_relative_to_downhill": "DOWNHILL", "downhill_body_direction": "FRONT",
            "orientation_basis": "ANALYSIS_ASSUMPTION", "rationale": "explicit offline heading assumption"}}
           if "ROLLAWAY" in model else {}),
        "basis_refs": ["MF.description"], "geometry": dict(road_user_type="VEHICLE", collision_type="SIDE" if lateral else "FRONTAL",
            object_position="LEFT" if lateral else "FRONT", ego_longitudinal_direction="FORWARD", object_id="object1",
            object_extent="EXTENDED_ALONG_TRAVEL_PATH" if lateral else "POINT_ON_TRAVEL_PATH"),
        "parameters": parameters, "conditions": ["stationary target"], "uncovered_conditions": ["moving targets; real capability bound unknown"],
        "failed_controls": ["steering" if lateral else "parking_hold" if "ROLLAWAY" in model else "drive"], "remaining_controls": []}], "uncovered_conditions": ["offline proposal; no live validation"]}


def mf():
    return MalfunctionCandidate("MF", "F", "More", "drive exceeds demand", "unintended acceleration", "impact", ["fault", "motion"],
        sources=[SourceRef("item_definition", "offline", "test", "synthetic fault definition")], status=ReviewStatus.FINALIZED)


def make_candidate(initial=0):
    return candidate(ego_speed_kph=initial, road_user_type="VEHICLE", scenario_atom_ids=["SO006"],
                     component_category="actuator_motor", driver_in_vehicle=True, function_type="parking",
                     remote_intervention_available=False, other_road_user_avoidance_possible=False)


def instantiate(method, raw=None, sc=None):
    plan = PhysicalScenarioPlan.from_dict(raw or raw_plan(), evidence_ids={"MF.description"})
    children, errors = PhysicalHypothesisInstantiationService(method).instantiate(sc or make_candidate(), mf(), plan,
        plan_key="offline-plan", response_model="offline-scripted")
    assert not errors, errors
    return children


def score(method, sc):
    view = {**sc.facts, "malfunction_id": "MF", "scenario_id": sc.scenario_id, "_fact_provenance": sc.fact_provenance}
    contexts = HazardousEventRiskContextService(method)
    context = contexts.build(malfunction_id="MF", scenario_id=sc.scenario_id, hazard_node_id="impact", scenario=view)
    view = contexts.scoring_facts(context, view)
    view["_exposure_dimension_coverage_decision"] = ExposureDimensionCoverageService(method).decide(
        assessment_key="test", function=None, operating_mode=sc.operating_mode)
    return StructuredRiskScoringService(method).score(view, "conditional contact")


def test_drive_from_rest_produces_conditional_impact_without_approving_inputs(method):
    raw = raw_plan()
    raw["hypotheses"][0]["remaining_controls"] = ["driver_braking"]
    sc = instantiate(method, raw)[0]
    assert sc.facts["ego_impact_speed_kph"] == pytest.approx(10.182338)
    assert sc.facts["ttc_s"] == pytest.approx(math.sqrt(2), abs=1e-6)
    assert sc.fact_provenance["longitudinal_acceleration_mps2"]["approval"] == "PENDING"
    assert sc.fact_provenance["longitudinal_acceleration_mps2"]["provenance"] == "LLM_INFERENCE"
    assert source_is_accepted_for(sc.fact_provenance["relative_speed_kph"], malfunction_id="MF", scenario_id=sc.scenario_id)
    result = score(method, sc)
    assert result["severity"]["severity_score"]
    assert result["controllability"]["controllability_score"]
    assert result["severity"]["engineering_status"] == "PENDING"
    assert result["branch_summary"]["calculation_nature"] == "CONDITIONAL"


def test_rollaway_slope_percent_and_zero_initial_are_distinct(method):
    sc = instantiate(method, raw_plan("GRADE_ROLLAWAY_TO_IMPACT_DISTANCE"))[0]
    assert sc.facts["longitudinal_acceleration_mps2"] == pytest.approx(9.80665 * math.sin(math.atan(.15)))
    assert sc.facts["ego_impact_speed_kph"] > 0
    assert sc.facts["ttc_s"] > 0
    assert score(method, sc)["severity"]["severity_score"]


def test_lateral_contact_keeps_e_and_s_but_does_not_invent_longitudinal_c(method):
    sc = instantiate(method, raw_plan("CONSTANT_LATERAL_ACCELERATION_TO_CONTACT"), make_candidate(3.6))[0]
    assert sc.facts["first_contact_time_s"] == 2
    assert sc.fact_provenance["first_contact_time_s"]["analysis_value_semantic"] == "FIRST_CONTACT_TIME"
    assert "ttc_s" not in sc.facts
    result = score(method, sc)
    assert result["severity"]["severity_score"]
    assert not result["controllability"]["controllability_score"]
    assert result["controllability"]["pending_reason"] == "LATERAL_C_METHOD_NOT_AVAILABLE"
    assert result["branch_evaluations"][0]["first_contact_time_s"] == 2
    assert result["branch_evaluations"][0]["ttc_s"] is None
    from hara_agent.workflow.nodes.scoring import _scoring_evidence
    assert "LATERAL_C_METHOD_NOT_AVAILABLE" in _scoring_evidence(result["controllability"], "controllability_score").review_reason
    changed = branch_physics_view({**sc.facts, "scenario_id": sc.scenario_id, "malfunction_id": "MF", "_fact_provenance": sc.fact_provenance},
        {"model_valid_until_s": 1, "_fact_provenance": {"model_valid_until_s": sc.fact_provenance["fault_motion_model"]}})
    assert "first_contact_time_s" not in changed
    assert "ego_impact_speed_kph" not in changed


def test_ranges_are_bounded_sensitivity_not_a_real_upper_limit(method):
    raw = raw_plan()
    raw["hypotheses"][0]["parameters"][-1].update(kind="RANGE", lower=0, upper=4)
    cases = instantiate(method, raw)
    assert 2 < len(cases) <= 9
    assert not cases[0].facts["physical_analysis"]["sensitivity"]["vehicle_upper_bound_proven"]
    assert cases[0].facts["physical_analysis"]["sensitivity"]["method_boundaries"]


def test_symbolic_acceleration_reports_method_boundaries_without_inventing_value(method):
    raw = raw_plan()
    p = raw["hypotheses"][0]["parameters"][-1]
    p.pop("value"); p.update(kind="SYMBOLIC", symbol="a_fault")
    sc = instantiate(method, raw)[0]
    assert "ego_impact_speed_kph" not in sc.facts
    assert sc.facts["physical_analysis"]["sensitivity"]["method_boundaries"]


@pytest.mark.parametrize("field,value", [("severity", "S3"), ("approved", True), ("collision_branches", [])])
def test_llm_cannot_inject_ratings_or_approval(field, value):
    raw = raw_plan(); raw["hypotheses"][0][field] = value
    with pytest.raises(ValueError):
        PhysicalScenarioPlan.from_dict(raw, evidence_ids={"MF.description"})


def test_conditional_causal_proof_uses_generated_world_and_preserves_input_pending(method):
    sc = instantiate(method)[0]
    payload = dict(scenario_id=sc.scenario_id, physically_feasible=True, functionally_relevant=True,
        causally_relevant=True, breakpoint="NONE", hazardous_event="Under this hypothesis the vehicle contacts the object in front",
        causal_chain={"m_to_b": dict(claim="defined excessive drive", basis_type="DIRECT_FACT", evidence_refs=["MF.description"]),
                      "b_to_i": dict(claim="under the modeled acceleration, first contact occurs", basis_type="DERIVED_PHYSICS", evidence_refs=["DERIVED.ego_impact_speed_kph"]),
                      "i_to_h": dict(claim="conditional collision at this speed", basis_type="DERIVED_PHYSICS", evidence_refs=["DERIVED.relative_speed_kph"])},
        risk_dimension_changes=[], confidence=.9, status="FINALIZED")
    assessment = ScenarioFeasibilityAgent._parse(mf(), payload, scenario=sc)
    assert assessment.retain
    assert assessment.causal_assessment.review_status is ReviewStatus.FINALIZED
    assert sc.fact_provenance["fault_motion_model"]["approval"] == "PENDING"


def test_raw_response_is_saved_and_reparsed_without_new_call(tmp_path):
    class Client:
        calls = 0
        def complete_json(self, request):
            self.calls += 1
            return LLMResponse(raw_plan(), "offline-scripted", usage={"prompt_tokens": 10})
    client = Client()
    agent = PhysicalScenarioAgent(client, response_checkpoint_dir=tmp_path)
    context = {"plan_key": "key", "evidence": {"MF.description": "drive fault"}}
    agent.propose(context)
    _, audit = agent.propose(context)
    assert client.calls == 1 and audit["response_cache_hits"] == 1
    assert json.loads(next(tmp_path.glob('*.json')).read_text())["response"]["data"] == raw_plan()


@pytest.mark.parametrize("kind,payload,required", [
    ("SYMBOLIC", {"value": "a_fault"}, "symbol"),
    ("POINT", {}, "value"),
    ("RANGE", {"lower": 0, "upper": 2}, "value"),
])
def test_parameter_payload_error_identifies_field_and_required_key(kind, payload, required):
    raw = raw_plan()
    p = raw["hypotheses"][0]["parameters"][-1]
    p.pop("value"); p.update(kind=kind, **payload)
    before = deepcopy(raw)
    with pytest.raises(ValueError, match=r"parameters\[longitudinal_acceleration_mps2\]") as error:
        PhysicalScenarioPlan.from_dict(raw, evidence_ids={"MF.description"})
    assert required in str(error.value) and "missing=" in str(error.value)
    assert raw == before


def test_schema_repair_receives_original_invalid_payload_and_dynamic_contract(tmp_path):
    bad = raw_plan()
    p = bad["hypotheses"][0]["parameters"][-1]
    p.update(kind="SYMBOLIC", value="a_fault")
    good = deepcopy(bad)
    p = good["hypotheses"][0]["parameters"][-1]
    p["symbol"] = p.pop("value")
    requests, records = [], []
    class Client:
        def complete_json(self, request):
            requests.append(json.loads(request.user_prompt))
            return LLMResponse(deepcopy(bad if len(requests) == 1 else good), "OFFLINE")
    agent = PhysicalScenarioAgent(Client(), response_checkpoint_dir=tmp_path)
    context = {"plan_key": "repair", "evidence": {"MF.description": "drive fault"},
               "risk_vocabulary": {"road_user_type": {"VEHICLE": "vehicle"},
                                   "collision_type": {"FRONTAL": "frontal"}}}
    plan, audit = agent.propose(context, on_response=records.append)
    assert len(requests) == 2 and audit["schema_repairs"] == 1
    contract = requests[0]["output_contract"]
    assert contract["geometry_values"] == {"road_user_type": ["VEHICLE"], "collision_type": ["FRONTAL"]}
    assert contract["basis_ref_ids"] == ["MF.description"]
    assert requests[1]["previous_invalid_response"] == bad
    assert "longitudinal_acceleration_mps2" in requests[1]["repair_errors"][0]
    assert records[0]["response"]["data"] == bad
    assert plan.to_dict() == good


def test_unknown_basis_error_lists_supplied_ids_without_accepting_invented_path():
    raw = raw_plan(); raw["hypotheses"][0]["basis_refs"] = ["PROJECT.slope_percent"]
    with pytest.raises(ValueError, match="basis_refs must use supplied IDs") as error:
        PhysicalScenarioPlan.from_dict(raw, evidence_ids={"MF.description", "SCENARIO.conditions"})
    assert "SCENARIO.conditions" in str(error.value) and "PROJECT.slope_percent" in str(error.value)


@pytest.mark.parametrize("acceleration", [0, -.5])
def test_fault_motion_need_not_exceed_normal_speed(method, acceleration):
    sc = instantiate(method, raw_plan(initial=7, acceleration=acceleration), make_candidate(initial=7))[0]
    assert sc.facts["ego_impact_speed_kph"] <= 7


def test_remaining_limiter_is_a_separate_piecewise_physical_constraint(method):
    raw = raw_plan()
    raw["hypotheses"][0]["remaining_controls"] = ["speed_limiter"]
    raw["hypotheses"][0]["parameters"].append(dict(field="post_fault_speed_cap_kph", unit="km/h", kind="POINT", value=3.6, rationale="conditional independent remaining limiter"))
    sc = instantiate(method, raw)[0]
    assert sc.facts["ego_impact_speed_kph"] == pytest.approx(3.6)
    assert sc.facts["ttc_s"] == pytest.approx(2.25)


def test_branch_change_invalidates_new_model_dependencies(method):
    sc = instantiate(method, raw_plan("GRADE_ROLLAWAY_TO_IMPACT_DISTANCE"))[0]
    view = {**sc.facts, "scenario_id": sc.scenario_id, "malfunction_id": "MF", "_fact_provenance": sc.fact_provenance}
    child = branch_physics_view(view, {"slope_percent": 0, "_fact_provenance": {"slope_percent": sc.fact_provenance["slope_percent"]}})
    assert "ttc_s" not in child
    assert "relative_speed_kph" not in child


def test_plan_reuse_resume_and_context_change_are_separate_from_causal_reuse(method, tmp_path):
    from hara_agent.workflow import HARAState
    from hara_agent.workflow.physical_analysis import construct_physical_scenarios
    class Client:
        calls = 0
        def complete_json(self, request):
            self.calls += 1
            return LLMResponse(raw_plan(), "offline")
    client = Client()
    agent = PhysicalScenarioAgent(client, response_checkpoint_dir=tmp_path)
    a = make_candidate(); b = deepcopy(a); b.scenario_id = "SC-driver2"
    a.facts["driver_position"] = "in_driver_seat"; b.facts["driver_position"] = "outside_driver_seat"
    state = HARAState("offline")
    def run():
        return construct_physical_scenarios(state, {"MF": [a, b]}, [mf()], method=method, agent=agent, project_facts=None)
    out = run()
    assert len(out["MF"]) == 2 and client.calls == 1
    assert state.item_definition["physical_analysis_metrics"]["physical_plan_reuse"] == 1
    assert "complete_causal_reuse" not in state.item_definition["physical_analysis_metrics"]
    assert state.item_definition["physical_analysis_metrics"]["physical_solve_points"] == 2
    state = HARAState.read_committed(json.loads(json.dumps(state.to_dict())))
    run(); assert client.calls == 1
    a.fact_provenance["ego_speed_kph"]["source_refs"][0]["excerpt"] = "changed relevant evidence"
    run(); assert client.calls == 2


def test_budget_stop_preserves_completed_physical_instances_before_next_plan(method, tmp_path):
    from hara_agent.workflow import HARAState, CheckpointRepository
    from hara_agent.workflow.physical_analysis import construct_physical_scenarios
    from hara_agent.infrastructure.llm.provider_budget import ProviderAttemptBudgetExceeded
    class Client:
        calls = 0
        def complete_json(self, request):
            self.calls += 1
            if self.calls > 1:
                raise ProviderAttemptBudgetExceeded(limit=1, attempts=1, ledger_path=tmp_path/'ledger')
            return LLMResponse(raw_plan(), 'OFFLINE')
    state = HARAState('physical-persistence')
    repo = CheckpointRepository(tmp_path/'state')
    a = make_candidate(); b = deepcopy(a); b.scenario_id = 'second'; b.facts['where'] = 'different condition'
    with pytest.raises(ProviderAttemptBudgetExceeded):
        construct_physical_scenarios(state, {'MF': [a, b]}, [mf()], method=method,
            agent=PhysicalScenarioAgent(Client()), project_facts=None, checkpoint=repo.save)
    restored = repo.load(state.run_id)
    done = next(e for e in restored.item_definition['physical_analysis_plans'].values() if e.get('instantiations'))
    record = done['instantiations'][a.scenario_id]
    child = record['generated_scenarios'][0]
    assert record['generated_ids'] == [child['scenario_id']]
    assert child['facts']['motion_model_outcome']['status'] == 'CONTACT'
    assert child['facts']['ego_impact_speed_kph'] > 0
    assert child['fact_provenance']['fault_motion_model']['approval'] == 'PENDING'


def test_existing_direct_path_never_requests_a_physical_plan(method):
    from hara_agent.workflow import HARAState
    from hara_agent.workflow.physical_analysis import construct_physical_scenarios
    class Client:
        def complete_json(self, request):
            raise AssertionError("resolved path must not call LLM")
    sc = make_candidate()
    sc.facts["emergency_braking_available"] = True
    sc.fact_provenance["emergency_braking_available"] = source()
    sc.facts["collision_branches"] = [{"branch_id": "existing"}]
    sc.fact_provenance["collision_branches"] = source()
    state = HARAState("offline")
    result = construct_physical_scenarios(state, {"MF": [sc]}, [mf()], method=method,
        agent=PhysicalScenarioAgent(Client()), project_facts=None)
    assert result["MF"] == [sc]
    assert state.item_definition["physical_analysis_metrics"]["existing_path_skipped"] == 1


def test_unsupported_rear_target_preserves_world_and_reuses_model_gap(method, tmp_path):
    from hara_agent.workflow import HARAState
    from hara_agent.workflow.physical_analysis import construct_physical_scenarios
    class Client:
        calls = 0
        def complete_json(self, request):
            self.calls += 1
            return LLMResponse({"hypotheses": [], "uncovered_conditions": [
                "Moving following vehicle rear impact is outside the stationary target models."]}, "offline")
    client = Client()
    a = make_candidate(7)
    a.facts.update(object_position="REAR", target_speed_kph=10)
    b = deepcopy(a); b.scenario_id = "second-driver"
    a.facts["driver_position"] = "in_driver_seat"
    b.facts["driver_position"] = "outside_driver_seat"
    original = [(deepcopy(c.facts), deepcopy(c.fact_provenance)) for c in (a, b)]
    state = HARAState("unsupported-model")
    agent = PhysicalScenarioAgent(client, response_checkpoint_dir=tmp_path)
    for _ in range(2):
        result = construct_physical_scenarios(state, {"MF": [a, b]}, [mf()], method=method,
            agent=agent, project_facts=None)
        assert result["MF"] == [a, b]
        assert client.calls == 1
        for c, (facts, provenance) in zip(result["MF"], original):
            assert c.facts == facts and c.fact_provenance == provenance
            assert c.analysis_instance["physical_plan_diagnostics"][0]["reason"] == "PHYSICAL_MODEL_NOT_APPLICABLE"
    metrics = state.item_definition["physical_analysis_metrics"]
    assert metrics["physical_plan_reuse"] == 1
    assert metrics["generated_instances"] == metrics["semantic_corrections"] == metrics["physical_solve_points"] == 0


def test_empty_physical_plan_requires_an_explanation():
    with pytest.raises(ValueError, match="applicability gap"):
        PhysicalScenarioPlan.from_dict({"hypotheses": [], "uncovered_conditions": []}, evidence_ids=set())


def test_parallel_physical_plans_equal_serial_and_resume_without_requests(method, tmp_path):
    from threading import Barrier, Lock
    from hara_agent.workflow import HARAState, CheckpointRepository
    from hara_agent.workflow.physical_analysis import construct_physical_scenarios
    class Client:
        def __init__(self, parallel=False):
            self.calls = 0
            self.lock = Lock()
            self.barrier = Barrier(2) if parallel else None
        def complete_json(self, request):
            with self.lock:
                self.calls += 1
            if self.barrier:
                self.barrier.wait(timeout=5)  # Both independent requests must overlap.
            return LLMResponse(raw_plan(), 'offline')
    malfunctions = [mf(), replace(mf(), malfunction_id='MF2')]
    candidates = {}
    for m in malfunctions:
        a = make_candidate(); a.scenario_id = m.malfunction_id + '-driver1'
        b = deepcopy(a); b.scenario_id = m.malfunction_id + '-driver2'
        a.facts['driver_position'] = 'in_driver_seat'; b.facts['driver_position'] = 'outside_driver_seat'
        candidates[m.malfunction_id] = [a, b]
    serial = construct_physical_scenarios(HARAState('serial'), deepcopy(candidates), malfunctions,
        method=method, agent=PhysicalScenarioAgent(Client()), project_facts=None)
    state = HARAState('parallel')
    repo = CheckpointRepository(tmp_path/'state')
    client = Client(parallel=True)
    agent = PhysicalScenarioAgent(client, response_checkpoint_dir=tmp_path/'responses')
    checkpoints = []
    def save(current):
        repo.save(current)
        restored = repo.load(current.run_id)
        checkpoints.append(len(restored.item_definition['physical_analysis_plans']))
    parallel = construct_physical_scenarios(state, deepcopy(candidates), malfunctions, method=method,
        agent=agent, project_facts=None, max_workers=2, checkpoint=save)
    assert parallel == serial and client.calls == 2
    assert len(state.item_definition['physical_analysis_plans']) == 2
    metrics = state.item_definition['physical_analysis_metrics']
    assert metrics['plan_requests'] == metrics['physical_plan_reuse'] == 2
    assert metrics['physical_solve_points'] == metrics['generated_instances'] == 4
    assert checkpoints == sorted(checkpoints) and checkpoints[-1] == 2
    restored = repo.load(state.run_id)
    resumed = construct_physical_scenarios(restored, deepcopy(candidates), malfunctions, method=method,
        agent=agent, project_facts=None, max_workers=2, checkpoint=save)
    assert resumed == serial and client.calls == 2
    assert restored.item_definition['physical_analysis_metrics']['plan_cache_hits'] == 2


def test_physical_resume_skips_unchanged_writes_but_persists_recomputed_values(method, tmp_path):
    from hara_agent.workflow import HARAState, CheckpointRepository
    from hara_agent.workflow.physical_analysis import construct_physical_scenarios
    class Client:
        calls = 0
        def complete_json(self, request):
            self.calls += 1
            return LLMResponse(raw_plan(), 'offline')
    client = Client(); agent = PhysicalScenarioAgent(client)
    repo = CheckpointRepository(tmp_path)
    state = HARAState('resume-unchanged')
    writes = []
    def save(s):
        writes.append(repo.save(s))
    def run():
        return construct_physical_scenarios(state, {'MF':[make_candidate()]}, [mf()],
            method=method, agent=agent, project_facts=None, checkpoint=save)
    original = run()
    assert writes and client.calls == 1
    state = repo.load(state.run_id)
    writes.clear()
    assert run() == original and not writes and client.calls == 1
    entry = next(iter(state.item_definition['physical_analysis_plans'].values()))
    saved = next(iter(entry['instantiations'].values()))['generated_scenarios'][0]
    saved['facts']['ego_impact_speed_kph'] = 999
    assert run() == original and len(writes) == 1 and client.calls == 1
    restored = repo.load(state.run_id)
    actual = next(iter(next(iter(restored.item_definition['physical_analysis_plans'].values()))['instantiations'].values()))
    assert actual['generated_scenarios'][0]['facts']['ego_impact_speed_kph'] == original['MF'][0].facts['ego_impact_speed_kph']


def test_parallel_physical_plans_share_one_hard_budget_and_keep_responses(method, tmp_path):
    from threading import Barrier
    from hara_agent.workflow import HARAState, CheckpointRepository
    from hara_agent.workflow.physical_analysis import construct_physical_scenarios
    from hara_agent.infrastructure.llm.provider_budget import ProviderAttemptBudget, ProviderAttemptBudgetExceeded
    class Client:
        attempt_budget = ProviderAttemptBudget(tmp_path/'ledger', run_id='budget-parallel', limit=1)
        barrier = Barrier(2)
        def complete_json(self, request):
            self.barrier.wait(timeout=5)
            self.attempt_budget.begin(request, model='offline')
            return LLMResponse(raw_plan(), 'offline')
    state = HARAState('budget-parallel')
    repo = CheckpointRepository(tmp_path/'state')
    malfunctions = [mf(), replace(mf(), malfunction_id='MF2')]
    with pytest.raises(ProviderAttemptBudgetExceeded):
        construct_physical_scenarios(state, {m.malfunction_id:[make_candidate()] for m in malfunctions},
            malfunctions, method=method, agent=PhysicalScenarioAgent(Client()), project_facts=None,
            max_workers=2, checkpoint=repo.save)
    restored = repo.load(state.run_id)
    assert Client.attempt_budget.attempts == 1
    assert sum(bool(p.get('instantiations')) for p in restored.item_definition['physical_analysis_plans'].values()) == 1
    assert restored.item_definition['physical_analysis_metrics']['provider_transport_attempts_this_stage'] == 1


def test_negative_physical_range_keeps_signed_order():
    raw = raw_plan(acceleration=-7)
    raw["hypotheses"][0]["parameters"][-1].update(kind="RANGE", lower=-8, upper=-6.5)
    PhysicalScenarioPlan.from_dict(raw, evidence_ids={"MF.description"})
    raw["hypotheses"][0]["parameters"][-1].update(lower=-6.5, upper=-8)
    with pytest.raises(ValueError, match="lower <= representative <= upper"):
        PhysicalScenarioPlan.from_dict(raw, evidence_ids={"MF.description"})


def test_scripted_source_sample_report_does_not_claim_a_full_project():
    from hara_agent.workflow import HARAState
    from hara_agent.services.reporting.projection import HARAReportProjectionService
    state = HARAState("offline", malfunctions=[asdict(mf())])
    state.record("offline_source_context_replay", available_malfunction_count=61)
    scope = HARAReportProjectionService._run_scope(state, [], 0)
    assert scope["run_scope"] == "OFFLINE SCRIPTED SOURCE SAMPLE"
    assert "1/61" in scope["scope_note"]
    assert "NOT MEASURED" in scope["full_project_population"]


def test_budget_stop_saves_raw_response_before_schema_repair(method, tmp_path):
    from hara_agent.workflow import HARAState, CheckpointRepository
    from hara_agent.workflow.physical_analysis import construct_physical_scenarios
    from hara_agent.infrastructure.llm.provider_budget import ProviderAttemptBudgetExceeded
    class Client:
        calls = 0
        def complete_json(self, request):
            self.calls += 1
            if self.calls == 1:
                return LLMResponse({"hypotheses": []}, "offline")
            raise ProviderAttemptBudgetExceeded(limit=1, attempts=1, ledger_path=tmp_path / "dummy")
    state = HARAState("offline")
    repository = CheckpointRepository(tmp_path / "checkpoints")
    with pytest.raises(ProviderAttemptBudgetExceeded):
        construct_physical_scenarios(state, {"MF": [make_candidate()]}, [mf()], method=method,
            agent=PhysicalScenarioAgent(Client(), response_checkpoint_dir=tmp_path / "responses"),
            project_facts=None, checkpoint=repository.save)
    restored = repository.load("offline")
    entry = next(iter(restored.item_definition["physical_analysis_plans"].values()))
    assert len(entry["responses"]) == 1 and entry["status"] == "REQUESTED"
    assert restored.item_definition["physical_analysis_metrics"]["cumulative_uncached_requests"] == 1


def test_lateral_steering_cannot_generate_motion_from_rest(method):
    raw = raw_plan("CONSTANT_LATERAL_ACCELERATION_TO_CONTACT")
    raw["hypotheses"][0]["parameters"][0]["value"] = 0
    plan = PhysicalScenarioPlan.from_dict(raw, evidence_ids={"MF.description"})
    children, errors = PhysicalHypothesisInstantiationService(method).instantiate(make_candidate(), mf(), plan,
        plan_key="offline", response_model="offline")
    assert not children and errors[0]["reason"] == "STEERING_LATERAL_MODEL_REQUIRES_LONGITUDINAL_MOTION"


def test_project_grade_limit_and_parameter_units_are_checked(method):
    raw = raw_plan("GRADE_ROLLAWAY_TO_IMPACT_DISTANCE")
    plan = PhysicalScenarioPlan.from_dict(raw, evidence_ids={"MF.description"})
    children, errors = PhysicalHypothesisInstantiationService(method).instantiate(make_candidate(), mf(), plan,
        plan_key="offline", response_model="offline", project_context={"slope_max_pct": 10})
    assert not children and errors[0]["reason"] == "SLOPE_OUTSIDE_PROJECT_ODD"
    raw["hypotheses"][0]["parameters"][2]["unit"] = "degrees"
    with pytest.raises(ValueError):
        PhysicalScenarioPlan.from_dict(raw, evidence_ids={"MF.description"})


def test_failed_braking_is_not_counted_as_an_available_avoidance_path(method):
    raw = raw_plan(); raw["hypotheses"][0]["failed_controls"].append("braking")
    sc = instantiate(method, raw)[0]
    result = score(method, sc)
    assert result["controllability"]["pending_reason"] == "FAILED_CONTROL_PROFILE_DEPENDENCY_UNRESOLVED"
    assert not result["branch_summary"]["highest_known_asil"]


def test_aggregate_compares_all_instantiated_points_without_splicing_sec(method):
    from hara_agent.workflow import HARAState
    from hara_agent.workflow.nodes.scoring import score_structured_scenarios
    from hara_agent.services.analysis import MethodContractASILService
    raw = raw_plan()
    raw["hypotheses"][0]["parameters"][-1].update(kind="RANGE", lower=.1, upper=20)
    cases = instantiate(method, raw)
    assessments = []
    for sc in cases:
        assessments.append(ScenarioFeasibilityAgent._parse(mf(), dict(scenario_id=sc.scenario_id,
            physically_feasible=True, functionally_relevant=True, causally_relevant=True,
            breakpoint="NONE", hazardous_event="offline conditional contact in front", risk_dimension_changes=[], confidence=.9,
            causal_chain={"m_to_b": dict(claim="defined fault", basis_type="DIRECT_FACT", evidence_refs=["MF.description"]),
                "b_to_i": dict(claim="computed trajectory", basis_type="DERIVED_PHYSICS", evidence_refs=["DERIVED.ego_impact_speed_kph"]),
                "i_to_h": dict(claim="contact", basis_type="DERIVED_PHYSICS", evidence_refs=["DERIVED.relative_speed_kph"])}), scenario=sc).to_dict())
    state = HARAState("offline-range", scenarios=cases, malfunctions=[asdict(mf())],
        item_definition={"scenario_assessments": assessments})
    score_structured_scenarios(state, StructuredRiskScoringService(method), MethodContractASILService(method))
    summary = state.item_definition["physical_hypothesis_summaries"][0]
    assert summary["evaluated_points"] == len(cases)
    if summary["highest_sampled_asil"]:
        row = next(r for r in state.risk_results if r.scenario_id == summary["deciding_asil_scenario"])
        assert summary["deciding_SEC"] == {"S": row.severity.value, "E": row.exposure.value, "C": row.controllability.value}
    assert not summary["vehicle_upper_bound_proven"]
    assert not summary["range_invariance_proven"]
