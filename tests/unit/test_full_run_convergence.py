"""Full-run regressions: production consumers with explicitly offline replies."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path

import pytest

from test_p6_scenario_pipeline import method, project, parent, malfunction, function
from test_autonomous_physics import raw_plan, make_candidate, instantiate, score, mf
from hara_agent.models import ItemDefinitionFacts, ScenarioCandidate, SourceRef
from hara_agent.services.extraction import DocumentReader
from hara_agent.services.semantic.item_definition_agent import ItemDefinitionNormalizer
from hara_agent.services.semantic.physical_scenario_agent import PhysicalScenarioAgent
from hara_agent.services.analysis import MethodScenarioCandidateService, ConstrainedScenarioSynthesisService, ProjectFactResolver
from hara_agent.infrastructure.llm import LLMResponse
from hara_agent.workflow import HARAState
from hara_agent.workflow.physical_analysis import construct_physical_scenarios


def scope_rows():
    return [dict(kind="table_row", location="table1.row2", section_path=["环境条件"],
                 text="停车场 | 室外停车场，室内停车场"),
            dict(kind="table_row", location="table1.row3", section_path=["环境条件"],
                 text="上下坡 | 上坡15%，下坡15%")]


def test_source_scope_is_allowed_range_not_actual_grade_and_is_idempotent():
    old = replace(project(), odd_locations=["停车场"], odd_road_types=["室外停车场", "室内停车场", "垂直车位"])
    before = asdict(old)
    facts = ItemDefinitionNormalizer.normalize_source_scope(old, scope_rows(), "item.docx")
    assert facts.odd_road_types == ["垂直车位"]
    assert set(facts.odd_locations) == {"停车场", "室内停车场", "室外停车场"}
    grade = facts.odd_scope_evidence[1]
    assert grade["quantity_semantic"] == "ALLOWED_ODD_RANGE"
    assert [v["reference_direction"] for v in grade["limits"]] == ["UNSPECIFIED"] * 2
    assert grade["sources"][0]["excerpt"] == scope_rows()[1]["text"]
    assert asdict(ItemDefinitionNormalizer.normalize_source_scope(facts, scope_rows(), "item.docx")) == asdict(facts)
    assert asdict(old) == before
    assert ItemDefinitionFacts.from_dict(asdict(facts)).odd_scope_evidence == facts.odd_scope_evidence


@pytest.mark.parametrize("change", [
    {"text": "上下坡 | 禁止上坡15%，下坡15%"},
    {"text": "上下坡 | 示例：上坡15%，下坡15%"},
    {"section_path": ["参考示例"]}, {"section_path": ["参考示例", "环境条件"]}, {"kind": "paragraph"},
])
def test_source_normalization_never_promotes_qualified_or_example_text(change):
    row = {**scope_rows()[1], **change}
    assert not ItemDefinitionNormalizer.normalize_source_scope(project(), [row], "item").odd_scope_evidence


def test_where_scope_refinement_weather_is_not_road_and_slope_represented(method):
    facts = ItemDefinitionNormalizer.normalize_source_scope(
        replace(project(), odd_locations=["停车场"], odd_road_types=["室内停车场", "室外停车场"],
                odd_weather_conditions=["晴朗", "雨天"]), scope_rows(), "item.docx")
    service = MethodScenarioCandidateService(method)
    speed = ProjectFactResolver().resolve_speed_context(facts, 'parking vehicle control', allow_aggregate_fallback=True)
    parents, _ = service.generate(project_facts=facts, operating_mode='parking vehicle control', speed_resolution=speed)
    assert {p.operating_scenario for p in parents} == {"室内停车场", "室外停车场"}
    assert all(p.facts['method_scenario_dimensions']['ROAD']['resolution_status'] != 'RESOLVED' for p in parents)
    children, audit = ConstrainedScenarioSynthesisService(method).generate_candidates(
        malfunction=asdict(malfunction()), parent=parent(), project_context=asdict(facts), function=asdict(function()))
    slopes = [s for s in children if s.facts.get("source_operating_conditions", {}).get("source_slope_pct") == 15]
    assert slopes
    assert {s.facts["source_operating_conditions"]["source_slope_direction"] for s in slopes} == {"上坡", "下坡"}
    assert all(s.facts["method_scenario_dimensions"]["ROAD"]["atom_id"] == "FB003" for s in slopes)
    assert all(s.fact_provenance["source_slope_pct"]["approval"] == "PENDING" for s in slopes)


def test_symbolic_geometry_retains_range_relation_without_repeated_same_regime(method):
    raw = raw_plan()
    params = raw["hypotheses"][0]["parameters"]
    params[1].pop("value"); params[1].update(kind="SYMBOLIC", symbol="d")
    params[2].update(kind="RANGE", lower=1, value=2, upper=3)
    children = instantiate(method, raw)
    assert len(children) == 1
    analysis = children[0].facts["physical_analysis"]
    assert "ego_impact_speed_kph" not in children[0].facts
    # The original range survives; a single scenario is not a proof that all
    # values have one rating or a declaration of the vehicle's physical limit.
    assert '"lower": 1' in json.dumps(analysis)
    assert 'FULL_PARAMETER_RELATION_RETAINED_NO_NEW_NUMERIC_CONTACT_RESULT' in json.dumps(analysis)
    params[2].update(lower=-1, value=0, upper=1)
    assert len(instantiate(method, raw)) > 1


def test_selected_source_boundary_is_not_silently_changed_by_plan(method):
    from hara_agent.contracts.physical_scenario_plan import PhysicalScenarioPlan
    from hara_agent.services.analysis.physical_hypothesis_instantiation_service import PhysicalHypothesisInstantiationService
    sc = make_candidate()
    sc.facts.update(source_slope_pct=15, source_grade_semantic='ALLOWED_ODD_BOUNDARY_REPRESENTATIVE')
    raw = raw_plan('GRADE_ROLLAWAY_TO_IMPACT_DISTANCE')
    assert instantiate(method, raw, sc)[0].facts['slope_percent'] == 15
    next(p for p in raw['hypotheses'][0]['parameters'] if p['field'] == 'slope_percent')['value'] = 10
    children, errors = PhysicalHypothesisInstantiationService(method).instantiate(sc, mf(),
        PhysicalScenarioPlan.from_dict(raw, evidence_ids={'MF.description'}), plan_key='test', response_model='offline')
    assert not children
    assert 'SLOPE_CONFLICTS_WITH_SELECTED_BOUNDARY' in json.dumps(errors)


def test_original_real_proposal_replays_through_production_physics_and_sec(method):
    from hara_agent.models import MalfunctionCandidate, ReviewStatus, FunctionDefinition
    from hara_agent.services.analysis import StructuredRiskScoringService, ExposureDimensionCoverageService
    from hara_agent.services.analysis.hazardous_event_risk_context_service import HazardousEventRiskContextService
    fixture = json.loads((Path(__file__).parents[1] / 'fixtures/report_numeric_physics_e2e_20261019.json').read_text())
    frozen = fixture['physical_replay']
    parent = HARAState._scenario_from_dict(frozen['candidate'])
    raw = fixture['state']['malfunctions'][0]
    fault = MalfunctionCandidate(**{**raw, 'sources': [SourceRef(**r) for r in raw['sources']], 'status': ReviewStatus(raw['status'])})
    class Client:
        calls = 0
        def complete_json(self, request):
            self.calls += 1
            return LLMResponse(deepcopy(frozen['response']['data']), 'OFFLINE_SAVED_REAL_RESPONSE')
    client = Client(); state = HARAState('real-proposal-offline', functions=fixture['state']['functions'])
    out = construct_physical_scenarios(state, {fault.malfunction_id: [parent]}, [fault], method=method,
        agent=PhysicalScenarioAgent(client), project_facts=ItemDefinitionFacts.from_dict(frozen['project']))
    sc = out[fault.malfunction_id][0]
    assert client.calls == 1
    old = fixture['state']['scenarios'][0]['facts']
    for key in ('ego_impact_speed_kph', 'relative_speed_kph', 'ttc_s'):
        assert sc.facts[key] == old[key]
    view = {**sc.facts, 'malfunction_id': fault.malfunction_id, 'scenario_id': sc.scenario_id,
            'component_category': fault.component_category, '_fact_provenance': sc.fact_provenance}
    contexts = HazardousEventRiskContextService(method)
    context = contexts.build(malfunction_id=fault.malfunction_id, scenario_id=sc.scenario_id, hazard_node_id='impact', scenario=view)
    view = contexts.scoring_facts(context, view)
    raw_function = next(f for f in fixture['state']['functions'] if f['function_id'] == fault.function_id)
    func = FunctionDefinition(**{**raw_function, 'sources': [SourceRef(**r) for r in raw_function['sources']],
                                 'status': ReviewStatus(raw_function['status'])})
    view['_exposure_dimension_coverage_decision'] = ExposureDimensionCoverageService(method).decide(
        assessment_key='replay', function=func, operating_mode=sc.operating_mode)
    result = StructuredRiskScoringService(method).score(view, 'unmodified real proposal replay')
    assert tuple(result[key][field] for key, field in (
        ('severity', 'severity_score'), ('exposure', 'exposure_score'), ('controllability', 'controllability_score'))) == ('S1', 'E3', 'C3')
    assert result['branch_summary']['highest_known_asil'] == 'A'
    assert result['branch_summary']['calculation_nature'] == 'CONDITIONAL'


def test_sample_private_policy_cannot_change_production_strategy_or_request(method):
    class Client:
        def __init__(self): self.requests = []
        def complete_json(self, request):
            self.requests.append(asdict(request))
            return LLMResponse(raw_plan(), "offline")
    requests, outputs = [], []
    for private in (None, "Use a private sample-only numeric shortcut"):
        state = HARAState("same-input")
        if private: state.item_definition["physical_validation_policy"] = private
        client = Client()
        out = construct_physical_scenarios(state, {"MF": [make_candidate()]}, [mf()],
            method=method, agent=PhysicalScenarioAgent(client), project_facts=None)
        requests.append(client.requests); outputs.append([asdict(c) for c in out["MF"]])
        context = next(iter(state.item_definition["physical_analysis_plans"].values()))["context"]
        assert context["analysis_strategy"] == PhysicalScenarioAgent.ANALYSIS_STRATEGY
    assert requests[0] == requests[1] and outputs[0] == outputs[1]


@pytest.mark.parametrize("remote,expected", [(True, "C1"), (False, "")])
def test_lateral_active_override_survives_without_longitudinal_ttc(method, remote, expected):
    candidate = make_candidate(3.6)
    candidate.facts["remote_intervention_available"] = remote
    sc = instantiate(method, raw_plan("CONSTANT_LATERAL_ACCELERATION_TO_CONTACT"), candidate)[0]
    result = score(method, sc)
    assert result["controllability"]["controllability_score"] == expected
    assert "ttc_s" not in sc.facts
    if expected:
        assert result["controllability"]["decision_status"] == "OVERRIDE_MATCH"
        assert result["controllability"]["engineering_status"] == "PENDING"
        assert all(row['scores']['controllability']['controllability_score'] == expected
                   for row in result['branch_evaluations'])
    else:
        assert result["controllability"]["pending_reason"] == "LATERAL_C_METHOD_NOT_AVAILABLE"


@pytest.mark.parametrize('external,expected', [(True, 'C1'), (False, '')])
def test_override_does_not_restore_failed_brake_actuator(method, external, expected):
    candidate = make_candidate()
    candidate.facts.update(remote_intervention_available=True, other_road_user_avoidance_possible=external)
    raw = raw_plan(); raw['hypotheses'][0]['failed_controls'] = ['braking']
    child = instantiate(method, raw, candidate)[0]
    assert child.facts['emergency_braking_available'] is False
    result = score(method, child)
    assert result['controllability']['controllability_score'] == expected
    assert all(row['scores']['controllability']['controllability_score'] == expected for row in result['branch_evaluations'])


@pytest.mark.parametrize('change', ['none', 'source', 'object', 'phase', 'query', 'driver'])
def test_scoped_source_absence_reuse_and_invalidation(change, tmp_path):
    from test_scenario_risk_fact_agent import _spec
    from hara_agent.contracts import FactOrigin, FactType
    from hara_agent.services.semantic import ScenarioRiskFactAgent
    class Client:
        calls = 0
        def complete_json(self, request):
            self.calls += 1
            payload = json.JSONDecoder().raw_decode(request.user_prompt.split('fact_requests=', 1)[1])[0]
            return LLMResponse({'results': [{**{k: q[k] for k in ('malfunction_id', 'scenario_id', 'fact_type')},
                                             'status': 'NOT_FOUND'} for q in payload]}, 'offline')
    source = SourceRef('item_definition', 'item.docx', 'table1', 'parking allowed')
    first = ScenarioCandidate('A', 'parking', 'object in front', 'same initial phase', sources=[source],
                              facts={'object_id': 'target', 'time_role': 'INITIAL', 'driver_position': 'inside'})
    second = replace(first, scenario_id='B', facts=deepcopy(first.facts))
    if change == 'source': second.sources = [replace(source, excerpt='changed source')]
    if change == 'object': second.facts['object_id'] = 'other'
    if change == 'phase': second.facts['time_role'] = 'POST_FAULT'
    if change == 'query': second.situational_description = 'object behind vehicle'
    if change == 'driver': second.facts['driver_position'] = 'outside'
    def assessments(candidates):
        return [dict(malfunction_id='MF', scenario_id=c.scenario_id, physically_feasible=True,
                     functionally_relevant=True, causally_relevant=True, status='FINALIZED') for c in candidates]
    spec = _spec(FactType.COLLISION_TYPE, FactOrigin.SCENARIO_FACT)
    client = Client(); agent = ScenarioRiskFactAgent(client, batch_max_results=1); cache = {}
    agent.interpret(assessments([first]), [first], [mf()], [spec], batch_cache=cache)
    # Persistence round trip, not an in-memory-only counter.
    cache = json.loads(json.dumps(cache))
    facts, audit = agent.interpret(assessments([second]), [second], [mf()], [spec], batch_cache=cache)
    assert not facts
    assert client.calls == (1 if change == 'none' else 2)
    assert audit['scoped_absence_reuse_count'] == (1 if change == 'none' else 0)
    assert audit['resolution_counts']['ABSENT_IN_EVIDENCE'] == 1
    assert audit['fact_resolutions'][0]['scenario_id'] == 'B'
