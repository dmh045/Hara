"""Source-linked fault premises through existing production consumers; no Provider."""
from copy import deepcopy
from dataclasses import asdict, replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from hara_agent.infrastructure.llm import LLMResponse
from hara_agent.models import FunctionDefinition, GuidewordAssessment, ReviewStatus, SourceRef
from hara_agent.models import evaluate_risk_eligibility
from hara_agent.services.semantic import MalfunctionHazardAgent
from hara_agent.services.semantic.scenario_agent import ScenarioFeasibilityAgent
from hara_agent.services.semantic.scenario_evidence import (
    CausalEvidenceSelector, build_fact_registry, ScenarioEvidenceContractError,
)
from test_p7_closure import method
from test_autonomous_physics import raw_plan, make_candidate


def inputs(subject="横向引导", *, function_id="renamed-function"):
    function = FunctionDefinition(function_id, "半自动泊车", subject,
        "接管纵向和横向引导，支持泊入/泊出。",
        preconditions=["驾驶员持续监控"], triggers=["驾驶员确认泊车"],
        odd_constraints=["不支持公共道路自动驾驶"],
        sources=[SourceRef("item_definition", "renamed-input.pdf", "p3",
            "Supports semi-automated parking and parking-out with lateral and longitudinal guidance.")],
        status=ReviewStatus.FINALIZED)
    assessment = GuidewordAssessment(function_id, "More", True, "输出幅值可能超过需求",
        sources=function.sources, status=ReviewStatus.FINALIZED)
    return function, assessment


def proposal(quantity="转向角", subject="横向引导"):
    return dict(malfunction_id="model-local", guideword="More",
        description=f"{subject}在泊入/泊出过程中输出的{quantity}超过目标需求",
        functional_effect="车辆运动偏离期望轨迹", vehicle_level_hazard="与轨迹上的其他车辆接触",
        causal_chain=["功能输出偏大", "运动偏离期望轨迹"], confidence=.8, status="PENDING",
        derivation=dict(function_refs=["output"], subject=subject, quantity=quantity,
            basis="功能接管引导，More对应所述输出幅值超出需求",
            assumptions=["以转向角表达横向引导偏差；未假定执行器硬件损坏"]))


class Scripted:
    def __init__(self, *batches):
        self.batches, self.requests = list(batches), []

    def complete_json(self, request):
        self.requests.append(request)
        assert self.batches, "unexpected extra semantic request"
        return LLMResponse({"candidates": self.batches.pop(0)}, "offline-scripted")


@pytest.mark.parametrize("limit", [4096, 16384, 65536])
def test_malfunction_request_honors_configured_output_allowance(limit):
    function, assessment = inputs()
    client = Scripted([proposal()])
    client.config = SimpleNamespace(max_tokens=limit)
    candidates, audit = MalfunctionHazardAgent(client).generate(function, [assessment])
    assert len(candidates) == 1 and audit["llm_call_count"] == 1
    assert client.requests[0].max_tokens == limit


def test_source_bound_single_quantity_uses_one_request_and_preserves_limits():
    function, assessment = inputs()
    client = Scripted([proposal()])
    candidates, audit = MalfunctionHazardAgent(client).generate(function, [assessment])
    assert len(candidates) == 1 and audit["llm_call_count"] == 1
    assert candidates[0].status is ReviewStatus.PENDING and candidates[0].is_analysis_premise
    assert "泊入/泊出" in candidates[0].description  # no operation Cartesian product
    assert candidates[0].sources == function.sources
    request = client.requests[0]
    for text in ("驾驶员持续监控", "驾驶员确认泊车", "不支持公共道路自动驾驶", "parking-out"):
        assert text in request.user_prompt
    assert "不机械展开" in request.system_prompt


@pytest.mark.parametrize("bad", [
    proposal("转向角或转向速率"), proposal("转向角/转向速率"), proposal(subject="横向执行器"),
])
def test_compound_quantity_or_invented_component_is_repaired_in_existing_request(bad):
    function, assessment = inputs()
    client = Scripted([bad], [proposal()])
    values, audit = MalfunctionHazardAgent(client).generate(function, [assessment])
    assert len(values) == 1 and values[0].derivation["quantity"] == "转向角"
    assert audit["llm_call_count"] == 2 and audit["coverage_repair_count"] == 1
    assert audit["rejected_candidates"]
    assert "RejectedCandidates=" in client.requests[1].user_prompt


def test_invalid_alternative_is_not_hidden_by_a_valid_candidate_of_same_guideword():
    function, assessment = inputs()
    second = proposal("转向速率")
    client = Scripted([proposal(), proposal("转向角或转向速率")], [second])
    values, audit = MalfunctionHazardAgent(client).generate(function, [assessment])
    assert {v.derivation["quantity"] for v in values} == {"转向角", "转向速率"}
    assert len({v.malfunction_id for v in values}) == 2
    assert audit["llm_call_count"] == 2 and not audit["coverage_missing_before_repair"]


def test_actual_source_component_is_accepted_without_project_name_or_id_rules():
    function, assessment = inputs("横向执行器", function_id="different-project-function")
    values, audit = MalfunctionHazardAgent(Scripted([proposal(subject="横向执行器")])).generate(function, [assessment])
    assert values[0].function_id == function.function_id and audit["llm_call_count"] == 1


def test_identical_valid_candidates_are_removed_locally_without_semantic_repair():
    function, assessment = inputs()
    client = Scripted([proposal(), proposal()])
    values, audit = MalfunctionHazardAgent(client).generate(function, [assessment])
    assert len(values) == 1 and audit["llm_call_count"] == 1
    assert not audit["rejected_candidates"]


@pytest.mark.parametrize("mutate, error", [
    (lambda p: p["derivation"].update(function_refs=["missing"]), "FunctionContext"),
    (lambda p: p["derivation"].update(function_refs=["source[0]"], subject="renamed-input.pdf"), "absent"),
    (lambda p: p["derivation"].update(assumptions="not a list"), "array"),
])
def test_invalid_source_links_do_not_establish_a_fault_premise(mutate, error):
    function, _ = inputs()
    value = proposal(); mutate(value)
    with pytest.raises(ValueError, match=error):
        MalfunctionHazardAgent._derivation(function, value)


def test_provider_cannot_approve_the_generated_premise():
    function, assessment = inputs()
    value = proposal(); value["status"] = "FINALIZED"
    candidates, _ = MalfunctionHazardAgent(Scripted([value])).generate(function, [assessment])
    assert candidates[0].status is ReviewStatus.PENDING


def causal_payload(scenario):
    return dict(scenario_id=scenario.scenario_id, physically_feasible=True, functionally_relevant=True,
        causally_relevant=True, breakpoint="NONE", hazardous_event="在该故障假设下车辆与前方对象接触",
        causal_chain={
            "m_to_b": dict(claim="驱动输出偏大", basis_type="DIRECT_FACT", evidence_refs=["MF.description", "MF.functional_effect"]),
            "b_to_i": dict(claim="按所建轨迹发生接触", basis_type="DERIVED_PHYSICS", evidence_refs=["DERIVED.ego_impact_speed_kph"]),
            "i_to_h": dict(claim="按该速度形成碰撞", basis_type="DERIVED_PHYSICS", evidence_refs=["DERIVED.relative_speed_kph"])},
        risk_dimension_changes=[], confidence=.9, status="FINALIZED")


def conditional_world(method):
    from hara_agent.contracts.physical_scenario_plan import PhysicalScenarioPlan
    from hara_agent.services.analysis.physical_hypothesis_instantiation_service import PhysicalHypothesisInstantiationService
    function, guideword = inputs("驱动控制")
    raw = proposal("驱动扭矩", "驱动控制")
    raw.update(functional_effect="非预期加速", causal_chain=["驱动输出过大", "非预期加速"], component_category="actuator_propulsion")
    raw["derivation"].update(basis="驱动控制输出过大导致非预期加速", assumptions=[])
    malfunction = MalfunctionHazardAgent(Scripted([raw])).generate(function, [guideword])[0][0]
    malfunction.malfunction_id = "MF"
    plan = raw_plan(); plan["hypotheses"][0]["remaining_controls"] = ["driver_braking"]
    children, errors = PhysicalHypothesisInstantiationService(method).instantiate(make_candidate(), malfunction,
        PhysicalScenarioPlan.from_dict(plan, evidence_ids={"MF.description"}), plan_key="scripted-source-premise", response_model="offline-scripted")
    assert not errors and len(children) == 1
    return function, malfunction, children[0]


def test_conditional_fault_reaches_physics_causality_scoring_and_report(method, tmp_path):
    from hara_agent.workflow import HARAState
    from hara_agent.workflow.nodes.scoring import score_structured_scenarios
    from hara_agent.workflow.review_artifacts import ReviewArtifactWriter
    from hara_agent.services.analysis.risk_services import RiskScoringServices
    from hara_agent.services.reporting import HARAReportProjectionService, load_report_schema
    function, malfunction, scenario = conditional_world(method)
    assessment = ScenarioFeasibilityAgent._parse(malfunction, causal_payload(scenario), scenario=scenario)
    assert assessment.retain and assessment.status is ReviewStatus.PENDING
    selected = CausalEvidenceSelector().select(build_fact_registry(malfunction, scenario))
    assert "analysis_premise; approval=PENDING; not_item_fact" in selected.compact_view
    state = HARAState("source-premise", functions=[asdict(function)], malfunctions=[asdict(malfunction)],
        scenarios=[scenario], item_definition={"scenario_assessments": [asdict(assessment)]})
    services = RiskScoringServices.from_method(method)
    writer = ReviewArtifactWriter(state.run_id, tmp_path)
    score_structured_scenarios(state, services.scoring, services.asil, review_artifact_writer=writer)
    assert len(state.risk_results) == 1
    risk = state.risk_results[0]
    assert risk.severity.value == "S1" and risk.controllability.value == "C3"
    # This minimal physics fixture has no complete exposure selection. Do not
    # fabricate E/ASIL merely because the new fault premise is computable.
    assert not risk.exposure.value and "EXPOSURE_ATOM_BINDING_INCOMPLETE" in risk.exposure.review_reason
    assert all(getattr(risk, field).status is ReviewStatus.PENDING for field in ("severity", "exposure", "controllability", "asil"))
    trace = json.loads((writer.directory / "risk_execution_trace.json").read_text())
    view = HARAReportProjectionService(load_report_schema()).project(state, method, risk_trace=trace)
    assert view.rows and view.rows[0].severity == risk.severity.value
    assert "失效推导依据" in view.malfunction_inventory[0].coverage_summary
    restored = HARAState.read_committed(json.loads(json.dumps(state.to_dict())))
    assert restored.malfunctions == state.malfunctions
    assert restored.risk_results == state.risk_results


def test_fault_premise_cannot_prove_downstream_interaction_by_itself(method):
    _, malfunction, scenario = conditional_world(method)
    payload = causal_payload(scenario)
    payload["causal_chain"]["b_to_i"].update(basis_type="DIRECT_FACT", evidence_refs=["MF.description"])
    with pytest.raises(ScenarioEvidenceContractError, match="SELF_REFERENTIAL"):
        ScenarioFeasibilityAgent._parse(malfunction, payload, scenario=scenario)


def test_pending_downstream_evidence_cannot_borrow_fault_premise_eligibility(method):
    _, malfunction, scenario = conditional_world(method)
    assessment = ScenarioFeasibilityAgent._parse(malfunction, causal_payload(scenario), scenario=scenario)
    assert evaluate_risk_eligibility(assessment).eligible
    causal = assessment.causal_assessment
    bad_bindings = tuple(replace(b, status=ReviewStatus.PENDING) if b.edge_id == "B_TO_I" else b
                         for b in causal.evidence_bindings)
    altered = replace(assessment, causal_assessment=replace(causal, evidence_bindings=bad_bindings))
    assert not evaluate_risk_eligibility(altered).eligible
    assert not evaluate_risk_eligibility(replace(assessment, status=ReviewStatus.REJECTED)).eligible


def test_legacy_task_serialization_omits_empty_derivation_and_new_premises_isolate():
    from test_p6_scenario_pipeline import malfunction
    from hara_agent.services.analysis.scenario_equivalence import signature
    fault = malfunction()
    old = asdict(fault); old.pop("derivation")
    assert fault.to_dict() == old and signature(fault.to_dict()) == signature(old)
    fault.derivation = {"basis": "newly scoped premise"}
    assert signature(fault.to_dict()) != signature(old)


def test_frozen_real_numeric_case_retains_values_without_approving_new_fault(method, tmp_path):
    from hara_agent.workflow import HARAState
    from hara_agent.workflow.nodes.scoring import score_structured_scenarios
    from hara_agent.workflow.nodes.safety_goals import aggregate_safety_goals
    from hara_agent.workflow.review_artifacts import ReviewArtifactWriter
    from hara_agent.workflow.risk_rescoring import clean_risk_stage
    from hara_agent.services.analysis.risk_services import RiskScoringServices
    from hara_agent.services.analysis.method_safety_goal_service import MethodSafetyGoalService
    from hara_agent.services.reporting import HARAReportProjectionService, load_report_schema
    from hara_agent.services.reporting.canonical_renderer import HARAReportWorkbookRenderer
    fixture = json.loads((Path(__file__).parents[1] / "fixtures/report_numeric_physics_e2e_20261019.json").read_text())
    state = HARAState.read_committed(fixture["state"])
    state.item_definition["typed"] = fixture["physical_replay"]["project"]
    clean_risk_stage(state)
    services = RiskScoringServices.from_method(method)
    legacy = deepcopy(state)
    score_structured_scenarios(legacy, services.scoring, services.asil, risk_fact_binding=services.binding)
    raw_function = deepcopy(state.functions[0])
    raw_function["sources"] = [SourceRef(**s) for s in raw_function["sources"]]
    raw_function["status"] = ReviewStatus(raw_function["status"])
    function = FunctionDefinition(**raw_function)
    saved_mf = state.malfunctions[0]
    proposal_fields = ("malfunction_id", "guideword", "description", "functional_effect", "vehicle_level_hazard",
                      "causal_chain", "confidence", "component_category", "failure_type")
    raw = {k: saved_mf[k] for k in proposal_fields}
    raw.update(status="PENDING", derivation=dict(function_refs=["description"], subject="MDC",
        quantity="轮端扭矩控制信号", basis="所述扭矩控制信号在More偏差下高于需求；这是失效分析前提",
        assumptions=[]))
    guideword = GuidewordAssessment(function.function_id, raw["guideword"], True, "输出幅值偏大",
        sources=function.sources, status=ReviewStatus.FINALIZED)
    fault = MalfunctionHazardAgent(Scripted([raw])).generate(function, [guideword])[0][0]
    state.malfunctions = [asdict(fault)]
    # Keep the saved trajectory; only the fault premise's approval nature
    # changes. Recompile the frozen causal claims against that same world.
    for i, saved in enumerate(state.item_definition["scenario_assessments"]):
        scenario = next(s for s in state.scenarios if s.scenario_id == saved["scenario_id"])
        payload = {**saved, "risk_dimension_changes": saved["causal_assessment"]["risk_dimension_changes"]}
        payload.pop("risk_dimensions_changed", None)
        state.item_definition["scenario_assessments"][i] = asdict(ScenarioFeasibilityAgent._parse(fault, payload, scenario=scenario))
    writer = ReviewArtifactWriter(state.run_id, tmp_path)
    score_structured_scenarios(state, services.scoring, services.asil, risk_fact_binding=services.binding,
                              review_artifact_writer=writer)
    assert len(state.risk_results) == len(legacy.risk_results) > 0
    for current, previous in zip(state.risk_results, legacy.risk_results):
        for field in ("severity", "exposure", "controllability", "asil"):
            assert getattr(current, field).value == getattr(previous, field).value
            assert getattr(current, field).value
            assert getattr(current, field).status is ReviewStatus.PENDING
        assert [(b['S'], b['E'], b['C'], b['ASIL']) for b in current.branch_evaluations] == [
            (b['S'], b['E'], b['C'], b['ASIL']) for b in previous.branch_evaluations]
    aggregate_safety_goals(state, MethodSafetyGoalService(method))
    assert state.safety_goals and all(g.status is ReviewStatus.PENDING for g in state.safety_goals)
    trace = json.loads((writer.directory / "risk_execution_trace.json").read_text())
    schema = load_report_schema()
    view = HARAReportProjectionService(schema).project(state, method, risk_trace=trace)
    assert view.rows[0].asil == state.risk_results[0].asil.value
    HARAReportWorkbookRenderer().render(view, Path("references/HARA_Template_AI_20260327.xlsx"), tmp_path / "source-premise.xlsx", schema)
    assert (tmp_path / "source-premise.xlsx").exists()


@pytest.mark.parametrize("mode, requests", [("exact", 1), ("local", 1), ("driver_relevant", 2)])
def test_source_premise_keeps_exact_and_proven_driver_reuse(mode, requests):
    from test_reuse_reporting_acceptance import pair, CountingClient
    from test_p6_scenario_pipeline import malfunction
    from hara_agent.workflow.nodes.scenarios import _assess_instantiated_scenarios
    a, b = pair()
    if mode == "exact":
        b = deepcopy(a); b.scenario_id = "duplicate-identity"
    fault = malfunction()
    fault.status = ReviewStatus.PENDING
    fault.derivation = {"validation_status": "SOURCE_LINKED_ANALYSIS_PREMISE", "basis": "fixture function deviation"}
    client = CountingClient([a, b], mode)
    values, _ = _assess_instantiated_scenarios(ScenarioFeasibilityAgent(client, batch_max_items=1), fault, [a, b], {}, None)
    assert len(client.calls) == requests
    assert len(values) == 2 and all(v.retain for v in values)
    assert all(v.status is ReviewStatus.PENDING for v in values)
