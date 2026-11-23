"""A validated singleton request is a condition, never a new source fact."""
from copy import deepcopy
import json

import pytest

from test_autonomous_physics import method, make_candidate, mf, score
from test_scoped_timing_and_avoidance import timed_plan
from test_geometry_completion import Client
from hara_agent.contracts.physical_scenario_plan import PhysicalScenarioPlan
from hara_agent.services.analysis.physical_hypothesis_instantiation_service import PhysicalHypothesisInstantiationService
from hara_agent.services.analysis.scenario_equivalence import signature, scenario_conditions
from hara_agent.services.semantic.physical_scenario_agent import PhysicalScenarioAgent
from hara_agent.workflow import HARAState
from hara_agent.workflow.physical_analysis import _physical_conditions, construct_physical_scenarios


def unknown_parent():
    parent = make_candidate(3.6)
    for field in ("driver_position", "driver_in_vehicle"):
        parent.facts.pop(field, None)
        parent.fact_provenance.pop(field, None)
    return parent


def signed_context(producer, parent, positions):
    context = {"conditions": producer.physical_conditions(parent), "driver_configurations": positions,
        "malfunction": mf().to_dict()}
    context["plan_key"] = signature(context)
    return context


def instantiate(producer, parent, raw, context):
    return producer.instantiate(parent, mf(), PhysicalScenarioPlan.from_dict(raw, evidence_ids={"MF.description"}),
        plan_key=context["plan_key"], response_model="frozen-response", request_context=context)


def test_singleton_request_projects_pending_driver_scope_and_changes_identity(method):
    producer = PhysicalHypothesisInstantiationService(method)
    parent, raw = unknown_parent(), timed_plan()
    context = signed_context(producer, parent, ["in_driver_seat"])
    before = deepcopy((parent, raw, context))
    legacy, errors = producer.instantiate(parent, mf(), PhysicalScenarioPlan.from_dict(raw, evidence_ids={"MF.description"}),
        plan_key=context["plan_key"], response_model="frozen-response")
    assert not errors and legacy[0].facts.get("driver_in_vehicle") is None
    children, errors = instantiate(producer, parent, raw, context)
    assert not errors and len(children) == 1
    child = children[0]
    assert child.scenario_id != legacy[0].scenario_id
    assert child.facts["driver_position"] == "in_driver_seat" and child.facts["driver_in_vehicle"] is True
    for field in ("motion_model_outcome", "relative_speed_kph", "ttc_s"):
        assert child.facts[field] == legacy[0].facts[field]
    metadata = child.fact_provenance["driver_in_vehicle"]
    assert metadata["approval"] == "PENDING" and metadata["selection_basis"] == "VALIDATED_REQUEST_DRIVER_SCOPE"
    assert any(r["location"] == "context.driver_configurations" for r in metadata["source_refs"])
    assert score(method, child)["controllability"]["controllability_score"]
    assert before == (parent, raw, context)


@pytest.mark.parametrize("positions", [[], ["in_driver_seat", "outside_driver_seat"], ["unknown"]])
def test_unknown_or_multiple_request_configs_do_not_default_inside(method, positions):
    producer = PhysicalHypothesisInstantiationService(method); parent = unknown_parent()
    children, errors = instantiate(producer, parent, timed_plan(), signed_context(producer, parent, positions))
    assert not errors and children[0].facts.get("driver_in_vehicle") is None
    assert not children[0].facts.get("driver_position")


@pytest.mark.parametrize("mutation", ["unsigned_config", "other_malfunction", "other_conditions"])
def test_changed_or_foreign_request_context_cannot_project_driver(method, mutation):
    producer = PhysicalHypothesisInstantiationService(method); parent = unknown_parent()
    context = signed_context(producer, parent, ["in_driver_seat"])
    if mutation == "unsigned_config":
        context["driver_configurations"] = ["outside_driver_seat"]
    else:
        if mutation == "other_malfunction": context["malfunction"]["malfunction_id"] = "OTHER"
        else: context["conditions"]["facts"]["ego_speed_kph"] = 99
        context["plan_key"] = signature({k:v for k,v in context.items() if k != "plan_key"})
    children, errors = instantiate(producer, parent, timed_plan(), context)
    assert children[0].facts.get("driver_in_vehicle") is None
    assert any(e["reason"] == "REQUEST_DRIVER_SCOPE_CONTEXT_NOT_APPLICABLE" for e in errors)


@pytest.mark.parametrize("kind", ["unknown", "outside_fact", "conflicting_response"])
def test_explicit_personnel_limit_takes_precedence_over_request_scope(method, kind):
    producer = PhysicalHypothesisInstantiationService(method); parent = unknown_parent(); raw = timed_plan()
    if kind == "outside_fact":
        from test_p7_closure import source
        parent.facts["driver_in_vehicle"] = False; parent.fact_provenance["driver_in_vehicle"] = source()
    elif kind == "unknown": raw["hypotheses"][0]["unresolved_controls"] = {"driver_in_vehicle": "Vehicle location unestablished"}
    else:
        raw["hypotheses"][0]["intervention_assumptions"] = {"driver_in_vehicle": {
            "value": False, "rationale": "Explicit outside condition", "basis_refs": ["MF.description"], "conditions": ["outside vehicle"]}}
    children, errors = instantiate(producer, parent, raw, signed_context(producer, parent, ["in_driver_seat"]))
    assert children[0].facts.get("driver_in_vehicle") is not True
    assert children[0].facts.get("driver_position") != "in_driver_seat"
    if kind == "conflicting_response": assert any(e["reason"] == "REQUEST_DRIVER_SCOPE_CONFLICT" for e in errors)


def test_outside_seat_request_never_implies_outside_vehicle(method):
    producer = PhysicalHypothesisInstantiationService(method); parent = unknown_parent()
    children, errors = instantiate(producer, parent, timed_plan(), signed_context(producer, parent, ["outside_driver_seat"]))
    assert not errors and children[0].facts["driver_position"] == "outside_driver_seat"
    assert children[0].facts.get("driver_in_vehicle") is None


@pytest.mark.parametrize("presence", [None, True, False])
def test_canonical_request_view_migration_preserves_frozen_identity(method, presence):
    parent = unknown_parent()
    if presence is not None:
        from test_p7_closure import source
        parent.facts["driver_in_vehicle"] = presence; parent.fact_provenance["driver_in_vehicle"] = source()
    original = scenario_conditions(parent, without_driver=True)
    if (not PhysicalHypothesisInstantiationService.driver_position(parent)
            and parent.facts.get("driver_in_vehicle") is not None):
        full = scenario_conditions(parent)
        for section in ("facts", "fact_authority"):
            if "driver_in_vehicle" in full[section]: original[section]["driver_in_vehicle"] = full[section]["driver_in_vehicle"]
    assert original == _physical_conditions(parent) == PhysicalHypothesisInstantiationService.physical_conditions(parent)
    assert signature(original) == signature(_physical_conditions(parent))


def test_production_request_and_old_checkpoint_restore_use_singleton_without_new_requests(method, tmp_path):
    parent = unknown_parent(); client = Client([timed_plan()]); agent = PhysicalScenarioAgent(client, response_checkpoint_dir=tmp_path)
    state = HARAState("singleton-request-test")
    def run(current):
        return construct_physical_scenarios(current, {"MF":[parent]}, [mf()], method=method, agent=agent,
            project_facts=None, _driver_configurations=["in_driver_seat"])
    children = run(state)
    assert len(client.requests) == 1 and children["MF"][0].facts["driver_in_vehicle"] is True
    restored = HARAState.read_committed(json.loads(json.dumps(state.to_dict())))
    assert run(restored) == children and len(client.requests) == 1
