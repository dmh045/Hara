"""Selected C references fill bounded method gaps without manufacturing facts."""
from copy import deepcopy
from dataclasses import replace

import pytest

from test_autonomous_physics import method, raw_plan, make_candidate, instantiate, score
from test_lateral_c_reference import steering_candidate
from hara_agent.services.analysis.scenario_physics import branch_physics_view
from hara_agent.services.analysis.structured_risk_scoring_service import StructuredRiskScoringService
from hara_agent.services.analysis.asil_service import MethodContractASILService
from hara_agent.services.analysis.risk_execution_trace_service import RiskExecutionTraceService
from hara_agent.services.reporting.engineering_text_mapper import EngineeringReportTextMapper


@pytest.mark.parametrize("model", [
    "CONSTANT_ACCELERATION_TO_IMPACT_DISTANCE", "GRADE_ROLLAWAY_TO_IMPACT_DISTANCE",
    "CONSTANT_LATERAL_ACCELERATION_TO_CONTACT",
])
def test_outside_reference_preserves_unknown_capabilities_and_branch_matrix(method, model):
    parent = make_candidate(7 if "LATERAL" in model else 0)
    parent.facts.update(driver_in_vehicle=False, operating_mode="parking_vehicle_control",
        component_category="computing", scenario_atom_ids=["CN_parking_multistory", "FB001", "FA040"])
    parent.facts.pop("remote_intervention_available")
    parent.facts.pop("other_road_user_avoidance_possible")
    raw = raw_plan(model, initial=parent.facts["ego_speed_kph"])
    raw["hypotheses"][0]["remaining_controls"] = ["braking"]
    sc = instantiate(method, raw, parent)[0]
    before = deepcopy(sc)
    result = score(method, sc)
    c = result["controllability"]
    assert c["controllability_score"] == "C3"
    assert c["engineering_rule_id"] == "IAV_UNATTENDED_PARKING_REFERENCE"
    assert c["engineering_status"] == "PENDING"
    assert sc == before
    assert "remote_intervention_available" not in sc.facts
    assert "other_road_user_avoidance_possible" not in sc.facts
    for branch in result["branch_evaluations"]:
        assert branch["C"] == "C3"
        assert branch["ASIL"] == MethodContractASILService(method).determine(branch["S"], branch["E"], branch["C"])
    trace = RiskExecutionTraceService(method)._controllability_trace(sc.facts, c)
    assert trace["decision_tree_stage"] == "REFERENCE"
    assert not trace["ttc_branch_eligible"]
    assert "其他交通参与者不能避让" not in EngineeringReportTextMapper.controllability_branch({"controllability": trace})
    assert "不在驾驶位" in EngineeringReportTextMapper.controllability_branch({"controllability": trace})


@pytest.mark.parametrize("failure", ["loss", "degraded", "stuck", "delayed"])
def test_additional_steering_reference_is_scoped_and_keeps_risk_tuple(method, failure):
    sc = steering_candidate(method)
    sc.facts["failure_type"] = failure
    result = score(method, sc)
    assert result["controllability"]["controllability_score"] == "C1"
    assert result["controllability"]["engineering_rule_id"] == "IAV_LOW_SPEED_STEERING_DEVIATION_BRAKING_REFERENCE"
    # An unrelated function name cannot undo an evidenced steering consequence;
    # absence of that consequence must, even with the same lateral geometry.
    sc.facts["steering_control_available"] = True
    sc.facts["_function_output"] = "drive torque"
    assert not score(method, sc)["controllability"]["controllability_score"]


def test_explicit_braking_loss_does_not_erase_independent_evasion(method):
    raw = raw_plan()
    raw["hypotheses"][0]["failed_controls"] = ["braking"]
    parent = make_candidate()
    parent.facts.update(operating_mode="parking_vehicle_control")
    sc = instantiate(method, raw, parent)[0]
    result = score(method, sc)
    assert result["controllability"]["controllability_score"] == "C3"
    assert result["controllability"]["engineering_rule_id"] == "IAV_UNAVAILABLE_BRAKING_REFERENCE"
    sc.facts["other_road_user_avoidance_possible"] = True
    result = score(method, sc)
    assert result["controllability"]["controllability_score"] == "C1"
    assert result["controllability"]["decision_status"] == "OVERRIDE_MATCH"


@pytest.mark.parametrize("mutation", ["unknown_driver", "unapproved_driver", "wrong_scope", "no_contact", "no_time", "search"])
def test_reference_does_not_fill_arbitrary_pending_inputs(method, mutation):
    sc = steering_candidate(method)
    view = branch_physics_view({**sc.facts, "scenario_id": sc.scenario_id, "malfunction_id": "MF",
                               "_fact_provenance": deepcopy(sc.fact_provenance)}, sc.facts["collision_branches"][0])
    view["driver_in_vehicle"] = False
    if mutation == "unknown_driver":
        view.pop("driver_in_vehicle")
    elif mutation == "unapproved_driver":
        view["_fact_provenance"]["driver_in_vehicle"]["approval"] = "PENDING"
    elif mutation == "wrong_scope":
        view["_fact_provenance"]["driver_in_vehicle"]["applicable_scope"] = {"scenario_id": "different"}
    elif mutation == "no_contact":
        view["motion_model_outcome"] = {"status": "NO_CONTACT"}
    elif mutation == "no_time":
        view.pop("first_contact_time_s", None)
        view.pop("ttc_s", None)
    elif mutation == "search":
        view["operating_mode"] = "parking_search"
    assert StructuredRiskScoringService(method)._conditional_reference_control(
        view, phase="AFTER_APPLICABLE_METHODS") is None


def test_successful_ttc_and_positive_override_keep_priority(method):
    raw = raw_plan()
    raw["hypotheses"][0]["remaining_controls"] = ["braking"]
    sc = instantiate(method, raw)[0]
    metadata = deepcopy(method.metadata)
    metadata["project_analysis_policy"]["controllability"]["conditional_reference_rules"] = []
    assert score(method, sc) == score(replace(method, metadata=metadata), sc)
    sc.facts.update(driver_in_vehicle=False, remote_intervention_available=True)
    assert score(method, sc)["controllability"]["decision_status"] == "OVERRIDE_MATCH"


def crossing_candidate(method, *, offset=4):
    from test_bounded_interaction_coverage import moving_plan
    parent = make_candidate(3.6)
    parent.facts.update(operating_mode="parking_vehicle_control", component_category="computing",
        scenario_atom_ids=["CN_parking_multistory", "FB001", "FA040"])
    return instantiate(method, moving_plan(target_direction="RIGHT", x=offset, y=offset,
        position="LEFT", collision="SIDE"), parent)[0]


@pytest.mark.parametrize("offset,expected", [(3.5, "C3"), (4, "C2"), (5, "C1"), (6, "C0")])
def test_crossing_contact_uses_active_profile_without_fabricating_longitudinal_ttc(method, offset, expected):
    sc = crossing_candidate(method, offset=offset)
    before = deepcopy(sc)
    result = score(method, sc)
    c = result["controllability"]
    assert c["controllability_score"] == expected
    assert c["engineering_status"] == "PENDING"
    assert c["calculation_nature"] == "CONDITIONAL"
    lookup = c["reference_rule"]["profile_lookup"]
    assert lookup["time_input"] == "first_contact_time_s"
    assert lookup["time_value_s"] == offset - .5
    assert lookup["band_rule_id"] in {b.rule_id for b in method.structured_risk_method.controllability_profile.bands}
    assert "ttc_s" not in sc.facts and "ttc_s" not in c["inputs_used"]
    assert sc == before
    for branch in result["branch_evaluations"]:
        assert branch["C"] == expected
        assert branch["ASIL"] == MethodContractASILService(method).determine(branch["S"], branch["E"], branch["C"])


@pytest.mark.parametrize("mutation", ["unknown_braking", "failed_shared_braking", "no_contact", "unscoped_time", "rear_target", "search"])
def test_crossing_transfer_does_not_bypass_applicability_or_provenance(method, mutation):
    sc = crossing_candidate(method)
    view = branch_physics_view({**sc.facts, "scenario_id": sc.scenario_id, "malfunction_id": "MF",
        "_fact_provenance": deepcopy(sc.fact_provenance)}, sc.facts["collision_branches"][0])
    if mutation == "unknown_braking":
        view["emergency_braking_available"] = None
    elif mutation == "failed_shared_braking":
        view["emergency_braking_available"] = False
    elif mutation == "no_contact":
        view["motion_model_outcome"] = {"status": "NO_CONTACT_WITHIN_MODEL"}
    elif mutation == "unscoped_time":
        metadata = view["_fact_provenance"]["first_contact_time_s"]
        while metadata.get("input_fact_metadata"):
            metadata = metadata["input_fact_metadata"][0]
        metadata["applicable_scope"] = {"malfunction_id": "MF", "scenario_id": "other"}
    elif mutation == "rear_target":
        view["object_longitudinal_direction"] = "FORWARD"
    else:
        view["operating_mode"] = "parking_search"
    assert StructuredRiskScoringService(method)._conditional_reference_control(view) is None
