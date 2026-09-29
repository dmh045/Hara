import pytest

from hara_agent.models import MalfunctionCandidate, ScenarioCandidate
from hara_agent.services.semantic.scenario_evidence import (
    ScenarioEvidenceContractError, ScenarioEvidenceErrorCode,
    build_fact_registry, validate_evidence_contract,
)


def context():
    malfunction = MalfunctionCandidate(
        "MF-EVAL", "FUN-1", "loss", "braking command lost", "no deceleration",
        "vehicle continues", ["command lost", "vehicle continues"],
    )
    scenario = ScenarioCandidate(
        "SCN-EVAL", "Parking", "atomic", "pedestrian ahead",
        {"relative_distance": "0.5 m", "relative_speed_kph": 5.0,
         "closing_speed_kph": 5.0,
         "object_position": "ahead"}, semantic_fingerprint="fp-eval",
    )
    return malfunction, scenario, build_fact_registry(malfunction, scenario)


def negative(dimensions=None):
    return {
        "scenario_id": "SCN-EVAL", "physically_feasible": True,
        "functionally_relevant": True, "causally_relevant": False,
        "breakpoint": "I_TO_H", "causal_chain": {
            "m_to_b": {"claim": "effect", "basis_type": "DIRECT_FACT", "evidence_refs": ["MF.functional_effect"]},
            "b_to_i": {"claim": "interaction", "basis_type": "DIRECT_FACT", "evidence_refs": ["SCN.object_position"]},
            "i_to_h": {"claim": "unsupported", "basis_type": "ASSUMPTION", "evidence_refs": []},
        },
        "risk_dimension_changes": dimensions or [], "hazardous_event": "",
        "potential_harm": "", "rationale": "breakpoint", "confidence": 0.8,
    }


def validate(item):
    malfunction, scenario, registry = context()
    return validate_evidence_contract(
        malfunction=malfunction, scenario=scenario, item=item, registry=registry,
        prompt_version="scenario-feasibility-v9", batch="1/1",
        split_path="root", split_depth=0,
    )


def test_structured_error_attributes_and_cross_field_code():
    item = negative([{
        "dimension": "distance", "evidence_refs": ["SCN.relative_distance"],
        "reason": "distance present",
    }])
    with pytest.raises(ScenarioEvidenceContractError) as caught:
        validate(item)
    error = caught.value
    assert error.code is ScenarioEvidenceErrorCode.CAUSAL_FALSE_WITH_DIMENSIONS
    assert error.hop == "risk_dimension_changes"
    assert error.malfunction_id == "MF-EVAL"
    assert error.scenario_id == "SCN-EVAL"
    assert error.semantic_fingerprint == "fp-eval"


def test_unknown_ref_has_machine_readable_code():
    item = negative()
    item["causal_chain"]["b_to_i"]["evidence_refs"] = ["SCN.driver_panic"]
    with pytest.raises(ScenarioEvidenceContractError) as caught:
        validate(item)
    assert caught.value.code is ScenarioEvidenceErrorCode.UNRESOLVED_EVIDENCE_REF
    assert caught.value.invalid_evidence_refs == ["SCN.driver_panic"]


def test_positive_assumption_has_machine_readable_code():
    item = negative()
    item.update({"causally_relevant": True, "breakpoint": "NONE",
                 "hazardous_event": "hazard", "potential_harm": "harm",
                 "risk_dimension_changes": [{"dimension": "distance", "evidence_refs": ["SCN.relative_distance"], "reason": "distance"}]})
    item["causal_chain"]["h_to_harm"] = {
        "claim": "harm", "basis_type": "DIRECT_FACT", "evidence_refs": ["MF.vehicle_level_hazard"],
    }
    with pytest.raises(ScenarioEvidenceContractError) as caught:
        validate(item)
    assert caught.value.code is ScenarioEvidenceErrorCode.ASSUMPTION_IN_POSITIVE_CHAIN


def test_current_contract_limitation_mixed_direct_and_derived_evidence_is_classified():
    item = negative()
    item["causal_chain"]["b_to_i"] = {
        "claim": "mixed support", "basis_type": "DERIVED_PHYSICS",
        "evidence_refs": ["SCN.relative_distance", "DERIVED.ttc_s"],
    }
    with pytest.raises(ScenarioEvidenceContractError) as caught:
        validate(item)
    assert caught.value.code is ScenarioEvidenceErrorCode.DERIVED_PHYSICS_KIND_MISMATCH


@pytest.mark.parametrize(("mutate", "expected"), [
    (lambda item: item["causal_chain"]["m_to_b"].update({"claim": ""}),
     ScenarioEvidenceErrorCode.EMPTY_HOP_CLAIM),
    (lambda item: item.update({"risk_dimension_changes": [{
        "dimension": "unknown", "evidence_refs": ["MF.description"], "reason": "x",
    }]}), ScenarioEvidenceErrorCode.UNKNOWN_RISK_DIMENSION),
    (lambda item: item.update({"risk_dimension_changes": [
        {"dimension": "distance", "evidence_refs": ["MF.description"], "reason": "x"},
        {"dimension": "distance", "evidence_refs": ["MF.description"], "reason": "x"},
    ]}), ScenarioEvidenceErrorCode.DUPLICATE_RISK_DIMENSION),
    (lambda item: item.update({"risk_dimension_changes": [{
        "dimension": "distance", "evidence_refs": ["MF.description"], "reason": "",
    }]}), ScenarioEvidenceErrorCode.MISSING_RISK_DIMENSION_REASON),
])
def test_distinct_fail_closed_branches_have_non_overlapping_codes(mutate, expected):
    item = negative()
    mutate(item)
    with pytest.raises(ScenarioEvidenceContractError) as caught:
        validate(item)
    assert caught.value.code is expected
