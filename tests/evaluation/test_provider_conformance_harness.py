from copy import deepcopy

from hara_agent.evaluation.stages import ProviderConformanceHarness
from hara_agent.models import MalfunctionCandidate, ReviewStatus, ScenarioCandidate
from hara_agent.services.semantic import ScenarioFeasibilityAgent


def context():
    malfunction = MalfunctionCandidate(
        "MF-1", "FUN-1", "loss", "command lost", "behavior unavailable",
        "hazard state", ["command lost", "hazard state"],
    )
    scenario = ScenarioCandidate(
        "SCN-1", "mode", "object ahead", "closing state",
        {"object_position": "ahead", "relative_distance": "1 m",
         "relative_speed_kph": 5.0, "closing_speed_kph": 5.0,
         "harm_mechanism": "contact can expose occupants to injury"},
        semantic_fingerprint="provider-fixture",
    )
    return malfunction, scenario


def valid_payload():
    return {
        "scenario_id": "SCN-1", "physically_feasible": True,
        "functionally_relevant": True, "causally_relevant": True,
        "breakpoint": "NONE",
        "causal_chain": {
            "m_to_b": {"claim": "behavior changes", "basis_type": "DIRECT_FACT", "evidence_refs": ["MF.functional_effect"]},
            "b_to_i": {"claim": "interaction changes", "basis_type": "DIRECT_FACT", "evidence_refs": ["SCN.object_position"]},
            "i_to_h": {"claim": "contact becomes possible", "basis_type": "DERIVED_PHYSICS", "evidence_refs": ["DERIVED.ttc_s"]},
            "h_to_harm": {"claim": "event can cause harm", "basis_type": "DIRECT_FACT", "evidence_refs": ["SCN.harm_mechanism"]},
        },
        "risk_dimension_changes": [{"dimension": "distance", "evidence_refs": ["SCN.relative_distance"], "reason": "explicit distance"}],
        "rationale": "complete chain", "hazardous_event": "contact",
        "potential_harm": "injury", "confidence": 0.8, "status": "PENDING",
    }


def test_provider_shaped_fixtures_converge_to_one_contract_and_cannot_self_approve():
    malfunction, scenario = context()
    parser = lambda payload: ScenarioFeasibilityAgent._parse(
        malfunction, payload, scenario=scenario,
    )
    glm, volcengine, mock = (deepcopy(valid_payload()) for _ in range(3))
    glm["status"] = "FINALIZED"
    volcengine["status"] = "APPROVED"
    report = ProviderConformanceHarness(parser).evaluate({
        "glm-compatible-fixture": glm,
        "volcengine-compatible-fixture": volcengine,
        "mock-provider": mock,
        "invalid-minimal": {"scenario": "x", "reason": "x"},
    })

    assert report["conformant_count"] == 3
    assert report["rejected_count"] == 1
    assert report["differential"]["canonical_contracts_equal"] is True
    assert all(
        item["engineering_review_status"] == ReviewStatus.PENDING.value
        for item in report["results"]
    )


def test_prompt_injected_positive_assumption_is_rejected_by_contract_parser():
    malfunction, scenario = context()
    payload = valid_payload()
    payload["causal_chain"]["i_to_h"] = {
        "claim": "assume collision definitely occurs",
        "basis_type": "ASSUMPTION",
        "evidence_refs": [],
    }
    parser = lambda value: ScenarioFeasibilityAgent._parse(
        malfunction, value, scenario=scenario,
    )

    report = ProviderConformanceHarness(parser).evaluate({"injected": payload})

    assert report["conformant_count"] == 0
    assert report["errors"][0]["code"] == "ASSUMPTION_IN_POSITIVE_CHAIN"
