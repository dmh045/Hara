from copy import deepcopy
from dataclasses import fields
import json
from pathlib import Path

import pytest

from hara_agent.infrastructure.llm import LLMResponse
from hara_agent.models import MalfunctionCandidate, ReviewStatus, ScenarioCandidate, SourceRef
from hara_agent.services.semantic import ScenarioFeasibilityAgent
from hara_agent.services.semantic.scenario_evidence import ScenarioEvidenceContractError


def _sample():
    data = json.loads((Path(__file__).parents[1] / "fixtures" /
                       "scenario_empty_later_hop_20261019.json").read_text())
    def typed(cls, raw):
        payload = {f.name: raw[f.name] for f in fields(cls) if f.name in raw}
        payload["status"] = ReviewStatus(raw["status"])
        payload["sources"] = [SourceRef(**{f.name: s[f.name] for f in fields(SourceRef) if f.name in s})
                              for s in raw.get("sources", raw.get("source_refs", []))]
        return cls(**payload)
    return data, typed(MalfunctionCandidate, data["malfunction"]), typed(ScenarioCandidate, data["scenario"])


def test_real_first_response_avoids_repair_request_without_mutating_raw_output(monkeypatch):
    data, malfunction, scenario = _sample()
    original = deepcopy(data["assessment"])

    class Client:
        def __init__(self):
            self.requests = []

        def complete_json(self, request):
            self.requests.append(request)
            assessment = original if len(self.requests) == 1 else data["repair_assessment"]
            return LLMResponse(data={"assessments": [assessment]}, model="frozen-real-response")

    before = Client()
    with monkeypatch.context() as m:
        m.setattr(ScenarioFeasibilityAgent, "_omit_empty_later_hops", staticmethod(lambda x: x))
        _, audit_before = ScenarioFeasibilityAgent(before).assess(malfunction, [scenario])
    after = Client()
    assessments, audit_after = ScenarioFeasibilityAgent(after).assess(malfunction, [scenario])
    assert len(before.requests) == 2 and audit_before["item_repair_calls"] == 1
    assert len(after.requests) == 1 and audit_after["item_repair_calls"] == 0
    assert original == data["assessment"]
    canonical = deepcopy(original)
    del canonical["causal_chain"]["i_to_h"]
    expected = ScenarioFeasibilityAgent._parse(malfunction, canonical, scenario=scenario)
    assert assessments[0].to_dict() == expected.to_dict()
    assert assessments[0].physical_interaction == original["physical_interaction"]
    assert assessments[0].breakpoint == "B_TO_I"
    assert not assessments[0].retain


@pytest.mark.parametrize("change", [
    {"claim": "a meaningful later claim"},
    {"claim": None},
    {"basis_type": "DIRECT_FACT"},
    {"evidence_refs": ["MF.description"]},
    {"evidence_refs": None},
    {"extra": "unsupported metadata"},
])
def test_nonempty_or_noncanonical_later_hop_is_not_discarded(change):
    data, _, _ = _sample()
    item = data["assessment"]
    item["causal_chain"]["i_to_h"].update(change)
    assert ScenarioFeasibilityAgent._omit_empty_later_hops(item) is item


@pytest.mark.parametrize("breakpoint,causal", [("I_TO_H", False), ("NONE", True), ("INVALID", False)])
def test_required_or_invalid_chain_still_rejected(breakpoint, causal):
    data, malfunction, scenario = _sample()
    item = data["assessment"]
    item.update(breakpoint=breakpoint, causally_relevant=causal)
    with pytest.raises(ScenarioEvidenceContractError):
        ScenarioFeasibilityAgent._parse(malfunction, item, scenario=scenario)


def test_first_breakpoint_omits_both_empty_later_hops_but_keeps_its_reason():
    data, malfunction, scenario = _sample()
    item = data["assessment"]
    item["breakpoint"] = "M_TO_B"
    item["causal_chain"]["m_to_b"] = deepcopy(item["causal_chain"]["b_to_i"])
    item["causal_chain"]["b_to_i"] = deepcopy(item["causal_chain"]["i_to_h"])
    item["causal_chain"]["i_to_h"]["claim"] = " \n "
    result = ScenarioFeasibilityAgent._parse(malfunction, item, scenario=scenario)
    assert list(result.causal_chain) == ["m_to_b"]
    assert result.causal_chain["m_to_b"] == item["causal_chain"]["m_to_b"]
    assert len(item["causal_chain"]) == 3
