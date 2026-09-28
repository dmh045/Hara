from __future__ import annotations

from pathlib import Path

import pytest

from hara_agent.method_sources import YamlBaselineCompiler
from hara_agent.models import ScenarioCandidate
from hara_agent.services.analysis.analytical_physics_instantiation_service import (
    AnalyticalPhysicsInstantiationService,
)
from hara_agent.services.analysis.scenario_physics import (
    derive_scenario_physics, longitudinal_closing_speed_kph,
    select_ego_speed_from_policy,
)
from hara_agent.template import TemplateRoleCompiler


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def method():
    report = TemplateRoleCompiler().compile_method(
        ROOT / "references/HARA_Template_AI_20260327.xlsx"
    ).report_contract
    return YamlBaselineCompiler().compile(
        ROOT / "method_assets/fusa_baseline_v1/manifest.yaml",
        report_contract=report,
    )


def _scenario(lower=0.0, upper=7.0, *, inclusive=True, approval="FINALIZED"):
    return ScenarioCandidate(
        scenario_id="SC-POLICY", operating_scenario="parking",
        situational_description="parking", situational_detailing="parking",
        facts={"ego_speed_constraint": {
            "min_kph": lower, "max_kph": upper,
            "upper_inclusive": inclusive,
        }},
        fact_provenance={"ego_speed_constraint": {
            "provenance": "PROJECT_INPUT", "approval": approval,
            "source_refs": [{
                "source_type": "item_document", "source_id": "ItemDef.docx",
                "location": "环境条件", "excerpt": "operating range",
            }],
        }},
    )


@pytest.mark.parametrize("upper", [7.0, 30.0])
def test_project_upper_bound_is_scoped_and_source_linked(method, upper):
    scenario = _scenario(upper=upper)
    selected = select_ego_speed_from_policy(
        scenario, policy=method.metadata["project_analysis_policy"],
        malfunction_id="MF-1",
    )
    assert selected is not None
    value, metadata = selected
    assert value == upper
    assert metadata["effective_range_kph"] == [0.0, upper]
    assert metadata["analysis_assumption_scope"]["scenario_id"] == "SC-POLICY"
    assert metadata["analysis_assumption_scope"]["malfunction_id"] == "MF-1"
    assert len(metadata["source_refs"]) == 2
    audit = AnalyticalPhysicsInstantiationService(method).instantiate(
        scenario=scenario, malfunction={"malfunction_id": "MF-1"},
        causal_status="CAUSAL_REVALIDATED",
    )
    speed = next(item for item in audit["inputs"] if item["field"] == "ego_speed_kph")
    assert speed["value"] == upper
    assert speed["selection_basis"] == "GOVERNED_CLOSED_UPPER_BOUND"


@pytest.mark.parametrize("scenario", [
    _scenario(upper=None),
    _scenario(upper=float("inf")),
    _scenario(upper=7.0, inclusive=False),
    _scenario(upper=7.0, approval="PENDING"),
    _scenario(lower=8.0, upper=7.0),
])
def test_upper_bound_fails_closed_when_range_is_not_approved_finite_closed(method, scenario):
    assert select_ego_speed_from_policy(
        scenario, policy=method.metadata["project_analysis_policy"],
        malfunction_id="MF-1",
    ) is None


@pytest.mark.parametrize("ego,obj,ego_dir,obj_dir,position,expected", [
    (7, 0, "FORWARD", "STATIONARY", "front", 7),
    (7, 3, "FORWARD", "REVERSE", "front", 10),
    (7, 3, "FORWARD", "FORWARD", "front", 4),
    (7, 10, "FORWARD", "FORWARD", "front", 0),
    (7, 10, "FORWARD", "FORWARD", "rear", 3),
    (7, 0, "FORWARD", "STATIONARY", "left", None),
])
def test_ttc_uses_approach_not_relative_speed_magnitude(
    ego, obj, ego_dir, obj_dir, position, expected,
):
    closing = longitudinal_closing_speed_kph(
        ego, obj, ego_direction=ego_dir, object_direction=obj_dir,
        object_position=position, collision_type="FRONTAL",
    )
    assert closing == expected
    facts = {
        "ego_speed_kph": ego, "object_speed_kph": obj,
        "ego_longitudinal_direction": ego_dir,
        "object_longitudinal_direction": obj_dir,
        "object_position": position, "collision_type": "FRONTAL",
        "relative_distance_m": 10.0,
    }
    scenario = ScenarioCandidate(
        "SC-PHYSICS", "parking", "collision approach", "",
        facts=facts,
        fact_provenance={
            key: {"provenance": "PROJECT_INPUT", "approval": "FINALIZED",
                  "source_refs": [{"source_type": "test", "source_id": "physics",
                                   "location": key}]}
            for key in facts
        },
    )
    derived = {item.evidence_ref: item.value for item in derive_scenario_physics(scenario)}
    assert ("DERIVED.ttc_s" in derived) is (expected is not None and expected > 0)
    if expected == 0:
        assert derived["DERIVED.relative_speed_kph"] == abs(ego - obj)
        assert derived["DERIVED.closing_speed_kph"] == 0


def test_explicit_relative_speed_does_not_force_lateral_ttc():
    scenario = ScenarioCandidate(
        "SC-LATERAL", "parking", "side interaction", "",
        facts={"relative_speed_kph": 8.0, "relative_distance_m": 5.0,
               "collision_type": "SIDE"},
    )
    assert "DERIVED.ttc_s" not in {
        item.evidence_ref for item in derive_scenario_physics(scenario)
    }
