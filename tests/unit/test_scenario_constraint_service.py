from hara_agent.contracts import (
    NormativeStrength, ScenarioConstraintDisposition,
    ScenarioConstraintPredicate, ScenarioConstraintRule, SourceRef,
)
from hara_agent.services.analysis import (
    ScenarioConstraintExecutor, ScenarioConstraintStatus,
)
import pytest


@pytest.mark.parametrize("left,right,conflict", [
    ({"range_pct": [2, 8]}, {"pct_min": 8, "min_inclusive": False}, True),
    ({"range_pct": [2, 8]}, {"pct_min": 8, "min_inclusive": True}, False),
    ({"pct_min": 8, "min_inclusive": False}, {"range_pct": [2, 8]}, True),
    ({"range_pct": [0, 2]}, {"range_pct": [5, 8]}, True),
    ({"range_pct": [8, 15], "min_inclusive": False}, {"pct_min": 8, "min_inclusive": False}, False),
])
def test_same_road_slope_ranges_respect_open_endpoints(left, right, conflict):
    atoms = [{"atom_id": key, "physical_semantics": {"slope": value}}
             for key, value in (("road", left), ("ego-on-road", right))]
    assert bool(ScenarioConstraintExecutor.physical_semantic_conflicts(atoms)) is conflict


def _source(name: str) -> SourceRef:
    return SourceRef.create(
        workbook="method.xlsx", template_hash="hash", sheet="Scenario Rules",
        range=name, raw_text=name,
    )


def _rule(rule_id: str, disposition: ScenarioConstraintDisposition):
    return ScenarioConstraintRule(
        rule_id=rule_id,
        predicates=(
            ScenarioConstraintPredicate("ROAD", ("HIGHWAY",)),
            ScenarioConstraintPredicate("OBJECT", ("PEDESTRIAN",)),
        ),
        disposition=disposition,
        reason=rule_id,
        normative_strength=NormativeStrength.NORMATIVE,
        source_ref=_source(rule_id),
    )


def test_rare_but_feasible_combination_is_not_deleted():
    result = ScenarioConstraintExecutor().evaluate(
        {"ROAD": "HIGHWAY", "OBJECT": "PEDESTRIAN"},
        (_rule("RARE-1", ScenarioConstraintDisposition.RARE_BUT_FEASIBLE),),
    )

    assert result.status is ScenarioConstraintStatus.KEEP_RARE
    assert result.matched_rule_ids == ("RARE-1",)


def test_only_explicit_normative_impossibility_is_deleted():
    result = ScenarioConstraintExecutor().evaluate(
        {"ROAD": "HIGHWAY", "OBJECT": "PEDESTRIAN"},
        (_rule("IMP-1", ScenarioConstraintDisposition.PHYSICALLY_IMPOSSIBLE),),
    )

    assert result.status is ScenarioConstraintStatus.DROP


def test_conflicting_normative_constraints_fail_closed():
    result = ScenarioConstraintExecutor().evaluate(
        {"ROAD": "HIGHWAY", "OBJECT": "PEDESTRIAN"},
        (
            _rule("IMP-1", ScenarioConstraintDisposition.PHYSICALLY_IMPOSSIBLE),
            _rule("RARE-1", ScenarioConstraintDisposition.RARE_BUT_FEASIBLE),
        ),
    )

    assert result.status is ScenarioConstraintStatus.CONFLICT
