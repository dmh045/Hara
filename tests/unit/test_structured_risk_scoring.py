from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from hara_agent.contracts import (
    CalculationStatus, ExposureDimensionCoverage,
    ExposureDimensionCoverageDecision, ExposureDimensionCoverageStatus,
    ExposureDimensionRequirementStatus, SeveritySemanticResolution, SpeedSemantic,
    UnknownOverridePolicy,
)
from hara_agent.method_sources import YamlBaselineCompiler
from hara_agent.services.analysis import (
    ExposureDimensionCoverageService, StructuredRiskScoringService,
)
from hara_agent.template import TemplateRoleCompiler


ROOT = Path(__file__).resolve().parents[2]


def _service() -> StructuredRiskScoringService:
    report = TemplateRoleCompiler().compile_method(
        ROOT / "references/HARA_Template_AI_20260327.xlsx"
    ).report_contract
    method = YamlBaselineCompiler().compile(
        ROOT / "method_assets/fusa_baseline_v1/manifest.yaml",
        report_contract=report,
    )
    return StructuredRiskScoringService(method)


def _resolved_coverage(service: StructuredRiskScoringService) -> ExposureDimensionCoverageDecision:
    dimensions = []
    for item in service.method.scenario_model.dimensions:
        status = (
            ExposureDimensionRequirementStatus.REQUIRED
            if item.canonical_name in {"WHERE", "EGO_ACTION"}
            else ExposureDimensionRequirementStatus.NOT_APPLICABLE
        )
        dimensions.append(ExposureDimensionCoverage(
            dimension=item.canonical_name,
            status=status,
            rule_id="TEST-COVERAGE",
            source_ref="tests/test_structured_risk_scoring.py#TEST-COVERAGE",
        ))
    return ExposureDimensionCoverageDecision(
        assessment_key="MF-1::SC-1",
        method_contract_hash=str(service.method.metadata["template_hash"]),
        coverage_status=ExposureDimensionCoverageStatus.RESOLVED,
        dimensions=tuple(dimensions),
        coverage_rule_ids=("TEST-COVERAGE",),
        granularity="TEST",
    )


def _scenario(service: StructuredRiskScoringService, relative_speed: float) -> dict:
    return {
        "scenario_id": "SC-1", "malfunction_id": "MF-1",
        "relative_speed_kph": relative_speed, "collision_type": "FRONTAL",
        "road_user_type": "VEHICLE", "component_category": "sensor_camera",
        "_fact_provenance": {"relative_speed_kph": {
            "provenance": "PROJECT_INPUT",
            "source_refs": [{"location": "tests/test_structured_risk_scoring.py"}],
        }},
        "scenario_atom_ids": ["SO010", "PH005"],
        "method_scenario_dimensions": {
            "WHERE": {"resolution_status": "RESOLVED", "atom_id": "SO010"},
            "EGO_ACTION": {"resolution_status": "RESOLVED", "atom_id": "PH005"},
        },
        "_exposure_dimension_coverage_decision": _resolved_coverage(service),
        "ttc_s": 4.1, "driver_in_vehicle": True,
        "remote_intervention_available": False,
        "other_road_user_avoidance_possible": False,
    }


def test_structured_severity_boundaries_and_c_profile():
    service = _service()
    assert service.score(_scenario(service, 3.9), "hazard")["severity"]["severity_score"] == "S0"
    scored = service.score(_scenario(service, 4.0), "hazard")
    assert scored["severity"]["severity_score"] == "S1"
    assert scored["exposure"]["exposure_score"] in {"E1", "E2", "E3", "E4"}
    assert scored["controllability"]["controllability_score"] == "C1"


@pytest.mark.parametrize(("road_user", "speed", "expected"), [
    ("PEDESTRIAN", 7.0, "S2"),
    ("CYCLIST", 7.0, "S2"),
    ("MOTORCYCLIST", 7.0, "S2"),
])
def test_vru_any_bands_accept_canonical_frontal_collision(road_user, speed, expected):
    service = _service()
    scenario = _scenario(service, speed)
    scenario["road_user_type"] = road_user
    severity = service.score(scenario, "hazard")["severity"]
    assert severity["severity_score"] == expected
    assert severity["engineering_rule_id"].endswith("-any-2")


def test_vehicle_bands_still_require_configured_collision_type():
    service = _service()
    scenario = _scenario(service, 7.0)
    scenario["collision_type"] = "UNMAPPED"
    severity = service.score(scenario, "hazard")["severity"]
    assert severity["severity_score"] == ""
    assert severity["calculation_status"] == CalculationStatus.PENDING_INPUT.value


def test_structured_c_override_precedes_ttc():
    service = _service()
    scenario = _scenario(service, 20.0)
    scenario.update({
        "ttc_s": 10.0, "driver_in_vehicle": False,
        "remote_intervention_available": False,
        "other_road_user_avoidance_possible": False,
    })
    scored = service.score(scenario, "hazard")
    assert scored["controllability"]["controllability_score"] == "C3"
    assert scored["controllability"]["engineering_rule_id"] == "driver_outside_no_intervention"


def test_current_project_unknown_override_policy_routes_to_ttc_without_fabricating_facts():
    service = _service()
    scenario = _scenario(service, 20.0)
    scenario.pop("driver_in_vehicle")
    scored = service.score(scenario, "hazard")["controllability"]
    assert scored["calculation_status"] == CalculationStatus.FINALIZED.value
    assert scored["unknown_override_policy"] == UnknownOverridePolicy.SKIP_TO_TTC.value
    assert scored["unknown_policy_action"] == "SKIP_TO_TTC"
    assert scored["decision_status"] == "TTC_AFTER_UNKNOWN_OVERRIDE"
    assert "driver_in_vehicle" not in scenario


def test_skip_to_ttc_fixture_preserves_unknown_in_runtime_trace():
    service = _service()
    service.structured = replace(
        service.structured,
        controllability_branch_policy=replace(
            service.structured.controllability_branch_policy,
            unknown_override_policy=UnknownOverridePolicy.SKIP_TO_TTC,
        ),
    )
    scenario = _scenario(service, 20.0)
    scenario.pop("driver_in_vehicle")
    scored = service.score(scenario, "hazard")["controllability"]
    assert scored["controllability_score"] == "C1"
    assert scored["unknown_policy_action"] == "SKIP_TO_TTC"
    assert scored["decision_status"] == "TTC_AFTER_UNKNOWN_OVERRIDE"
    assert scored["rule_match_states"][0]["state"] == "UNKNOWN"


def test_structured_exposure_missing_atom_is_pending_input():
    service = _service()
    scenario = _scenario(service, 20.0)
    scenario["scenario_atom_ids"] = []
    scored = service.score(scenario, "hazard")
    assert scored["exposure"]["calculation_status"] == CalculationStatus.PENDING_INPUT.value
    assert scored["exposure"]["executor_invoked"] is False
    assert scored["exposure"]["pending_reason"] == "EXPOSURE_SCENARIO_ATOM_SET_EMPTY"


def test_pending_coverage_is_diagnostic_only_for_fusa_v1(monkeypatch):
    service = _service()
    scenario = _scenario(service, 20.0)
    scenario["scenario_atom_ids"] = ["FA001"]
    # Coverage stays diagnostic for fusa_v1.  This fixture deliberately has
    # no asserted Scenario-dimension binding; it is not a partial binding.
    scenario["method_scenario_dimensions"] = {}
    scenario["_exposure_dimension_coverage_decision"] = (
        ExposureDimensionCoverageService(service.method).decide(
            assessment_key="MF-1::SC-1", function=None, operating_mode="Active",
        )
    )
    calls = 0
    original = service.exposure.lookup

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        return original(*args, **kwargs)

    monkeypatch.setattr(service.exposure, "lookup", counted)
    scored = service.score(scenario, "hazard")

    assert calls == 1
    assert scored["exposure"]["calculation_status"] == CalculationStatus.FINALIZED.value
    assert scored["exposure"]["exposure_score"] == "E4"
    assert scored["exposure"]["exposure_result"] == "E4"
    assert scored["exposure"]["executor_invoked"] is True
    assert scored["exposure"]["pending_reason"] == ""


def test_pending_coverage_still_blocks_a_future_method_that_requires_it(monkeypatch):
    service = _service()
    service.structured = replace(
        service.structured,
        exposure=replace(
            service.structured.exposure,
            aggregation_policy=replace(
                service.structured.exposure.aggregation_policy,
                policy_id="future_coverage_required",
            ),
        ),
    )
    scenario = _scenario(service, 20.0)
    scenario["_exposure_dimension_coverage_decision"] = (
        ExposureDimensionCoverageService(service.method).decide(
            assessment_key="MF-1::SC-1", function=None, operating_mode="Active",
        )
    )
    calls = 0

    def counted(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("coverage-required method must not invoke Exposure")

    monkeypatch.setattr(service.exposure, "lookup", counted)
    scored = service.score(scenario, "hazard")

    assert calls == 0
    assert scored["exposure"]["calculation_status"] == CalculationStatus.PENDING_METHOD_SEMANTICS.value
    assert scored["exposure"]["missing_method_semantics"] == "EXPOSURE_DIMENSION_COVERAGE"


def test_s0_short_circuits_before_the_diagnostic_coverage_decision():
    service = _service()
    scenario = _scenario(service, 3.9)
    scenario["_exposure_dimension_coverage_decision"] = (
        ExposureDimensionCoverageService(service.method).decide(
            assessment_key="MF-1::SC-1", function=None, operating_mode="Active",
        )
    )

    scored = service.score(scenario, "hazard")

    assert scored["severity"]["severity_score"] == "S0"
    assert scored["exposure"]["exposure_score"] == "E0"
    assert scored["exposure"]["engineering_rule_id"] == "ASIL-ZERO-SHORT-CIRCUIT"


def test_structured_scoring_requires_typed_coverage_decision():
    service = _service()
    scenario = _scenario(service, 20.0)
    scenario.pop("_exposure_dimension_coverage_decision")

    with pytest.raises(ValueError, match="ExposureDimensionCoverageDecision"):
        service.score(scenario, "hazard")


def test_approved_source_conflict_prevents_severity_executor_invocation(monkeypatch):
    original = _service()
    severity = original.method.structured_risk_method.severity
    semantic = replace(
        severity.semantic,
        compiled_semantic=SpeedSemantic.UNRESOLVED,
        semantic_resolution=SeveritySemanticResolution.APPROVED_SOURCE_INTERNAL_CONFLICT,
    )
    method = replace(
        original.method,
        structured_risk_method=replace(
            original.method.structured_risk_method,
            severity=replace(severity, speed_semantic=SpeedSemantic.UNRESOLVED, semantic=semantic),
        ),
    )
    service = StructuredRiskScoringService(method)
    calls = 0

    def forbidden(*args, **kwargs):
        nonlocal calls
        calls += 1
        raise AssertionError("Severity executor must not run for source conflict")

    monkeypatch.setattr(service.severity, "lookup", forbidden)
    scenario = _scenario(original, 8.0)
    scenario["delta_v_kph"] = 8.0
    scored = service.score(scenario, "hazard")

    assert calls == 0
    assert scored["severity"]["calculation_status"] == CalculationStatus.PENDING_METHOD_SEMANTICS.value
    assert scored["severity"]["executor_invoked"] is False
    assert scored["severity"]["pending_reason"] == "APPROVED_SOURCE_SEMANTIC_CONFLICT"
