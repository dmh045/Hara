"""Parity fixtures transcribed from the original FUSA ``compute_exposure``.

The source of truth inspected for these cases is
``Desktop/fusa_agent/core/hara/evaluators/exposure_evaluator.py``.  The
runtime only invokes V13's compiled ExposureMethodExecutor; this test keeps a
small, reviewable table of the original executable outcomes rather than
embedding a second evaluator.
"""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from hara_agent.contracts import (
    ExposureAtom, ExposureDimensionCoverage, ExposureDimensionCoverageDecision,
    ExposureDimensionCoverageStatus, ExposureDimensionRequirementStatus,
    ExposureDomainRule, ExposureMethodDomain,
)
from hara_agent.method_sources import YamlBaselineCompiler
from hara_agent.services.analysis import ExposureMethodExecutor, StructuredRiskScoringService
from hara_agent.template import TemplateRoleCompiler


ROOT = Path(__file__).resolve().parents[2]


def _baseline():
    report = TemplateRoleCompiler().compile_method(
        ROOT / "references/HARA_Template_AI_20260327.xlsx"
    ).report_contract
    return YamlBaselineCompiler().compile(
        ROOT / "method_assets/fusa_baseline_v1/manifest.yaml", report_contract=report,
    )


def _exposure(levels: list[tuple[str, str]]):
    baseline = _baseline()
    exposure = baseline.structured_risk_method.exposure
    source = exposure.source_refs[0]
    atoms = tuple(
        ExposureAtom(
            atom_id=f"A{index}", dimensions=(f"D{index}",), label=f"atom {index}",
            duration_level=z, frequency_level=f, source_ref=source,
        )
        for index, (z, f) in enumerate(levels, start=1)
    )
    return replace(
        exposure,
        atoms=atoms,
        domain_rules=(ExposureDomainRule("TEST-Z", ("test",), ExposureMethodDomain.TIME, source),),
        strong_couplings=(),
    )


@pytest.mark.parametrize(
    ("levels", "coupling", "legacy"),
    [
        ([("E4", "E4"), ("E4", "E4")], None,
         {"value": "E4", "actual_domain": "Z", "dimension_fallback": False,
          "aggregation_rule": "all_e4", "coupling_consumed": False}),
        ([("E3", "E3"), ("E4", "E4")], None,
         {"value": "E3", "actual_domain": "Z", "dimension_fallback": False,
          "aggregation_rule": "e3_e4_mix", "coupling_consumed": False}),
        ([("E2", "E2"), ("E4", "E4")], None,
         {"value": "E2", "actual_domain": "Z", "dimension_fallback": False,
          "aggregation_rule": "min_when_unequal", "coupling_consumed": False}),
        ([("E3", "E3"), ("E3", "E3")], "independent",
         {"value": "E2", "actual_domain": "Z", "dimension_fallback": False,
          "aggregation_rule": "same_independent_minus_one", "coupling_consumed": True}),
        ([("E3", "E3"), ("E3", "E3")], "coupled",
         {"value": "E3", "actual_domain": "Z", "dimension_fallback": False,
          "aggregation_rule": "same_coupled_no_change", "coupling_consumed": True}),
        ([("", "E2"), ("", "E3")], None,
         {"value": "E2", "actual_domain": "F", "dimension_fallback": True,
          "aggregation_rule": "min_when_unequal", "coupling_consumed": False}),
    ],
)
def test_v13_matches_legacy_fusa_finalized_cases(levels, coupling, legacy):
    scenario = {
        "component_category": "test",
        "scenario_atom_ids": [f"A{index}" for index in range(1, len(levels) + 1)],
    }
    if coupling is not None:
        scenario["atoms_coupling"] = coupling

    result = ExposureMethodExecutor().lookup(scenario, _exposure(levels))

    assert result["status"].value == "FINALIZED"
    assert {key: result[key] for key in legacy} == legacy
    assert all(atom["used"] for atom in result["atom_details"])


def test_v13_matches_legacy_both_domains_missing_case():
    result = ExposureMethodExecutor().lookup(
        {"component_category": "test", "scenario_atom_ids": ["A1", "A2"]},
        _exposure([("", ""), ("", "")]),
    )

    assert result["status"].value == "PENDING_INPUT"
    assert result["pending_reason"] == "MISSING_EXPOSURE_VALUE_BOTH_DOMAINS"
    assert result["requested_domain"] == result["actual_domain"] == "Z"
    assert result["dimension_fallback"] is False


def test_v13_matches_legacy_unequal_ranks_without_coupling():
    result = ExposureMethodExecutor().lookup(
        {"component_category": "test", "scenario_atom_ids": ["A1", "A2"]},
        _exposure([("E2", "E2"), ("E4", "E4")]),
    )

    assert result["status"].value == "FINALIZED"
    assert result["value"] == "E2"
    assert result["aggregation_rule"] == "min_when_unequal"
    assert result["coupling_consumed"] is False


def test_v13_matches_legacy_same_rank_missing_coupling_case():
    result = ExposureMethodExecutor().lookup(
        {"component_category": "test", "scenario_atom_ids": ["A1", "A2"]},
        _exposure([("E3", "E3"), ("E3", "E3")]),
    )

    assert result["status"].value == "PENDING_INPUT"
    assert result["pending_reason"] == "MISSING_ATOMS_COUPLING"
    assert result["aggregation_rule"] == ""
    assert result["coupling_consumed"] is False


def test_s0_short_circuit_matches_legacy_before_coverage_or_atoms():
    method = _baseline()
    service = StructuredRiskScoringService(method)
    dimensions = tuple(ExposureDimensionCoverage(
        dimension=item.canonical_name,
        status=ExposureDimensionRequirementStatus.PENDING_METHOD_SEMANTICS,
    ) for item in method.scenario_model.dimensions)
    decision = ExposureDimensionCoverageDecision(
        assessment_key="MF-1::SC-1", method_contract_hash=str(method.metadata["template_hash"]),
        coverage_status=ExposureDimensionCoverageStatus.PENDING_METHOD_SEMANTICS,
        dimensions=dimensions,
    )
    result = service.score({
        "scenario_id": "SC-1", "malfunction_id": "MF-1", "relative_speed_kph": 3.9,
        "road_user_type": "VEHICLE", "collision_type": "FRONTAL",
        "component_category": "sensor_camera", "scenario_atom_ids": [],
        "_fact_provenance": {"relative_speed_kph": {
            "provenance": "PROJECT_INPUT", "approval": "FINALIZED",
            "source_refs": [{"location": "test"}],
        }},
        "_exposure_dimension_coverage_decision": decision,
    }, "hazard")

    assert result["severity"]["severity_score"] == "S0"
    assert result["exposure"]["exposure_score"] == ""
    assert result["controllability"]["controllability_score"] == ""
