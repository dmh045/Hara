from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from hara_agent.contracts import ExposureAtom, ExposureDomainRule, ExposureMethodDomain
from hara_agent.method_sources import YamlBaselineCompiler
from hara_agent.services.analysis import ExposureInputReadinessService
from hara_agent.template import TemplateRoleCompiler


ROOT = Path(__file__).resolve().parents[2]


def _method(levels: list[tuple[str, str]]):
    report = TemplateRoleCompiler().compile_method(
        ROOT / "references/HARA_Template_AI_20260327.xlsx"
    ).report_contract
    method = YamlBaselineCompiler().compile(
        ROOT / "method_assets/fusa_baseline_v1/manifest.yaml", report_contract=report,
    )
    source = method.structured_risk_method.exposure.source_refs[0]
    atoms = tuple(
        ExposureAtom(
            atom_id=f"A{index}", dimensions=(f"D{index}",), label=f"atom {index}",
            duration_level=z, frequency_level=f, source_ref=source,
        )
        for index, (z, f) in enumerate(levels, start=1)
    )
    exposure = replace(
        method.structured_risk_method.exposure,
        atoms=atoms,
        domain_rules=(ExposureDomainRule("TEST-Z", ("test",), ExposureMethodDomain.TIME, source),),
        strong_couplings=(),
    )
    return replace(
        method,
        structured_risk_method=replace(method.structured_risk_method, exposure=exposure),
    )


def _scenario(*, atoms: list[str], bindings: dict[str, dict[str, object]]):
    return {
        "component_category": "test",
        "scenario_atom_ids": atoms,
        "method_scenario_dimensions": bindings,
    }


def test_complete_relevant_atom_set_is_ready_complete():
    readiness = ExposureInputReadinessService(_method([("E4", "E4")])).assess(
        _scenario(atoms=["A1"], bindings={
            "D1": {"resolution_status": "RESOLVED", "atom_id": "A1"},
        })
    )
    assert readiness["status"] == "READY_COMPLETE"


def test_unresolved_irrelevant_dimension_is_ready():
    readiness = ExposureInputReadinessService(_method([("E4", "E4")])).assess(
        _scenario(atoms=["A1"], bindings={
            "D1": {"resolution_status": "RESOLVED", "atom_id": "A1"},
            "NOT_AN_EXPOSURE_DIMENSION": {"resolution_status": "PENDING"},
        })
    )
    assert readiness["status"] == "READY_METHOD_IRRELEVANT_GAPS"


def test_unresolved_dimension_that_can_lower_e4_is_pending():
    readiness = ExposureInputReadinessService(_method([("E4", "E4"), ("E2", "E2")])).assess(
        _scenario(atoms=["A1"], bindings={
            "D1": {"resolution_status": "RESOLVED", "atom_id": "A1"},
            "D2": {"resolution_status": "PENDING", "candidate_atom_ids": ["A2"]},
        })
    )
    assert readiness["status"] == "PENDING_RELEVANT_DIMENSION"
    assert readiness["unresolved_relevant_dimensions"] == ["D2"]
    assert readiness["dimension_assessments"][0]["change_witnesses"][0]["result"]["value"] == "E2"


def test_resolved_atom_absent_from_atom_set_is_pending_binding():
    readiness = ExposureInputReadinessService(_method([("E4", "E4")])).assess(
        _scenario(atoms=[], bindings={
            "D1": {"resolution_status": "RESOLVED", "atom_id": "A1"},
        })
    )
    assert readiness["status"] == "PENDING_ATOM_BINDING"
    assert readiness["reason_code"] == "EXPOSURE_SCENARIO_ATOM_SET_EMPTY"


def test_ambiguous_atom_binding_is_pending_ambiguous():
    readiness = ExposureInputReadinessService(_method([("E4", "E4"), ("E2", "E2")])).assess(
        _scenario(atoms=["A1"], bindings={
            "D1": {"resolution_status": "RESOLVED", "atom_id": "A1"},
            "D2": {
                "resolution_status": "PENDING", "binding_status": "AMBIGUOUS",
                "candidate_atom_ids": ["A2", "A3"],
            },
        })
    )
    assert readiness["status"] == "PENDING_AMBIGUOUS_ATOM_SET"


def test_not_applicable_dimension_with_lower_atom_does_not_block():
    readiness = ExposureInputReadinessService(_method([("E4", "E4"), ("E2", "E2")])).assess(
        _scenario(atoms=["A1"], bindings={
            "D1": {"resolution_status": "RESOLVED", "atom_id": "A1"},
            "D2": {"resolution_status": "NOT_APPLICABLE",
                   "applicability_status": "NOT_APPLICABLE",
                   "candidate_atom_ids": ["A2"]},
        })
    )
    assert readiness["status"] == "READY_METHOD_IRRELEVANT_GAPS"
    assert readiness["unresolved_relevant_dimensions"] == []


def test_native_whole_scenario_domain_fallback_is_preserved():
    readiness = ExposureInputReadinessService(_method([("dash", "E3")])).assess(
        _scenario(atoms=["A1"], bindings={
            "D1": {"resolution_status": "RESOLVED", "atom_id": "A1"},
        })
    )
    assert readiness["status"] == "READY_COMPLETE"
    assert readiness["baseline_exposure"]["requested_domain"] == "Z"
    assert readiness["baseline_exposure"]["actual_domain"] == "F"


def test_source_defined_compound_atom_is_supplied_once():
    method = _method([("E3", "E3")])
    exposure = method.structured_risk_method.exposure
    atom = replace(exposure.atoms[0], dimensions=("D1", "D2"))
    method = replace(method, structured_risk_method=replace(
        method.structured_risk_method,
        exposure=replace(exposure, atoms=(atom,)),
    ))
    readiness = ExposureInputReadinessService(method).assess(
        _scenario(atoms=["A1"], bindings={
            "D1": {"resolution_status": "RESOLVED", "atom_id": "A1",
                   "filled_dimensions": ["D1", "D2"]},
            "D2": {"resolution_status": "RESOLVED", "atom_id": "A1",
                   "filled_dimensions": ["D1", "D2"]},
        })
    )
    assert readiness["status"] == "READY_COMPLETE"
    assert readiness["scenario_atom_ids"] == ["A1"]
