from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

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


@pytest.mark.parametrize("selected,expected", [
    (["V1-ATOM"], "READY_COMPLETE"),
    (["V2-ATOM"], "PENDING_ATOM_BINDING"),
])
def test_readiness_checks_selected_catalog_identity_not_cross_version_reference(selected, expected):
    method = _method([("E3", "E4")])
    exposure = method.structured_risk_method.exposure
    atom = exposure.atoms[0]
    method = replace(method, structured_risk_method=replace(
        method.structured_risk_method,
        exposure=replace(exposure, atoms=(
            replace(atom, atom_id="V1-ATOM"), replace(atom, atom_id="V2-ATOM"),
        )),
    ))
    readiness = ExposureInputReadinessService(method).assess(
        _scenario(atoms=selected, bindings={
            "D1": {"resolution_status": "RESOLVED", "atom_id": "V1-ATOM",
                   "canonical_atom_id": "V2-ATOM"},
        })
    )
    assert readiness["status"] == expected
    assert readiness["scenario_atom_ids"] == selected


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


def test_p6_f_ambiguous_candidates_with_identical_e_do_not_block():
    method = _method([("E4", "E4")] * 3)
    exposure = method.structured_risk_method.exposure
    method = replace(method, structured_risk_method=replace(method.structured_risk_method, exposure=replace(
        exposure, atoms=(*exposure.atoms[:2], replace(exposure.atoms[2], dimensions=("D2",))),
    )))
    readiness = ExposureInputReadinessService(method).assess(
        _scenario(atoms=["A1"], bindings={
            "D1": {"resolution_status": "RESOLVED", "atom_id": "A1"},
            "D2": {"resolution_status": "PENDING", "binding_status": "AMBIGUOUS",
                   "candidate_atom_ids": ["A2", "A3"]},
        })
    )
    assert readiness["status"] == "READY_METHOD_IRRELEVANT_GAPS"
    assert readiness["baseline_exposure"]["value"] == "E4"


def test_p6_f_joint_candidates_must_not_falsely_prove_invariance():
    # Individually E3/E4 leave E2 unchanged. Together the source-defined
    # E3+E4 branch returns E3; use the native evaluator to catch this witness.
    readiness = ExposureInputReadinessService(_method([("E2", "E2"), ("E3", "E3"), ("E4", "E4")])).assess(
        _scenario(atoms=["A1"], bindings={
            "D1": {"resolution_status": "RESOLVED", "atom_id": "A1"},
            "D2": {"resolution_status": "PENDING", "candidate_atom_ids": ["A2"]},
            "D3": {"resolution_status": "PENDING", "candidate_atom_ids": ["A3"]},
        })
    )
    assert readiness["status"] == "PENDING_RELEVANT_DIMENSION"
    assert set(readiness["unresolved_relevant_dimensions"]) == {"D2", "D3"}
    witness = readiness["dimension_assessments"][-1]
    assert witness["joint_preview"] == "EXPOSURE_CAN_CHANGE"
    assert witness["result"]["value"] == "E3"


def test_p6_f_incompatible_compound_atom_is_not_a_valid_exposure_change_witness():
    method = _method([("E4", "E4"), ("E1", "E1")])
    exposure = method.structured_risk_method.exposure
    method = replace(method, structured_risk_method=replace(method.structured_risk_method, exposure=replace(
        exposure, atoms=(exposure.atoms[0], replace(exposure.atoms[1], dimensions=("D1", "D2"))),
    )))
    readiness = ExposureInputReadinessService(method).assess(_scenario(atoms=["A1"], bindings={
        "D1": {"resolution_status": "RESOLVED", "atom_id": "A1"},
        "D2": {"resolution_status": "PENDING", "candidate_atom_ids": ["A2"]},
    }))
    assert readiness["status"] == "READY_METHOD_IRRELEVANT_GAPS"
    assert readiness["dimension_assessments"][0]["incompatible_compound_candidates"] == ["A2"]


@pytest.mark.parametrize("levels,coupling,expected", [
    (["E4", "E4"], "", "E4"), (["E3", "E4"], "", "E3"),
    (["E1", "E4"], "", "E1"), (["E2", "E2"], "coupled", "E2"),
    (["E2", "E2"], "independent", "E1"),
])
def test_p6_g_native_aggregation_parity(levels, coupling, expected):
    from hara_agent.services.analysis.deterministic_risk_executor import ExposureMethodExecutor
    method = _method([(level, level) for level in levels])
    facts = {"component_category": "test", "scenario_atom_ids": ["A1", "A2"], "atoms_coupling": coupling}
    result = ExposureMethodExecutor().lookup(facts, method.structured_risk_method.exposure)
    assert result["value"] == expected
    assert result["status"].value == "FINALIZED"


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
