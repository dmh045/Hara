from __future__ import annotations

from pathlib import Path

import pytest

from hara_agent.contracts import CausalAssessmentStatus
from hara_agent.method_sources import YamlBaselineCompiler
from hara_agent.models import (
    EvidenceKind, EvidenceRecord, FactProvenance, ItemDefinitionFacts,
    MalfunctionCandidate, ReviewStatus, ScenarioCandidate, SourceRef,
)
from hara_agent.services.analysis import (
    ExposureDimensionCoverageService, MethodRuleScoringService,
    PotentialHarmResolver,
)
from hara_agent.services.semantic import (
    CausalEvidenceSelector, ScenarioFeasibilityAgent,
    build_project_evidence_registry,
)
from hara_agent.services.semantic.scenario_batching import build_scenario_user_prompt
from hara_agent.services.semantic.scenario_evidence import (
    FactRegistry, ScenarioEvidenceContractError, build_fact_registry,
)
from hara_agent.template import TemplateRoleCompiler


ROOT = Path(__file__).resolve().parents[2]


def _source() -> SourceRef:
    return SourceRef("item_definition", "ItemDef.docx", "p1", "explicit engineering fact")


def _malfunction() -> MalfunctionCandidate:
    source = _source()
    return MalfunctionCandidate(
        "MF-1", "FUN-1", "loss", "braking command lost", "no deceleration",
        "vehicle continues toward the explicit object", ["failure", "behavior"],
        sources=[source], status=ReviewStatus.FINALIZED,
    )


def _scenario() -> ScenarioCandidate:
    source = _source()
    provenance = {
        key: {
            "provenance": "PROJECT_INPUT", "approval": "FINALIZED",
            "source_refs": [source],
        }
        for key in ("relative_distance", "relative_speed_kph", "closing_speed_kph",
                    "collision_type", "road_user_type")
    }
    return ScenarioCandidate(
        "SCN-1", "parking", "object ahead", "closing object",
        {
            "relative_distance": "10 m", "relative_speed_kph": 18.0,
            "closing_speed_kph": 18.0,
            "collision_type": "FRONTAL", "road_user_type": "VEHICLE",
        }, fact_provenance=provenance, status=ReviewStatus.FINALIZED,
        sources=[source], semantic_fingerprint="fp-1",
    )


def _positive_payload() -> dict:
    return {
        "scenario_id": "SCN-1", "physically_feasible": True,
        "functionally_relevant": True, "causally_relevant": True,
        "breakpoint": "NONE",
        "causal_chain": {
            "m_to_b": {"claim": "failure removes braking command", "basis_type": "DIRECT_FACT", "evidence_refs": ["MF.functional_effect"]},
            "b_to_i": {"claim": "vehicle continues while the object is ahead", "basis_type": "DIRECT_FACT", "evidence_refs": ["SCN.relative_distance"]},
            "i_to_h": {"claim": "closing motion creates the hazardous vehicle state", "basis_type": "DERIVED_PHYSICS", "evidence_refs": ["DERIVED.ttc_s"]},
        },
        "risk_dimension_changes": [], "hazardous_event": "vehicle continues toward object",
        "rationale": "M to B to I to H is source linked", "confidence": 0.8,
    }


def test_pre_causal_registry_contains_bounded_physics_and_no_guess_for_missing_inputs():
    registry = build_fact_registry(_malfunction(), _scenario())
    assert registry.resolve_record("DERIVED.ttc_s").value == 2.0
    assert registry.resolve_record("DERIVED.relative_distance_m").value == 10.0
    assert registry.resolve_record("DERIVED.relative_speed_kph").value == 18.0

    incomplete = ScenarioCandidate("SCN-2", "parking", "object", "detail", {"relative_distance": "10 m"})
    incomplete_registry = build_fact_registry(_malfunction(), incomplete)
    assert incomplete_registry.resolve_record("DERIVED.ttc_s") is None
    assert incomplete_registry.resolve_record("DERIVED.relative_speed_kph") is None


def test_pair_scoped_project_risk_facts_cannot_cross_scenario_or_malfunction():
    project = FactRegistry()
    for evidence_ref, malfunction_id, scenario_id in (
        ("PROJECT.risk.collision_exact", "MF-1", "SCN-1"),
        ("PROJECT.risk.collision_other_scenario", "MF-1", "SCN-OTHER"),
        ("PROJECT.risk.collision_other_malfunction", "MF-OTHER", "SCN-1"),
    ):
        project.register(EvidenceRecord(
            evidence_ref, "FRONTAL", EvidenceKind.DIRECT_FACT,
            FactProvenance.PROJECT_INPUT, ReviewStatus.FINALIZED, (_source(),),
            {"parameter": "COLLISION_TYPE", "context": {
                "malfunction_id": malfunction_id, "scenario_id": scenario_id,
            }},
        ))

    registry = build_fact_registry(_malfunction(), _scenario(), project)

    assert registry.resolve("PROJECT.risk.collision_exact") is not None
    assert registry.resolve("PROJECT.risk.collision_other_scenario") is None
    assert registry.resolve("PROJECT.risk.collision_other_malfunction") is None
    selection = CausalEvidenceSelector(max_evidence=20).select(registry)
    assert "PROJECT.risk.collision_exact" in selection.selected_context_evidence_refs
    assert all("other_" not in ref for ref in selection.selected_context_evidence_refs)


def test_v4_scenario_causal_contract_stops_at_hazardous_event():
    result = ScenarioFeasibilityAgent._parse(_malfunction(), _positive_payload(), scenario=_scenario())
    assert result.causal_assessment is not None
    assert result.causal_assessment.status is CausalAssessmentStatus.VALIDATED
    assert result.potential_harm == ""
    assert "h_to_harm" not in result.causal_assessment.to_dict()["causal_graph"]


def test_vehicle_level_hazard_alone_still_cannot_prove_i_to_h():
    payload = _positive_payload()
    payload["causal_chain"]["i_to_h"]["evidence_refs"] = ["MF.vehicle_level_hazard"]
    payload["causal_chain"]["i_to_h"]["basis_type"] = "DIRECT_FACT"
    with pytest.raises(ScenarioEvidenceContractError) as caught:
        ScenarioFeasibilityAgent._parse(_malfunction(), payload, scenario=_scenario())
    assert caught.value.code.value == "SELF_REFERENTIAL_CAUSAL_EVIDENCE"
    assert caught.value.hop == "i_to_h"


def test_method_evidence_provider_exposes_yaml_and_template_rules():
    report = TemplateRoleCompiler().compile_method(
        ROOT / "references/HARA_Template_AI_20260327.xlsx"
    ).report_contract
    yaml_method = YamlBaselineCompiler().compile(
        ROOT / "method_assets/fusa_baseline_v1/manifest.yaml",
        report_contract=report,
    )
    template_method = TemplateRoleCompiler().compile_method(
        ROOT / "references/HARA_Template_AI_20260327.xlsx"
    )
    facts = ItemDefinitionFacts(system_description="system", item_boundary="vehicle", sources=[_source()])
    yaml_registry = build_project_evidence_registry(facts, yaml_method)
    template_registry = build_project_evidence_registry(facts, template_method)
    assert any(item.evidence_ref.startswith("METHOD.") for item in yaml_registry.records)
    assert any(item.evidence_ref.startswith("METHOD.") for item in template_registry.records)
    assert all(item.kind is EvidenceKind.APPROVED_RULE for item in yaml_registry.records if item.namespace == "METHOD")
    assert {item.metadata.get("method_source_hash") for item in yaml_registry.records if item.namespace == "METHOD"} == {yaml_method.structured_risk_method.method_source_hash}


def test_potential_harm_is_resolved_from_method_severity_semantics():
    report = TemplateRoleCompiler().compile_method(
        ROOT / "references/HARA_Template_AI_20260327.xlsx"
    ).report_contract
    method = YamlBaselineCompiler().compile(
        ROOT / "method_assets/fusa_baseline_v1/manifest.yaml",
        report_contract=report,
    )
    scenario = {
        "scenario_id": "SCN-1", "malfunction_id": "MF-1", "relative_speed_kph": 20.0,
        "collision_type": "FRONTAL", "road_user_type": "VEHICLE",
        "_fact_provenance": {"relative_speed_kph": {
            "provenance": "PROJECT_INPUT",
            "source_refs": [{"location": "tests/test_p0b_pre_causal_risk_boundary.py"}],
        }},
    }
    scenario["_exposure_dimension_coverage_decision"] = (
        ExposureDimensionCoverageService(method).decide(
            assessment_key="MF-1::SCN-1", function=None, operating_mode="Active",
        )
    )
    scored = MethodRuleScoringService(method).score(scenario, "hazard")
    registry = build_project_evidence_registry(
        ItemDefinitionFacts(system_description="system", item_boundary="vehicle", sources=[_source()]), method,
    )
    resolved = PotentialHarmResolver().resolve(
        method=method, severity_result=scored["severity"], scenario=scenario,
        registry=registry,
    )
    assert resolved.potential_harm
    assert resolved.status.value == "FINALIZED"
    assert resolved.rule_id


def test_causal_prompt_uses_bounded_view_and_excludes_downstream_method_metadata():
    scenario = _scenario()
    scenario.facts.update({
        "driver_in_vehicle": False,
        "operating_mode": "ACTIVE",
        "e_z": 3,
        "e_f": 2,
        "asil": "B",
    })
    scenario.fact_provenance.update({
        key: {"provenance": "PROJECT_INPUT", "approval": "FINALIZED", "source_refs": [_source()]}
        for key in ("driver_in_vehicle", "operating_mode", "e_z", "e_f", "asil")
    })
    registry = build_fact_registry(_malfunction(), scenario)
    selection = CausalEvidenceSelector(max_evidence=2).select(registry)
    assert selection.selected_evidence_count == 2
    assert selection.candidate_evidence_count > selection.selected_evidence_count
    assert all("e_z" not in ref and "e_f" not in ref for ref in selection.selected_context_evidence_refs)
    assert all(ref.startswith(("DERIVED.", "SCN.")) for ref in selection.selected_context_evidence_refs)

    prompt = build_scenario_user_prompt(_malfunction(), [scenario], causal_evidence_budget=2)
    assert "causal_evidence_view" in prompt
    assert '"fact_registry"' not in prompt
    assert '"e_z"' not in prompt
    assert '"e_f"' not in prompt
    assert "[kind=DERIVED_PHYSICS]" in prompt


def test_finalized_malfunction_anchors_are_mandatory_and_do_not_consume_context_budget():
    selection = CausalEvidenceSelector(max_evidence=1).select(
        build_fact_registry(_malfunction(), _scenario())
    )

    assert selection.mandatory_evidence_refs == (
        "MF.description", "MF.functional_effect",
    )
    assert len(selection.selected_context_evidence_refs) == 1
    assert selection.total_prompt_evidence_count == 3
    assert "MF.description [kind=DIRECT_FACT]" in selection.compact_view
    assert "MF.functional_effect [kind=DIRECT_FACT]" in selection.compact_view
    assert selection.selected_context_evidence_refs[0].startswith(("SCN.", "DERIVED."))
    prompt = build_scenario_user_prompt(_malfunction(), [_scenario()], causal_evidence_budget=1)
    assert "MF.description" in prompt
    assert "MF.functional_effect" in prompt
    assert "mandatory DIRECT_FACT anchors" in prompt


def test_selector_deduplicates_canonical_derived_copy_of_direct_fact():
    scenario = _scenario()
    scenario.facts["ego_speed_kph"] = 20.0
    scenario.fact_provenance["ego_speed_kph"] = {
        "provenance": "PROJECT_INPUT", "approval": "FINALIZED",
        "source_refs": [_source()],
    }
    selection = CausalEvidenceSelector(max_evidence=20).select(
        build_fact_registry(_malfunction(), scenario)
    )

    assert not (
        "SCN.ego_speed_kph" in selection.selected_context_evidence_refs
        and "DERIVED.ego_speed_kph" in selection.selected_context_evidence_refs
    )


def test_m_to_b_anchor_survives_missing_scenario_trigger_context():
    payload = _positive_payload()
    payload["causally_relevant"] = False
    payload["breakpoint"] = "B_TO_I"
    payload["causal_chain"] = {
        "m_to_b": {
            "claim": "the malfunction removes the braking command",
            "basis_type": "DIRECT_FACT",
            "evidence_refs": ["MF.description", "MF.functional_effect"],
        },
        "b_to_i": {
            "claim": "the required interaction context is not established",
            "basis_type": "ASSUMPTION",
            "evidence_refs": [],
        },
    }
    payload["hazardous_event"] = ""

    result = ScenarioFeasibilityAgent._parse(
        _malfunction(), payload, scenario=_scenario(),
    )

    assert result.breakpoint == "B_TO_I"
    assert result.causal_chain["m_to_b"]["evidence_refs"] == [
        "MF.description", "MF.functional_effect",
    ]


def test_m_to_b_repair_boundary_does_not_downgrade_valid_anchor():
    from hara_agent.infrastructure.llm import LLMResponse

    class Client:
        def __init__(self):
            self.requests = []

        def complete_json(self, request):
            self.requests.append(request)
            if len(self.requests) == 1:
                initial = _positive_payload()
                initial["causal_chain"]["b_to_i"]["evidence_refs"] = [
                    "SCN.not_registered",
                ]
                return LLMResponse(
                    data={"assessments": [initial]}, model="fake",
                )
            repaired = _positive_payload()
            repaired["causally_relevant"] = False
            repaired["breakpoint"] = "M_TO_B"
            repaired["causal_chain"] = {
                "m_to_b": {
                    "claim": "unsupported",
                    "basis_type": "ASSUMPTION",
                    "evidence_refs": [],
                },
            }
            repaired["hazardous_event"] = ""
            return LLMResponse(
                data={"assessments": [repaired]}, model="fake",
            )

    client = Client()
    assessments, audit = ScenarioFeasibilityAgent(
        client, batch_max_chars=50000, batch_max_items=12,
    ).assess(_malfunction(), [_scenario()])

    assert len(client.requests) == 2
    assert assessments[0].status.value == "PENDING"
    assert audit["repair_failure_by_code"]["REPAIR_BREAKPOINT_DOWNGRADE"] == 1


def test_mixed_direct_and_derived_refs_fail_until_basis_matches_registry_kind():
    scenario = _scenario()
    scenario.facts["operating_mode"] = "ACTIVE"
    scenario.facts["ego_speed_kph"] = 20.0
    payload = _positive_payload()
    payload["causal_chain"]["b_to_i"]["evidence_refs"] = [
        "SCN.operating_mode", "SCN.ego_speed_kph", "DERIVED.ego_speed_kph",
    ]
    with pytest.raises(ScenarioEvidenceContractError) as caught:
        ScenarioFeasibilityAgent._parse(_malfunction(), payload, scenario=scenario)
    assert caught.value.code.value == "DIRECT_FACT_KIND_MISMATCH"
    assert caught.value.hop == "b_to_i"

    payload["causal_chain"]["b_to_i"]["evidence_refs"] = [
        "SCN.operating_mode", "SCN.ego_speed_kph",
    ]
    result = ScenarioFeasibilityAgent._parse(
        _malfunction(), payload, scenario=scenario,
    )
    assert result.causal_assessment is not None


def test_debug_kind_diagnostics_records_expected_and_actual_evidence_kinds():
    scenario = _scenario()
    scenario.facts["operating_mode"] = "ACTIVE"
    scenario.facts["ego_speed_kph"] = 20.0
    payload = _positive_payload()
    payload["causal_chain"]["b_to_i"]["evidence_refs"] = [
        "SCN.operating_mode", "SCN.ego_speed_kph", "DERIVED.ego_speed_kph",
    ]
    with pytest.raises(ScenarioEvidenceContractError) as caught:
        ScenarioFeasibilityAgent._parse(_malfunction(), payload, scenario=scenario)
    diagnostics = ScenarioFeasibilityAgent._evidence_kind_diagnostics(
        caught.value, build_fact_registry(_malfunction(), scenario),
    )
    assert diagnostics["validator_expected_kind"] == "DIRECT_FACT"
    assert diagnostics["actual_evidence_kinds"] == [
        "DERIVED_PHYSICS", "DIRECT_FACT",
    ]
    assert next(
        item for item in diagnostics["resolved_evidence"]
        if item["evidence_ref"] == "DERIVED.ego_speed_kph"
    )["kind"] == "DERIVED_PHYSICS"
