from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import socket

import pytest
import yaml

from hara_agent.contracts import ExposureAtom
from hara_agent.contracts.situation_selection import SituationReference
from hara_agent.method_sources import MethodSourceResolver
from hara_agent.method_sources.situation_selection_compiler import compile_situation_selection
from hara_agent.models import (
    FactProvenance, FunctionDefinition, ItemDefinitionFacts, MalfunctionCandidate,
    ReviewStatus, RiskFact, ScenarioCandidate, SourceRef,
)
from hara_agent.services.analysis import ExposureMethodExecutor, ScenarioMethodService
from hara_agent.services.analysis.malfunction_situation_selection import MalfunctionSituationSelectionService


ROOT = Path(__file__).resolve().parents[2]
ASSET = ROOT / "method_assets/fusa_baseline_v1/normalized/malfunction_situation_selection.yaml"
SOURCE = SourceRef("item_document", "synthetic-project", "test:scope", "Explicit synthetic project evidence")


@pytest.fixture(scope="module")
def method():
    return MethodSourceResolver().resolve(
        template_path=None,
        baseline_manifest_path=ROOT / "method_assets/fusa_baseline_v1/manifest.yaml",
        report_template_path=ROOT / "references/HARA_Template_AI_20260327.xlsx",
    ).method


@pytest.fixture(autouse=True)
def no_provider(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("P5-O offline regression attempted network/Provider access")
    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr("hara_agent.infrastructure.llm.factory.create_llm_client", forbidden)


def malfunction(**values):
    return MalfunctionCandidate(**{
        "malfunction_id": "MF-TEST", "function_id": "F-TEST", "guideword": "More",
        "guideword_id": "GW-MORE", "description": "Synthetic braking output deviation",
        "functional_effect": "Synthetic output exceeds request", "vehicle_level_hazard": "Synthetic sudden deceleration",
        "causal_chain": ["M", "B"], "component_category": "computing", "failure_type": "excessive",
        "sources": [SOURCE], **values,
    })


def function():
    return FunctionDefinition("F-TEST", "Brake request", "Brake command", sources=[SOURCE])


def project(method, *, output="braking_torque", scope="WITHIN_FUNCTION", extras=()):
    context = {
        "project_scope": method.metadata["project_analysis_policy"]["project_scope"],
        "function_id": "F-TEST", "malfunction_id": "MF-TEST", "function_output": function().output,
    }
    facts = [RiskFact(
        fact_id=key, parameter=key, value=value, context=context,
        source_refs=[SOURCE], provenance=FactProvenance.HUMAN_CONFIRMATION,
        approval=ReviewStatus.FINALIZED,
    ) for key, value in [("FUNCTIONAL_OUTPUT", output), ("FUNCTION_OPERATING_SCOPE", scope), *extras]]
    return ItemDefinitionFacts("synthetic project", "synthetic boundary", risk_facts=facts, sources=[SOURCE])


def scenario(identity="SC-TEST", atoms=("PH005",), mode="PARKING_CONTROL"):
    return ScenarioCandidate(
        identity, "synthetic scenario", "synthetic description", "synthetic detail",
        facts={"scenario_atom_ids": list(atoms), "operating_mode": mode},
        fact_provenance={"operating_mode": {
            "provenance": "PROJECT_INPUT", "approval": "FINALIZED", "source_refs": [asdict(SOURCE)],
            "applicable_scope": {
                "malfunction_id": "MF-TEST", "scenario_id": identity, "function_id": "F-TEST",
                "project_scope": "current_avp_item_definition",
            },
        }},
        semantic_fingerprint=identity,
    )


def select(method, mf=None, candidates=None, facts=None):
    return MalfunctionSituationSelectionService(method).select(
        mf or malfunction(), candidates or [scenario()],
        function=function(), project_facts=facts if facts is not None else project(method),
    )


def test_within_braking_includes_supported_parking_without_excluding_other_situations(method):
    parents = [scenario(), scenario("SC-OTHER", ("FV100",))]
    result, audit = select(method, candidates=parents)
    assert result == parents
    assert (audit["include_count"], audit["exclude_count"], audit["unresolved_passthrough_count"]) == (1, 0, 1)
    assert audit["records"][0]["rule_ids"] == ["RHP-SIT-P06"]
    assert audit["records"][0]["references"][0]["rows"] == (22, 23)
    assert audit["base_candidate_count"] == audit["include_count"] + audit["exclude_count"] + audit["unresolved_passthrough_count"]
    assert audit["scenarios_after_selection"] == audit["include_count"] + audit["unresolved_passthrough_count"]
    assert audit["provider_calls_added_by_selection"] == 0


def test_outside_unwarranted_braking_is_not_within_rule_or_highway_generator(method):
    parent = scenario()
    result, audit = select(method, malfunction(failure_type="unintended_activation"), [parent], project(method, scope="OUTSIDE_FUNCTION"))
    assert result == [parent]
    record = audit["records"][0]
    assert record["status"] == "UNRESOLVED"
    assert record["reference_rule_ids"] == ["RHP-SIT-P05"]
    assert "SOURCE_BROAD_SCOPE_NOT_APPLICABLE" in record["diagnostic"]


@pytest.mark.parametrize("output,scope,failure", [
    ("driver_display", "WITHIN_FUNCTION", "loss"),
    ("flasher", "OUTSIDE_FUNCTION", "loss"),
    ("remote_engine_on", "WITHIN_FUNCTION", "loss"),
    ("unknown_output", "WITHIN_FUNCTION", "excessive"),
])
def test_reference_none_or_unknown_does_not_exclude(method, output, scope, failure):
    _, audit = select(method, malfunction(failure_type=failure), facts=project(method, output=output, scope=scope))
    assert audit["exclude_count"] == 0
    assert audit["unresolved_passthrough_count"] == 1


@pytest.mark.parametrize("issue", ["missing_scope", "wrong_mf", "pending", "conflicting_scope", "wrong_output_binding", "narrower_context"])
def test_guideword_and_similar_prose_never_supply_missing_scope(method, issue):
    facts = project(method)
    if issue == "missing_scope":
        facts.risk_facts = facts.risk_facts[:1]
    elif issue == "wrong_mf":
        facts.risk_facts = [replace(f, context={**f.context, "malfunction_id": "OTHER"}) for f in facts.risk_facts]
    elif issue == "pending":
        facts.risk_facts = [replace(f, approval=ReviewStatus.PENDING) for f in facts.risk_facts]
    elif issue == "wrong_output_binding":
        facts.risk_facts = [replace(f, context={**f.context, "function_output": "Other command"}) for f in facts.risk_facts]
    elif issue == "narrower_context":
        facts.risk_facts = [replace(f, context={**f.context, "scenario_id": "OTHER-SCENARIO"}) for f in facts.risk_facts]
    else:
        facts.risk_facts.append(replace(facts.risk_facts[1], fact_id="conflict", value="OUTSIDE_FUNCTION"))
    _, audit = select(method, malfunction(description="More unwarranted parking 泊车 within"), facts=facts)
    assert audit["unresolved_passthrough_count"] == 1
    assert audit["include_count"] == audit["exclude_count"] == 0


def exclusion_method(method):
    # Explicit synthetic approval only; this fixture is not current-project authority.
    scope = method.metadata["project_analysis_policy"]["project_scope"]
    base = next(r for r in method.situation_selection.rules if r.rule_id == "RHP-SIT-P06")
    exclusion = replace(
        base, rule_id="TEST-APPROVED-CONSTRAINT", action="EXCLUDE",
        authority="APPROVED_PROJECT_CONSTRAINT", approval="APPROVED",
        runtime_exclusion_authority=True, project_scope=scope,
        constraint_parameter="FUNCTION_ALLOWED_MODE", scenario_field="operating_mode",
        supported_families=(), source_refs=(SituationReference(
            "TEST-PROJECT", "Synthetic approval", (1,), "A", "Only the approved mode can activate this synthetic effect",
            "CURRENT_PROJECT_APPROVED", "NOT_APPLICABLE", scope,
        ),),
    )
    return replace(method, situation_selection=replace(method.situation_selection, rules=(exclusion,)))


def test_exclusion_requires_explicit_project_approval_and_structured_contradiction(method):
    scoped = exclusion_method(method)
    parents = [scenario(), scenario("SC-OUTSIDE", ("FV100",), "SEARCH")]
    result, audit = select(scoped, candidates=parents, facts=project(method, extras=(("FUNCTION_ALLOWED_MODE", "PARKING_CONTROL"),)))
    assert result == parents[:1]
    assert audit["exclude_count"] == 1
    evidence = audit["exclude_rule_breakdown"][0]["exclusion_evidence"][0]
    assert evidence["actual"] == "SEARCH" and evidence["required"] == "PARKING_CONTROL"
    assert evidence["project_constraint_facts"][0]["source_refs"]
    assert audit["scenarios_after_selection"] == 1


@pytest.mark.parametrize("issue", ["pending_rule", "wrong_project", "missing_fact_source", "missing_constraint", "conflict"])
def test_exclusion_cannot_be_inferred_or_first_match(method, issue):
    scoped = exclusion_method(method)
    parent = scenario(mode="SEARCH")
    facts = project(method, extras=(("FUNCTION_ALLOWED_MODE", "PARKING_CONTROL"),))
    rule = scoped.situation_selection.rules[0]
    if issue == "pending_rule":
        scoped = replace(scoped, situation_selection=replace(scoped.situation_selection, rules=(replace(rule, approval="REFERENCE_ONLY"),)))
    elif issue == "wrong_project":
        scoped = replace(scoped, situation_selection=replace(scoped.situation_selection, rules=(replace(rule, project_scope="other"),)))
    elif issue == "missing_fact_source":
        parent.fact_provenance = {}
    elif issue == "missing_constraint":
        facts.risk_facts = facts.risk_facts[:2]
    else:
        include = next(r for r in method.situation_selection.rules if r.rule_id == "RHP-SIT-P06")
        scoped = replace(scoped, situation_selection=replace(scoped.situation_selection, rules=(rule, include)))
    result, audit = select(scoped, candidates=[parent], facts=facts)
    assert result == [parent]
    assert audit["unresolved_passthrough_count"] == 1
    assert audit["exclude_count"] == 0


def test_version_mapping_preserves_native_ids_and_unresolved_meanings(method):
    families = {f.family_id: f for f in method.situation_selection.families}
    assert (families["PARKING_IN_OUT"].source_id, families["PARKING_IN_OUT"].candidate_canonical_id) == ("FV010", "PH005")
    assert families["FREE_DRIVING"].source_id == "FV100"
    assert families["GARAGE_PARKING"].source_id == "SO020"
    for name in ("FREE_DRIVING", "GARAGE_PARKING", "HOLD_ON_SLOPE"):
        assert families[name].candidate_canonical_id == ""
    assert families["HOLD_ON_SLOPE"].mapping_status == "MAPPING_REVIEW_REQUIRED"
    assert not any(r.runtime_exclusion_authority for r in method.situation_selection.rules)


@pytest.mark.parametrize("issue", ["schema", "unknown_field", "family", "version", "invented_mapping", "unapproved_exclude", "failure_type", "duplicate"])
def test_compiler_rejects_unrecognized_or_unapproved_rule_semantics(method, issue):
    data = yaml.safe_load(ASSET.read_text())
    if issue == "schema": data["schema_version"] = "2.0"
    elif issue == "unknown_field": data["rules"][0]["asil_max"] = "D"
    elif issue == "family": data["rules"][0]["supported_families"] = ["UNKNOWN"]
    elif issue == "version": data["families"][0]["source_version"] = "V9"
    elif issue == "invented_mapping": data["families"][0]["candidate_canonical_id"] = "FB002"
    elif issue == "unapproved_exclude": data["rules"][0]["action"] = "EXCLUDE"
    elif issue == "failure_type": data["rules"][0]["canonical_failure_types"] = ["NEW_GUIDEWORD"]
    else: data["rules"].append(deepcopy(data["rules"][0]))
    atoms = yaml.safe_load((ASSET.parents[1] / "raw/vda702_atoms.yaml").read_text())["atoms"]
    with pytest.raises(ValueError):
        compile_situation_selection(data, asset=str(ASSET), source_hash="a" * 64,
                                    atoms={x["id"]: x for x in atoms},
                                    failure_types={x.canonical_id for x in method.scenario_model.scenario_method.failure_mode_selector_taxonomy.failure_types})


def test_frozen_assets_and_eight_guidewords_are_exactly_unchanged(method):
    fixture = json.loads((ROOT / "tests/fixtures/method_alignment/p5o_frozen_assets.json").read_text())
    for filename, digest in fixture["sha256"].items():
        assert hashlib.sha256((ROOT / filename).read_bytes()).hexdigest() == digest, filename
    assert [{"id": g.guideword_id, "name": g.name, "description": g.description} for g in method.guidewords.guidewords] == fixture["guidewords"]
    assert len(method.guidewords.guidewords) == 8


def test_all_seven_native_fusa_operands_and_separate_reference_difference(method):
    exposure = method.structured_risk_method.exposure
    results = []
    for case in method.situation_selection.exposure_parity_cases:
        atoms = tuple(ExposureAtom(str(i), (str(i),), str(i), level, level, exposure.source_refs[0]) for i, level in enumerate(case.operands))
        value = ExposureMethodExecutor().lookup({
            "component_category": "computing", "scenario_atom_ids": [a.atom_id for a in atoms],
            **({"atoms_coupling": case.coupling} if case.coupling else {}),
        }, replace(exposure, atoms=atoms, strong_couplings=()))
        assert value["value"] == case.expected
        results.append(value["value"])
    assert results == ["E4", "E3", "E3", "E3", "E1", "E1", "E2"]
    example = next(e for e in method.situation_selection.assessment_examples if e.case_id == "GOLD-E-MIX-01")
    assert example.reference_result == "E1"
    assert example.source_refs[0].reference_status == "CONFIRMED_IN_SOURCE"
    assert "METHOD_RULE_DIFFERENCE" in example.allowed_use


def test_zf_definitions_are_auditable_without_overriding_component_route(method):
    _, audit = select(method)
    trigger = audit["exposure_trigger_audit"]
    assert trigger["trigger_semantics_status"] == "TRIGGER_SEMANTICS_NOT_PROVEN"
    assert trigger["runtime_route_status"] == "COMPONENT_HEURISTIC_USED"
    assert trigger["requested_domain"] == "Z"
    assert {(p["trigger_semantics"], p["exposure_domain"]) for p in trigger["reference_principles"]} == {
        ("IMMEDIATE_VEHICLE_EFFECT", "Z"), ("SITUATION_TRIGGERED", "F"),
    }
    _, with_fact = select(method, facts=project(method, extras=(("VEHICLE_EFFECT_TRIGGER", "SITUATION_TRIGGERED"),)))
    assert with_fact["exposure_trigger_audit"]["requested_domain"] == "Z"
    assert with_fact["exposure_trigger_audit"]["numeric_behavior_changed"] is False


def test_template_options_unchanged_when_selection_is_unresolved(method):
    # Same contract hash isolates preselection from the deliberate new bundle hash.
    before = replace(method, situation_selection=None)
    mf = malfunction(failure_type="degraded", component_category="actuator_longitudinal")
    children_before, _ = ScenarioMethodService(before).instantiate_analytical_candidates(mf, [scenario()])
    children_after, audit = ScenarioMethodService(method).instantiate_analytical_candidates(mf, [scenario()])
    assert [x.to_dict() for x in children_before] == [x.to_dict() for x in children_after]
    assert len(children_after) == 4
    assert audit["situation_selection"]["unresolved_passthrough_count"] == 1
    assert audit["situation_selection"]["post_template_options_count"] == 4
    assert audit["situation_selection"]["post_driver_branch_count"] == 4


def test_explicit_synthetic_exclusion_approval_compiles(method):
    data = yaml.safe_load(ASSET.read_text())
    rule = exclusion_method(method).situation_selection.rules[0]
    data["rules"] = json.loads(json.dumps([asdict(rule)]))
    data["documents"].append({
        "document_key": "TEST-PROJECT", "filename": "synthetic-approval.txt",
        "sha256": "a" * 64, "source_kind": "CURRENT_PROJECT_APPROVAL",
    })
    atoms = yaml.safe_load((ASSET.parents[1] / "raw/vda702_atoms.yaml").read_text())["atoms"]
    compiled = compile_situation_selection(
        data, asset="test-only", source_hash="b" * 64, atoms={x["id"]: x for x in atoms},
        failure_types={x.canonical_id for x in method.scenario_model.scenario_method.failure_mode_selector_taxonomy.failure_types},
    )
    assert compiled.rules[0].runtime_exclusion_authority


class OfflineCausalFixture:
    """Explicit synthetic M→B→I→H evidence; no transport/client implementation."""
    prompt_version = "p5o-offline-fixture"
    assessment_contract_version = "scenario-causal-assessment-v4"
    client = object()

    def assess(self, malfunction, candidates, project_registry=None):
        from hara_agent.contracts import (
            CausalBreakpoint, CausalEdge, CausalGraph, CausalNode, CausalNodeType,
            CausalRelation, EvidenceBinding, ScenarioCausalAssessment,
        )
        from hara_agent.models import EvidenceKind, ScenarioFeasibilityAssessment
        nodes = tuple(CausalNode(name, kind, name) for name, kind in (
            ("M", CausalNodeType.MALFUNCTION), ("B", CausalNodeType.SYSTEM_BEHAVIOR_CHANGE),
            ("I", CausalNodeType.OPERATIONAL_CONSEQUENCE), ("H", CausalNodeType.HAZARD),
        ))
        edges = tuple(CausalEdge(edge, a, b, CausalRelation.CAUSES, "Synthetic evidence", (f"TEST.{edge}",))
                      for edge, a, b in (("M_TO_B", "M", "B"), ("B_TO_I", "B", "I"), ("I_TO_H", "I", "H")))
        results = []
        for candidate in candidates:
            causal = ScenarioCausalAssessment(
                candidate.scenario_id, CausalGraph(nodes, edges), ("M", "B", "I", "H"), CausalBreakpoint.NONE,
                tuple(EvidenceBinding(e.edge_id, e.evidence_refs, EvidenceKind.DIRECT_FACT,
                                      status=ReviewStatus.FINALIZED) for e in edges),
                (), (), f"Synthetic hazard {candidate.scenario_id}",
                review_status=ReviewStatus.FINALIZED,
            )
            results.append(ScenarioFeasibilityAssessment(
                malfunction.malfunction_id, candidate.scenario_id, True, True, True, [],
                "Explicit offline test evidence", hazardous_event=causal.hazardous_event,
                breakpoint="NONE", causal_assessment=causal,
                status=ReviewStatus.FINALIZED,
            ))
        return results, {
            "malfunction_id": malfunction.malfunction_id, "llm_calls": 0,
            "initial_batch_count": 0, "leaf_batch_count": 0,
            "leaf_items": [], "leaf_input_chars": [],
        }


def test_production_causal_scoring_and_report_groups_identical_without_matching_rules(method, tmp_path):
    from hara_agent.services.analysis import MethodContractASILService, MethodRuleScoringService
    from hara_agent.services.reporting.projection import HARAReportProjectionService
    from hara_agent.services.reporting.report_schema import load_report_schema
    from hara_agent.workflow import HARAState, ReviewArtifactWriter
    from hara_agent.workflow.nodes.scenarios import assess_scenarios
    from hara_agent.workflow.nodes.scoring import score_structured_scenarios
    results = []
    for label, active in (("before", replace(method, situation_selection=None)), ("after", method)):
        mf = malfunction(failure_type="stuck")
        parents = [scenario("SC-1"), scenario("SC-2")]
        state = HARAState(run_id=label, functions=[asdict(function())], malfunctions=[asdict(mf)])
        writer = ReviewArtifactWriter(label, root=tmp_path)
        assess_scenarios(state, OfflineCausalFixture(), [mf], parents, method=active, review_artifact_writer=writer)
        assert len(state.scenarios) == 2
        audit = json.loads((tmp_path / label / "malfunction_situation_selection.json").read_text())["per_malfunction"][0]
        assert (audit["base_candidate_count"], audit["unresolved_passthrough_count"], audit["causal_attempted_count"], audit["causal_retained_count"]) == (2, 2, 2, 2)
        score_structured_scenarios(state, MethodRuleScoringService(active), MethodContractASILService(active))
        report = HARAReportProjectionService(load_report_schema()).project(state, active)
        assert len(state.risk_results) == len(report.rows) == 2
        results.append({
            "scenarios": [x.to_dict() for x in state.scenarios],
            "causal": state.item_definition["scenario_assessments"],
            "risks": [asdict(x) for x in state.risk_results],
            "report_rows": [x.to_dict() for x in report.rows],
        })
    assert results[0] == results[1]


def test_synthesis_annotation_does_not_grant_exclusion_or_change_candidates(method):
    audit = MalfunctionSituationSelectionService(method).annotate_synthesis(
        asdict(malfunction()), scenario(), asdict(function()), asdict(project(method)),
    )
    assert audit["mode"] == "READ_ONLY_SYNTHESIS_ANNOTATION"
    assert audit["applied_to_candidate_sets"] is False
    assert audit["include_count"] == 1


def test_unavailable_atom_and_exclusion_provenance_pass_through(method):
    parent = scenario(mode="SEARCH")
    parent.facts["scenario_atom_ids"] = None
    parent.fact_provenance["operating_mode"] = None
    result, audit = select(exclusion_method(method), candidates=[parent],
                           facts=project(method, extras=(("FUNCTION_ALLOWED_MODE", "PARKING_CONTROL"),)))
    assert result == [parent]
    assert audit["unresolved_passthrough_count"] == 1
