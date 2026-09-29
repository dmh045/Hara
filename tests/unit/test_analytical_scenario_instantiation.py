from copy import deepcopy
from dataclasses import replace
from pathlib import Path

import pytest

from hara_agent.contracts import FailureModeTaxonomyValue, RiskContextFactStatus
from hara_agent.method_sources import YamlBaselineCompiler
from hara_agent.models import (
    MalfunctionCandidate, ReviewStatus, ScenarioCandidate,
    ScenarioFeasibilityAssessment, SourceRef,
)
from hara_agent.services.analysis import (
    HazardousEventRiskContextService, ScenarioMethodService,
)
from hara_agent.services.analysis.scenario_physics import derive_scenario_physics
from hara_agent.services.semantic.scenario_evidence import (
    CausalEvidenceSelector, ScenarioEvidenceContractError, ScenarioEvidenceErrorCode,
    build_fact_registry, validate_evidence_contract,
)
from hara_agent.services.semantic import ScenarioFeasibilityAgent
from hara_agent.template import TemplateRoleCompiler
from hara_agent.workflow import HARAState, WorkflowStage
from hara_agent.workflow.nodes.scenarios import assess_scenarios


ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture(scope="module")
def baseline_method():
    report = TemplateRoleCompiler().compile_method(
        ROOT / "references/HARA_Template_AI_20260327.xlsx"
    ).report_contract
    return YamlBaselineCompiler().compile(
        ROOT / "method_assets/fusa_baseline_v1/manifest.yaml",
        report_contract=report,
    )


@pytest.fixture(scope="module")
def method(baseline_method):
    base = baseline_method
    taxonomy = base.scenario_model.scenario_method.failure_mode_selector_taxonomy
    assert taxonomy is not None
    expanded = replace(
        taxonomy,
        component_categories=taxonomy.component_categories + (
            FailureModeTaxonomyValue(
                canonical_id="parking_brake", aliases=(),
                source_ref=taxonomy.component_categories[0].source_ref,
            ),
        ),
        failure_types=taxonomy.failure_types + (
            FailureModeTaxonomyValue(
                canonical_id="unintended_deactivation", aliases=(),
                source_ref=taxonomy.failure_types[0].source_ref,
            ),
        ),
    )
    return replace(
        base,
        scenario_model=replace(
            base.scenario_model,
            scenario_method=replace(
                base.scenario_model.scenario_method,
                failure_mode_selector_taxonomy=expanded,
            ),
        ),
    )


def _malfunction() -> MalfunctionCandidate:
    return MalfunctionCandidate(
        malfunction_id="MF-P3-B", function_id="F-P3", guideword="No/Loss",
        description="parking_brake_release",
        functional_effect="confirmed output is unavailable",
        vehicle_level_hazard="vehicle behavior may be unsafe",
        causal_chain=["M", "B"], status=ReviewStatus.FINALIZED,
        confidence=0.9, component_category="parking_brake",
        failure_type="unintended_deactivation",
    )


def _baseline_synthetic_malfunction(identity: str = "MF-P3-B-BASELINE") -> MalfunctionCandidate:
    """Synthetic fixture using only the untouched compiled baseline vocabulary."""
    source = SourceRef("item_definition", "ItemDef.docx", "causal-fixture", "source")
    return MalfunctionCandidate(
        malfunction_id=identity, function_id="F-P3", guideword="Less",
        description="longitudinal actuator degraded output",
        functional_effect="confirmed output is unavailable",
        vehicle_level_hazard="vehicle behavior may be unsafe",
        causal_chain=["M", "B"], status=ReviewStatus.FINALIZED,
        confidence=0.9, component_category="actuator_longitudinal",
        failure_type="degraded", sources=[source],
    )


def _parent() -> ScenarioCandidate:
    return ScenarioCandidate(
        scenario_id="SCN-PARENT", operating_scenario="parking",
        situational_description="项目基础场景", situational_detailing="项目基础场景",
        facts={"ego_speed_constraint": {"min_kph": 0.0, "max_kph": 10.0}},
        semantic_fingerprint="parent-fingerprint", status=ReviewStatus.FINALIZED,
    )


def _causal_parent() -> ScenarioCandidate:
    source = SourceRef("item_definition", "ItemDef.docx", "causal-fixture", "source")
    facts = {
        "operating_scenario": "parking lot",
        "vehicle_state": "active",
        "ego_speed_kph": 20.0,
        "harm_mechanism": "loss of trajectory control can expose occupants to impact injury",
        "ego_speed_constraint": {"min_kph": 0.0, "max_kph": 10.0},
    }
    provenance = {
        field: {
            "provenance": "PROJECT_INPUT", "approval": "FINALIZED",
            "source_refs": [{
                "source_type": source.source_type, "source_id": source.source_id,
                "location": source.location, "excerpt": source.excerpt,
            }],
        }
        for field in facts
    }
    return ScenarioCandidate(
        scenario_id="SCN-CAUSAL-PARENT", operating_scenario="parking",
        situational_description="project causal scenario",
        situational_detailing="project causal scenario", facts=facts,
        fact_provenance=provenance, semantic_fingerprint="causal-parent-fingerprint",
        status=ReviewStatus.FINALIZED, sources=[source],
    )


def _materialize(candidate: ScenarioCandidate) -> dict:
    scenario = {**candidate.facts, "_fact_provenance": dict(candidate.fact_provenance)}
    for record in derive_scenario_physics(candidate):
        key = record.evidence_ref.split(".", 1)[1]
        scenario[key] = record.value
        scenario["_fact_provenance"][key] = {
            "provenance": record.provenance.value,
            "approval": record.approval_status.value,
            "source_refs": [source.__dict__ for source in record.source_refs],
            **record.metadata,
        }
    return scenario


def test_strong_template_creates_two_isolated_analysis_instances_with_source_contract(method):
    parent = _parent()
    instances, audit = ScenarioMethodService(method).instantiate_analytical_candidates(
        _malfunction(), [parent],
    )

    assert audit["selection_mode"] == "STRONG_TEMPLATE_ANALYTICAL_INSTANCES"
    assert len(instances) == 2
    assert len({item.scenario_id for item in instances}) == 2
    assert parent.facts == {"ego_speed_constraint": {"min_kph": 0.0, "max_kph": 10.0}}
    assert all(item.source_scenario_id == "SCN-PARENT" for item in instances)
    assert all(item.analysis_instance["malfunction_id"] == "MF-P3-B" for item in instances)
    assert all(item.analysis_instance["validation_status"] == "VALIDATED" for item in instances)
    assert all(item.facts["relative_distance_m"] == 0.3 for item in instances)
    assert all(item.facts["object_speed_kph"] == 0.0 for item in instances)
    assert all("relative_speed_kph" not in item.facts for item in instances)
    assert all("项目速度约束" in item.situational_description for item in instances)
    assert all("本分析场景设定" in item.situational_detailing for item in instances)
    for instance in instances:
        for assumption in instance.analysis_instance["assumptions"]:
            assert assumption["origin"] == "SCENARIO_DEFINED"
            assert assumption["source_template_id"] == "FM_TEMPLATE_006"
            assert assumption["source_option_id"]
            assert assumption["source_hash"]
            assert assumption["method_contract_hash"]
            assert assumption["applicable_scope"]["scenario_id"] == instance.scenario_id
            assert instance.fact_provenance[assumption["field"]]["approval"] == "PENDING"
    assert audit["unmapped_value_count"] == 0
    assert all(
        item.analysis_instance["risk_vocabulary_mappings"]
        for item in instances
    )


def test_untouched_baseline_matcher_instantiates_official_vocabulary(baseline_method):
    malfunction = _baseline_synthetic_malfunction()
    service = ScenarioMethodService(baseline_method)

    match = service.match_fm_template(malfunction)
    instances, audit = service.instantiate_analytical_candidates(
        malfunction, [_parent()],
    )

    assert match.injectable
    assert match.template is not None
    assert match.template.template_id == "FM_TEMPLATE_001"
    assert audit["selection_mode"] == "STRONG_TEMPLATE_ANALYTICAL_INSTANCES"
    assert len(instances) == 4
    assert audit["unmapped_value_count"] == 0
    assert all(item.facts["relative_distance_m"] == 0.3 for item in instances)
    assert all(item.facts["object_speed_kph"] == 0.0 for item in instances)
    assert all("relative_speed_kph" not in item.facts for item in instances)
    assert all(
        item.analysis_instance["malfunction_id"] == malfunction.malfunction_id
        for item in instances
    )


def test_risk_context_accepts_only_same_instance_validated_analysis_assumptions(method):
    instance = ScenarioMethodService(method).instantiate_analytical_candidates(
        _malfunction(), [_parent()],
    )[0][0]
    service = HazardousEventRiskContextService(method)
    context = service.build(
        malfunction_id="MF-P3-B", scenario_id=instance.scenario_id,
        hazard_node_id="H", scenario=_materialize(instance),
    )

    assert context.object_speed_kph.status is RiskContextFactStatus.AVAILABLE
    assert context.object_speed_kph.source_type.value == "DIRECT_SCENARIO_FACT"
    assert context.object_speed_kph.source_provenance == "SCENARIO_DEFINED"
    assert context.relative_distance_m.status is RiskContextFactStatus.AVAILABLE
    assert context.relative_distance_m.source_type.value == "DERIVED_PHYSICS"
    assert context.relative_distance_m.source_provenance == "SCENARIO_DEFINED"
    assert context.to_dict()["relative_distance_m"]["source_provenance"] == (
        "SCENARIO_DEFINED"
    )
    assert context.relative_speed_kph.status is RiskContextFactStatus.UNAVAILABLE


@pytest.mark.parametrize("approval", ["PENDING", "FINALIZED", "APPROVED"])
@pytest.mark.parametrize("scope_issue", ["malfunction_id", "scenario_id", "missing"])
def test_analysis_assumption_scope_cannot_be_bypassed_by_approval(
    method, approval, scope_issue,
):
    instance = ScenarioMethodService(method).instantiate_analytical_candidates(
        _malfunction(), [_parent()],
    )[0][0]
    provenance = deepcopy(instance.fact_provenance)
    metadata = provenance["relative_distance_m"]
    metadata["approval"] = approval
    if scope_issue == "missing":
        metadata.pop("applicable_scope")
    else:
        metadata["applicable_scope"][scope_issue] = "WRONG-MF" if scope_issue == "malfunction_id" else "SCN-OTHER"
    altered = replace(instance, fact_provenance=provenance)

    context = HazardousEventRiskContextService(method).build(
        malfunction_id="MF-P3-B", scenario_id=instance.scenario_id,
        hazard_node_id="H", scenario=_materialize(altered),
    )

    assert context.relative_distance_m.status is RiskContextFactStatus.UNAVAILABLE


def test_ttc_retains_analytical_assumption_source_chain_for_explicit_closing_speed(method):
    instance = ScenarioMethodService(method).instantiate_analytical_candidates(
        _malfunction(), [_parent()],
    )[0][0]
    speed_metadata = deepcopy(instance.fact_provenance["object_speed_kph"])
    speed_metadata["field"] = "closing_speed_kph"
    speed_metadata["source_refs"] = [{
        "source_type": "method_contract", "source_id": "closing-speed-source",
        "location": "template!closing-speed", "excerpt": "configured test input",
    }]
    synthetic = replace(
        instance,
        facts={**instance.facts, "closing_speed_kph": 5.0},
        fact_provenance={
            **instance.fact_provenance,
            "closing_speed_kph": speed_metadata,
        },
    )

    ttc = next(
        item for item in derive_scenario_physics(synthetic)
        if item.evidence_ref == "DERIVED.ttc_s"
    )
    assert ttc.value == 0.216
    assert ttc.metadata["analysis_assumption_origin"] == "SCENARIO_DEFINED"
    assert ttc.metadata["analysis_assumption_scope"]["scenario_id"] == instance.scenario_id
    assert [item["field"] for item in ttc.metadata["input_fact_metadata"]] == [
        "relative_distance_m", "closing_speed_kph",
    ]
    assert [item["input_field"] for item in ttc.metadata["analysis_assumption_inputs"]] == [
        "relative_distance_m", "closing_speed_kph",
    ]
    assert {item.location for item in ttc.source_refs} == {
        instance.sources[-1].location, "template!closing-speed",
    }

    context = HazardousEventRiskContextService(method).build(
        malfunction_id="MF-P3-B", scenario_id=instance.scenario_id,
        hazard_node_id="H", scenario=_materialize(synthetic),
    )
    assert context.ttc_s.status is RiskContextFactStatus.AVAILABLE
    assert context.ttc_s.value == 0.216
    assert context.ttc_s.source_type.value == "DERIVED_PHYSICS"
    assert context.ttc_s.source_provenance == "SCENARIO_DEFINED"


def test_ttc_with_valid_analysis_distance_and_pending_project_speed_is_unavailable(method):
    instance = ScenarioMethodService(method).instantiate_analytical_candidates(
        _malfunction(), [_parent()],
    )[0][0]
    synthetic = replace(
        instance,
        facts={**instance.facts, "closing_speed_kph": 5.0},
        fact_provenance={
            **instance.fact_provenance,
            "closing_speed_kph": {
                "provenance": "PROJECT_INPUT", "approval": "PENDING",
                "source_refs": [{
                    "source_type": "item_definition", "source_id": "ItemDef.docx",
                    "location": "closing_speed", "excerpt": "unconfirmed speed",
                }],
            },
        },
    )
    context = HazardousEventRiskContextService(method).build(
        malfunction_id="MF-P3-B", scenario_id=instance.scenario_id,
        hazard_node_id="H", scenario=_materialize(synthetic),
    )
    assert context.ttc_s.status is RiskContextFactStatus.UNAVAILABLE


def test_ttc_cannot_combine_analytical_inputs_from_different_instances(method):
    instance = ScenarioMethodService(method).instantiate_analytical_candidates(
        _malfunction(), [_parent()],
    )[0][0]
    speed_metadata = deepcopy(instance.fact_provenance["object_speed_kph"])
    speed_metadata["applicable_scope"]["scenario_id"] = "SCN-OTHER"
    synthetic = replace(
        instance,
        facts={**instance.facts, "closing_speed_kph": 5.0},
        fact_provenance={
            **instance.fact_provenance,
            "closing_speed_kph": speed_metadata,
        },
    )

    context = HazardousEventRiskContextService(method).build(
        malfunction_id="MF-P3-B", scenario_id=instance.scenario_id,
        hazard_node_id="H", scenario=_materialize(synthetic),
    )

    assert context.ttc_s.status is RiskContextFactStatus.UNAVAILABLE


def test_ttc_accepts_finalized_project_speed_with_current_analysis_distance(method):
    instance = ScenarioMethodService(method).instantiate_analytical_candidates(
        _malfunction(), [_parent()],
    )[0][0]
    synthetic = replace(
        instance,
        facts={**instance.facts, "closing_speed_kph": 5.0},
        fact_provenance={
            **instance.fact_provenance,
            "closing_speed_kph": {
                "provenance": "PROJECT_INPUT", "approval": "FINALIZED",
                "source_refs": [{
                    "source_type": "item_definition", "source_id": "ItemDef.docx",
                    "location": "closing_speed", "excerpt": "confirmed speed",
                }],
            },
        },
    )

    context = HazardousEventRiskContextService(method).build(
        malfunction_id="MF-P3-B", scenario_id=instance.scenario_id,
        hazard_node_id="H", scenario=_materialize(synthetic),
    )

    assert context.ttc_s.status is RiskContextFactStatus.AVAILABLE
    assert context.ttc_s.value == 0.216
    assert context.ttc_s.source_provenance == "SCENARIO_DEFINED"


@pytest.mark.parametrize("approval", ["FINALIZED", "APPROVED"])
@pytest.mark.parametrize("source_ref_form", ["empty", "missing"])
def test_ttc_rejects_finalized_project_speed_without_its_own_source_reference(
    method, approval, source_ref_form,
):
    instance = ScenarioMethodService(method).instantiate_analytical_candidates(
        _malfunction(), [_parent()],
    )[0][0]
    speed_metadata = {
        "provenance": "PROJECT_INPUT", "approval": approval,
    }
    if source_ref_form == "empty":
        speed_metadata["source_refs"] = []
    synthetic = replace(
        instance,
        facts={**instance.facts, "closing_speed_kph": 5.0},
        fact_provenance={
            **instance.fact_provenance,
            "closing_speed_kph": speed_metadata,
        },
    )

    context = HazardousEventRiskContextService(method).build(
        malfunction_id="MF-P3-B", scenario_id=instance.scenario_id,
        hazard_node_id="H", scenario=_materialize(synthetic),
    )

    assert context.ttc_s.status is RiskContextFactStatus.UNAVAILABLE
    assert context.ttc_s.reason == "MISSING_SOURCE_PROVENANCE"


@pytest.mark.parametrize("source_ref_form", ["empty", "missing"])
def test_ttc_rejects_analysis_speed_without_its_own_source_reference(
    method, source_ref_form,
):
    instance = ScenarioMethodService(method).instantiate_analytical_candidates(
        _malfunction(), [_parent()],
    )[0][0]
    speed_metadata = deepcopy(instance.fact_provenance["object_speed_kph"])
    if source_ref_form == "empty":
        speed_metadata["source_refs"] = []
    else:
        speed_metadata.pop("source_refs")
    synthetic = replace(
        instance,
        facts={**instance.facts, "closing_speed_kph": 5.0},
        fact_provenance={
            **instance.fact_provenance,
            "relative_distance_m": {
                "provenance": "PROJECT_INPUT", "approval": "FINALIZED",
                "source_refs": [{
                    "source_type": "item_definition", "source_id": "ItemDef.docx",
                    "location": "relative_distance", "excerpt": "confirmed distance",
                }],
            },
            "closing_speed_kph": speed_metadata,
        },
    )

    context = HazardousEventRiskContextService(method).build(
        malfunction_id="MF-P3-B", scenario_id=instance.scenario_id,
        hazard_node_id="H", scenario=_materialize(synthetic),
    )

    assert context.ttc_s.status is RiskContextFactStatus.UNAVAILABLE
    assert context.ttc_s.reason == "MISSING_SOURCE_PROVENANCE"


def test_nested_derived_fact_rejects_a_missing_source_leaf(method):
    source = {
        "source_type": "item_definition", "source_id": "ItemDef.docx",
        "location": "valid", "excerpt": "confirmed input",
    }
    finalized = {
        "provenance": "PROJECT_INPUT", "approval": "FINALIZED", "source_refs": [source],
    }
    missing_source = {
        "provenance": "PROJECT_INPUT", "approval": "FINALIZED", "source_refs": [],
    }
    nested = {
        "provenance": "DERIVED", "approval": "FINALIZED", "source_refs": [source],
        "inputs": ["SCN.distance", "SCN.speed"],
        "input_fact_metadata": [finalized, missing_source],
    }
    scenario = {
        "ttc_s": 0.216,
        "_fact_provenance": {
            "ttc_s": {
                "provenance": "DERIVED", "approval": "FINALIZED",
                "source_refs": [source], "inputs": ["DERIVED.inner"],
                "input_fact_metadata": [nested],
            },
        },
    }

    context = HazardousEventRiskContextService(method).build(
        malfunction_id="MF-P3-B", scenario_id="SCN-P3-B", hazard_node_id="H",
        scenario=scenario,
    )

    assert context.ttc_s.status is RiskContextFactStatus.UNAVAILABLE


def test_analysis_assumption_cannot_prove_a_positive_causal_hop(method):
    malfunction = _malfunction()
    instance = ScenarioMethodService(method).instantiate_analytical_candidates(
        malfunction, [_parent()],
    )[0][0]
    registry = build_fact_registry(malfunction, instance)
    item = {
        "scenario_id": instance.scenario_id,
        "physically_feasible": True, "functionally_relevant": True,
        "causally_relevant": True, "breakpoint": "NONE",
        "causal_chain": {
            "m_to_b": {
                "claim": "output is unavailable", "basis_type": "DIRECT_FACT",
                "evidence_refs": ["MF.functional_effect"],
            },
            "b_to_i": {
                "claim": "interaction occurs in the configured distance", "basis_type": "DIRECT_FACT",
                "evidence_refs": ["SCN.relative_distance_m"],
            },
            "i_to_h": {
                "claim": "hazard", "basis_type": "DIRECT_FACT",
                "evidence_refs": ["MF.description"],
            },
        },
        "risk_dimension_changes": [{
            "dimension": "distance", "evidence_refs": ["SCN.relative_distance_m"],
            "reason": "configured analysis condition",
        }],
        "hazardous_event": "hazard", "potential_harm": "",
        "rationale": "test", "confidence": 0.8,
    }
    with pytest.raises(ScenarioEvidenceContractError) as caught:
        validate_evidence_contract(
            malfunction=malfunction, scenario=instance, item=item, registry=registry,
            prompt_version="p3-b-test", batch="1/1", split_path="root", split_depth=0,
        )
    assert caught.value.code is ScenarioEvidenceErrorCode.ASSUMPTION_IN_POSITIVE_CHAIN


def test_analytical_ttc_is_scoring_only_not_positive_causal_evidence(method):
    malfunction = _malfunction()
    instance = ScenarioMethodService(method).instantiate_analytical_candidates(
        malfunction, [_parent()],
    )[0][0]
    speed_metadata = deepcopy(instance.fact_provenance["object_speed_kph"])
    synthetic = replace(
        instance,
        facts={**instance.facts, "closing_speed_kph": 5.0},
        fact_provenance={
            **instance.fact_provenance,
            "closing_speed_kph": speed_metadata,
        },
    )
    registry = build_fact_registry(malfunction, synthetic)
    item = {
        "scenario_id": synthetic.scenario_id,
        "physically_feasible": True, "functionally_relevant": True,
        "causally_relevant": True, "breakpoint": "NONE",
        "causal_chain": {
            "m_to_b": {
                "claim": "output is unavailable", "basis_type": "DIRECT_FACT",
                "evidence_refs": ["MF.functional_effect"],
            },
            "b_to_i": {
                "claim": "behavior changes", "basis_type": "DIRECT_FACT",
                "evidence_refs": ["MF.description"],
            },
            "i_to_h": {
                "claim": "configured TTC proves hazard", "basis_type": "DERIVED_PHYSICS",
                "evidence_refs": ["DERIVED.ttc_s"],
            },
        },
        "risk_dimension_changes": [], "hazardous_event": "hazard",
        "rationale": "test", "confidence": 0.8,
    }

    assert "DERIVED.ttc_s" not in CausalEvidenceSelector(max_evidence=20).select(
        registry,
    ).selected_context_evidence_refs
    with pytest.raises(ScenarioEvidenceContractError) as caught:
        validate_evidence_contract(
            malfunction=malfunction, scenario=synthetic, item=item, registry=registry,
            prompt_version="p3-b-test", batch="1/1", split_path="root", split_depth=0,
        )
    assert caught.value.code is ScenarioEvidenceErrorCode.ASSUMPTION_IN_POSITIVE_CHAIN


def test_analysis_instances_are_isolated_by_malfunction_and_survive_checkpoint(method):
    service = ScenarioMethodService(method)
    first, _ = service.instantiate_analytical_candidates(_malfunction(), [_parent()])
    second_malfunction = replace(_malfunction(), malfunction_id="MF-P3-B-SECOND")
    second, _ = service.instantiate_analytical_candidates(second_malfunction, [_parent()])

    assert set(item.scenario_id for item in first).isdisjoint(
        item.scenario_id for item in second
    )
    assert all(
        item.analysis_instance["applicable_scope"]["malfunction_id"] == "MF-P3-B"
        for item in first
    )
    assert all(
        item.analysis_instance["applicable_scope"]["malfunction_id"] == "MF-P3-B-SECOND"
        for item in second
    )

    state = HARAState(
        run_id="p3-b-checkpoint", stage=WorkflowStage.SCENARIOS,
        scenarios=[first[0], second[0]],
    )
    restored = HARAState.from_dict(state.to_dict())
    for original, recovered in zip(state.scenarios, restored.scenarios):
        assert recovered.analysis_instance == original.analysis_instance
        assert recovered.semantic_fingerprint == original.semantic_fingerprint
        assert recovered.fact_provenance["relative_distance_m"]["origin"] == "SCENARIO_DEFINED"
        assert recovered.fact_provenance["relative_distance_m"]["applicable_scope"] == (
            original.fact_provenance["relative_distance_m"]["applicable_scope"]
        )


class _Client:
    class Config:
        provider = "test"
        base_url = "local"
        model = "offline"

    config = Config()


class _OfflineAgent:
    prompt_version = "p3-b-test"
    assessment_contract_version = "scenario-causal-assessment-v2"

    def __init__(self):
        self.client = _Client()
        self.candidate_ids: list[str] = []

    def assess(self, malfunction, candidates, project_registry=None):
        self.candidate_ids = [item.scenario_id for item in candidates]
        return [
            ScenarioFeasibilityAssessment(
                malfunction_id=malfunction.malfunction_id,
                scenario_id=candidate.scenario_id,
                physically_feasible=False,
                functionally_relevant=False,
                causally_relevant=False,
                risk_dimensions_changed=[], rationale="offline P3-B wiring test",
                status=ReviewStatus.PENDING,
            )
            for candidate in candidates
        ], {
            "malfunction_id": malfunction.malfunction_id,
            "candidate_count": len(candidates), "retained_count": 0,
            "initial_batch_count": 1, "leaf_batch_count": 1,
            "llm_calls": 0, "leaf_items": [len(candidates)], "leaf_input_chars": [0],
        }


class _ValidatedOfflineAgent:
    prompt_version = "p3-b-validated-offline"
    assessment_contract_version = "scenario-causal-assessment-v2"

    def __init__(self):
        self.client = _Client()

    def assess(self, malfunction, candidates, project_registry=None):
        assessments = [
            ScenarioFeasibilityAgent._parse(
                malfunction,
                {
                    "scenario_id": candidate.scenario_id,
                    "physically_feasible": True,
                    "functionally_relevant": True,
                    "causally_relevant": True,
                    "breakpoint": "NONE",
                    "causal_chain": {
                        "m_to_b": {
                            "claim": "confirmed malfunction changes output",
                            "basis_type": "DIRECT_FACT",
                            "evidence_refs": ["MF.functional_effect"],
                        },
                        "b_to_i": {
                            "claim": "the behavior occurs while the vehicle is active",
                            "basis_type": "DIRECT_FACT",
                            "evidence_refs": ["SCN.vehicle_state"],
                        },
                        "i_to_h": {
                            "claim": "the active vehicle operates at the stated project speed",
                            "basis_type": "DIRECT_FACT",
                            "evidence_refs": ["SCN.ego_speed_kph"],
                        },
                        "h_to_harm": {
                            "claim": "the hazard can expose occupants to impact injury",
                            "basis_type": "DIRECT_FACT",
                            "evidence_refs": ["SCN.harm_mechanism"],
                        },
                    },
                    "risk_dimension_changes": [],
                    "rationale": "existing direct project facts support the causal chain",
                    "hazardous_event": "loss of intended trajectory control while active",
                    "potential_harm": "impact injury",
                    "confidence": 0.8,
                    "status": "PENDING",
                },
                scenario=candidate, project_registry=project_registry,
            )
            for candidate in candidates
        ]
        return assessments, {
            "malfunction_id": malfunction.malfunction_id,
            "candidate_count": len(candidates), "retained_count": len(assessments),
            "initial_batch_count": 1, "leaf_batch_count": 1, "llm_calls": 0,
            "leaf_items": [len(candidates)], "leaf_input_chars": [0],
        }


def test_workflow_uses_per_malfunction_instances_without_provider_calls(method):
    state = HARAState(run_id="p3-b-wiring", stage=WorkflowStage.MALFUNCTIONS)
    agent = _OfflineAgent()

    completed = assess_scenarios(
        state, agent, [_malfunction()], [_parent()], max_workers=1, method=method,
    )

    assert completed.stage is WorkflowStage.SCORING
    assert len(agent.candidate_ids) == 2
    assert all(identity.startswith("SCN-ANALYTICAL-") for identity in agent.candidate_ids)
    assert {
        item["scenario_id"] for item in completed.item_definition["scenario_assessments"]
    } == set(agent.candidate_ids)
    event = next(
        item for item in completed.audit_trail
        if item["event"] == "scenario_feasibility_assessed"
    )
    assert event["analytical_scenario_instantiation"]["instance_count"] == 2


def test_retained_baseline_instance_passes_causal_validation_and_risk_context(
    baseline_method,
):
    malfunction = _baseline_synthetic_malfunction()
    state = HARAState(run_id="p3-b-positive-handoff", stage=WorkflowStage.MALFUNCTIONS)

    completed = assess_scenarios(
        state, _ValidatedOfflineAgent(), [malfunction], [_causal_parent()],
        max_workers=1, method=baseline_method,
    )

    assert completed.stage is WorkflowStage.SCORING
    assert len(completed.scenarios) == 4
    retained = completed.scenarios[0]
    assessment = next(
        item for item in completed.item_definition["scenario_assessments"]
        if item["scenario_id"] == retained.scenario_id
    )
    assert assessment["risk_eligibility_status"] == "ELIGIBLE"
    assert assessment["causal_assessment"]["status"] == "VALIDATED"

    context = HazardousEventRiskContextService(baseline_method).build(
        malfunction_id=malfunction.malfunction_id, scenario_id=retained.scenario_id,
        hazard_node_id="H", scenario=_materialize(retained),
    )
    assert context.scenario_id == retained.scenario_id
    assert context.relative_distance_m.status is RiskContextFactStatus.AVAILABLE
    assert context.relative_distance_m.source_provenance == "SCENARIO_DEFINED"
