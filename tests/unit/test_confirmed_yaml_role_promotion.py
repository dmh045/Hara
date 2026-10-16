from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from hara_agent.method_sources import YamlBaselineCompileError, YamlBaselineCompiler
from hara_agent.method_sources.yaml_loader import load_yaml_mapping
from hara_agent.models import (
    FunctionDefinition, GuidewordAssessment, MalfunctionCandidate, ReviewStatus,
    SourceRef,
)
from hara_agent.models import ScenarioCandidate
from hara_agent.contracts import FailureModeTaxonomyValue
from hara_agent.services.analysis import (
    FMSelectorSemanticAuditService, FMTemplateSelectorAdapterResolver,
    FMTemplateAmbiguityAuditService,
    FailureModeSelectorResolver,
    ScenarioMethodService,
)
from hara_agent.services.semantic.scenario_batching import build_scenario_user_prompt
from hara_agent.services.semantic.scenario_evidence import build_fact_registry
from hara_agent.template import TemplateRoleCompiler
from hara_agent.workflow.nodes.malfunctions import derive_malfunctions
from hara_agent.workflow.state import HARAState


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "method_assets/fusa_baseline_v1/manifest.yaml"
REPORT_TEMPLATE = ROOT / "references/HARA_Template_AI_20260327.xlsx"


def _method():
    return YamlBaselineCompiler().compile(
        MANIFEST,
        report_contract=TemplateRoleCompiler().compile_method(REPORT_TEMPLATE).report_contract,
    )


def _malfunction(*, description: str, component: str = "", failure_type: str = ""):
    return MalfunctionCandidate(
        malfunction_id="MF-TEST-001", function_id="F-TEST", guideword="No/Loss",
        description=description, functional_effect="confirmed output is unavailable",
        vehicle_level_hazard="vehicle behavior may be unsafe", causal_chain=["M", "B"],
        status=ReviewStatus.FINALIZED, confidence=0.9,
        component_category=component, failure_type=failure_type,
    )


def _method_allowing_template_selector_values():
    """Isolate C9D qualification from the baseline's intentionally separate vocabularies."""
    method = _method()
    scenario_method = method.scenario_model.scenario_method
    taxonomy = scenario_method.failure_mode_selector_taxonomy
    assert taxonomy is not None
    selector_taxonomy = replace(
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
        method,
        scenario_model=replace(
            method.scenario_model,
            scenario_method=replace(
                scenario_method, failure_mode_selector_taxonomy=selector_taxonomy,
            ),
        ),
    )


def test_confirmed_fm_templates_and_domain_slices_are_typed_contract_data():
    method = _method()
    scenario_method = method.scenario_model.scenario_method
    assert scenario_method.fm_template_catalog is not None
    assert len(scenario_method.fm_template_catalog.templates) == 8
    assert scenario_method.fm_template_catalog.templates[0].source_role == "SCENARIO_TEMPLATE_CONSTRAINT"
    assert scenario_method.failure_mode_selector_taxonomy is not None
    assert len(scenario_method.failure_mode_selector_taxonomy.component_categories) == 15
    assert len(scenario_method.failure_mode_selector_taxonomy.failure_types) == 9
    assert scenario_method.fm_template_selector_adapter is not None
    assert len(scenario_method.fm_template_selector_adapter.mappings) == 24
    assert sum(item.runtime_status == "ACTIVE"
               for item in scenario_method.fm_template_selector_adapter.mappings) == 16
    assert sum(item.runtime_status == "UNRESOLVED"
               for item in scenario_method.fm_template_selector_adapter.mappings) == 8
    assert scenario_method.domain_knowledge is not None
    assert len(scenario_method.domain_knowledge.triggering_state_mappings) == 7
    assert scenario_method.domain_knowledge.kinematic_defaults is not None
    assert all(item.target_status == "TARGET_DIMENSION_PENDING"
               for item in scenario_method.domain_knowledge.fallback_dimensions)
    assert scenario_method.domain_knowledge.numeric_sections == (
        "exposure", "controllability", "severity",
    )


def test_template_match_is_source_defined_and_invalid_selector_never_becomes_structured_match():
    service = ScenarioMethodService(_method())
    matched = service.match_fm_template(_malfunction(
        description="parking_brake_release", component="parking_brake",
        failure_type="unintended_deactivation",
    ))
    assert matched.status == "WEAK_MATCH"
    assert matched.template is not None
    assert matched.template.template_id == "FM_TEMPLATE_006"
    assert matched.selector_resolution is not None
    assert matched.selector_resolution.component_resolution_status == "INVALID_TAXONOMY_VALUE"
    assert matched.selector_resolution.failure_type_resolution_status == "INVALID_TAXONOMY_VALUE"
    assert matched.matched_by == ("KEYWORD",)

    missing = service.match_fm_template(_malfunction(description="HMI text truncation"))
    assert missing.status == "NO_MATCH"
    assert missing.template is None


def test_scenario_method_service_does_not_read_raw_yaml_at_runtime():
    source = (ROOT / "src/hara_agent/services/analysis/scenario_method_service.py").read_text(
        encoding="utf-8"
    )
    assert "import yaml" not in source
    assert "load_yaml" not in source


def test_fm_selector_semantic_audit_is_offline_and_flags_early_taxonomy_gap():
    records = {
        "function": [{
            "function_id": "F-TEST", "name": "State transition",
            "output": "System switches operating modes", "description": "",
        }],
        "guideword_assessment": [{
            "function_id": "F-TEST", "guideword_id": "GW-EARLY",
            "guideword": "Early", "disposition": "DOWNSTREAM_CANDIDATE",
        }, {
            "function_id": "F-TEST", "guideword_id": "GW-OTHER",
            "guideword": "As well as/Other than",
            "disposition": "DOWNSTREAM_CANDIDATE",
        }],
        "malfunction": [{
            "malfunction_id": "MF-EARLY", "function_id": "F-TEST",
            "guideword_id": "GW-EARLY", "guideword": "Early",
            "description": "The correct state transition occurs 提前", "functional_effect": "mode switches too early",
            "vehicle_level_hazard": "vehicle behavior is unsafe", "causal_chain": ["M", "B"],
            "status": "FINALIZED", "confidence": 0.9,
            "component_category": "computing", "failure_type": "unintended_activation",
        }, {
            "malfunction_id": "MF-OTHER", "function_id": "F-TEST",
            "guideword_id": "GW-OTHER", "guideword": "As well as/Other than",
            "description": "System emits an 错误 value", "functional_effect": "wrong output",
            "vehicle_level_hazard": "vehicle behavior is unsafe", "causal_chain": ["M", "B"],
            "status": "FINALIZED", "confidence": 0.9,
            "component_category": "computing", "failure_type": "wrong_value",
        }],
    }

    payload = FMSelectorSemanticAuditService(_method()).generate(records)

    assert payload["scope"] == "OFFLINE_REVIEW_ONLY"
    assert payload["llm_calls_added"] == 0
    assert payload["contract_status"]["canonical_failure_type_count"] == 2
    assert payload["early_audit"]["taxonomy_expressiveness_gap_count"] == 1
    assert payload["other_than_audit"]["semantic_distinction_observed"] is False
    assert payload["prompt_inspection"]["classification"] == "MODEL_LEARNED_CORRELATION"
    resolver = (ROOT / "src/hara_agent/services/analysis/failure_mode_selector_resolver.py").read_text(
        encoding="utf-8"
    )
    assert "import yaml" not in resolver
    assert "load_yaml" not in resolver
    assert "fm_template_selector_map.yaml" not in resolver


def test_template_ambiguity_is_fail_closed_while_recording_original_first_precedence():
    method = _method_allowing_template_selector_values()
    catalog = method.scenario_model.scenario_method.fm_template_catalog
    assert catalog is not None
    duplicate = replace(catalog.templates[5], template_id="FM_TEMPLATE_999", original_precedence=99)
    scenario_method = replace(
        method.scenario_model.scenario_method,
        fm_template_catalog=replace(catalog, templates=(catalog.templates[5], duplicate)),
    )
    ambiguous_method = replace(
        method, scenario_model=replace(method.scenario_model, scenario_method=scenario_method),
    )
    result = ScenarioMethodService(ambiguous_method).match_fm_template(_malfunction(
        description="parking_brake_release", component="parking_brake",
        failure_type="unintended_deactivation",
    ))
    assert result.status == "AMBIGUOUS"
    assert result.template is None
    assert result.original_precedence_template_id == "FM_TEMPLATE_006"


def test_fallback_never_overrides_project_fact_or_invents_target_dimension():
    service = ScenarioMethodService(_method())
    assert service.fallback_context_for_missing_project_fact(
        source_dimension="road_state", explicit_project_value="straight road",
    )["status"] == "EXPLICIT_PROJECT_FACT_PRESENT"
    pending = service.fallback_context_for_missing_project_fact(
        source_dimension="surrounding", explicit_project_value="",
    )
    assert pending["status"] == "TARGET_DIMENSION_PENDING"
    assert pending["target_dimension"] == ""


def test_conflicting_confirmed_geometry_fails_closed():
    compiler = YamlBaselineCompiler()
    base = ROOT / "method_assets/fusa_baseline_v1"
    assets = {
        "component_taxonomy": load_yaml_mapping(base / "raw/component_taxonomy.yaml"),
        "failure_type_taxonomy": load_yaml_mapping(base / "raw/failure_type_taxonomy.yaml"),
        "fm_template_selector_map": load_yaml_mapping(base / "normalized/fm_template_selector_map.yaml"),
        "scenario_templates": load_yaml_mapping(base / "raw/fm_scenario_templates.yaml"),
        "scenario_domain": load_yaml_mapping(base / "raw/domain_rules/avp_low_speed.yaml"),
        "coupling_examples": load_yaml_mapping(base / "raw/coupling_examples.yaml"),
        "infeasible_examples": load_yaml_mapping(base / "raw/infeasible_examples.yaml"),
    }
    assets["scenario_templates"]["avp_odd"]["gap_front_m"] = 0.4
    with pytest.raises(YamlBaselineCompileError, match="METHOD_SOURCE_CONFLICT"):
        compiler._compile_scenario_method(
            bundle_hash_value="test", assets=assets,
            paths={
                "component_taxonomy": "raw/component_taxonomy.yaml",
                "failure_type_taxonomy": "raw/failure_type_taxonomy.yaml",
                "fm_template_selector_map": "normalized/fm_template_selector_map.yaml",
                "scenario_templates": "raw/fm_scenario_templates.yaml",
                "scenario_domain": "raw/domain_rules/avp_low_speed.yaml",
                "coupling_examples": "raw/coupling_examples.yaml",
                "infeasible_examples": "raw/infeasible_examples.yaml",
            },
        )


def _scenario(**facts):
    return ScenarioCandidate(
        scenario_id="SCN-TEST", operating_scenario="parking", situational_description="parking",
        situational_detailing="parking", facts=facts, semantic_fingerprint="fixed",
    )


def test_keyword_only_retains_base_candidates_without_instantiating_template_options():
    service = ScenarioMethodService(_method())
    malfunction = _malfunction(description="parking_brake_release")
    result = service.match_fm_template(malfunction)
    assert result.status == "WEAK_MATCH"
    assert result.matched_by == ("KEYWORD",)
    scenario = _scenario()
    contextual, audit = service.instantiate_analytical_candidates(malfunction, [scenario])
    assert contextual == [scenario]
    assert audit["selection_mode"] == "BASE_CANDIDATES_ONLY"
    assert audit["selection_basis"] == "WEAK_KEYWORD_ONLY"


def test_strong_template_options_become_pair_scoped_analytical_instances_not_causal_evidence():
    service = ScenarioMethodService(_method_allowing_template_selector_values())
    malfunction = _malfunction(
        description="parking_brake_release", component="parking_brake",
        failure_type="unintended_deactivation",
    )
    scenario = _scenario()
    contextual, audit = service.instantiate_analytical_candidates(malfunction, [scenario])
    assert audit["selection_mode"] == "STRONG_TEMPLATE_ANALYTICAL_INSTANCES"
    assert len(contextual) == 2
    assert all(item is not scenario for item in contextual)
    assert all(item.analysis_instance["source_template_id"] == "FM_TEMPLATE_006" for item in contextual)
    assert all(item.analysis_instance["origin"] == "SCENARIO_DEFINED" for item in contextual)
    assert all(item.fact_provenance["relative_distance_m"]["approval"] == "PENDING" for item in contextual)
    assert build_fact_registry(malfunction, contextual[0]).resolve("SCN.relative_distance_m") is not None
    prompt = build_scenario_user_prompt(malfunction, contextual)
    assert "analysis_scenario_assumptions" in prompt
    assert "SCENARIO_DEFINED analysis conditions only" in prompt
    assert "method_template_context" not in prompt


def test_strong_template_conflict_fails_closed_without_overwriting_item_fact():
    service = ScenarioMethodService(_method_allowing_template_selector_values())
    malfunction = _malfunction(
        description="parking_brake_release", component="parking_brake",
    )
    scenario = _scenario(object_type="truck", collision_type="front")
    contextual, audit = service.instantiate_analytical_candidates(malfunction, [scenario])
    assert contextual == []
    assert audit["conflict_count"] == 2
    assert scenario.facts["object_type"] == "truck"


def test_selector_taxonomy_resolves_only_exact_or_source_declared_aliases():
    method = _method()
    taxonomy = method.scenario_model.scenario_method.failure_mode_selector_taxonomy
    assert taxonomy is not None
    resolver = FailureModeSelectorResolver(method)
    exact = resolver.resolve(_malfunction(
        description="x", component="computing", failure_type="loss",
    ))
    assert exact.component_resolution_status == "RESOLVED_EXACT"
    assert exact.canonical_component_category == "computing"
    assert exact.failure_type_resolution_status == "RESOLVED_EXACT"

    base = ROOT / "method_assets/fusa_baseline_v1"
    assets = {
        "component_taxonomy": load_yaml_mapping(base / "raw/component_taxonomy.yaml"),
        "failure_type_taxonomy": load_yaml_mapping(base / "raw/failure_type_taxonomy.yaml"),
        "fm_template_selector_map": load_yaml_mapping(base / "normalized/fm_template_selector_map.yaml"),
        "scenario_templates": load_yaml_mapping(base / "raw/fm_scenario_templates.yaml"),
        "scenario_domain": load_yaml_mapping(base / "raw/domain_rules/avp_low_speed.yaml"),
        "coupling_examples": load_yaml_mapping(base / "raw/coupling_examples.yaml"),
        "infeasible_examples": load_yaml_mapping(base / "raw/infeasible_examples.yaml"),
    }
    assets["component_taxonomy"]["categories"][0]["aliases"] = ["longitudinal_alias"]
    compiled = YamlBaselineCompiler()._compile_scenario_method(
        bundle_hash_value="test", assets=assets,
        paths={
            "component_taxonomy": "raw/component_taxonomy.yaml",
            "failure_type_taxonomy": "raw/failure_type_taxonomy.yaml",
            "fm_template_selector_map": "normalized/fm_template_selector_map.yaml",
            "scenario_templates": "raw/fm_scenario_templates.yaml",
            "scenario_domain": "raw/domain_rules/avp_low_speed.yaml",
            "coupling_examples": "raw/coupling_examples.yaml",
            "infeasible_examples": "raw/infeasible_examples.yaml",
        },
    )
    assert compiled.failure_mode_selector_taxonomy is not None
    assert compiled.failure_mode_selector_taxonomy.component_categories[0].aliases == (
        "longitudinal_alias",)
    aliased_method = replace(
        method,
        scenario_model=replace(
            method.scenario_model,
            scenario_method=replace(
                method.scenario_model.scenario_method,
                failure_mode_selector_taxonomy=compiled.failure_mode_selector_taxonomy,
            ),
        ),
    )
    alias = FailureModeSelectorResolver(aliased_method).resolve(_malfunction(
        description="x", component="longitudinal_alias",
    ))
    assert alias.component_resolution_status == "RESOLVED_ALIAS"
    assert alias.canonical_component_category == "actuator_longitudinal"


def test_selector_taxonomy_never_guesses_missing_or_undeclared_values():
    resolver = FailureModeSelectorResolver(_method())
    missing = resolver.resolve(_malfunction(description="x"))
    assert missing.component_resolution_status == "PENDING_MISSING_SOURCE_VALUE"
    assert missing.failure_type_resolution_status == "PENDING_MISSING_SOURCE_VALUE"
    invalid = resolver.resolve(_malfunction(
        description="x", component="longitudinal", failure_type="loss_of_function",
    ))
    assert invalid.component_resolution_status == "INVALID_TAXONOMY_VALUE"
    assert invalid.canonical_component_category == ""
    assert invalid.failure_type_resolution_status == "INVALID_TAXONOMY_VALUE"

    baseline_method = _method()
    ungoverned_method = replace(
        baseline_method,
        scenario_model=replace(
            baseline_method.scenario_model,
            scenario_method=replace(
                baseline_method.scenario_model.scenario_method,
                failure_mode_selector_taxonomy=None,
            ),
        ),
    )
    ungoverned = FailureModeSelectorResolver(ungoverned_method).resolve(_malfunction(
        description="x", component="computing", failure_type="loss",
    ))
    assert ungoverned.component_resolution_status == "PENDING_NO_GOVERNED_MAPPING"
    assert ungoverned.failure_type_resolution_status == "PENDING_NO_GOVERNED_MAPPING"


def test_template_selector_adapter_enables_reviewed_parent_mapping_with_provenance():
    service = ScenarioMethodService(_method())
    result = service.match_fm_template(_malfunction(
        description="unrelated wording",
        component="actuator_propulsion",
        failure_type="excessive",
    ))
    assert result.status == "STRONG_MATCH"
    assert result.template is not None
    assert result.template.template_id == "FM_TEMPLATE_004"
    assert result.matched_by == ("COMPONENT_CATEGORY", "FAILURE_TYPE")
    component = next(
        item for item in result.template_selector_resolution
        if item.selector_type == "COMPONENT_CATEGORY"
    )
    assert component.raw_template_selector == "drive_actuator"
    assert component.canonical_template_selector == "actuator_propulsion"
    assert component.mapping_id == "FMSEL-COMP-004"
    assert component.mapping_semantics == "FUNCTIONAL_PARENT"
    assert component.resolution_status == "RESOLVED_TEMPLATE_ADAPTER"
    assert component.template_rule_ref and component.taxonomy_rule_ref


def test_qualification_keeps_generic_failure_overlap_fail_closed():
    result = ScenarioMethodService(_method()).match_fm_template(_malfunction(
        description="unrelated wording", component="computing", failure_type="loss",
    ))
    assert result.status == "AMBIGUOUS"
    assert result.template is None


def test_qualification_promotes_unique_component_and_failure_support():
    result = ScenarioMethodService(_method()).match_fm_template(_malfunction(
        description="unrelated wording", component="actuator_propulsion", failure_type="excessive",
    ))
    assert result.status == "STRONG_MATCH"
    assert result.template is not None
    assert result.template.template_id == "FM_TEMPLATE_004"
    assert result.qualification_tier == "TIER_1_COMPONENT_AND_FAILURE"


def test_qualification_promotes_unique_component_without_failure_selector():
    result = ScenarioMethodService(_method()).match_fm_template(_malfunction(
        description="unrelated wording", component="actuator_lateral", failure_type="intermittent",
    ))
    assert result.status == "STRONG_MATCH"
    assert result.template is not None
    assert result.template.template_id == "FM_TEMPLATE_005"
    assert result.qualification_tier == "TIER_2_UNIQUE_COMPONENT"


def test_qualification_keeps_conflicting_structured_selectors_ambiguous():
    result = ScenarioMethodService(_method()).match_fm_template(_malfunction(
        description="unrelated wording", component="actuator_lateral", failure_type="degraded",
    ))
    assert result.status == "AMBIGUOUS"
    assert result.template is None
    assert result.qualification_tier == "STRUCTURED_SELECTOR_CONFLICT"


def test_generic_failure_type_does_not_authorize_unrelated_braking_geometry():
    result = ScenarioMethodService(_method()).match_fm_template(_malfunction(
        description="path_deviation", component="computing", failure_type="degraded",
    ))
    assert result.status == "WEAK_MATCH"
    assert result.template is not None
    assert result.template.template_id == "FM_TEMPLATE_001"
    assert result.qualification_tier == "FAILURE_TYPE_WITHOUT_MECHANISM_SUPPORT"
    assert not result.injectable


def test_ambiguity_audit_retains_parent_subtype_and_never_uses_first_match():
    records = {
        "malfunction": [{
            "malfunction_id": "MF-PARENT", "function_id": "F-TEST",
            "guideword_id": "GW-MORE", "guideword": "More",
            "description": "unrelated wording", "functional_effect": "excessive output",
            "vehicle_level_hazard": "vehicle behavior unsafe", "causal_chain": ["M", "B"],
            "status": "FINALIZED", "confidence": 0.9,
            "component_category": "actuator_longitudinal", "failure_type": "excessive",
        }],
    }
    payload = FMTemplateAmbiguityAuditService(_method()).generate(records)
    assert payload["llm_calls_added"] == 0
    assert payload["ambiguity_summary"]["total_ambiguous"] == 1
    row = payload["ambiguous_matches"][0]
    assert row["primary_root_cause"] == "FUNCTIONAL_PARENT_COLLAPSE"
    assert {"brake_actuator", "epb"}.issubset(
        row["original_raw_replay"]["functional_parent_source_subtypes"]
    )
    assert row["original_first_match_template"] == "FM_TEMPLATE_001"
    assert row["qualification_preview"]["status"] == "AMBIGUOUS"


def test_ambiguous_template_selector_never_becomes_structured_support():
    service = ScenarioMethodService(_method())
    adapter = FMTemplateSelectorAdapterResolver(_method())
    unresolved = adapter.resolve("COMPONENT_CATEGORY", "apa_controller")
    assert unresolved.resolution_status == "SELECTOR_MAPPING_AMBIGUOUS"
    assert unresolved.mapping_id == "FMSEL-COMP-012"
    assert unresolved.canonical_template_selector == ""
    result = service.match_fm_template(_malfunction(
        description="path_deviation",
        component="computing",
        failure_type="intermittent",
    ))
    assert result.status == "WEAK_MATCH"
    assert result.matched_by == ("KEYWORD",)
    assert result.template_selector_resolution == ()


def test_new_malfunction_review_output_contains_non_semantic_selector_provenance():
    method = _method()
    source = SourceRef("item_definition", "item.docx", "p1", "function evidence")
    function = FunctionDefinition(
        "F-TEST", "test function", "test output", "test description",
        sources=[source], status=ReviewStatus.FINALIZED,
    )
    assessment = GuidewordAssessment(
        "F-TEST", "No/Loss", True, "credible deviation",
        sources=[source], status=ReviewStatus.FINALIZED, confidence=0.9,
    )
    candidate = _malfunction(description="x", component="computing", failure_type="loss")

    class Agent:
        def generate(self, _function, _assessments):
            return [candidate], {"function_id": "F-TEST", "skipped": False, "llm_call_count": 0}

    class Writer:
        def __init__(self):
            self.rows = []

        def record_malfunction(self, malfunction):
            self.rows.append(malfunction)
            return True

    writer = Writer()
    state = derive_malfunctions(
        HARAState(run_id="selector-taxonomy"), Agent(), [function], [assessment],
        review_artifact_writer=writer, selector_resolver=FailureModeSelectorResolver(method),
    )
    resolution = state.malfunctions[0]["selector_resolution"]
    assert resolution["component_resolution_status"] == "RESOLVED_EXACT"
    assert resolution["failure_type_resolution_status"] == "RESOLVED_EXACT"
    assert writer.rows[0].selector_resolution["canonical_component_category"] == "computing"
    assert writer.rows[0].selector_resolution["canonical_failure_type"] == "loss"
