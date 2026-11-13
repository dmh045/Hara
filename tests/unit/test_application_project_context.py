from __future__ import annotations

from dataclasses import asdict, replace
from pathlib import Path

import pytest

from hara_agent.application import HARAApplication
from hara_agent.config import RunConfig
from hara_agent.contracts import FactType
from hara_agent.models import (
    FactProvenance,
    ItemDefinitionFacts,
    MethodRiskFactBinding,
    ReviewStatus,
    RiskFact,
    ScenarioCandidate,
    SourceRef,
    SpeedEnvelope,
)
from hara_agent.services.analysis import (
    MethodScenarioCandidateService,
    UnresolvedProjectContextError,
)
from hara_agent.services.semantic.scenario_contract import SCENARIO_CONTRACT_VERSION
from hara_agent.template import TemplateRoleCompiler
from hara_agent.workflow import HARAState


ROOT = Path(__file__).resolve().parents[2]
TEMPLATE = ROOT / "references" / "HARA_Template_AI_20260327.xlsx"


def _config(**changes) -> RunConfig:
    base = RunConfig(
        item_path=Path("ItemDef.docx"),
        template_path=Path("template.xlsx"),
        output_path=Path("output.xlsx"),
        run_dir=Path("runtime"),
        operating_mode="parking",
    )
    return replace(base, **changes)


def test_analyze_and_embedded_workflow_share_worker_default_and_allow_override():
    from hara_agent.cli import build_parser
    from hara_agent.workflow.semantic_graph import SemanticWorkflowInputs

    parser = build_parser()
    args = ["analyze", "--item", "input.pdf", "--output", "report.xlsx"]
    assert parser.parse_args(args).max_workers == _config().max_workers == 8
    assert SemanticWorkflowInputs("input.pdf", []).max_workers == 8
    assert parser.parse_args(args + ["--max-workers", "4"]).max_workers == 4
    assert _config(max_workers=4).max_workers == 4


def _facts(*, contextual: bool = True, aggregate: bool = True) -> ItemDefinitionFacts:
    source = SourceRef("item_definition", "ItemDef.docx", "table[14]", "mode speeds")
    return ItemDefinitionFacts(
        system_description="AVP",
        item_boundary="vehicle motion control",
        operating_modes=["parking"],
        odd_locations=["停车场"],
        odd_weather_conditions=["晴天"],
        odd_road_surfaces=["干燥路面"],
        speed_min_kph=0 if aggregate else None,
        speed_max_kph=30 if aggregate else None,
        speed_envelopes=(
            [
                SpeedEnvelope("search", 0, 30, sources=[source]),
                SpeedEnvelope("parking", 0, 5, sources=[source]),
                SpeedEnvelope("control", 0, 7, sources=[source]),
            ]
            if contextual else []
        ),
        sources=[source],
    )


def _state(facts: ItemDefinitionFacts) -> HARAState:
    return HARAState(
        run_id="project-context-test",
        item_definition={"source_id": "ItemDef.docx", "typed": asdict(facts)},
    )


def _service() -> MethodScenarioCandidateService:
    method = TemplateRoleCompiler().compile_method(TEMPLATE, use_manifest=False)
    return MethodScenarioCandidateService(method)


@pytest.mark.parametrize(
    ("operating_mode", "expected_speed"),
    [("parking", 5.0), ("search", 30.0), ("control", 7.0)],
)
def test_application_resolves_structured_mode_without_llm(operating_mode, expected_speed):
    app = HARAApplication(_config(operating_mode=operating_mode), object())

    result = app.resolve_project_speed_context(_state(_facts()))

    assert result is not None
    assert result.resolved_value == expected_speed
    assert result.operating_mode == operating_mode


def test_application_candidate_path_uses_parking_envelope_and_provenance():
    app = HARAApplication(_config(), object())

    candidates, audit = app.prepare_scenario_candidates(_state(_facts()), _service())

    assert candidates
    assert audit["ego_speed_value_kph"] is None
    assert audit["ego_speed_constraint"]["speed_max_kph"] == 5.0
    assert audit["speed_resolution"]["resolution_source"] == "SpeedEnvelope"
    assert audit["combination_strategy"] == (
        "primary_dimension_anchor_with_pending_unbound_dimensions"
    )
    assert audit["dimension_compatibility_status"] == "PENDING_NO_COMPILED_RULES"
    assert audit["function_phase_binding_status"] == "NO_FUNCTION_CONTEXT"
    assert all(item.operating_mode == "parking" for item in candidates)
    assert all("ego_speed_kph" not in item.facts for item in candidates)
    assert all(item.facts["ego_speed_constraint"]["speed_max_kph"] == 5.0 for item in candidates)
    assert all(
        item.fact_provenance["ego_speed_constraint"]["provenance"] == "PROJECT_INPUT"
        for item in candidates
    )
    assert all(
        source.source_type != "domain_profile"
        for item in candidates for source in item.sources
    )


def test_draft_without_source_speed_produces_unbound_candidates_not_default_speed():
    facts = _facts(contextual=False, aggregate=False)
    app = HARAApplication(_config(allow_draft=True), object())
    candidates, audit = app.prepare_scenario_candidates(_state(facts), _service())
    assert candidates
    assert audit["speed_resolution"]["resolved_value"] is None
    assert audit["speed_resolution"]["resolution_status"] == "UNRESOLVED_PROJECT_CONTEXT"
    assert audit["speed_resolution"]["fallback_used"] is False
    assert audit["speed_resolution"]["source_refs"] == ()
    for candidate in candidates:
        assert "ego_speed_kph" not in candidate.facts
        assert candidate.facts["ego_speed_constraint"] == {}
        assert candidate.fact_provenance["ego_speed_constraint"]["approval"] == "PENDING"
        assert candidate.facts["method_scenario_dimensions"]["VEHICLE_SPEED"]["unresolved_reason"] == "NO_ITEM_SPEED_ENVELOPE"
        assert candidate.status is ReviewStatus.PENDING
    with pytest.raises(UnresolvedProjectContextError):
        HARAApplication(_config(), object()).prepare_scenario_candidates(_state(facts), _service())


def test_draft_missing_speed_support_does_not_swallow_mismatched_or_ambiguous_facts():
    app = HARAApplication(_config(allow_draft=True), object())
    for facts in (
        _facts(contextual=False, aggregate=True),
        replace(_facts(), speed_envelopes=[SpeedEnvelope("search", 0, 30)]),
        replace(_facts(), speed_envelopes=[SpeedEnvelope("parking", 0, 5, condition="phase A"), SpeedEnvelope("parking", 0, 7, condition="phase B")]),
    ):
        with pytest.raises(UnresolvedProjectContextError):
            app.prepare_scenario_candidates(_state(facts), _service())


def test_method_speed_bands_have_deterministic_template_boundaries():
    service = _service()
    speed_dimension = next(
        item for item in service.method.scenario_model.dimensions
        if item.canonical_name == "VEHICLE_SPEED"
    )

    matches_at_zero = [
        value for value in speed_dimension.values
        if service._speed_value_matches(value, 0.0)
    ]
    matches_at_fifteen = [
        value for value in speed_dimension.values
        if service._speed_value_matches(value, 15.0)
    ]
    matches_at_thirty = [
        value for value in speed_dimension.values
        if service._speed_value_matches(value, 30.0)
    ]

    assert len(matches_at_zero) == 1 and "Standstill" in matches_at_zero[0]
    assert len(matches_at_fifteen) == 1 and "0 < v" in matches_at_fifteen[0]
    assert len(matches_at_thirty) == 1 and "15 < v" in matches_at_thirty[0]


def test_unmatched_project_dimension_remains_pending_without_synonym_guessing():
    facts = _facts()
    facts.odd_weather_conditions = ["正常天气"]
    app = HARAApplication(_config(), object())

    candidates, audit = app.prepare_scenario_candidates(_state(facts), _service())

    assert audit["unresolved_binding_candidate_count"] == len(candidates)
    assert all(
        item.facts["method_scenario_dimensions"]["WEATHER"]["binding_status"]
        == "UNRESOLVED"
        for item in candidates
    )
    assert all(
        item.fact_provenance["weather_conditions"]["approval"] == "FINALIZED"
        and item.fact_provenance["weather_conditions"]["provenance"] == "PROJECT_INPUT"
        for item in candidates
    )


def test_missing_unrelated_project_field_does_not_taint_exact_field_evidence():
    facts = _facts()
    facts.odd_road_surfaces = []
    facts.status = ReviewStatus.PENDING
    app = HARAApplication(_config(), object())

    candidates, audit = app.prepare_scenario_candidates(_state(facts), _service())

    assert audit["unresolved_binding_candidate_count"] == len(candidates)
    assert all(item.status is ReviewStatus.PENDING for item in candidates)
    assert all(
        item.facts["method_scenario_dimensions"]["ROAD_SURFACE"]["binding_status"]
        == "MISSING"
        for item in candidates
    )
    for candidate in candidates:
        assert candidate.fact_provenance["road_surface_conditions"]["approval"] == "PENDING"
        assert candidate.fact_provenance["operating_scenario"]["approval"] == "FINALIZED"
        assert candidate.fact_provenance["vehicle_state"]["approval"] == "FINALIZED"
        assert candidate.fact_provenance["weather_conditions"]["approval"] == "FINALIZED"
        assert candidate.fact_provenance["operating_mode"]["approval"] == "FINALIZED"


def test_application_candidate_carries_hash_bound_canonical_risk_fact():
    service = _service()
    facts = _facts()
    source = SourceRef(
        "human_confirmation", "review-1", "EXPOSURE", "confirmed F"
    )
    facts.risk_facts = [RiskFact(
        fact_id="RF-EXPOSURE-PARKING",
        parameter="exposure method",
        value="F",
        context={"operating_mode": "parking"},
        source_refs=[source],
        provenance=FactProvenance.PROJECT_INPUT,
        approval=ReviewStatus.FINALIZED,
        produced_by="source_extraction",
    )]
    facts.method_risk_fact_bindings = [MethodRiskFactBinding(
        source_fact_id="RF-EXPOSURE-PARKING",
        target_fact_type=FactType.EXPOSURE.value,
        method_contract_hash=str(service.method.metadata["template_hash"]),
        source_refs=[source],
        provenance=FactProvenance.HUMAN_CONFIRMATION,
        approval=ReviewStatus.FINALIZED,
        binding_method="engineering_review",
    )]

    candidates, audit = HARAApplication(_config(), object()).prepare_scenario_candidates(
        _state(facts), service
    )

    assert audit["bound_risk_fact_count"] == len(candidates)
    assert all(item.facts["exposure_method"] == "F" for item in candidates)
    assert all(
        item.fact_provenance["exposure_method"]["provenance"]
        == "HUMAN_CONFIRMATION"
        for item in candidates
    )


def test_application_fails_closed_without_or_with_unknown_structured_mode():
    with pytest.raises(UnresolvedProjectContextError):
        HARAApplication(_config(operating_mode=None), object()).resolve_project_speed_context(
            _state(_facts())
        )
    with pytest.raises(UnresolvedProjectContextError) as captured:
        HARAApplication(_config(operating_mode="highway"), object()).resolve_project_speed_context(
            _state(_facts())
        )
    assert "available_speed_envelope_modes=['control', 'parking', 'search']" in str(
        captured.value
    )
    assert "declared_operating_modes=['parking']" in str(captured.value)


def test_application_aggregate_fallback_requires_explicit_authorization():
    state = _state(_facts(contextual=False, aggregate=True))
    with pytest.raises(UnresolvedProjectContextError):
        HARAApplication(_config(), object()).resolve_project_speed_context(state)

    result = HARAApplication(
        _config(allow_aggregate_speed_fallback=True), object()
    ).resolve_project_speed_context(state)
    assert result is not None
    assert result.provenance is FactProvenance.DERIVED
    assert result.fallback_used is True


def test_scenario_context_and_provenance_survive_checkpoint_roundtrip():
    candidate = ScenarioCandidate(
        scenario_id="SCN-PARKING",
        operating_scenario="parking",
        situational_description="parking",
        situational_detailing="speed 5 km/h",
        facts={"ego_speed_kph": 5.0, "operating_mode": "parking"},
        operating_mode="parking",
        context_resolution={"ego_speed_kph": {"resolved_value": 5.0}},
        fact_provenance={"ego_speed_kph": {"provenance": "PROJECT_INPUT"}},
        source_scenario_id="VRU_crossing",
        atomic_variant="crossing",
        semantic_fingerprint="abc123",
        scenario_contract_version=SCENARIO_CONTRACT_VERSION,
    )
    state = HARAState(run_id="roundtrip", scenarios=[candidate])

    restored = HARAState.from_dict(state.to_dict()).scenarios[0]

    assert restored.operating_mode == "parking"
    assert restored.context_resolution == candidate.context_resolution
    assert restored.fact_provenance == candidate.fact_provenance
    assert restored.source_scenario_id == "VRU_crossing"
    assert restored.atomic_variant == "crossing"
    assert restored.semantic_fingerprint == "abc123"
