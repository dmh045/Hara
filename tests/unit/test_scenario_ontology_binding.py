from __future__ import annotations

from pathlib import Path

from hara_agent.method_sources import YamlBaselineCompiler
from hara_agent.models import (
    FunctionDefinition,
    ItemDefinitionFacts,
    ReviewStatus,
    SourceRef,
    SpeedEnvelope,
)
from hara_agent.services.analysis import (
    MethodScenarioCandidateService,
    ProjectFactResolver,
)
from hara_agent.template import TemplateRoleCompiler


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "method_assets/fusa_baseline_v1/manifest.yaml"
REPORT_TEMPLATE = ROOT / "references/HARA_Template_AI_20260327.xlsx"


def _source() -> SourceRef:
    return SourceRef("item_definition", "ItemDef.docx", "p1", "AVP ODD")


def _service() -> MethodScenarioCandidateService:
    report_contract = TemplateRoleCompiler().compile_method(REPORT_TEMPLATE).report_contract
    method = YamlBaselineCompiler().compile(MANIFEST, report_contract=report_contract)
    return MethodScenarioCandidateService(method)


def _facts() -> ItemDefinitionFacts:
    source = _source()
    return ItemDefinitionFacts(
        system_description="AVP",
        item_boundary="automated parking",
        operating_modes=["Active"],
        odd_locations=[
            "indoor parking garage", "outdoor parking lot", "parking aisle",
        ],
        odd_weather_conditions=["clear", "light rain"],
        speed_envelopes=[SpeedEnvelope(
            "Active", 0, 20, sources=[source], status=ReviewStatus.FINALIZED,
        )],
        sources=[source],
        status=ReviewStatus.FINALIZED,
    )


def _function(function_id: str, name: str, *, trigger: str) -> FunctionDefinition:
    return FunctionDefinition(
        function_id,
        name,
        "parking output",
        preconditions=["AVP Active"],
        triggers=[trigger],
        odd_constraints=["parking area"],
        sources=[_source()],
        status=ReviewStatus.FINALIZED,
    )


def test_baseline_uses_primary_dimension_anchor_not_unconstrained_cartesian_product():
    service = _service()
    facts = _facts()
    speed = ProjectFactResolver().resolve_speed_context(facts, "Active")

    candidates, audit = service.generate(
        project_facts=facts,
        operating_mode="active",
        speed_resolution=speed,
        functions=[
            _function("F01", "AVP entry", trigger="enter self-map area"),
            _function("F05", "parking complete", trigger="parking action complete"),
            _function("F08", "park-out", trigger="park-out activated"),
            _function("F10", "park-out end", trigger="target proximity"),
        ],
    )

    # Three location alternatives are preserved; weather alternatives remain
    # explicitly pending instead of producing 3 x 2 unsupported worlds.
    assert len(candidates) == 3
    assert audit["combination_strategy"] == (
        "primary_dimension_anchor_with_pending_unbound_dimensions"
    )
    assert all("ego_speed_kph" not in item.facts for item in candidates)
    assert all(item.facts["ego_speed_constraint"]["speed_max_kph"] == 20 for item in candidates)
    assert all(
        item.context_resolution["dimension_compatibility"]["status"]
        == "PENDING_COMPATIBILITY"
        for item in candidates
    )
    for candidate in candidates:
        bindings = candidate.context_resolution["dimension_bindings"]
        assert bindings["EGO_X_ROAD"]["unresolved_reason"] == "NO_ITEM_FACT"
        assert bindings["TRAFFIC_PATTERN"]["unresolved_reason"] == "NO_ITEM_FACT"
        assert bindings["OBJECT"]["unresolved_reason"] == "NO_ITEM_FACT"
        assert bindings["ROAD"]["unresolved_reason"] == "AMBIGUOUS_BINDING"
        assert bindings["EGO_DYNAMICS"]["resolution_status"] == "AMBIGUOUS"
        assert bindings["EGO_DYNAMICS"]["unresolved_reason"] == "MULTI_ATOM_RANGE_AMBIGUITY"
        assert not bindings["EGO_DYNAMICS"]["atom_id"]


def test_function_context_is_bound_without_turning_it_into_method_dimensions():
    service = _service()
    facts = _facts()
    speed = ProjectFactResolver().resolve_speed_context(facts, "Active")
    functions = [
        _function("F01", "AVP entry", trigger="enter self-map area"),
        _function("F05", "parking complete", trigger="parking action complete"),
        _function("F08", "park-out", trigger="park-out activated"),
        _function("F10", "park-out end", trigger="target proximity"),
    ]

    candidates, _ = service.generate(
        project_facts=facts,
        operating_mode="active",
        speed_resolution=speed,
        functions=functions,
    )

    binding = candidates[0].context_resolution["function_phase_binding"]
    assert binding["status"] == "BOUND_FUNCTION_CONTEXTS"
    assert binding["bindings"]["F01"]["trigger_conditions"] == ["enter self-map area"]
    assert binding["bindings"]["F05"]["trigger_conditions"] == ["parking action complete"]
    assert binding["bindings"]["F08"]["trigger_conditions"] == ["park-out activated"]
    assert binding["bindings"]["F10"]["trigger_conditions"] == ["target proximity"]
    assert set(binding["bindings"]["F08"]).isdisjoint({"OBJECT", "TRAFFIC_PATTERN"})
