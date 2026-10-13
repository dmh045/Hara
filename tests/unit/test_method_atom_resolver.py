from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

from hara_agent.method_sources import YamlBaselineCompiler
from hara_agent.services.analysis import MethodAtomResolver, MethodScenarioCandidateService
from hara_agent.template import TemplateRoleCompiler
from hara_agent.workflow import ReviewArtifactWriter


ROOT = Path(__file__).resolve().parents[2]
MANIFEST = ROOT / "method_assets/fusa_baseline_v1/manifest.yaml"
REPORT_TEMPLATE = ROOT / "references/HARA_Template_AI_20260327.xlsx"


def _method():
    report = TemplateRoleCompiler().compile_method(REPORT_TEMPLATE).report_contract
    return YamlBaselineCompiler().compile(MANIFEST, report_contract=report)


def test_resolver_uses_only_compiled_atom_identity_label_and_explicit_mapping():
    resolver = MethodAtomResolver(_method())

    exact = resolver.resolve("FA005", "EGO_ACTION")
    assert exact.status == "RESOLVED"
    assert exact.reason == "EXACT_ATOM_ID"
    assert exact.filled_dimensions == ("EGO_ACTION", "EGO_DYNAMICS")

    v1 = resolver.resolve("FO010", "WHERE")
    assert v1.status == "RESOLVED"
    assert v1.reason == "EXPLICIT_V1_V2_MAPPING"
    assert v1.canonical_atom_id == "VD006"

    label = resolver.resolve("Motorway / Highway (V1)", "WHERE")
    assert label.status == "RESOLVED"
    assert label.reason == "EXACT_CANONICAL_LABEL"

    unknown = resolver.resolve("parking", "EGO_ACTION")
    assert unknown.status == "PENDING"
    assert unknown.reason == "NO_EXPLICIT_METHOD_ALIAS"


def test_resolver_rejects_ambiguous_exact_alias_without_guessing():
    resolver = MethodAtomResolver(SimpleNamespace(metadata={
        "scenario_atom_catalog": [
            {"atom_id": "A1", "label": "one", "aliases": ["same"], "filled_dimensions": ["OBJECT"]},
            {"atom_id": "A2", "label": "two", "aliases": ["same"], "filled_dimensions": ["OBJECT"]},
        ],
    }))

    result = resolver.resolve("same", "OBJECT")

    assert result.status == "AMBIGUOUS"
    assert result.reason == "AMBIGUOUS_EXPLICIT_ALIAS"
    assert result.candidate_atom_ids == ("A1", "A2")


def test_compound_atom_fills_its_dimensions_and_intersects_speed_range():
    method = _method()
    service = MethodScenarioCandidateService(method)
    action = next(
        item for item in method.scenario_model.dimensions
        if item.canonical_name == "EGO_ACTION"
    )
    dynamics = next(
        item for item in method.scenario_model.dimensions
        if item.canonical_name == "EGO_DYNAMICS"
    )
    bindings = {
        "EGO_ACTION": service._text_binding(action, "FA005"),
        "EGO_DYNAMICS": {
            "project_value": "0..20 km/h",
            "binding_status": "UNRESOLVED",
            "resolution_status": "PENDING",
            "unresolved_reason": "SPEED_CONSTRAINT_NOT_CONCRETE",
            "speed_constraint": {"min_kph": 0, "max_kph": 20, "unit": "km/h"},
            "dimension_source": f"{dynamics.source_ref.sheet}!{dynamics.source_ref.range}",
        },
    }

    service._apply_compound_fills(bindings)

    assert bindings["EGO_DYNAMICS"]["resolution_status"] == "RESOLVED"
    assert bindings["EGO_DYNAMICS"]["filled_by_atom_id"] == "FA005"
    assert bindings["EGO_DYNAMICS"]["speed_constraint"]["min_kph"] == 0
    assert bindings["EGO_DYNAMICS"]["speed_constraint"]["max_kph"] == 15

    incompatible = service.atom_resolver.intersect_speed_range(
        minimum_kph=0, maximum_kph=20, atom_id="FA003",
    )
    assert incompatible.status == "PENDING"
    assert incompatible.reason == "RANGE_INCOMPATIBLE"


def test_speed_interval_requires_one_governed_atom_to_contain_the_full_range():
    resolver = MethodAtomResolver(_method())

    ambiguous = resolver.resolve_speed_range(minimum_kph=0, maximum_kph=20)
    # Explicit speed inequalities in catalog labels are now compiled too;
    # the project interval alone no longer uniquely implies acceleration.
    assert ambiguous.status == "AMBIGUOUS"
    assert len(ambiguous.candidate_atom_ids) > 1
    contained = MethodAtomResolver(SimpleNamespace(metadata={
        "scenario_atom_catalog": [resolver.by_id["FA001"]],
    })).resolve_speed_range(minimum_kph=0, maximum_kph=20)
    assert contained.status == "RESOLVED"
    assert contained.reason == "RANGE_CONTAINMENT"
    assert contained.atom_id == "FA001"
    assert contained.compatible_range_kph == (0, 20)

    split = MethodAtomResolver(SimpleNamespace(metadata={
        "scenario_atom_catalog": [
            {"atom_id": "A", "filled_dimensions": ["EGO_DYNAMICS"], "speed_range_kph": [0, 10]},
            {"atom_id": "B", "filled_dimensions": ["EGO_DYNAMICS"], "speed_range_kph": [10, 20]},
        ],
    })).resolve_speed_range(minimum_kph=0, maximum_kph=20)
    assert split.status == "PENDING"
    assert split.reason == "MULTI_ATOM_RANGE_AMBIGUITY"
    assert split.candidate_atom_ids == ("A", "B")


def test_compiler_excludes_non_normative_scenario_examples_from_runtime_catalog():
    method = _method()

    assert method.scenario_model.constraint_rules == ()
    assert method.metadata["scenario_atom_catalog"]
    assert all(
        item["source_asset"].endswith("raw/vda702_atoms.yaml")
        for item in method.metadata["scenario_atom_catalog"]
    )
    assert all(
        "coupling_examples" not in item["source_asset"]
        and "infeasible_examples" not in item["source_asset"]
        and "fm_scenario_templates" not in item["source_asset"]
        and "avp_low_speed" not in item["source_asset"]
        for item in method.metadata["scenario_atom_catalog"]
    )


def test_binding_gap_artifact_is_deterministic_review_output():
    root = Path("runtime/test-review-atom-gaps")
    shutil.rmtree(root, ignore_errors=True)
    try:
        writer = ReviewArtifactWriter("atom-gaps", root)
        writer.write_scenario_binding_gaps(
            {"dimensions": {"OBJECT": {"resolved": 0, "pending": 1}}},
            [{"dimension": "OBJECT", "reason": "NO_ITEM_FACT"}],
        )
        payload = json.loads(
            (root / "atom-gaps" / "scenario_binding_gaps.json").read_text(
                encoding="utf-8"
            )
        )
        assert payload["artifact_version"] == "scenario-binding-gaps-v1"
        assert payload["gaps"] == [{"dimension": "OBJECT", "reason": "NO_ITEM_FACT"}]
    finally:
        shutil.rmtree(root, ignore_errors=True)
