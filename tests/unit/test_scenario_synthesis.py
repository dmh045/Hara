from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import itertools
import json
from pathlib import Path

import pytest

from hara_agent.infrastructure.llm.protocol import LLMResponse
from hara_agent.infrastructure.llm.provider_budget import ProviderAttemptBudgetExceeded
from hara_agent.contracts import (
    CandidateOrigin, CoverageLabel, ScenarioSynthesisAssessment,
    SynthesisValidationStatus,
)
from hara_agent.method_sources import YamlBaselineCompiler
from hara_agent.models import (
    MalfunctionCandidate, ReviewStatus, ScenarioCandidate,
    ScenarioFeasibilityAssessment, SourceRef,
)
from hara_agent.services.analysis import (
    AnalyticalPhysicsInstantiationService, ConstrainedScenarioSynthesisService,
    RiskScoreabilityService, ScenarioSynthesisValidationError,
)
from hara_agent.services.semantic.scenario_synthesis_agent import (
    BoundedScenarioSynthesisAgent,
)
from hara_agent.services.semantic.scenario_batching import (
    select_scenario_causal_evidence,
)
from hara_agent.workflow.scenario_causal_revalidation import (
    ScenarioCausalRevalidationRunner,
)
from hara_agent.workflow.checkpoints import CheckpointRepository
from hara_agent.workflow.state import HARAState, WorkflowStage
from hara_agent.template import TemplateRoleCompiler
from hara_agent.workflow.scenario_synthesis import ScenarioSynthesisRunner


ROOT = Path(__file__).resolve().parents[2]


def test_causal_budget_exhaustion_does_not_become_deferred(tmp_path, method, monkeypatch):
    import hara_agent.workflow.scenario_causal_revalidation as causal_module

    child = replace(
        _parent(), scenario_id="SCN-CHILD", source_scenario_id="SCN-PARENT",
        analysis_instance={"validation_status": "VALIDATED", "malfunction_id": "MF-1"},
    )
    run_dir = tmp_path / "agent"
    review_root = tmp_path / "review"
    CheckpointRepository(run_dir).save(HARAState(
        run_id="source", stage=WorkflowStage.COMPLETE,
        item_definition={"typed": {
            "system_description": "parking automation",
            "item_boundary": "vehicle motion controller",
            "sources": [{"source_type": "item_definition", "source_id": "item.docx"}],
        }},
        scenarios=[child], malfunctions=[{**_malfunction(), "causal_chain": ["M", "H"]}],
    ))
    queue_path = review_root / "source" / "causal_delta_queue.json"
    queue_path.parent.mkdir(parents=True)
    queue_path.write_text(json.dumps({"records": [{
        "child_scenario_id": "SCN-CHILD",
        "status": "CAUSAL_REVALIDATION_REQUIRED",
    }]}), encoding="utf-8")

    class BudgetAgent:
        def __init__(self, *args, **kwargs):
            pass

        def assess(self, *args):
            raise ProviderAttemptBudgetExceeded(
                limit=2, attempts=2, ledger_path=tmp_path / "attempts.jsonl",
            )

    monkeypatch.setattr(causal_module, "ScenarioFeasibilityAgent", BudgetAgent)
    runner = ScenarioCausalRevalidationRunner(
        method=method, client=object(), run_dir=run_dir, review_root=review_root,
    )
    with pytest.raises(ProviderAttemptBudgetExceeded):
        runner.run(source_run_id="source", target_run_id="causal", max_workers=1)
    assert not CheckpointRepository(run_dir).path_for("causal").exists()


def test_causal_resume_recovers_only_complete_validated_malfunction_audits(tmp_path):
    assessment = ScenarioFeasibilityAssessment(
        malfunction_id="MF-1", scenario_id="SCN-PARENT",
        physically_feasible=True, functionally_relevant=True,
        causally_relevant=True, risk_dimensions_changed=["operating_mode"],
        rationale="source-linked causal chain remains valid",
        hazardous_event="bounded hazardous event", status=ReviewStatus.FINALIZED,
        confidence=0.9,
    )
    trace = tmp_path / "provider.json"
    trace.write_text(json.dumps({
        "artifact_version": "scenario-causal-revalidation-provider-trace-v1",
        "source_run_id": "source", "run_id": "target",
        "completed_malfunctions": 1, "target_malfunctions": 1,
        "audits": [{
            "malfunction_id": "MF-1",
            "item_salvage_audit": [{
                "scenario_id": "SCN-PARENT",
                "parsed_assessment": assessment.to_dict(),
            }],
        }],
    }), encoding="utf-8")

    recovered = ScenarioCausalRevalidationRunner._recover_completed(
        trace, source_run_id="source", target_run_id="target",
        candidates_by_malfunction={"MF-1": [_parent()]},
    )

    assert list(recovered) == ["MF-1"]
    assert recovered["MF-1"][0][0].scenario_id == "SCN-PARENT"

    payload = json.loads(trace.read_text(encoding="utf-8"))
    payload["audits"][0]["item_salvage_audit"] = []
    trace.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="incomplete"):
        ScenarioCausalRevalidationRunner._recover_completed(
            trace, source_run_id="source", target_run_id="target",
            candidates_by_malfunction={"MF-1": [_parent()]},
        )


@pytest.fixture(scope="module")
def method():
    report = TemplateRoleCompiler().compile_method(
        ROOT / "references/HARA_Template_AI_20260327.xlsx"
    ).report_contract
    return YamlBaselineCompiler().compile(
        ROOT / "method_assets/fusa_baseline_v1/manifest.yaml",
        report_contract=report,
    )


def _parent() -> ScenarioCandidate:
    return ScenarioCandidate(
        scenario_id="SCN-PARENT", operating_scenario="室内停车场",
        situational_description="AVP在停车场低速泊入，附近有行人和车辆",
        situational_detailing="AVP在停车场低速泊入，附近有行人和车辆",
        operating_mode="active",
        facts={
            "ego_speed_constraint": {"min_kph": 0.0, "max_kph": 20.0},
            "object_type": "pedestrian", "road_user_type": "PEDESTRIAN",
            "method_scenario_dimensions": {
                "EGO_DYNAMICS": {
                    "resolution_status": "RESOLVED", "atom_id": "FA001",
                    "canonical_atom_id": "FA001", "project_value": "0..20 km/h",
                },
            },
            "scenario_atom_ids": ["FA001"],
        },
        semantic_fingerprint="parent", status=ReviewStatus.FINALIZED,
    )


def _malfunction() -> dict:
    return {
        "malfunction_id": "MF-1", "function_id": "F-1", "guideword": "No/Loss",
        "description": "泊入时行人识别丢失", "functional_effect": "车辆未对行人制动",
        "vehicle_level_hazard": "停车场内与行人碰撞",
        "component_category": "perception", "failure_type": "loss",
    }


def _assessment() -> dict:
    return {
        "malfunction_id": "MF-1", "scenario_id": "SCN-PARENT",
        "hazardous_event_id": "HE-1", "hazardous_event": "停车场内碰撞行人",
        "causal_assessment": {
            "status": "VALIDATED", "hazardous_event": "停车场内碰撞行人",
            "causal_chain": ["M", "B", "I", "H"],
            "risk_dimension_changes": [],
        },
    }


def _project() -> dict:
    return {
        "odd_locations": ["室外停车场", "室内停车场"],
        "odd_road_types": ["停车场道路", "停车位"],
        "odd_weather_conditions": ["晴朗", "小雨"],
        "odd_road_surfaces": [
            "停车场路面（支持上坡15%、下坡15%，超出JGJ100车库标准坡度功能退出）",
        ],
        "speed_min_kph": 0.0, "speed_max_kph": 20.0,
    }


def _input(method):
    return ConstrainedScenarioSynthesisService(method).build_input(
        malfunction=_malfunction(), parent=_parent(), assessment=_assessment(),
        project_context=_project(),
    )


def _valid_payload(synthesis_input) -> dict:
    service = ConstrainedScenarioSynthesisService.__new__(
        ConstrainedScenarioSynthesisService
    )
    # The caller's real service owns validation; this helper only enumerates
    # candidate-set products and preserves compound memberships.
    service.dimensions = tuple(
        item.dimension for item in synthesis_input.dimension_candidate_sets
    )
    choices = []
    for candidate_set in synthesis_input.dimension_candidate_sets:
        if candidate_set.locked_atom_ids:
            choices.append([candidate_set.locked_atom_ids])
        elif candidate_set.applicability.status.value == "NOT_APPLICABLE":
            choices.append([()])
        elif candidate_set.candidates:
            values = [(item.atom_id,) for item in candidate_set.candidates[:3]]
            if candidate_set.applicability.status.value == "OPTIONAL":
                values.append(())
            choices.append(values)
        else:
            choices.append([()])
    candidate_by_id = {
        candidate.atom_id: candidate
        for candidate_set in synthesis_input.dimension_candidate_sets
        for candidate in candidate_set.candidates
    }
    valid = []
    for product in itertools.product(*choices):
        selected = dict(zip(service.dimensions, product, strict=True))
        consistent = True
        for atom_ids in selected.values():
            for atom_id in atom_ids:
                candidate = candidate_by_id[atom_id]
                if any(
                    selected.get(dimension) != (atom_id,)
                    for dimension in candidate.dimensions
                    if dimension in selected
                ):
                    consistent = False
        if consistent:
            valid.append(selected)
    primary = synthesis_input.coverage_plan.primary_variation_dimensions
    selected_variants = []
    primary_signatures = set()
    for selected in valid:
        signature = tuple(selected.get(dimension, ()) for dimension in primary)
        if selected_variants and signature in primary_signatures:
            continue
        selected_variants.append(selected)
        primary_signatures.add(signature)
        if len(selected_variants) == synthesis_input.coverage_plan.desired_variant_count:
            break
    assert len(selected_variants) == synthesis_input.coverage_plan.desired_variant_count
    return {
        "variants": [{
            "coverage_label": intent["coverage_label"],
            "selected_atom_ids": sorted({
                atom_id for atom_ids in selected.values() for atom_id in atom_ids
            }),
            "semantic_rationale": "Uses the supplied parking, pedestrian and ODD context.",
            "context_refs": [
                "PROJECT.ODD", "HE.hazardous_event", "PARENT.scenario",
                "METHOD.scenario_atom_catalog",
            ],
        } for intent, selected in zip(
            synthesis_input.coverage_plan.variant_intents,
            selected_variants, strict=True,
        )],
    }


def test_candidate_generation_reads_method_dimensions_and_locks_exact_atom(method):
    synthesis_input = _input(method)
    assert tuple(item.dimension for item in synthesis_input.dimension_candidate_sets) == (
        "WHERE", "ROAD", "EGO_ACTION", "EGO_X_ROAD", "TRAFFIC_PATTERN",
        "EGO_DYNAMICS", "OBJECT",
    )
    by_dimension = {item.dimension: item for item in synthesis_input.dimension_candidate_sets}
    assert by_dimension["EGO_DYNAMICS"].locked_atom_ids == ("FA001",)
    assert len(by_dimension["WHERE"].candidates) < by_dimension["WHERE"].catalog_size
    assert all("motorway" not in item.label.casefold() for item in by_dimension["WHERE"].candidates)
    assert "PH014" not in {
        item.atom_id for item in by_dimension["EGO_X_ROAD"].candidates
    }
    assert "FB007" not in {
        item.atom_id for item in by_dimension["ROAD"].candidates
    }


def test_zero_where_match_retains_bounded_unknown_candidates(method):
    service = ConstrainedScenarioSynthesisService(method)
    empty = service.build_input(
        malfunction={
            **_malfunction(), "description": "opaque", "functional_effect": "opaque",
            "vehicle_level_hazard": "opaque",
        },
        parent=ScenarioCandidate(
            scenario_id="SCN-OPAQUE", operating_scenario="opaque",
            situational_description="opaque", situational_detailing="opaque",
            facts={}, status=ReviewStatus.PENDING,
        ),
        assessment={**_assessment(), "scenario_id": "SCN-OPAQUE", "hazardous_event": "opaque"},
        project_context={"odd_locations": ["不可映射语义"]},
    )
    where = next(item for item in empty.dimension_candidate_sets if item.dimension == "WHERE")
    assert where.generation_status == "CANDIDATES_AVAILABLE"
    assert where.candidates
    assert all(item.semantic_compatibility.value == "UNKNOWN" for item in where.candidates)
    assert all(item.ranking_scores["source_evidence_tier"] <= 1 for item in where.candidates)
    assert where.catalog_size > 0


def test_provider_output_rejects_invented_atom(method):
    service = ConstrainedScenarioSynthesisService(method)
    synthesis_input = _input(method)
    payload = _valid_payload(synthesis_input)
    payload["variants"][0]["selected_atom_ids"] = ["INVENTED"]
    with pytest.raises(ScenarioSynthesisValidationError) as caught:
        service.validate_provider_payload(synthesis_input, payload)
    assert caught.value.code == "INVENTED_LOGICAL_ATOM_ID:INVENTED"


def test_provider_output_rejects_catalog_atom_outside_logical_registry(method):
    service = ConstrainedScenarioSynthesisService(method)
    synthesis_input = _input(method)
    supplied = set(service.logical_candidate_registry(synthesis_input))
    outside_id = next(
        atom_id for atom_id in service.by_id if atom_id not in supplied
    )
    outside = _valid_payload(synthesis_input)
    outside["variants"][0]["selected_atom_ids"] = [outside_id]
    with pytest.raises(ScenarioSynthesisValidationError) as caught:
        service.validate_provider_payload(synthesis_input, outside)
    assert caught.value.code == f"LOGICAL_ATOM_OUTSIDE_REGISTRY:{outside_id}"


def test_provider_output_rejects_missing_required_field(method):
    service = ConstrainedScenarioSynthesisService(method)
    synthesis_input = _input(method)
    payload = _valid_payload(synthesis_input)
    del payload["variants"][0]["semantic_rationale"]
    with pytest.raises(ScenarioSynthesisValidationError) as caught:
        service.validate_provider_payload(synthesis_input, payload)
    assert caught.value.code == "SCHEMA_VARIANT_FIELDS"


class _Client:
    class Config:
        model = "configured-model"
        scenario_thinking = "disabled"

    config = Config()

    def __init__(self, responses):
        self.responses = list(responses)

    def complete_json(self, request):
        data = self.responses.pop(0)
        return LLMResponse(
            data=data, model="resolved-model", request_id=f"REQ-{len(self.responses)}",
            usage={
                "finish_reason": "stop", "reasoning_characters": 0,
                "latency_seconds": 0.01,
            },
        )


def test_bounded_provider_selection_allows_one_repair(method):
    service = ConstrainedScenarioSynthesisService(method)
    synthesis_input = _input(method)
    invalid = deepcopy(_valid_payload(synthesis_input))
    invalid["variants"][0]["selected_atom_ids"] = ["INVENTED"]
    agent = BoundedScenarioSynthesisAgent(
        _Client([invalid, _valid_payload(synthesis_input)]), service,
    )
    assessments, trace = agent.select(synthesis_input)
    assert len(assessments) == synthesis_input.coverage_plan.desired_variant_count
    assert trace["status"] == "PASS"
    assert trace["repairs"] == 1
    assert len(trace["calls"]) == 2
    assert trace["calls"][0]["raw_logical_selection"][0]["selected_atom_ids"] == [
        "INVENTED"
    ]


def test_bounded_provider_selection_second_failure_stays_pending(method):
    service = ConstrainedScenarioSynthesisService(method)
    synthesis_input = _input(method)
    invalid = deepcopy(_valid_payload(synthesis_input))
    invalid["variants"][0]["selected_atom_ids"] = ["INVENTED"]
    agent = BoundedScenarioSynthesisAgent(_Client([invalid, invalid]), service)
    assessments, trace = agent.select(synthesis_input)
    assert assessments == ()
    assert trace["status"] == "PENDING_SCENARIO_SYNTHESIS"
    assert trace["repairs"] == 1
    assert len(trace["calls"]) == 2


def test_provider_budget_exhaustion_stops_synthesis_without_repair(method, tmp_path):
    class BudgetClient(_Client):
        def __init__(self):
            super().__init__([])
            self.calls = 0

        def complete_json(self, request):
            self.calls += 1
            raise ProviderAttemptBudgetExceeded(
                limit=2, attempts=2, ledger_path=tmp_path / "attempts.jsonl",
            )

    client = BudgetClient()
    agent = BoundedScenarioSynthesisAgent(
        client, ConstrainedScenarioSynthesisService(method),
    )
    with pytest.raises(ProviderAttemptBudgetExceeded):
        agent.select(_input(method))
    assert client.calls == 1


def test_odd_speed_incompatible_atom_is_pruned(method):
    service = ConstrainedScenarioSynthesisService(method)
    assert service._speed_compatible(
        {"speed_range_kph": [130, 200]}, (0, 20),
    ) is False
    dynamics = next(
        item for item in _input(method).dimension_candidate_sets
        if item.dimension == "EGO_DYNAMICS"
    )
    assert all(
        service._speed_compatible(service.by_id[item.atom_id], (0, 20))
        for item in dynamics.candidates
    )


def _compound_input(method):
    service = ConstrainedScenarioSynthesisService(method)
    synthesis_input = _input(method)
    compound = service._candidate(
        service.by_id["FA005"], origin=CandidateOrigin.METHOD_TEMPLATE,
        refs=("METHOD.scenario_atom_catalog",),
        reason="Test fixture uses an explicit source-defined compound.",
    )
    independent = service._candidate(
        service.by_id["FA001"], origin=CandidateOrigin.DIRECT_PROJECT_BINDING,
        refs=("METHOD.scenario_atom_catalog",),
        reason="Test fixture uses an independent dynamics atom.",
    )
    candidate_sets = []
    for item in synthesis_input.dimension_candidate_sets:
        if item.dimension == "EGO_ACTION":
            item = replace(item, candidates=(compound,), locked_atom_ids=())
        elif item.dimension == "EGO_DYNAMICS":
            item = replace(
                item, candidates=(compound, independent), locked_atom_ids=(),
                generation_status="CANDIDATES_AVAILABLE",
            )
        candidate_sets.append(item)
    return service, replace(
        synthesis_input, dimension_candidate_sets=tuple(candidate_sets),
    )


def test_compound_atom_conflict_rejected(method):
    service, synthesis_input = _compound_input(method)
    payload = _valid_payload(synthesis_input)
    payload["variants"][0]["selected_atom_ids"].append("FA001")
    with pytest.raises(ScenarioSynthesisValidationError) as caught:
        service.validate_provider_payload(synthesis_input, payload)
    assert caught.value.code.startswith("COMPOUND_ATOM_CONFLICT:")


def test_valid_compound_atom_fills_all_declared_dimensions(method):
    service, synthesis_input = _compound_input(method)
    assessment = service.validate_provider_payload(
        synthesis_input, _valid_payload(synthesis_input),
    )[0]
    assert assessment.selected_atoms["EGO_ACTION"] == ("FA005",)
    assert assessment.selected_atoms["EGO_DYNAMICS"] == ("FA005",)


def test_provider_schema_and_payload_expose_each_logical_atom_once(method):
    service, synthesis_input = _compound_input(method)
    agent = BoundedScenarioSynthesisAgent(None, service)
    schema = agent._schema(synthesis_input)
    variant = schema["properties"]["variants"]["items"]
    assert "selected_atom_ids" in variant["properties"]
    assert "selected_atoms" not in variant["properties"]
    logical = agent._user_payload(synthesis_input)[
        "logical_atom_candidates"
    ]
    assert [item["atom_id"] for item in logical].count("FA005") == 1
    compound = next(item for item in logical if item["atom_id"] == "FA005")
    assert compound["filled_dimensions"] == ["EGO_ACTION", "EGO_DYNAMICS"]
    assert "FA001" in compound["conflicts_with_atom_ids"]


def test_compound_repair_exposes_exact_forbidden_pair_and_preferred_atom(method):
    service, synthesis_input = _compound_input(method)
    agent = BoundedScenarioSynthesisAgent(None, service)
    request = agent._request(
        synthesis_input,
        repair_error="COMPOUND_ATOM_CONFLICT:EGO_DYNAMICS:FA001,FA005",
    )
    payload = json.loads(request.user_prompt)
    assert payload["repair_constraints"]["forbidden_together"] == [
        "FA001", "FA005",
    ]
    assert "containing FA005" in payload["repair_constraints"]["correction"]
    assert "without FA001" in payload["repair_constraints"]["correction"]


def test_missing_dimension_repair_requires_complete_exact_cover_example(method):
    service, synthesis_input = _compound_input(method)
    agent = BoundedScenarioSynthesisAgent(None, service)
    request = agent._request(
        synthesis_input,
        repair_error="REQUIRED_DIMENSION_EMPTY:EGO_DYNAMICS",
    )
    payload = json.loads(request.user_prompt)
    correction = payload["repair_constraints"]["correction"]
    assert "whole, unchanged" in correction
    assert "coverage_valid_logical_atom_set_assignments" in correction


def test_whole_logical_assignments_satisfy_coverage_diversity(method):
    service = ConstrainedScenarioSynthesisService(method)
    synthesis_input = _input(method)
    assignments = service.logical_selection_assignments(synthesis_input)
    assert assignments
    for assignment in assignments:
        selections = [
            service.expand_logical_atom_ids(synthesis_input, bundle)
            for bundle in assignment
        ]
        assert not service.diversity_validator.reasons(
            synthesis_input.coverage_plan, selections,
        )
    payload = BoundedScenarioSynthesisAgent(None, service)._user_payload(
        synthesis_input
    )
    assert payload["coverage_valid_logical_atom_set_assignments"]


def test_historical_smoke_variant_count_is_a_scoped_non_mutating_override(method):
    runner = ScenarioSynthesisRunner(method=method, client=None)
    synthesis_input = _input(method)
    synthesis_input = replace(
        synthesis_input,
        coverage_plan=replace(
            synthesis_input.coverage_plan,
            desired_variant_count=3,
            variant_intents=(
                {"coverage_label": "typical"},
                {"coverage_label": "boundary"},
                {"coverage_label": "extreme"},
            ),
        ),
    )
    assert synthesis_input.coverage_plan.desired_variant_count == 3
    identity = {
        "malfunction_id": synthesis_input.malfunction_id,
        "parent_scenario_id": synthesis_input.parent_scenario_id,
        "hazardous_event_id": synthesis_input.hazardous_event_id,
        "requested_variant_count": 2,
    }
    overridden = runner._apply_smoke_plan_overrides([synthesis_input], [identity])[0]
    assert overridden.coverage_plan.desired_variant_count == 2
    assert len(overridden.coverage_plan.variant_intents) == 2
    assert synthesis_input.coverage_plan.desired_variant_count == 3


def test_overlapping_logical_atoms_fail_as_compound_conflict(method):
    service, synthesis_input = _compound_input(method)
    with pytest.raises(ScenarioSynthesisValidationError) as caught:
        service.expand_logical_atom_ids(synthesis_input, ["FA005", "FA001"])
    assert caught.value.code.startswith("COMPOUND_ATOM_CONFLICT:")
    assert caught.value.details


def test_parking_odd_removes_expressway_compound_globally(method):
    synthesis_input = _input(method)
    for candidate_set in synthesis_input.dimension_candidate_sets:
        assert "CN_peds_across_expressway" not in {
            item.atom_id for item in candidate_set.candidates
        }


def test_parent_expressway_context_cannot_widen_parking_project_odd(method):
    service = ConstrainedScenarioSynthesisService(method)
    parent = replace(
        _parent(), operating_scenario="expressway",
        facts={**_parent().facts, "operating_scenario": "expressway"},
    )
    synthesis_input = service.build_input(
        malfunction=_malfunction(), parent=parent, assessment=_assessment(),
        project_context=_project(),
    )
    query = synthesis_input.structured_semantic_query
    assert query["project_location_categories"] == ["LOCATION_PARKING"]
    assert query["parent_location_categories"] == ["LOCATION_EXPRESSWAY"]
    assert "CN_peds_across_expressway" not in service.logical_candidate_registry(
        synthesis_input
    )


def test_candidate_order_does_not_depend_on_e_rank_metadata(method):
    baseline_method = deepcopy(method)
    changed_method = deepcopy(method)
    for index, atom in enumerate(changed_method.metadata["scenario_atom_catalog"]):
        atom["e_rank"] = "E4" if index % 2 else "E0"
    before = _input(baseline_method)
    after = _input(changed_method)
    assert {
        item.dimension: [candidate.atom_id for candidate in item.candidates]
        for item in before.dimension_candidate_sets
    } == {
        item.dimension: [candidate.atom_id for candidate in item.candidates]
        for item in after.dimension_candidate_sets
    }


def test_materialization_is_child_isolated_and_never_mutates_parent(method):
    service = ConstrainedScenarioSynthesisService(method)
    parent = _parent()
    before = parent.to_dict()
    synthesis_input = _input(method)
    assessment = service.validate_provider_payload(
        synthesis_input, _valid_payload(synthesis_input),
    )[0]
    child, instance = service.materialize(
        synthesis_input=synthesis_input, assessment=assessment,
        parent=parent, provider_evidence={"request_id": "REQ"},
    )
    assert parent.to_dict() == before
    assert child.scenario_id.startswith("SCN-ANALYTICAL-")
    assert child.source_scenario_id == parent.scenario_id
    assert instance.parent_scenario_id == parent.scenario_id
    assert child.facts["ego_speed_constraint"] == parent.facts["ego_speed_constraint"]
    assert "ego_speed_kph" not in child.facts
    scope = child.fact_provenance["scenario_atom_ids"]["applicable_scope"]
    assert scope["malfunction_id"] == "MF-1"
    assert scope["scenario_id"] == child.scenario_id


def test_materialization_promotes_exact_fm_option_physics_with_method_authority(method):
    service = ConstrainedScenarioSynthesisService(method)
    parent = _parent()
    synthesis_input = replace(_input(method), fm_scenario_template={
        "template_id": "FM_TEMPLATE_TEST",
        "source_governed_constraints": [],
        "active_option": {
            "source_option_id": "FM_TEMPLATE_TEST:OPTION:1",
            "label": "front pedestrian",
            "obj_type": "pedestrian", "obj_position": "front",
            "obj_distance_m": 0.3, "obj_v_kph": 0.0,
            "collision_type": "front",
            "source": {
                "source_asset": "raw/fm_scenario_templates.yaml",
                "source_rule": "yaml!templates[test].required_scenarios[0]",
                "source_hash": "abc",
                "source_excerpt": "obj_type: pedestrian; obj_v_kph: 0.0",
            },
        },
    })
    assessment = service.validate_provider_payload(
        synthesis_input, _valid_payload(synthesis_input),
    )[0]

    child, _ = service.materialize(
        synthesis_input=synthesis_input, assessment=assessment,
        parent=parent, provider_evidence={"request_id": "REQ-PHYSICS"},
    )

    assert child.facts["relative_distance_m"] == 0.3
    assert child.facts["object_speed_kph"] == 0.0
    assert child.facts["road_user_type"] == "PEDESTRIAN"
    assert child.facts["collision_type"] == "FRONTAL"
    assert child.facts["object_longitudinal_direction"] == "STATIONARY"
    assert "ego_speed_kph" not in child.facts
    for field in (
        "relative_distance_m", "object_speed_kph", "road_user_type", "collision_type",
    ):
        assert child.fact_provenance[field]["origin"] == "METHOD_DEFINED"
        assert child.fact_provenance[field]["approval"] == "FINALIZED"
        assert child.fact_provenance[field]["source_refs"][0]["location"].startswith("yaml!")
    projection = child.analysis_instance["method_physical_projection"]
    assert projection["option_resolution"] == "EXACT_ACTIVE_OPTION"
    assert not projection["conflicts"]


def test_template_projection_keeps_unknown_motion_unknown_and_retains_conflicts(method):
    service = ConstrainedScenarioSynthesisService(method)
    parent = replace(
        _parent(), facts={**_parent().facts, "object_speed_kph": 7.0},
    )
    synthesis_input = replace(_input(method), fm_scenario_template={
        "template_id": "FM_TEMPLATE_TEST",
        "source_governed_constraints": [],
        "active_option": {
            "source_option_id": "FM_TEMPLATE_TEST:OPTION:1",
            "label": "moving pedestrian", "obj_type": "pedestrian",
            "obj_position": "front", "obj_distance_m": 1.0,
            "obj_v_kph": 5.0, "collision_type": "side",
            "source": {
                "source_rule": "yaml!templates[test].required_scenarios[0]",
                "source_excerpt": "obj_v_kph: 5.0; collision_type: side",
            },
        },
    })
    assessment = service.validate_provider_payload(
        synthesis_input, _valid_payload(synthesis_input),
    )[0]

    child, _ = service.materialize(
        synthesis_input=synthesis_input, assessment=assessment,
        parent=parent, provider_evidence={"request_id": "REQ-CONFLICT"},
    )

    assert child.facts["object_speed_kph"] == 7.0
    assert "object_longitudinal_direction" not in child.facts
    assert child.facts["collision_type"] == "SIDE"
    conflicts = child.analysis_instance["method_physical_projection"]["conflicts"]
    assert conflicts == [{
        "field": "object_speed_kph", "existing_value": 7.0, "method_value": 5.0,
    }]


def test_template_option_matching_requires_one_unique_structured_match(method):
    service = ConstrainedScenarioSynthesisService(method)
    base_input = _input(method)
    object_set = next(
        item for item in base_input.dimension_candidate_sets if item.dimension == "OBJECT"
    )
    candidate = replace(
        object_set.candidates[0],
        method_semantics={"object": {"type": "pedestrian", "position": "front"}},
    )
    template = {
        "template_id": "FM_TEMPLATE_TEST", "active_option": {},
        "source_governed_constraints": [
            {
                "source_option_id": "OPT-PED", "obj_type": "pedestrian",
                "obj_position": "front", "collision_type": "front",
            },
            {
                "source_option_id": "OPT-CAR", "obj_type": "passenger_car",
                "obj_position": "front", "collision_type": "front",
            },
        ],
    }
    synthesis_input = replace(base_input, fm_scenario_template=template)

    option, resolution, matches = service._selected_template_option(
        synthesis_input, {candidate.atom_id: candidate},
    )

    assert option["source_option_id"] == "OPT-PED"
    assert resolution == "UNIQUE_STRUCTURED_OPTION_MATCH"
    assert matches == ["OPT-PED"]

    ambiguous = replace(synthesis_input, fm_scenario_template={
        **template,
        "source_governed_constraints": [
            *template["source_governed_constraints"],
            {
                "source_option_id": "OPT-PED-2", "obj_type": "pedestrian",
                "obj_position": "front", "collision_type": "rear",
            },
        ],
    })
    option, resolution, matches = service._selected_template_option(
        ambiguous, {candidate.atom_id: candidate},
    )
    assert option is None
    assert resolution == "AMBIGUOUS_FM_TEMPLATE_OPTION"
    assert matches == ["OPT-PED", "OPT-PED-2"]


def test_child_speed_intersection_remains_a_range_without_automatic_point(method):
    service = ConstrainedScenarioSynthesisService(method)
    parent = replace(
        _parent(), facts={"ego_speed_constraint": {"min_kph": 0.0, "max_kph": 20.0}},
    )
    malfunction = {
        **_malfunction(),
        "description": "reversing reverse backing vehicle collision",
        "functional_effect": "vehicle reverses unexpectedly",
        "vehicle_level_hazard": "reverse collision",
    }
    synthesis_input = service.build_input(
        malfunction=malfunction, parent=parent, assessment=_assessment(),
        project_context=_project(),
    )
    raw_ids = _valid_payload(synthesis_input)["variants"][0]["selected_atom_ids"]
    raw_ids = [
        atom_id for atom_id in raw_ids
        if not set(service.by_id[atom_id].get("filled_dimensions", []))
        & {"EGO_ACTION", "EGO_DYNAMICS"}
    ]
    selected = service.expand_logical_atom_ids(
        synthesis_input, [*raw_ids, "FA033"],
    )
    assessment = ScenarioSynthesisAssessment(
        semantic_group_id=synthesis_input.semantic_group_id,
        coverage_label=CoverageLabel.TYPICAL,
        selected_atoms=selected,
        semantic_rationale="Select explicit reverse <=10 km/h Method atom.",
        context_refs=("PARENT.scenario", "METHOD.scenario_atom_catalog"),
        validation_status=SynthesisValidationStatus.VALIDATED,
    )

    child, _ = service.materialize(
        synthesis_input=synthesis_input, assessment=assessment,
        parent=parent, provider_evidence={"request_id": "REQ-RANGE"},
    )

    assert child.facts["ego_speed_constraint"] == {"min_kph": 0.0, "max_kph": 10.0}
    assert "ego_speed_kph" not in child.facts
    metadata = child.fact_provenance["ego_speed_constraint"]
    assert metadata["speed_intersection_kph"] == [0.0, 10.0]
    assert len(metadata["source_refs"]) == 2


def test_repeated_child_materialization_keeps_nested_facts_isolated(method):
    service = ConstrainedScenarioSynthesisService(method)
    parent = _parent()
    synthesis_input = _input(method)
    assessments = service.validate_provider_payload(
        synthesis_input, _valid_payload(synthesis_input),
    )
    assessment = assessments[0]
    child_a, _ = service.materialize(
        synthesis_input=synthesis_input, assessment=assessment,
        parent=parent, provider_evidence={"request_id": "REQ-A"},
    )
    child_b, _ = service.materialize(
        synthesis_input=synthesis_input, assessment=assessment,
        parent=parent, provider_evidence={"request_id": "REQ-B"},
    )
    child_a.facts["ego_speed_constraint"]["max_kph"] = 999
    assert parent.facts["ego_speed_constraint"]["max_kph"] == 20.0
    assert child_b.facts["ego_speed_constraint"]["max_kph"] == 20.0
    assert child_a.scenario_id == child_b.scenario_id


def test_physics_instantiation_has_no_distance_speed_or_geometry_defaults():
    result = AnalyticalPhysicsInstantiationService().instantiate(
        scenario=ScenarioCandidate(
            scenario_id="SCN-1", operating_scenario="parking",
            situational_description="parking", situational_detailing="parking",
            facts={"ego_speed_constraint": {"min_kph": 0, "max_kph": 20}},
        ),
        malfunction={"malfunction_id": "MF-1"},
        causal_status="CAUSAL_REVALIDATED",
    )
    inputs = {item["field"]: item for item in result["inputs"]}
    assert inputs["ego_speed_kph"]["authority"] == "ENGINEERING_ANALYSIS_ASSUMPTION"
    assert inputs["ego_speed_kph"]["value"] is None
    assert inputs["object_speed_kph"]["authority"] == "UNAVAILABLE"
    assert inputs["relative_distance_m"]["authority"] == "UNAVAILABLE"
    assert result["derived"] == []


def _physics_scenario(**overrides):
    facts = {
        "ego_speed_kph": 20.0, "object_speed_kph": 5.0,
        "relative_distance_m": 10.0, "road_user_type": "VEHICLE",
        "collision_type": "FRONTAL", "ego_longitudinal_direction": "FORWARD",
        "object_longitudinal_direction": "FORWARD",
    }
    facts.update(overrides)
    return ScenarioCandidate(
        scenario_id="SCN-PHYSICS", operating_scenario="parking",
        situational_description="physics fixture",
        situational_detailing="physics fixture", facts=facts,
        fact_provenance={
            field: {"origin": "PROJECT_FACT", "approval": "FINALIZED"}
            for field in facts if field != "method_scenario_dimensions"
        },
    )


def test_explicit_stationary_object_atom_derives_zero_speed(method):
    scenario = ScenarioCandidate(
        scenario_id="SCN-STATIONARY", operating_scenario="parking",
        situational_description="stationary object",
        situational_detailing="stationary object",
        facts={
            "scenario_atom_ids": ["TEST-STATIONARY"],
            "method_scenario_dimensions": {
                "OBJECT": {
                    "atom_id": "TEST-STATIONARY",
                    "resolution_status": "RESOLVED",
                    "atom_provenance": {
                        "source_asset": "raw/vda702_atoms.yaml",
                        "source_rule": "TEST-STATIONARY",
                    },
                    "method_semantics": {"object": {"motion": "STATIONARY"}},
                },
            },
        },
    )
    result = AnalyticalPhysicsInstantiationService(method).instantiate(
        scenario=scenario, malfunction={"malfunction_id": "MF-1"},
        causal_status="CAUSAL_REVALIDATED",
    )
    speed = next(item for item in result["inputs"] if item["field"] == "object_speed_kph")
    assert speed["value"] == 0.0
    assert speed["authority"] == "DERIVED"
    assert speed["selection_basis"] == "METHOD_ATOM_EXPLICIT_STATIONARY_OBJECT"


def test_relative_speed_requires_both_longitudinal_directions():
    scenario = _physics_scenario()
    scenario.facts.pop("object_longitudinal_direction")
    scenario.fact_provenance.pop("object_longitudinal_direction")
    result = AnalyticalPhysicsInstantiationService().instantiate(
        scenario=scenario, malfunction={"malfunction_id": "MF-1"},
        causal_status="CAUSAL_REVALIDATED",
    )
    assert not any(item["field"] == "relative_speed_kph" for item in result["derived"])


def test_side_collision_does_not_use_longitudinal_relative_speed_formula():
    result = AnalyticalPhysicsInstantiationService().instantiate(
        scenario=_physics_scenario(collision_type="SIDE"),
        malfunction={"malfunction_id": "MF-1"},
        causal_status="CAUSAL_REVALIDATED",
    )
    assert result["derived"] == []


def test_missing_dependency_metadata_requires_causal_revalidation(method):
    runner = ScenarioSynthesisRunner(method=method, client=None)
    synthesis_input = _input(method)
    service = runner.synthesis
    assessment = service.validate_provider_payload(
        synthesis_input, _valid_payload(synthesis_input),
    )[0]
    child, _ = service.materialize(
        synthesis_input=synthesis_input, assessment=assessment,
        parent=_parent(), provider_evidence={},
    )
    delta = runner._causal_delta(
        synthesis_input=synthesis_input, parent_assessment=_assessment(), child=child,
    )
    assert delta["status"] == "CAUSAL_REVALIDATION_REQUIRED"
    assert delta["causal_reuse_basis"] == ""


def test_explicit_noninterference_proof_allows_deterministic_causal_reuse(method):
    runner = ScenarioSynthesisRunner(method=method, client=None)
    synthesis_input = _input(method)
    service = runner.synthesis
    assessment = service.validate_provider_payload(
        synthesis_input, _valid_payload(synthesis_input),
    )[0]
    child, _ = service.materialize(
        synthesis_input=synthesis_input, assessment=assessment,
        parent=_parent(), provider_evidence={},
    )
    parent_assessment = _assessment()
    parent_assessment["dependency_metadata"] = {
        "complete": True,
        "causal_evidence_fields": ["MF.description"],
        "physical_feasibility_fields": ["MF.functional_effect"],
        "scenario_identity_fields": ["malfunction_id"],
        "child_subset_refinement": "Explicit source proof: changed Scenario dimensions are noninterfering.",
    }
    delta = runner._causal_delta(
        synthesis_input=synthesis_input,
        parent_assessment=parent_assessment, child=child,
    )
    assert delta["status"] == "CAUSAL_REUSE_PROVEN"
    assert delta["provider_required"] is False


def test_validated_analytical_atom_facts_are_visible_to_causal_revalidation(method):
    synthesis = ConstrainedScenarioSynthesisService(method)
    synthesis_input = _input(method)
    assessment = synthesis.validate_provider_payload(
        synthesis_input, _valid_payload(synthesis_input),
    )[0]
    child, _ = synthesis.materialize(
        synthesis_input=synthesis_input, assessment=assessment,
        parent=_parent(), provider_evidence={},
    )
    projected = ScenarioCausalRevalidationRunner(
        method=method, client=None,
    )._causal_projection(child)
    selection = select_scenario_causal_evidence(
        MalfunctionCandidate(
            malfunction_id="MF-1", function_id="F-1", guideword="No/Loss",
            description="泊入时行人识别丢失",
            functional_effect="车辆未对行人制动",
            vehicle_level_hazard="停车场内与行人碰撞",
            causal_chain=["M", "B", "I", "H"],
            sources=[SourceRef("item_definition", "ITEM", "p1", "source")],
            status=ReviewStatus.FINALIZED,
        ),
        projected,
    )
    assert "SCN.road_surface_conditions" in selection.selected_refs
    assert "SCN.vehicle_state" in selection.selected_refs
    assert "SCN.ego_road_relation" not in selection.selected_refs
    assert "SCN.operating_scenario" in selection.selected_refs
    assert "SCN.vehicle_state" in selection.selected_refs
    assert "SCN.ego_dynamics" in selection.selected_refs
    assert "SCN.scenario_object_atom" in selection.selected_refs


def test_risk_scoreability_exposes_distinct_p5d_statuses():
    statuses = RiskScoreabilityService.readiness_statuses({
        "scenario_synthesis_status": "PENDING_SCENARIO_SYNTHESIS",
        "causal_delta_status": "CAUSAL_REVALIDATION_REQUIRED",
        "S": {"ready": False, "blockers": ["EGO_POINT_SPEED_ENGINEERING_ASSUMPTION_REQUIRED"]},
        "C": {"ready": False, "blockers": [], "override_finalized": False, "ttc_ready": False},
    })
    assert statuses == (
        "SCENARIO_SYNTHESIS_BLOCKED", "CAUSAL_REVALIDATION_BLOCKED",
        "PHYSICS_ASSUMPTION_BLOCKED",
    )
