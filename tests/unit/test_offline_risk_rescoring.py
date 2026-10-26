from copy import deepcopy
from dataclasses import asdict
import json
from pathlib import Path
from unittest.mock import patch
import zipfile

import pytest

from hara_agent.contracts import (
    CausalBreakpoint, CausalEdge, CausalGraph, CausalNode, CausalNodeType,
    CausalRelation, EvidenceBinding, ScenarioCausalAssessment,
)
from hara_agent.method_sources import MethodSourceResolver
from hara_agent.models import (
    EvidenceKind, EvidenceValue, ItemDefinitionFacts, ReviewStatus,
    RiskAssessment, ScenarioCandidate, SourceRef,
)
from hara_agent.services.analysis import MethodRuleScoringService
from hara_agent.workflow import HARAState, CheckpointRepository
from hara_agent.workflow.nodes.scoring import _validated_causal_assessment
from hara_agent.workflow.risk_rescoring import (
    OfflineRiskRescorer, apply_supplement, clean_risk_stage, file_hash,
)

ROOT = Path(__file__).resolve().parents[2]
BASELINE = ROOT / "method_assets/fusa_baseline_v1/manifest.yaml"
TEMPLATE = ROOT / "references/HARA_Template_AI_20260327.xlsx"


@pytest.fixture(scope="module")
def method():
    return MethodSourceResolver().resolve(template_path=None,
        baseline_manifest_path=BASELINE, report_template_path=TEMPLATE).method


@pytest.fixture
def source_state(tmp_path, method):
    # Deliberate synthetic engineering fixture, never a claimed r2 project fact.
    facts = {"relative_speed_kph": 4.0, "road_user_type": "VEHICLE", "collision_type": "FRONTAL",
             "driver_in_vehicle": False, "remote_intervention_available": False,
             "other_road_user_avoidance_possible": False}
    fixture = tmp_path / "engineering-fixture.json"
    fixture.write_text(json.dumps(facts), encoding="utf-8")
    source = SourceRef("test_fixture", str(fixture), "engineering-fixture", "explicit synthetic conditions")
    provenance = {key: {"provenance": "PROJECT_INPUT", "approval": "FINALIZED",
                        "source_refs": [asdict(source)]} for key in facts}
    nodes = tuple(CausalNode(identity, kind, identity) for identity, kind in (
        ("M", CausalNodeType.MALFUNCTION), ("B", CausalNodeType.SYSTEM_BEHAVIOR_CHANGE),
        ("I", CausalNodeType.OPERATIONAL_CONSEQUENCE), ("H", CausalNodeType.HAZARD)))
    edges = tuple(CausalEdge(identity, start, end, CausalRelation.CAUSES, identity, (ref,))
                  for identity, start, end, ref in (("M_TO_B", "M", "B", "MF.description"),
                      ("B_TO_I", "B", "I", "MF.functional_effect"),
                      ("I_TO_H", "I", "H", "SCN.relative_speed_kph")))
    causal = ScenarioCausalAssessment("SC-1", CausalGraph(nodes, edges), ("M", "B", "I", "H"),
        CausalBreakpoint.NONE, tuple(EvidenceBinding(edge.edge_id, edge.evidence_refs,
            EvidenceKind.DIRECT_FACT, (source,), ReviewStatus.FINALIZED) for edge in edges),
        (), (), "test hazardous event", review_status=ReviewStatus.FINALIZED)
    assessment = {"malfunction_id": "MF-1", "scenario_id": "SC-1", "status": "FINALIZED",
        "physically_feasible": True, "functionally_relevant": True, "causally_relevant": True,
        "final_retain": True, "breakpoint": "NONE", "risk_dimensions_changed": [],
        "hazardous_event": causal.hazardous_event, "causal_assessment": causal.to_dict()}
    state = HARAState(run_id="source", method_contract={
        "method_source_hash": method.metadata["method_source_hash"],
        "template_hash": method.metadata["template_hash"], "contract_version": method.contract_version,
        "compiler_version": method.compiler_version, "source_kind": "YAML_BASELINE"},
        item_definition={"typed": HARAState._serialize(asdict(ItemDefinitionFacts("test", "boundary", sources=[source]))),
                         "source_path": str(fixture), "scenario_assessments": [assessment]},
        functions=[{"function_id": "F-1", "name": "control", "output": "motion"}],
        guideword_assessments=[{"function_id": "F-1", "guideword_id": "Less", "guideword": "Less"}],
        malfunctions=[{"malfunction_id": "MF-1", "function_id": "F-1", "guideword": "Less",
                       "description": "output lost", "functional_effect": "motion uncontrolled",
                       "vehicle_level_hazard": "hazard", "causal_chain": ["M", "B"], "sources": [asdict(source)]}],
        scenarios=[ScenarioCandidate("SC-1", "parking", "停车场", "明确测试工况", facts=facts,
            fact_provenance=provenance, semantic_fingerprint="fixture-fingerprint",
            scenario_contract_version="atomic-scenario-v1")],
        scenario_contract_version="atomic-scenario-v1")
    state.risk_results = [RiskAssessment("OLD-1", "SC-1", *(EvidenceValue("", ReviewStatus.PENDING) for _ in range(4)),
                                          malfunction_id="MF-1", hazardous_event=causal.hazardous_event)]
    return state


@pytest.mark.parametrize("with_guidewords", [True, False])
def test_real_scoring_offline_preserves_source_and_writes_fresh_trace(tmp_path, method, source_state, with_guidewords):
    if not with_guidewords:
        source_state.guideword_assessments = []
    repository = CheckpointRepository(tmp_path / "runs")
    source_path = repository.save(source_state)
    digest = file_hash(source_path)
    original = MethodRuleScoringService.score
    calls = []
    def observed(self, *args, **kwargs):
        calls.append(args[0]["scenario_id"])
        return original(self, *args, **kwargs)
    with patch.object(MethodRuleScoringService, "score", observed), patch(
        "hara_agent.infrastructure.llm.OpenAICompatibleClient.__init__", side_effect=AssertionError("Provider"),
    ) as provider, patch("hara_agent.application.create_llm_client", side_effect=AssertionError("factory")) as factory:
        result = OfflineRiskRescorer().run(source_run_id="source", target_run_id="new", baseline=BASELINE,
            report_template=TEMPLATE, output=tmp_path / "new.xlsx", run_dir=repository.run_dir,
            review_root=tmp_path / "review")
    assert provider.call_count == factory.call_count == 0
    assert calls == ["SC-1"]
    assert result["values_before"] == dict.fromkeys(("severity", "exposure", "controllability", "asil"), 0)
    assert result["values_after"] == {"severity": 1, "exposure": 0, "controllability": 1, "asil": 0}
    new = HARAState.read_committed(json.loads(Path(result["target_checkpoint"]).read_text(encoding="utf-8")))
    assert (new.risk_results[0].severity.value, new.risk_results[0].controllability.value) == ("S1", "C3")
    assert not new.risk_results[0].exposure.value and not new.risk_results[0].asil.value
    assert file_hash(source_path) == digest
    assert new.scenarios == source_state.scenarios
    assert new.scenario_contract_version == "atomic-scenario-v1"
    assert result["safety_goals_status"] == "DETERMINISTIC_REAGGREGATED"
    assert len(new.safety_goals) == 1
    goal = new.safety_goals[0]
    assert goal.sg_id == new.risk_results[0].safety_goal_id
    assert goal.associated_scenario_ids == ["SC-1"]
    assert goal.status is ReviewStatus.PENDING and not goal.max_asil
    assert goal.safe_state == "待确定"
    trace = json.loads((tmp_path / "review/new/risk_execution_trace.json").read_text(encoding="utf-8"))
    assert trace["assessments"][0]["parent_assessment_id"] == "OLD-1"
    assert trace["risk_execution_id"] == result["risk_execution_id"]
    with zipfile.ZipFile(result["output"]) as workbook:
        assert any(result["risk_execution_id"].encode() in workbook.read(name)
                   for name in workbook.namelist() if name.endswith(".xml"))


def supplement(state, *, value=4.0):
    return {"source_run_id": state.run_id, "source_checkpoint_sha256": "digest",
        "scopes": {"one": [{"malfunction_id": "MF-1", "scenario_id": "SC-1"}]},
        "entries": [{"scope_id": "one", "field": "relative_speed_kph", "value": value, "unit": "km/h",
                     "provenance": deepcopy(state.scenarios[0].fact_provenance["relative_speed_kph"])}]}


def test_supplement_preserves_ranges_and_never_reuses_causal_for_new_conditions(source_state):
    before = deepcopy(source_state)
    assert apply_supplement(source_state, supplement(source_state), "digest")[0]["status"] == "UNCHANGED_EXISTING_FACT"
    for value in (20.0, {"min": 0.0, "max": 20.0}):
        result = apply_supplement(source_state, supplement(source_state, value=value), "digest")
        assert result[0]["status"] == "PENDING_DIFFERENTIAL_VALIDATION"
        assert result[0]["value"] == value
    assert source_state == before


@pytest.mark.parametrize("broken", ["source", "scope", "analysis_scope"])
def test_supplement_rejects_invalid_provenance_even_if_finalized(source_state, broken):
    payload = supplement(source_state)
    metadata = payload["entries"][0]["provenance"]
    if broken == "source":
        metadata["source_refs"] = []
    elif broken == "scope":
        payload["scopes"]["one"][0]["malfunction_id"] = "OTHER"
    else:
        metadata.update(provenance="SCENARIO_DEFINED", origin="SCENARIO_DEFINED",
                        applicable_scope={"malfunction_id": "OTHER", "scenario_id": "SC-1"},
                        validation_status="VALIDATED")
    with pytest.raises(ValueError):
        apply_supplement(source_state, payload, "digest")


def test_clear_downstream_harm_is_repeatable_and_preserves_unrelated_audit(source_state):
    assessment = source_state.item_definition["scenario_assessments"][0]
    causal = _validated_causal_assessment(assessment)
    assessment["causal_assessment"] = causal.with_harm(potential_harm="old injury",
        evidence_refs=("OLD.HARM",), source_refs=causal.evidence_bindings[0].source_refs,
        review_status=ReviewStatus.FINALIZED).to_dict()
    assessment["potential_harm"] = "old injury"
    source_state.audit_trail = [{"event": "source_preserved"}, {"event": "structured_risk_scoring_completed"}]
    source_state.pending_reviews = [{"field": "severity"}, {"field": "functions"}]
    clean_risk_stage(source_state)
    assert _validated_causal_assessment(assessment).is_validated
    assert len(assessment["causal_assessment"]["causal_graph"]["nodes"]) == 4
    assert source_state.audit_trail == [{"event": "source_preserved"}]
    assert source_state.pending_reviews == [{"field": "functions"}]
    again = deepcopy(source_state)
    clean_risk_stage(source_state)
    assert source_state == again


@pytest.mark.parametrize("failure", ["run_id", "contract", "queue", "fingerprint", "method"])
def test_invalid_source_fails_before_output(tmp_path, source_state, failure):
    if failure == "run_id":
        source_state.run_id = "other"
    elif failure == "contract":
        source_state.scenario_assessment_contract_version = "incompatible"
    elif failure == "queue":
        source_state.risk_results[0].scenario_id = "OTHER"
    elif failure == "fingerprint":
        source_state.scenarios[0].semantic_fingerprint = ""
    else:
        source_state.method_contract["method_source_hash"] = "mismatch"
    path = CheckpointRepository(tmp_path / "runs").save(source_state)
    with pytest.raises(ValueError):
        OfflineRiskRescorer().run(source_run_id="source", target_run_id="new", checkpoint=path,
            baseline=BASELINE, report_template=TEMPLATE, output=tmp_path / "new.xlsx",
            run_dir=tmp_path / "runs", review_root=tmp_path / "review")
    assert not (tmp_path / "new.xlsx").exists()
