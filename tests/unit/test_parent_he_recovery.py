from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from hara_agent.contracts import (
    CausalBreakpoint, CausalEdge, CausalGraph, CausalNode,
    CausalNodeType, CausalRelation, EvidenceBinding,
    ScenarioCausalAssessment,
)
from hara_agent.method_sources import YamlBaselineCompiler
from hara_agent.models import (
    EvidenceKind, ItemDefinitionFacts, MalfunctionCandidate, ReviewStatus,
    ScenarioCandidate, ScenarioFeasibilityAssessment, SourceRef, SpeedEnvelope,
)
from hara_agent.services.analysis import (
    MethodScenarioCandidateService, ProjectFactResolver,
)
from hara_agent.template import TemplateRoleCompiler
from hara_agent.workflow.checkpoints import CheckpointRepository
from hara_agent.workflow import parent_he_recovery as recovery_module
from hara_agent.workflow.parent_he_recovery import ParentHERecoveryRunner
from hara_agent.workflow.state import HARAState


ROOT = Path(__file__).resolve().parents[2]


def _method():
    report = TemplateRoleCompiler().compile_method(
        ROOT / "references/HARA_Template_AI_20260327.xlsx"
    ).report_contract
    return YamlBaselineCompiler().compile(
        ROOT / "method_assets/fusa_baseline_v1/manifest.yaml",
        report_contract=report,
    )


def test_recovery_regenerates_and_commits_parent_without_promoting_old_child(
    tmp_path, monkeypatch,
):
    method = _method()
    source = SourceRef("item_definition", "ItemDef.docx", "p1", "parking context")
    facts = ItemDefinitionFacts(
        system_description="AVP", item_boundary="vehicle",
        operating_modes=["Active"], odd_locations=["indoor parking garage"],
        speed_envelopes=[SpeedEnvelope(
            "Active", 0, 7, sources=[source], status=ReviewStatus.FINALIZED,
        )],
        sources=[source], status=ReviewStatus.FINALIZED,
    )
    speed = ProjectFactResolver().resolve_speed_context(facts, "Active")
    current, _ = MethodScenarioCandidateService(method).generate(
        project_facts=facts, operating_mode=speed.operating_mode,
        speed_resolution=speed, functions=[],
    )
    assert len(current) == 1
    malfunction = MalfunctionCandidate(
        "MF-1", "F-1", "More", "excess brake request",
        "excess deceleration", "occupant injury",
        ["brake request", "deceleration", "injury"],
        sources=[source], status=ReviewStatus.FINALIZED,
    )
    old_child = ScenarioCandidate(
        "OLD-CHILD", "parking", "old analytical child", "",
        source_scenario_id="OLD-PARENT",
        analysis_instance={"malfunction_id": "MF-1"},
    )
    state = HARAState(
        run_id="source",
        method_contract={"method_source_hash": method.metadata["method_source_hash"]},
        item_definition={"typed": HARAState._serialize(asdict(facts))},
        malfunctions=[HARAState._serialize(asdict(malfunction))],
        scenarios=[old_child],
    )
    run_dir = tmp_path / "run"
    review_dir = tmp_path / "review"
    CheckpointRepository(run_dir).save(state)
    (review_dir / "historical").mkdir(parents=True)
    historical = current[0].to_dict() | {
        "scenario_id": "OLD-PARENT",
        "generated_for_malfunction_ids": ["MF-1"],
    }
    (review_dir / "historical" / "scenario_candidates.jsonl").write_text(
        json.dumps(historical) + "\n"
    )
    prepared = ParentHERecoveryRunner(
        method=method, client=None, run_dir=run_dir, review_root=review_dir,
    ).prepare(
        source_run_id="source", historical_review_run_id="historical",
        malfunction_id="MF-1", historical_parent_scenario_ids=["OLD-PARENT"],
        operating_mode="Active",
    )
    assert prepared["mappings"][0]["historical_child_scenario_ids"] == ["OLD-CHILD"]
    assert prepared["mappings"][0]["current_parent_scenario_id"] == current[0].scenario_id
    assert prepared["candidates"][0].scenario_id != old_child.scenario_id

    nodes = tuple(CausalNode(identity, kind, identity) for identity, kind in (
        ("M", CausalNodeType.MALFUNCTION),
        ("B", CausalNodeType.SYSTEM_BEHAVIOR_CHANGE),
        ("I", CausalNodeType.OPERATIONAL_CONSEQUENCE),
        ("H", CausalNodeType.HAZARD),
    ))
    edges = tuple(CausalEdge(identity, start, end, CausalRelation.CAUSES,
                             identity, ("test-evidence",))
                  for identity, start, end in (
                      ("M_TO_B", "M", "B"),
                      ("B_TO_I", "B", "I"),
                      ("I_TO_H", "I", "H"),
                  ))
    causal = ScenarioCausalAssessment(
        current[0].scenario_id, CausalGraph(nodes, edges),
        ("M", "B", "I", "H"), CausalBreakpoint.NONE,
        tuple(EvidenceBinding(
            edge.edge_id, edge.evidence_refs, EvidenceKind.DIRECT_FACT,
            (source,), ReviewStatus.FINALIZED,
        ) for edge in edges),
        (), (), "bounded parent hazardous event",
        review_status=ReviewStatus.FINALIZED,
    )
    assessment = ScenarioFeasibilityAssessment(
        malfunction_id="MF-1", scenario_id=current[0].scenario_id,
        physically_feasible=True, functionally_relevant=True,
        causally_relevant=True, risk_dimensions_changed=[],
        rationale="source-linked causal chain", hazardous_event=causal.hazardous_event,
        status=ReviewStatus.FINALIZED, breakpoint="NONE",
        causal_assessment=causal,
    )

    class StubAgent:
        def __init__(self, *_args, **_kwargs):
            pass

        def assess(self, *_args):
            return [assessment], {"actual_llm_calls": 1}

    monkeypatch.setattr(recovery_module, "ScenarioFeasibilityAgent", StubAgent)
    source_path = CheckpointRepository(run_dir).path_for("source")
    source_sha = hashlib.sha256(source_path.read_bytes()).hexdigest()
    result = ParentHERecoveryRunner(
        method=method, client=object(), run_dir=run_dir, review_root=review_dir,
    ).run(
        source_run_id="source", historical_review_run_id="historical",
        target_run_id="recovered", malfunction_id="MF-1",
        historical_parent_scenario_ids=["OLD-PARENT"], operating_mode="Active",
    )
    child = CheckpointRepository(run_dir).load("recovered")
    committed = child.item_definition["scenario_assessments"]
    assert result["retained_parent_count"] == 1
    assert len(child.scenarios) == len(committed) == 1
    assert child.scenarios[0].scenario_id == current[0].scenario_id
    assert committed[0]["hazardous_event_id"] == (
        f"HE::MF-1::{current[0].scenario_id}::H"
    )
    assert child.audit_trail[0]["event"] == "parent_he_recovery_child_run_materialized"
    assert hashlib.sha256(source_path.read_bytes()).hexdigest() == source_sha
