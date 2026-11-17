from __future__ import annotations

import json
import shutil
from types import SimpleNamespace
from pathlib import Path

import pytest

from hara_agent.cli import main
from hara_agent.workflow import ReviewArtifactReader, ReviewArtifactWriter, render_review


@pytest.fixture
def review_root():
    root = Path("runtime/review")
    root.mkdir(parents=True, exist_ok=True)
    try:
        yield root
    finally:
        for run_id in ("run-1", "run-cli"):
            shutil.rmtree(root / run_id, ignore_errors=True)


def _writer(review_root):
    return ReviewArtifactWriter("run-1", review_root)


def _populate(writer: ReviewArtifactWriter) -> None:
    writer.record_function({
        "function_id": "F01", "name": "Parking", "description": "Park vehicle",
        "output": "parked vehicle", "preconditions": ["active"],
        "triggers": ["user request"], "odd_constraints": ["parking lot"],
        "fallback_behavior": "manual takeover", "status": "FINALIZED", "confidence": 0.9,
        "source_refs": [],
    })
    writer.record_guideword_assessment({
        "function_id": "F01", "guideword": "No", "applicable": False,
        "disposition": "NOT_APPLICABLE", "rationale": "No credible deviation",
        "status": "FINALIZED", "confidence": 0.8, "source_refs": [],
    })
    writer.record_malfunction({
        "malfunction_id": "MF-F01-001", "function_id": "F01", "guideword": "No",
        "description": "HMI does not show AVP", "functional_effect": "User cannot select",
        "vehicle_level_hazard": "Vehicle state may become unsafe", "status": "FINALIZED",
        "confidence": 0.8, "source_refs": [],
    })
    writer.record_scenario_candidate({
        "scenario_id": "SCN-1", "semantic_fingerprint": "fp-1",
        "operating_scenario": "parking lot", "situational_description": "object ahead",
        "situational_detailing": "vehicle moving", "facts": {"ego_speed_kph": 10},
    }, generated_for_malfunction_ids=["MF-F01-001"])
    writer.record_scenario_feasibility({
        "malfunction_id": "MF-F01-001", "scenario_id": "SCN-1",
        "physically_feasible": True, "functionally_relevant": True,
        "causally_relevant": True, "final_retain": True, "breakpoint": "NONE",
        "causal_chain": {"m_to_b": {"claim": "defined effect", "evidence_refs": ["MF.functional_effect"]}},
        "rationale": "supported", "status": "FINALIZED",
    }, function_id="F01", guideword="No")


def test_review_artifacts_append_immediately_and_resume_is_idempotent(review_root):
    writer = _writer(review_root)
    assert writer.record_function({"function_id": "F01", "name": "Parking"}) is True
    assert writer.record_function({"function_id": "F01", "name": "Parking updated"}) is False

    resumed = ReviewArtifactWriter("run-1", review_root)
    assert resumed.record_function({"function_id": "F01", "name": "Parking again"}) is False
    lines = (review_root / "run-1" / "functions.jsonl").read_text(encoding="utf-8").splitlines()
    assert len(lines) == 1
    assert json.loads(lines[0])["run_id"] == "run-1"


def test_reader_skips_one_malformed_jsonl_line_and_keeps_previous_records(review_root):
    writer = _writer(review_root)
    writer.record_function({"function_id": "F01", "name": "Parking"})
    path = review_root / "run-1" / "functions.jsonl"
    with path.open("a", encoding="utf-8") as stream:
        stream.write('{"function_id": "BROKEN"\n')
    records = ReviewArtifactReader("run-1", review_root)
    assert [item["function_id"] for item in records.read_all()["function"]] == ["F01"]
    assert records.warnings


def test_summary_and_views_are_read_only_and_keep_negative_guideword(review_root):
    writer = _writer(review_root)
    _populate(writer)
    writer.write_summary(SimpleNamespace(stage=SimpleNamespace(value="scenarios")))
    reader = ReviewArtifactReader("run-1", review_root)
    summary = reader.summary()
    assert summary["status"] == "IN_PROGRESS"
    assert summary["function_count"] == 1
    assert summary["guideword_filtered_count"] == 1
    assert summary["scenario_candidate_count"] == 1
    assert summary["scenario_feasibility_count"] == 1
    assert "FUNCTION F01" in render_review(reader, function_id="F01")
    function_view = render_review(reader, function_id="F01")
    assert "No | False | NOT_APPLICABLE" in function_view
    assert "MALFUNCTION MF-F01-001" in render_review(reader, malfunction_id="MF-F01-001")
    assert "SCENARIO CANDIDATE" in render_review(reader, scenario_id="SCN-1")


def test_cli_review_filters_run_without_llm(review_root, monkeypatch, capsys):
    writer = ReviewArtifactWriter("run-cli", review_root)
    _populate(writer)
    monkeypatch.setenv("HARA_REVIEW_ARTIFACT_DIR", str(review_root))

    assert main(["review", "--run-id", "run-cli", "--malfunction", "MF-F01-001"]) == 0
    output = capsys.readouterr().out
    assert "MALFUNCTION MF-F01-001" in output
    assert "SCN-1" in output


def test_control_only_child_replaces_its_parent_but_other_unassessed_parent_stays_pending(tmp_path):
    writer = ReviewArtifactWriter('control-scope', tmp_path)
    writer.record_malfunction({'malfunction_id': 'MF-control'})
    for identity in ('parent-a', 'parent-b'):
        writer.record_scenario_candidate({'scenario_id': identity, 'analysis_instance': {
            'p6_generation': True, 'parent_scenario_id': 'base-context'}},
            generated_for_malfunction_ids=['MF-control'])
    writer.record_scenario_candidate({'scenario_id': 'control-child', 'analysis_instance': {
        'p6_generation': True, 'parent_scenario_id': 'base-context',
        'control_analysis': {'parent_scenario_id': 'parent-a'}}},
        generated_for_malfunction_ids=['MF-control'])
    writer.record_scenario_feasibility({'malfunction_id': 'MF-control',
        'scenario_id': 'control-child', 'final_retain': True})
    state = SimpleNamespace(stage=SimpleNamespace(value='scenarios'))
    writer.write_summary(state)
    reader = ReviewArtifactReader('control-scope', tmp_path)
    item = reader.summary()['per_malfunction_summary']['MF-control']
    assert item['expected_scenario_count']==2 and item['total']==1
    assert item['assessment_status']=='PARTIAL'
    writer.record_scenario_feasibility({'malfunction_id': 'MF-control',
        'scenario_id': 'parent-b', 'final_retain': False})
    writer.write_summary(state)
    item = reader.summary()['per_malfunction_summary']['MF-control']
    assert item['expected_scenario_count']==item['total']==2
    assert item['assessment_status']=='COMPLETED'
    assert len(reader.read_all()['scenario_candidate'])==3  # Audit parents survive.
