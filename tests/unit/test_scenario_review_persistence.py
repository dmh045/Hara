from __future__ import annotations

import json
import shutil
import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

import pytest

from hara_agent.infrastructure.llm import LLMTransportError
from hara_agent.models import (
    MalfunctionCandidate,
    ReviewStatus,
    ScenarioCandidate,
    ScenarioFeasibilityAssessment,
)
from hara_agent.workflow import HARAState, ReviewArtifactReader, ReviewArtifactWriter, WorkflowStage, render_review
from hara_agent.workflow.nodes.scenarios import assess_scenarios
from hara_agent.workflow.semantic_graph import SemanticWorkflowInputs, _scenario_candidates


@pytest.fixture
def review_root():
    root = Path("runtime/review")
    root.mkdir(parents=True, exist_ok=True)
    run_ids = (
        "c51-candidates", "c51-failure", "c51-parallel", "c51-resume",
        "c51-status", "c51-projection", "c51-running",
    )
    try:
        yield root
    finally:
        for run_id in run_ids:
            shutil.rmtree(root / run_id, ignore_errors=True)


class _Client:
    class Config:
        provider = "test"
        base_url = "local"
        model = "scenario-review-test"

    config = Config()


@pytest.mark.parametrize("stage,can_publish,expected", [
    (WorkflowStage.QUALITY_GATE, False, "PENDING_REVIEW"),
    (WorkflowStage.QUALITY_GATE, True, "IN_PROGRESS"),
    (WorkflowStage.SCORING, False, "IN_PROGRESS"),
])
def test_review_summary_distinguishes_pending_gate_from_running(
    tmp_path, stage, can_publish, expected,
):
    state = SimpleNamespace(run_id="review-gate-status", stage=stage, can_publish=can_publish)
    writer = ReviewArtifactWriter(state.run_id, tmp_path)
    writer.write_summary(state)
    summary = json.loads((tmp_path / state.run_id / "summary.json").read_text())
    assert summary["status"] == expected


def _malfunction(identity: str) -> MalfunctionCandidate:
    return MalfunctionCandidate(
        identity, "F01", "loss", f"{identity} description", "lost output",
        "vehicle behavior changes", ["lost output", "vehicle behavior changes"],
        status=ReviewStatus.FINALIZED,
    )


def _candidates(count: int = 45) -> list[ScenarioCandidate]:
    return [
        ScenarioCandidate(
            f"SCN-{index:03d}", "Parking", f"scenario {index}", "active parking",
            {
                "operating_mode": "Active", "ego_speed_kph": 10,
                "weather_conditions": "dry", "road_surface_conditions": "dry",
                "object": "pedestrian", "method_scenario_dimensions": {"WHERE": "parking"},
            },
            semantic_fingerprint=f"fp-{index:03d}",
            context_resolution={"dimension_bindings": {"WHERE": {"binding_status": "RESOLVED"}}},
        )
        for index in range(count)
    ]


def _assessment(malfunction_id: str, scenario_id: str) -> ScenarioFeasibilityAssessment:
    return ScenarioFeasibilityAssessment(
        malfunction_id=malfunction_id,
        scenario_id=scenario_id,
        physically_feasible=False,
        functionally_relevant=False,
        causally_relevant=False,
        risk_dimensions_changed=[],
        rationale="No accepted causal classification in review persistence test.",
        status=ReviewStatus.PENDING,
        breakpoint="M_TO_B",
    )


class _Agent:
    prompt_version = "scenario-review-test-v1"
    assessment_contract_version = "scenario-causal-assessment-v4"
    client = _Client()

    def __init__(self, *, fail_ids: set[str] = frozenset(), barrier: threading.Barrier | None = None):
        self.fail_ids = fail_ids
        self.barrier = barrier
        self.calls: list[str] = []

    def assess(self, malfunction, candidates, project_registry=None):
        self.calls.append(malfunction.malfunction_id)
        if self.barrier is not None:
            self.barrier.wait(timeout=3)
        if malfunction.malfunction_id in self.fail_ids:
            raise LLMTransportError(
                "simulated DNS failure", attempts=2, category="dns",
                error_counts={"dns": 2},
            )
        return [
            _assessment(malfunction.malfunction_id, scenario.scenario_id)
            for scenario in candidates
        ], {
            "malfunction_id": malfunction.malfunction_id,
            "candidate_count": len(candidates),
            "retained_count": 0,
            "initial_batch_count": 1,
            "leaf_batch_count": 1,
            "llm_calls": 1,
            "leaf_items": [len(candidates)],
            "leaf_input_chars": [100],
        }


def _prepare_writer(writer: ReviewArtifactWriter, malfunctions, candidates) -> None:
    for malfunction in malfunctions:
        writer.record_malfunction(malfunction)
    for candidate in candidates:
        writer.record_scenario_candidate(candidate)


def test_candidate_generation_refreshes_summary_inside_long_scenario_node(review_root):
    writer = ReviewArtifactWriter("c51-candidates", review_root)
    state = HARAState(run_id="c51-candidates", stage=WorkflowStage.MALFUNCTIONS)
    state.malfunctions = [{"malfunction_id": "MF-F01-001"}]
    candidates = _candidates()
    inputs = SemanticWorkflowInputs(
        item_path="item.docx", guidewords=[],
        scenario_candidate_factory=lambda _state: (candidates, {
            "source": "test",
            "scenario_binding_coverage": {
                "dimensions": {"OBJECT": {"resolved": 0, "pending": 45}},
            },
            "scenario_binding_gaps": [{
                "dimension": "OBJECT", "reason": "NO_ITEM_FACT",
            }],
        }),
        review_artifact_writer=writer,
    )

    assert len(_scenario_candidates(state, inputs)) == 45
    summary = ReviewArtifactReader("c51-candidates", review_root).summary()
    assert summary["scenario_candidate_count"] == 45
    gap_payload = json.loads(
        (review_root / "c51-candidates" / "scenario_binding_gaps.json").read_text(
            encoding="utf-8"
        )
    )
    assert gap_payload["gaps"] == [{"dimension": "OBJECT", "reason": "NO_ITEM_FACT"}]


def test_completed_malfunction_is_persisted_before_later_transport_failure(review_root):
    writer = ReviewArtifactWriter("c51-failure", review_root)
    state = HARAState(run_id="c51-failure", stage=WorkflowStage.MALFUNCTIONS)
    malfunctions = [_malfunction("MF-A"), _malfunction("MF-B")]
    candidates = _candidates()
    _prepare_writer(writer, malfunctions, candidates)

    with pytest.raises(LLMTransportError, match="DNS"):
        assess_scenarios(
            state, _Agent(fail_ids={"MF-B"}), malfunctions, candidates,
            max_workers=1, review_artifact_writer=writer,
        )

    records = ReviewArtifactReader("c51-failure", review_root).read_all()
    assert len(records["scenario_feasibility"]) == 45
    assert {item["malfunction_id"] for item in records["scenario_feasibility"]} == {"MF-A"}
    summary = ReviewArtifactReader("c51-failure", review_root).summary()
    assert summary["scenario_candidate_count"] == 45
    assert summary["scenario_feasibility_count"] == 45
    assert summary["per_malfunction_summary"]["MF-A"]["assessment_status"] == "COMPLETED"
    assert summary["per_malfunction_summary"]["MF-B"]["assessment_status"] == "NOT_STARTED"


def test_completed_malfunction_is_readable_while_another_worker_is_running(review_root):
    persisted = threading.Event()
    release_slow_worker = threading.Event()
    slow_worker_started = threading.Event()

    class _Writer(ReviewArtifactWriter):
        def __init__(self):
            super().__init__("c51-running", review_root)
            self._persisted_count = 0

        def record_scenario_feasibility(self, assessment, **kwargs):
            appended = super().record_scenario_feasibility(assessment, **kwargs)
            if appended and assessment.malfunction_id == "MF-A":
                self._persisted_count += 1
                if self._persisted_count == 45:
                    persisted.set()
            return appended

    class _SlowAgent(_Agent):
        def assess(self, malfunction, candidates, project_registry=None):
            if malfunction.malfunction_id == "MF-B":
                slow_worker_started.set()
                assert release_slow_worker.wait(timeout=5)
            return super().assess(malfunction, candidates, project_registry)

    writer = _Writer()
    state = HARAState(run_id="c51-running", stage=WorkflowStage.MALFUNCTIONS)
    malfunctions = [_malfunction("MF-A"), _malfunction("MF-B")]
    candidates = _candidates()
    _prepare_writer(writer, malfunctions, candidates)
    failure: list[BaseException] = []

    def run() -> None:
        try:
            assess_scenarios(
                state, _SlowAgent(), malfunctions, candidates,
                max_workers=2, review_artifact_writer=writer,
            )
        except BaseException as exc:  # pragma: no cover - asserted below
            failure.append(exc)

    worker = threading.Thread(target=run)
    worker.start()
    assert slow_worker_started.wait(timeout=5)
    assert persisted.wait(timeout=5)
    assert worker.is_alive()
    assert len(
        ReviewArtifactReader("c51-running", review_root).read_all()[
            "scenario_feasibility"
        ]
    ) == 45
    release_slow_worker.set()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert failure == []


def test_parallel_completed_workers_append_valid_unique_jsonl(review_root):
    writer = ReviewArtifactWriter("c51-parallel", review_root)
    state = HARAState(run_id="c51-parallel", stage=WorkflowStage.MALFUNCTIONS)
    malfunctions = [_malfunction("MF-A"), _malfunction("MF-B")]
    candidates = _candidates(3)
    _prepare_writer(writer, malfunctions, candidates)

    assess_scenarios(
        state, _Agent(barrier=threading.Barrier(2)), malfunctions, candidates,
        max_workers=2, review_artifact_writer=writer,
    )

    path = review_root / "c51-parallel" / "scenario_feasibility.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()
    parsed = [json.loads(line) for line in lines]
    assert len(parsed) == 6
    assert len({(item["malfunction_id"], item["scenario_id"]) for item in parsed}) == 6


def test_resume_loads_existing_feasibility_without_duplicate_append(review_root):
    writer = ReviewArtifactWriter("c51-resume", review_root)
    state = HARAState(run_id="c51-resume", stage=WorkflowStage.MALFUNCTIONS)
    malfunctions = [_malfunction("MF-A")]
    candidates = _candidates(2)
    _prepare_writer(writer, malfunctions, candidates)
    serialized_checkpoint = {}

    def interrupt_after_checkpoint(current_state):
        serialized_checkpoint.update(current_state.to_dict())
        raise RuntimeError("stop after durable batch")

    with pytest.raises(RuntimeError, match="durable batch"):
        assess_scenarios(
            state, _Agent(), malfunctions, candidates,
            max_workers=1, review_artifact_writer=writer,
            checkpoint=interrupt_after_checkpoint,
        )

    resumed_writer = ReviewArtifactWriter("c51-resume", review_root)
    resumed = HARAState.from_dict(serialized_checkpoint)
    resumed_agent = _Agent()
    assert assess_scenarios(
        resumed, resumed_agent, malfunctions, candidates,
        max_workers=1, review_artifact_writer=resumed_writer,
    ).stage is WorkflowStage.SCORING
    assert resumed_agent.calls == []
    assert len(
        ReviewArtifactReader("c51-resume", review_root).read_all()[
            "scenario_feasibility"
        ]
    ) == 2


def test_cli_does_not_call_zero_coverage_malfunction_completed(review_root):
    writer = ReviewArtifactWriter("c51-status", review_root)
    candidates = _candidates(3)
    malfunctions = [_malfunction("MF-NONE"), _malfunction("MF-PARTIAL"), _malfunction("MF-DONE")]
    _prepare_writer(writer, malfunctions, candidates)
    writer.record_scenario_feasibility(_assessment("MF-PARTIAL", "SCN-000"))
    for candidate in candidates:
        writer.record_scenario_feasibility(_assessment("MF-DONE", candidate.scenario_id))
    writer.write_summary(HARAState(run_id="c51-status", stage=WorkflowStage.MALFUNCTIONS))

    rendered = render_review(ReviewArtifactReader("c51-status", review_root))
    assert "Completed: 1" in rendered
    assert "Partial/in-progress: 1" in rendered
    assert "Not started: 1" in rendered
    assert "MF-NONE: NOT_STARTED (0/3 assessed" in rendered
    assert "MF-PARTIAL: PARTIAL (1/3 assessed" in rendered
    assert "MF-DONE: COMPLETED (3/3 assessed" in rendered


@pytest.mark.parametrize("fm_origin", [False, True])
def test_driver_branches_replace_parent_in_review_progress(tmp_path, fm_origin):
    writer = ReviewArtifactWriter("driver-progress", tmp_path)
    parent = _candidates(1)[0]
    writer.record_malfunction(_malfunction("MF-DRIVER"))
    writer.record_scenario_candidate(parent, generated_for_malfunction_ids=["MF-DRIVER"])
    for index, position in enumerate(("in_driver_seat", "outside_driver_seat")):
        child = replace(
            parent, scenario_id=f"SCN-DRIVER-{index}",
            analysis_instance={
                "malfunction_id": "MF-DRIVER",
                "instance_id": f"SCN-DRIVER-{index}",
                "driver_configuration_branch": {
                    "parent_scenario_id": parent.scenario_id,
                    "driver_position": position,
                },
                **({"origin": "SCENARIO_DEFINED"} if fm_origin else {}),
            },
        )
        writer.record_scenario_candidate(child, generated_for_malfunction_ids=["MF-DRIVER"])
        writer.record_scenario_feasibility(_assessment("MF-DRIVER", child.scenario_id))
    writer.write_summary(HARAState(run_id="driver-progress", stage=WorkflowStage.MALFUNCTIONS))

    reader = ReviewArtifactReader("driver-progress", tmp_path)
    summary = reader.summary()
    assert summary["scenario_candidate_count"] == 3  # Keep the parent for audit.
    assert summary["per_malfunction_summary"]["MF-DRIVER"]["expected_scenario_count"] == 2
    assert summary["scenario_assessment_completed_malfunction_count"] == 1
    assert "MF-DRIVER: COMPLETED (2/2 assessed" in render_review(reader)


def test_candidate_review_projection_compacts_large_shared_method_excerpts(review_root):
    writer = ReviewArtifactWriter("c51-projection", review_root)
    giant_excerpt = "method asset " * 20000
    candidate = {
        "scenario_id": "SCN-1", "semantic_fingerprint": "fp-1",
        "operating_scenario": "Parking", "situational_description": "object ahead",
        "situational_detailing": "active parking", "operating_mode": "Active",
        "facts": {"ego_speed_kph": 10, "weather_conditions": "dry", "object": "pedestrian", "method_scenario_dimensions": {"WHERE": "parking"}},
        "context_resolution": {"dimension_bindings": {"WHERE": {"binding_status": "RESOLVED"}}},
        "sources": [{"source_type": "method_contract", "source_id": "method-hash", "location": "yaml!dimensions.WHERE", "excerpt": giant_excerpt}],
        "fact_provenance": {
            "method_scenario_dimensions": {
                "provenance": "DERIVED", "approval": "PENDING",
                "source_refs": [{"source_type": "method_contract", "source_id": "method-hash", "location": "yaml!dimensions.WHERE", "excerpt": giant_excerpt}],
            },
        },
    }
    raw_chars = len(json.dumps(candidate, ensure_ascii=False, separators=(",", ":")))
    assert writer.record_scenario_candidate(candidate)
    line = (review_root / "c51-projection" / "scenario_candidates.jsonl").read_text(encoding="utf-8").strip()
    projection = json.loads(line)
    assert raw_chars > 400000
    assert len(line) < 10000
    assert len(line) * 50 < raw_chars
    assert "sources" not in projection
    assert projection["method_source_hash"] == "method-hash"
    assert projection["facts"]["object"] == "pedestrian"
    ref = projection["source_refs"][0]
    assert ref["excerpt_sha256"]
    assert ref["excerpt_truncated"] is True
