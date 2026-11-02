from hara_agent.models import (
    MalfunctionCandidate,
    ReviewStatus,
    ScenarioCandidate,
    ScenarioFeasibilityAssessment,
)
from hara_agent.workflow import HARAState, WorkflowStage
from hara_agent.workflow.nodes.scenarios import assess_scenarios


class _Client:
    class Config:
        provider = "test"
        base_url = "local"
        model = "checkpoint-model"

    config = Config()


class _Agent:
    prompt_version = "scenario-test-v1"
    assessment_contract_version = "scenario-causal-assessment-v2"

    def __init__(self):
        self.client = _Client()
        self.calls = []

    def assess(self, malfunction, candidates, project_registry=None):
        self.calls.append(malfunction.malfunction_id)
        assessments = [ScenarioFeasibilityAssessment(
            malfunction_id=malfunction.malfunction_id,
            scenario_id=scenario.scenario_id,
            physically_feasible=False,
            functionally_relevant=False,
            causally_relevant=False,
            risk_dimensions_changed=[],
            rationale="No accepted causal classification in checkpoint test.",
            status=ReviewStatus.PENDING,
        ) for scenario in candidates]
        return assessments, {
            "malfunction_id": malfunction.malfunction_id,
            "candidate_count": len(candidates),
            "retained_count": 0,
            "initial_batch_count": 1,
            "leaf_batch_count": 1,
            "llm_calls": 1,
            "leaf_items": [len(candidates)],
            "leaf_input_chars": [100],
        }


def _malfunction(identity):
    return MalfunctionCandidate(
        identity, "F-1", "loss", f"{identity} description", "lost output",
        "vehicle behavior changes", ["lost output", "vehicle behavior changes"],
    )


def test_completed_malfunction_batch_survives_interruption_and_is_reused():
    state = HARAState(run_id="incremental", stage=WorkflowStage.MALFUNCTIONS)
    malfunctions = [_malfunction("MF-1"), _malfunction("MF-2")]
    candidates = [ScenarioCandidate(
        "SCN-1", "Parking", "parking", "active parking",
        {"operating_mode": "Active"}, semantic_fingerprint="scenario-fp",
    )]
    first_agent = _Agent()
    serialized_checkpoint = {}

    def interrupt_after_save(current_state):
        serialized_checkpoint.update(current_state.to_dict())
        raise RuntimeError("simulated interruption")

    try:
        assess_scenarios(
            state, first_agent, malfunctions, candidates,
            max_workers=1, checkpoint=interrupt_after_save,
        )
    except RuntimeError as error:
        assert str(error) == "simulated interruption"
    else:
        raise AssertionError("simulated interruption was not raised")

    restored = HARAState.from_dict(serialized_checkpoint)
    assert first_agent.calls == ["MF-1"]
    assert "MF-1" in restored.item_definition["scenario_assessment_batches"]

    resumed_agent = _Agent()
    completed = assess_scenarios(
        restored, resumed_agent, malfunctions, candidates,
        max_workers=1,
    )

    assert resumed_agent.calls == ["MF-2"]
    assert completed.stage is WorkflowStage.SCORING
    assert len(completed.item_definition["scenario_assessments"]) == 2
    assert "scenario_assessment_batches" not in completed.item_definition


def _signal_parallel_drain(monkeypatch, event):
    from concurrent.futures import ThreadPoolExecutor
    from hara_agent.workflow.nodes import parallel
    class Executor(ThreadPoolExecutor):
        def submit(self,*args,**kwargs):
            future=super().submit(*args,**kwargs);cancel=future.cancel
            def signal():
                event.set()
                return cancel()
            future.cancel=signal
            return future
    monkeypatch.setattr(parallel,'ThreadPoolExecutor',Executor)


def test_parallel_failure_still_persists_running_successful_workers(monkeypatch):
    from threading import Barrier,Event
    from hara_agent.workflow.nodes.parallel import ordered_parallel_map
    import pytest
    entered=Barrier(2);draining=Event();saved=[]
    _signal_parallel_drain(monkeypatch,draining)
    def worker(item):
        entered.wait(timeout=5)
        if item=='bad':
            raise ValueError('original worker failure')
        assert draining.wait(timeout=5)
        return 'completed result'
    with pytest.raises(ValueError,match='original worker failure'):
        ordered_parallel_map(['bad','good'],worker,max_workers=2,
            on_result=lambda item,result:saved.append((item,result)))
    assert saved==[('good','completed result')]


def test_parallel_drain_preserves_original_error_if_checkpoint_callback_fails(monkeypatch):
    from threading import Barrier,Event
    from hara_agent.workflow.nodes.parallel import ordered_parallel_map
    import pytest
    entered=Barrier(2);draining=Event()
    _signal_parallel_drain(monkeypatch,draining)
    def worker(item):
        entered.wait(timeout=5)
        if item=='bad':raise ValueError('original worker failure')
        assert draining.wait(timeout=5)
        return item
    def save(item,result):raise OSError('checkpoint not writable')
    with pytest.raises(ValueError,match='original worker failure') as caught:
        ordered_parallel_map(['bad','good'],worker,max_workers=2,on_result=save)
    assert any('checkpoint not writable' in x for x in caught.value.__notes__)
