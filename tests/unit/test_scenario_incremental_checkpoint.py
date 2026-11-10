from hara_agent.models import (
    MalfunctionCandidate,
    ReviewStatus,
    ScenarioCandidate,
    ScenarioFeasibilityAssessment,
)
from hara_agent.workflow import HARAState, WorkflowStage
from hara_agent.workflow.nodes.scenarios import assess_scenarios
from unittest.mock import patch
import pytest


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


def test_prepared_conditions_use_existing_causal_checkpoint_without_regeneration():
    state = HARAState(run_id="prepared", stage=WorkflowStage.MALFUNCTIONS)
    candidate = ScenarioCandidate("SCN-saved", "Parking", "parking", "active parking",
        {"driver_in_vehicle": True}, semantic_fingerprint="saved-world",
        analysis_instance={"malfunction_id": "MF-1"})
    agent = _Agent()
    saved = {}
    def interrupt(current):
        saved.update(current.to_dict())
        raise RuntimeError("saved interruption")
    kwargs = dict(malfunctions=[_malfunction("MF-1")], candidates=[],
        prepared_candidates_by_malfunction={"MF-1": [candidate]})
    with patch('hara_agent.workflow.nodes.scenarios.MethodScenarioCandidateService') as generator:
        with pytest.raises(RuntimeError, match='saved interruption'):
            assess_scenarios(state, agent, checkpoint=interrupt, **kwargs)
        restored = HARAState.from_dict(saved)
        assess_scenarios(restored, agent, **kwargs)
        assert agent.calls == ["MF-1"]
        generator.assert_not_called()
    assert restored.item_definition['scenario_assessments'][0]['scenario_id'] == 'SCN-saved'
    candidate.analysis_instance['malfunction_id'] = 'OTHER'
    with pytest.raises(ValueError, match='malfunction scope'):
        assess_scenarios(HARAState('wrong'), _Agent(), **kwargs)


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


def test_empty_selected_batches_do_not_checkpoint_or_count_as_causal_work(monkeypatch):
    from types import SimpleNamespace
    from dataclasses import replace
    from hara_agent.workflow.nodes import scenarios
    import pytest
    source = ScenarioCandidate('base', 'Parking', 'parking', 'active', {'operating_mode':'Active'})
    class Candidates:
        def __init__(self, method):pass
        def generate_for_malfunction(self, malfunction, candidates, **kwargs):
            return [replace(source, scenario_id=malfunction.malfunction_id)], {
                'situation_selection': {'malfunction_id': malfunction.malfunction_id}}
    monkeypatch.setattr(scenarios, 'MethodScenarioCandidateService', Candidates)
    method = SimpleNamespace(compiler_version='test', contract_version='test', metadata={})
    malfunctions = [_malfunction(f'MF-{i}') for i in range(5)]
    state = HARAState('empty-scoped-batches', stage=WorkflowStage.MALFUNCTIONS)
    saved = [];progress=[]
    def save(s):
        saved.append(s.to_dict())
        raise RuntimeError('interrupt after first nonempty batch')
    first = _Agent()
    with pytest.raises(RuntimeError, match='interrupt'):
        assess_scenarios(state, first, malfunctions, [source], method=method,
            selected_generated_scenario_ids=['MF-2','MF-4'], checkpoint=save)
    assert first.calls == ['MF-2'] and len(saved)==1
    assert set(saved[0]['item_definition']['scenario_assessment_batches']) == {'MF-2'}
    event = next(e for e in saved[0]['audit_trail'] if e['event']=='scenario_malfunction_batch_checkpointed')
    assert event['completed_malfunction_count']==1 and event['total_malfunction_count']==2
    restored = HARAState.from_dict(saved[0]);second = _Agent();saved.clear()
    assess_scenarios(restored, second, malfunctions, [source], method=method,
        selected_generated_scenario_ids=['MF-2','MF-4'], checkpoint=lambda s:saved.append(s.to_dict()),
        progress=lambda *event:progress.append(event))
    assert second.calls == ['MF-4'] and len(saved)==1
    assert progress == [('scenario',2,2)]
    rows=restored.item_definition['malfunction_situation_selection']['per_malfunction']
    assert {r['malfunction_id'] for r in rows if r['causal_status']=='NOT_SELECTED'} == {'MF-0','MF-1','MF-3'}
    assert {r['scenario_id'] for r in restored.item_definition['scenario_assessments']} == {'MF-2','MF-4'}
    assert restored.stage is WorkflowStage.SCORING
