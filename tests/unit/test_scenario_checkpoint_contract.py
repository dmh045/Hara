from hara_agent.services.semantic.scenario_evidence import (
    SCENARIO_ASSESSMENT_CONTRACT_VERSION,
)
from hara_agent.services.semantic.scenario_contract import SCENARIO_CONTRACT_VERSION
from hara_agent.workflow import HARAState, WorkflowStage


def test_old_causal_assessment_checkpoint_is_invalidated_without_losing_malfunctions():
    value = {
        "run_id": "contract-upgrade",
        "stage": "quality_gate",
        "item_definition": {
            "scenario_assessments": [{"legacy": True}],
            "typed": {
                "risk_facts": [
                    {"fact_id": "RF-SCENARIO", "produced_by": "scenario-risk-facts-v2"},
                    {"fact_id": "RF-PROJECT", "produced_by": "source_extraction"},
                ],
                "method_risk_fact_bindings": [
                    {"source_fact_id": "RF-SCENARIO"},
                    {"source_fact_id": "RF-PROJECT"},
                ],
            },
        },
        "functions": [{"function_id": "F01"}],
        "malfunctions": [{"malfunction_id": "MF-F01-001"}],
        "pending_reviews": [
            {"field": "item_definition", "reason": "preserve"},
            {"field": "scenarios", "reason": "stale"},
            {"field": "scenario_risk_facts", "reason": "stale"},
            {"assessment_id": "RA-1", "field": "severity", "reason": "stale"},
            {"safety_goal_id": "SG-1", "field": "safety_goal", "reason": "stale"},
        ],
        "scenario_contract_version": SCENARIO_CONTRACT_VERSION,
        "scenario_assessment_contract_version": "scenario-causal-assessment-v1",
    }

    state = HARAState.from_dict(value)

    assert SCENARIO_ASSESSMENT_CONTRACT_VERSION == "scenario-causal-assessment-v4"
    assert state.stage is WorkflowStage.MALFUNCTIONS
    assert state.functions == [{"function_id": "F01"}]
    assert state.malfunctions == [{"malfunction_id": "MF-F01-001"}]
    assert "scenario_assessments" not in state.item_definition
    assert state.item_definition["typed"]["risk_facts"] == [
        {"fact_id": "RF-PROJECT", "produced_by": "source_extraction"},
    ]
    assert state.item_definition["typed"]["method_risk_fact_bindings"] == [
        {"source_fact_id": "RF-PROJECT"},
    ]
    assert state.pending_reviews == [
        {"field": "item_definition", "reason": "preserve"},
    ]
    assert any(
        item.get("event") == "scenario_checkpoint_invalidated"
        for item in state.audit_trail
    )


def test_single_pass_checkpoint_matches_previous_encoding_and_keeps_snapshot_isolated(tmp_path):
    from copy import deepcopy
    from dataclasses import asdict, dataclass
    from enum import Enum
    import json
    from hara_agent.workflow import CheckpointRepository
    from hara_agent.models import ReviewStatus, SourceRef
    from test_autonomous_physics import make_candidate
    @dataclass
    class SavedAudit:
        value: float
        status: ReviewStatus
        source: SourceRef
    state = HARAState('serialization-equivalence', stage=WorkflowStage.MALFUNCTIONS,
        scenarios=[make_candidate()], functions=[{'function_id': 'F'}],
        item_definition={'audit': SavedAudit(-.0000001, ReviewStatus.PENDING,
            SourceRef('item_definition', 'source', 'paragraph', '原文与否定条件')),
            'interval': (0., 2.0000001), 'responses': [{'usage': {'prompt_tokens': 12}, 'data': [True, None]}]},
        errors=[{'reason': 'same exception'}])
    def legacy(value):
        if isinstance(value, Enum): return value.value
        if isinstance(value, dict): return {k: legacy(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)): return [legacy(v) for v in value]
        return value
    expected = legacy(asdict(state)); expected['can_publish'] = state.can_publish
    snapshot = state.to_dict()
    assert snapshot == expected
    assert json.dumps(snapshot, ensure_ascii=False, separators=(',', ':')) == json.dumps(expected, ensure_ascii=False, separators=(',', ':'))
    snapshot['item_definition']['responses'][0]['usage']['prompt_tokens'] = 999
    snapshot['scenarios'][0]['facts']['ego_speed_kph'] = 99
    assert state.to_dict() == expected
    repo = CheckpointRepository(tmp_path)
    before = deepcopy(expected)
    repo.save(state)
    assert json.loads(repo.path_for(state.run_id).read_text()) == before
    assert repo.load(state.run_id).to_dict() == before
    import pytest
    state.item_definition['not_json'] = {1, 2}
    with pytest.raises(TypeError):
        repo.save(state)
    assert json.loads(repo.path_for(state.run_id).read_text()) == before
