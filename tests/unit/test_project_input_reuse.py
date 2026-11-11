"""Input identity owns project conventions; shared methods retain their scope."""
from copy import deepcopy
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path

import pytest

from test_autonomous_physics import method, mf, make_candidate, raw_plan, instantiate
from hara_agent.models import MalfunctionCandidate, SourceRef
from hara_agent.services.analysis.project_fact_resolver import effective_project_policy
from hara_agent.services.analysis.scenario_method_service import ScenarioMethodService
from hara_agent.services.analysis.hazardous_event_risk_context_service import HazardousEventRiskContextService
from hara_agent.services.semantic.physical_scenario_agent import PhysicalScenarioAgent
from hara_agent.workflow import HARAState
from hara_agent.workflow.nodes.extraction import read_item_document
from hara_agent.workflow.physical_analysis import construct_physical_scenarios


@pytest.mark.parametrize('identity', ['', 'b' * 64, 'bound'])
def test_project_policy_needs_input_identity_not_its_own_project_name(method, identity):
    policy = method.metadata['project_analysis_policy']
    before = deepcopy(policy)
    digest = policy['bound_item_sha256'][0] if identity == 'bound' else identity
    active = effective_project_policy(policy, digest)
    assert bool(active['engineering_decisions']) == (identity == 'bound')
    assert bool(active.get('driver_location_convention')) == (identity == 'bound')
    for key in ('controllability', 'timing_analysis', 'ego_speed_point_selection', 'representative_motion_rules'):
        assert active[key] == policy[key]
    assert effective_project_policy(active, digest) == active
    assert policy == before


def test_same_malfunction_id_different_input_cannot_inherit_exact_engineering_decision(method):
    policy = method.metadata['project_analysis_policy']
    decision = next(d for d in policy['engineering_decisions'].values() if d.get('approved_template_id'))
    malfunction = MalfunctionCandidate(**{
        key: decision[key] for key in ('malfunction_id', 'function_id', 'guideword_id', 'description',
            'functional_effect', 'vehicle_level_hazard', 'component_category', 'failure_type')},
        guideword='More', causal_chain=['fault', 'effect'], sources=[SourceRef('item_document', 'renamed', 'p1', 'source')])
    bound = replace(method, metadata={**method.metadata, 'project_input_sha256': policy['bound_item_sha256'][0]})
    result = ScenarioMethodService(bound).match_fm_template(malfunction)
    assert result.injectable and result.decision_source_ref
    foreign = replace(bound, metadata={**bound.metadata, 'project_input_sha256': 'b' * 64})
    result = ScenarioMethodService(foreign).match_fm_template(malfunction)
    assert not result.decision_source_ref
    assert result.reason != 'GOVERNED_DECISION_SCOPE_MISMATCH'  # Other method paths still run.


def test_no_runtime_dependency_on_legacy_A5_decision_label(method):
    changed = deepcopy(method)
    changed.metadata['project_analysis_policy']['decisions'] = {}
    assert ScenarioMethodService(changed).driver_brancher.policy_id


def test_reading_renamed_input_preserves_content_identity(tmp_path):
    original = Path(__file__).parents[1] / 'fixtures/avp/ItemDef.docx'
    renamed = tmp_path / 'unrelated-project-name.docx'
    renamed.write_bytes(original.read_bytes())
    a = read_item_document(HARAState('a'), str(original))
    b = read_item_document(HARAState('b'), str(renamed))
    assert a.item_definition['item_sha256'] == b.item_definition['item_sha256'] == hashlib.sha256(original.read_bytes()).hexdigest()
    assert a.item_definition['source_selection']['fingerprint'] == b.item_definition['source_selection']['fingerprint']
    assert a.item_definition['source_id'] != b.item_definition['source_id']


@pytest.mark.parametrize('new_value', [True, False, None])
def test_legacy_control_completion_projects_its_own_braking_source(method, new_value):
    raw = raw_plan()
    raw['hypotheses'][0]['remaining_controls'] = ['braking']
    child = instantiate(method, raw)[0]
    child.facts.pop('emergency_braking_available')
    child.fact_provenance.pop('emergency_braking_available')
    child.facts['control_analysis'] = {'hypothesis': {
        'remaining_controls': ['braking'] if new_value is True else [],
        'failed_controls': ['braking'] if new_value is False else [],
        'unresolved_controls': {'braking': 'independence unknown'} if new_value is None else {},
    }}
    metadata = deepcopy(child.fact_provenance['physical_analysis'])
    metadata['source_refs'] = [dict(metadata['source_refs'][0], location='later-control-assessment')]
    child.fact_provenance['control_analysis'] = metadata
    before = deepcopy(child)
    view = {**child.facts, '_fact_provenance': child.fact_provenance}
    service = HazardousEventRiskContextService(method)
    context = service.build(malfunction_id='MF', scenario_id=child.scenario_id, hazard_node_id='H', scenario=view)
    scored = service.scoring_facts(context, view)
    assert scored.get('emergency_braking_available') is new_value
    if new_value is not None:
        assert scored['_fact_provenance']['emergency_braking_available'] == metadata
    assert child == before


def test_input_hash_is_cache_context_only_and_isolates_same_named_project(method):
    requests = []
    class Captured(Exception):
        pass
    class Client:
        def complete_json(self, request):
            requests.append(request)
            raise Captured
    for digest in ('a' * 64, 'b' * 64):
        state = HARAState('same-names', item_definition={'item_sha256': digest, 'source_id': 'item',
            'blocks': [{'block_id': 'p1', 'kind': 'paragraph', 'location': 'p1',
                'text': '制动踏板允许驾驶员人工制动', 'section_path': []}]})
        with pytest.raises(Captured):
            construct_physical_scenarios(state, {'MF': [make_candidate()]}, [mf()],
                method=method, agent=PhysicalScenarioAgent(Client()), project_facts=None)
        assert len(state.item_definition['physical_analysis_plans']) == 1
    assert requests[0].metadata['physical_plan_key'] != requests[1].metadata['physical_plan_key']
    for request in requests:
        assert 'project_input_sha256' not in request.user_prompt
        assert 'a' * 64 not in request.user_prompt and 'b' * 64 not in request.user_prompt
        assert 'bound_item_sha256' not in request.user_prompt
    # Only the audit plan identity differs; engineering prompt payloads do not.
    payloads = [json.loads(request.user_prompt) for request in requests]
    for payload in payloads:
        payload['context'].pop('plan_key', None)
    assert payloads[0] == payloads[1]


def test_unchanged_model_gap_reuses_plan_across_identity_labels_and_recovery(method, tmp_path):
    from hara_agent.infrastructure.llm import LLMResponse
    from hara_agent.workflow import CheckpointRepository
    class FrozenClient:
        calls = 0
        def complete_json(self, request):
            self.calls += 1
            return LLMResponse({'hypotheses': [], 'uncovered_conditions': ['No applicable physical model for the supplied interaction.']}, 'frozen')
    client = FrozenClient()
    state = HARAState('gap', item_definition={'item_sha256': 'c' * 64})
    parent = make_candidate()
    parents = [parent, replace(parent, scenario_id='display-only-copy', situational_description='another wording')]
    agent = PhysicalScenarioAgent(client, response_checkpoint_dir=tmp_path / 'responses')
    first = construct_physical_scenarios(state, {'MF': parents}, [mf()], method=method, agent=agent, project_facts=None)
    assert client.calls == 1 and len(first['MF']) == 2
    repo = CheckpointRepository(tmp_path / 'checkpoints')
    repo.save(state)
    restored = repo.load('gap')
    second = construct_physical_scenarios(restored, {'MF': parents}, [mf()], method=method, agent=agent, project_facts=None)
    assert client.calls == 1 and second == first
    assert restored.item_definition['physical_analysis_metrics']['physical_solve_points'] == 0
    assert restored.item_definition['physical_analysis_metrics']['semantic_corrections'] == 0
    # A different input with the same labels cannot reuse this proposal cache.
    restored.item_definition['item_sha256'] = 'd' * 64
    construct_physical_scenarios(restored, {'MF': parents}, [mf()], method=method, agent=agent, project_facts=None)
    assert client.calls == 2


@pytest.mark.parametrize('change', ['rename', 'changed_bytes', 'legacy_same_path'])
def test_ordinary_application_resume_checks_content_without_new_cli_input(method, tmp_path, monkeypatch, change):
    from hara_agent.application import HARAApplication
    from hara_agent.config import RunConfig, LLMConfig
    from hara_agent.workflow import CheckpointRepository, WorkflowStage
    original = Path(__file__).parents[1] / 'fixtures/avp/ItemDef.docx'
    item = tmp_path / 'source.docx'
    item.write_bytes(original.read_bytes())
    state = read_item_document(HARAState('resume-input'), str(item),
        project_analysis_policy=method.metadata['project_analysis_policy'])
    state.method_contract = {'template_hash': method.metadata['template_hash']}
    state.stage = WorkflowStage.COMPLETE
    if change == 'legacy_same_path':
        state.item_definition.pop('item_sha256')
    CheckpointRepository(tmp_path / 'checkpoints').save(state)
    if change == 'rename':
        renamed = tmp_path / 'different-project-label.docx'
        renamed.write_bytes(item.read_bytes())
        item = renamed
    elif change == 'changed_bytes':
        # ZIP still reads identical text; same path/fingerprint alone is insufficient.
        item.write_bytes(item.read_bytes() + b'changed-container')
    class NoProvider:
        config = LLMConfig(provider='openai-compatible', base_url='https://example.invalid', model='offline', api_key='unused')
        def complete_json(self, request):
            raise AssertionError('Resume must not request a Provider')
    monkeypatch.setenv('HARA_REVIEW_ARTIFACT_DIR', str(tmp_path / 'review'))
    app = HARAApplication(RunConfig(item_path=item, template_path=None, output_path=tmp_path / 'report.xlsx',
        run_dir=tmp_path / 'checkpoints', resume=True, allow_draft=True, run_id=state.run_id,
        operating_mode='parking_vehicle_control'), NoProvider())
    if change == 'changed_bytes':
        with pytest.raises(ValueError, match='Checkpoint Item Definition'):
            app.run()
    else:
        result = app.run()
        assert result.state.item_definition['item_sha256'] == hashlib.sha256(item.read_bytes()).hexdigest()
        assert result.state.item_definition['source_id'] == 'source.docx'
