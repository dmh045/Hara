"""Source qualification survives extraction, ordinary scheduling and replay."""
from copy import deepcopy
from dataclasses import asdict, replace
import json

import pytest

from hara_agent.models import ItemDefinitionFacts, SourceRef
from hara_agent.models.item_facts import scoped_odd_context
from hara_agent.services.semantic.item_definition_agent import ItemDefinitionNormalizer
from hara_agent.services.semantic.core_item_artifact_contract import normalize_core_item_artifacts
from hara_agent.services.analysis.physical_hypothesis_instantiation_service import PhysicalHypothesisInstantiationService
from hara_agent.contracts.physical_scenario_plan import PhysicalScenarioPlan
from hara_agent.infrastructure.llm import LLMResponse
from hara_agent.services.semantic.physical_scenario_agent import PhysicalScenarioAgent
from hara_agent.workflow import HARAState
from hara_agent.workflow.physical_analysis import construct_physical_scenarios
from test_autonomous_physics import raw_plan, make_candidate, mf
from test_p7_closure import method
from test_core_item_artifact_contract import _payload
from test_p6_scenario_pipeline import parent as p6_parent, project as p6_project, function as p6_function, malfunction as p6_malfunction
from hara_agent.services.analysis import MethodScenarioCandidateService


def scope(function="renamed-function", mode=""):
    return dict(field="odd_road_types", values=["垂直车位"], function_ids=[function] if function else [],
                operating_modes=[mode] if mode else [], source_location="p4",
                source_excerpt="Exploratory parking is supported in perpendicular parking spaces.")


def facts(function="renamed-function", mode=""):
    record = ItemDefinitionNormalizer.scope_evidence([scope(function, mode)], "renamed.pdf", [
        {"location":"p4", "text":scope()["source_excerpt"]}])[0]
    return ItemDefinitionFacts("parking", "vehicle", odd_road_types=["垂直车位"],
        odd_scope_evidence=[record], sources=[SourceRef("item_definition", "renamed.pdf", "p4")])


def test_scope_preserves_legacy_union_but_projects_function_and_phase_without_mutation():
    original = facts(mode="execution")
    before = deepcopy(original)
    assert original.for_scope("other", "execution").odd_road_types == []
    assert original.for_scope("other", "execution").odd_scope_evidence == []
    assert original.for_scope("renamed-function", "training").odd_road_types == []
    assert original.for_scope("renamed-function", "training").odd_scope_evidence == []
    assert original.for_scope("renamed-function", "execution").odd_road_types == ["垂直车位"]
    assert original.for_scope("renamed-function", "execution").odd_scope_evidence == original.odd_scope_evidence
    assert original == before
    restored = ItemDefinitionFacts.from_dict(json.loads(json.dumps(asdict(original))))
    assert restored.for_scope("other", "execution").odd_road_types == []
    legacy = replace(original, odd_scope_evidence=[])
    assert legacy.for_scope("other").odd_road_types == ["垂直车位"]


def test_independent_global_evidence_preserves_same_value_in_other_functions():
    value = asdict(facts())
    value["odd_scope_evidence"].append({"field":"odd_road_types", "values":["垂直车位"], "sources":[]})
    assert scoped_odd_context(value, "other")["odd_road_types"] == ["垂直车位"]
    assert scoped_odd_context(value, "other")["odd_scope_evidence"] == [value['odd_scope_evidence'][-1]]


def test_scope_projects_unqualified_and_duplicate_global_records_without_removing_authority():
    global_record={'field':'odd_road_types','values':['垂直车位'],'sources':[]}
    value=asdict(facts(mode='execution'))
    value['odd_scope_evidence'].extend([deepcopy(global_record),deepcopy(global_record)])
    before=deepcopy(value)
    view=scoped_odd_context(value,'another-function','training')
    assert view['odd_road_types']==['垂直车位']
    assert view['odd_scope_evidence']==[global_record,global_record]
    assert value==before
    legacy={'odd_road_types':['斜列车位']}
    assert scoped_odd_context(legacy,'any','any')==legacy


def test_function_and_mode_qualifiers_select_the_same_evidence_and_value_view():
    records=[{'field':'odd_weather_conditions','values':['rain'],'function_ids':['F1'],'sources':[]},
        {'field':'odd_weather_conditions','values':['snow'],'operating_modes':['training'],'sources':[]},
        {'field':'odd_weather_conditions','values':['fog'],'function_ids':['F1'],'operating_modes':['execution'],'sources':[]}]
    context={'odd_weather_conditions':['rain','snow','fog'],'odd_scope_evidence':records}
    for function,mode,expected in [('F1','execution',['rain','fog']),('F2','training',['snow']),('F2','execution',[])]:
        view=scoped_odd_context(context,function,mode)
        assert view['odd_weather_conditions']==expected
        assert [v for r in view['odd_scope_evidence'] for v in r['values']]==expected


def test_contract_and_normalizer_preserve_qualified_source_clause():
    payload = _payload()
    payload['item_definition']['odd']['scope_evidence'] = [scope("F01")]
    normalized, _ = normalize_core_item_artifacts(payload)
    value, warnings = ItemDefinitionNormalizer._parse_with_warnings(normalized['item_definition'], 'new.pdf',
        [{'location':'p4','text':scope()['source_excerpt']}])
    assert not warnings
    assert value.odd_scope_evidence[0]['function_ids'] == ['F01']
    assert value.odd_scope_evidence[0]['sources'][0]['excerpt'] == scope()['source_excerpt']
    bad = scope();bad['source_excerpt'] = 'Unsupported broader project scope'
    with pytest.raises(ValueError, match='source clause is not present'):
        ItemDefinitionNormalizer.scope_evidence([bad], 'new.pdf', [{'text':scope()['source_excerpt']}])


def test_instantiation_rejects_other_function_condition_without_changing_parent(method):
    raw = raw_plan(assessed=True)
    raw['hypotheses'][0]['operating_condition'] = dict(source_field='odd_road_types',value='垂直车位',
        physical_effect='垂直车位规定有限接触路径上的目标位置')
    plan = PhysicalScenarioPlan.from_dict(raw,evidence_ids={'MF.description'})
    parent = make_candidate();before=deepcopy(parent)
    children, errors = PhysicalHypothesisInstantiationService(method).instantiate(parent,mf(),plan,
        plan_key='scope',response_model='frozen',project_context=asdict(facts()))
    assert not children
    assert any('OPERATING_CONDITION_OUTSIDE_ITEM_SCOPE' in e['reason'] for e in errors)
    allowed, errors = PhysicalHypothesisInstantiationService(method).instantiate(parent,replace(mf(),function_id='renamed-function'),plan,
        plan_key='scope',response_model='frozen',project_context=asdict(facts()))
    assert len(allowed)==1 and not errors
    assert parent == before


@pytest.mark.parametrize('applicable', [False, True])
def test_ordinary_physics_projects_function_scope_and_resumes_without_extra_requests(method,tmp_path,applicable):
    class Client:
        def __init__(self):self.requests=[]
        def complete_json(self,request):
            self.requests.append(request)
            assert ('source_condition_focus' in request.user_prompt) is applicable
            raw = raw_plan(assessed=True)
            if applicable:
                raw['hypotheses'][0]['operating_condition'] = dict(source_field='odd_road_types',value='垂直车位',
                    physical_effect='垂直车位规定有限接触路径上的目标位置')
            else:
                assert '"odd_road_types":[]' in request.user_prompt
            return LLMResponse(raw,'offline')
    client=Client();agent=PhysicalScenarioAgent(client,response_checkpoint_dir=tmp_path)
    parent=make_candidate();state=HARAState('scope-case')
    fault=replace(mf(),function_id='renamed-function') if applicable else mf()
    out=construct_physical_scenarios(state,{'MF':[parent]},[fault],method=method,agent=agent,project_facts=facts())
    assert len(out['MF'])==1 and len(client.requests)==1
    context=next(iter(state.item_definition['physical_analysis_plans'].values()))['context']
    assert bool(context['project_context']['odd_scope_evidence']) is applicable
    restored=HARAState.read_committed(json.loads(json.dumps(state.to_dict())))
    replay=construct_physical_scenarios(restored,{'MF':[parent]},[fault],method=method,agent=agent,project_facts=facts())
    assert replay==out and len(client.requests)==1


def test_mixed_parent_modes_equal_separate_scope_generation_without_cross_mode_merge(method):
    project = p6_project()
    project.odd_road_types = ['垂直车位']
    record = ItemDefinitionNormalizer.scope_evidence([scope('F-P6','execution')], 'renamed.pdf')[0]
    project.odd_scope_evidence = [record]
    parents = [replace(p6_parent('training-parent'),operating_mode='training'),
               replace(p6_parent('execution-parent'),operating_mode='execution')]
    before = deepcopy(parents)
    service = MethodScenarioCandidateService(method)
    together,audit = service.generate_for_malfunction(p6_malfunction(),parents,project_facts=project,function=p6_function())
    separate = []
    raw_count = 0
    for parent in parents:
        generated,subaudit = service.generate_for_malfunction(p6_malfunction(),[parent],project_facts=project,function=p6_function())
        assert generated
        raw_count += subaudit['atom_generation']['raw_candidate_count']
        assert all(c.facts['source_road_type_scope'] == (['垂直车位'] if parent.operating_mode=='execution' else [])
                   for c in generated)
        assert all(c.facts['odd_scope_evidence'] == ([record] if parent.operating_mode=='execution' else [])
                   for c in generated)
        separate.extend(generated)
    assert {c.scenario_id for c in together} == {c.scenario_id for c in separate}
    assert audit['atom_generation']['raw_candidate_count'] == raw_count
    assert {row['operating_mode'] for row in audit['atom_generation']['source_focus_allocation']} == {'execution'}
    assert all(c.analysis_instance.get('source_condition_focus', {}).get('value') != '垂直车位'
               for c in together if c.operating_mode == 'training')
    assert set(audit['atom_generation']['source_boundary_parent_ids_by_mode']) == {'training','execution'}
    assert parents == before


@pytest.mark.parametrize('with_item', [False, True])
def test_offline_generation_replays_draft_without_inventing_missing_speed(method, tmp_path, monkeypatch, with_item):
    from pathlib import Path
    from types import SimpleNamespace
    import hashlib
    import scripts.benchmark_p6_offline as benchmark
    from hara_agent.application import HARAApplication

    project = p6_project()
    project.speed_min_kph = project.speed_max_kph = None
    item_path = tmp_path / 'relocated.pdf'
    item_path.write_bytes(b'Frozen original document identity')
    source_hash = hashlib.sha256(item_path.read_bytes()).hexdigest()
    saved = dict(run_id='original', item_definition={'typed': asdict(project), 'item_sha256': source_hash},
                 functions=[asdict(p6_function())], malfunctions=[asdict(p6_malfunction())],
                 scenarios=[], risk_results=[], audit_trail=[], method_contract={'method_source_hash': 'historical'})
    checkpoint = tmp_path / 'checkpoint.json'
    checkpoint.write_text(json.dumps(saved))
    before = checkpoint.read_bytes()
    monkeypatch.setattr(benchmark.MethodSourceResolver, 'resolve', lambda *args, **kwargs: SimpleNamespace(method=method))
    monkeypatch.setattr(benchmark.DocumentReader, 'read', lambda *args: SimpleNamespace(blocks=[], source_id='relocated.pdf'))
    calls = []
    prepare = HARAApplication.prepare_scenario_candidates

    def ordinary_prepare(application, state, service):
        calls.append(state.item_definition['item_sha256'])
        result = prepare(application, state, service)
        assert service.method.metadata['project_input_sha256'] == source_hash
        assert application.resolve_project_speed_context(state).matched_envelope == {}
        return result

    monkeypatch.setattr(HARAApplication, 'prepare_scenario_candidates', ordinary_prepare)
    result = benchmark.run(checkpoint, Path('unused-method'), Path('unused-template'), tmp_path / 'generation',
                           item=item_path if with_item else None,
                           generation_only=True, operating_mode='parking vehicle control')
    assert calls == [source_hash]
    assert result['provider_calls'] == result['new_committed_risk_records'] == 0
    assert result['candidate_preview']['candidate_scenarios_after_driver'] > 0
    assert result['new_causal_provider_validation'] == 'NOT_RUN'
    assert checkpoint.read_bytes() == before
    if with_item:
        item_path.write_bytes(b'Different content at the same path')
        with pytest.raises(ValueError, match='checkpoint input hash'):
            benchmark.run(checkpoint, Path('unused-method'), Path('unused-template'), tmp_path/'changed-item',
                          item=item_path, generation_only=True, operating_mode='parking vehicle control')
        assert calls == [source_hash]


@pytest.mark.parametrize('layout', ['parallel', 'diagonal'])
def test_ordinary_candidate_accepts_analytical_geometry_without_claiming_source_permission(method, tmp_path, layout):
    from pathlib import Path
    from hara_agent.application import HARAApplication
    from hara_agent.config import RunConfig
    from test_reference_interaction_physics import angle_plan, parameters

    project = facts(function='source-scoped-exploratory')
    state = HARAState('geometry-scope', item_definition={'typed': asdict(project)},
                      functions=[asdict(p6_function())])
    application = HARAApplication(RunConfig(item_path=Path('source.pdf'), template_path=None,
        run_dir=tmp_path, output_path=tmp_path/'unused.xlsx',
        operating_mode='parking_vehicle_control', allow_draft=True), None)
    service = MethodScenarioCandidateService(method)
    parents, _ = application.prepare_scenario_candidates(state, service)
    fault = replace(mf(), function_id=p6_function().function_id)
    generated, _ = service.generate_for_malfunction(fault, parents, project_facts=project, function=p6_function())
    parent = generated[0]
    before = deepcopy(parent)
    raw = angle_plan()
    h = raw['hypotheses'][0]
    h.pop('safe_state_proposal'); h.pop('hazard_timing')
    h.update(remaining_controls=[], unresolved_controls={'braking': 'No independent response is specified by this geometry fixture.'},
             driver_positions=['in_driver_seat'],
             condition_changes={'initial_state': True, 'target': True,
                 'rationale': 'A separate conditional geometric world, not a source-stated parking layout.'})
    h['geometry']['object_id'] = layout+'-assumed-target'
    h['geometry'].update(road_user_type='PEDESTRIAN', collision_type='VEHICLE_TO_ROAD_USER')
    h['rationale'] = 'Conditional '+layout+' parking geometry for the existing fault; not an extracted product permission.'
    h['conditions'] = [h['rationale']]
    if layout == 'parallel':
        h['geometry'].update(object_position='FRONT')
        parameters(raw, ego_heading_angle_deg=0, object_offset_x_m=5, object_offset_y_m=0)

    class Client:
        def __init__(self): self.requests=[]
        def complete_json(self, request):
            self.requests.append(request)
            payload=json.loads(request.user_prompt)
            from hara_agent.services.semantic.physical_scenario_agent import expand_physical_prompt_view
            context = expand_physical_prompt_view(payload['context'], payload['source_texts'])
            assert context['project_context']['odd_road_types'] == []
            assert 'source_condition_focus' not in context
            # The fixture chooses a compatible shared compound operation from
            # the actual bounded catalog, without fabricating a method identity.
            allowed = context['operating_atom_options']
            dynamic_ids = {row['atom_id'] for row in allowed['EGO_DYNAMICS']}
            operation = next(row['atom_id'] for row in allowed['EGO_ACTION']
                if row['atom_id'] in dynamic_ids and row.get('speed_range_kph')
                and row['speed_range_kph'][0] == 0 and row['speed_range_kph'][1] >= 7)
            response = deepcopy(raw)
            response['hypotheses'][0]['operating_atoms'] = {
                'EGO_ACTION': operation, 'EGO_DYNAMICS': operation, 'TRAFFIC_PATTERN': None}
            response['hypotheses'][0]['operating_atom_exclusions'] = {
                'TRAFFIC_PATTERN': 'No catalog-specific traffic relation in this stationary-target geometric assumption.'}
            return LLMResponse(response, 'OFFLINE')

    client = Client()
    agent = PhysicalScenarioAgent(client, response_checkpoint_dir=tmp_path/'responses')
    result = construct_physical_scenarios(state, {fault.malfunction_id:[parent]}, [fault],
        method=method, agent=agent, project_facts=project)
    child = result[fault.malfunction_id][0]
    assert len(client.requests) == 1, [json.loads(r.user_prompt).get('repair_errors') for r in client.requests]
    assert 'physical_analysis' in child.facts, state.item_definition['physical_analysis_metrics']
    assert child.facts['motion_model_outcome']['status'] == 'CONTACT'
    assert child.fact_provenance['ego_heading_angle_deg']['provenance'] == 'LLM_INFERENCE'
    assert child.fact_provenance['ego_heading_angle_deg']['approval'] == 'PENDING'
    assert 'selected_source_condition' not in child.facts
    assert child.facts['source_road_type_scope'] == []
    assert state.item_definition['typed']['odd_road_types'] == ['垂直车位']
    assert parent == before
    restored = HARAState.read_committed(json.loads(json.dumps(state.to_dict())))
    resumed = construct_physical_scenarios(restored, {fault.malfunction_id:[parent]}, [fault],
        method=method, agent=agent, project_facts=project)
    assert resumed == result and len(client.requests) == 1
