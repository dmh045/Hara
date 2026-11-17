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
    assert original.for_scope("renamed-function", "training").odd_road_types == []
    assert original.for_scope("renamed-function", "execution").odd_road_types == ["垂直车位"]
    assert original == before
    restored = ItemDefinitionFacts.from_dict(json.loads(json.dumps(asdict(original))))
    assert restored.for_scope("other", "execution").odd_road_types == []
    legacy = replace(original, odd_scope_evidence=[])
    assert legacy.for_scope("other").odd_road_types == ["垂直车位"]


def test_independent_global_evidence_preserves_same_value_in_other_functions():
    value = asdict(facts())
    value["odd_scope_evidence"].append({"field":"odd_road_types", "values":["垂直车位"], "sources":[]})
    assert scoped_odd_context(value, "other")["odd_road_types"] == ["垂直车位"]


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
        separate.extend(generated)
    assert {c.scenario_id for c in together} == {c.scenario_id for c in separate}
    assert audit['atom_generation']['raw_candidate_count'] == raw_count
    assert set(audit['atom_generation']['source_boundary_parent_ids_by_mode']) == {'training','execution'}
    assert parents == before
