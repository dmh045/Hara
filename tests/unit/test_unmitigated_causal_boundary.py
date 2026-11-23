"""Common causal-stage boundaries apply with and without driver comparison."""
from copy import deepcopy
from types import SimpleNamespace
import json
import pytest
from hara_agent.infrastructure.llm import LLMResponse
from hara_agent.services.semantic.scenario_agent import ScenarioFeasibilityAgent,ScenarioSemanticConsistencyError
from hara_agent.services.analysis.scenario_equivalence import exact_task_key
from hara_agent.services.analysis.scenario_physics import causal_physics_conflicts
from hara_agent.services.semantic.scenario_batching import _scenario_json
from test_autonomous_physics import method,instantiate,make_candidate,mf,raw_plan
from test_longitudinal_surface_cache import longitudinal_world,claim


@pytest.mark.parametrize('comparison',[False,True])
def test_available_response_does_not_change_unmitigated_contact_request(method,comparison):
    child=longitudinal_world(method)
    if comparison:child.analysis_instance['driver_reuse_comparison']={'scenario_ids':['outside']}
    before=deepcopy(child)
    class Client:
        config=SimpleNamespace()
        requests=[]
        def complete_json(self,request):
            self.requests.append(request)
            return LLMResponse(data={'assessments':[claim(child)]},model='frozen')
    client=Client();agent=ScenarioFeasibilityAgent(client)
    result,_=agent.assess(mf(),[child])
    assert len(client.requests)==1 and result[0].retain and child==before
    request=client.requests[0]
    assert '不证明已检测、请求或执行' in request.system_prompt
    assert '不得据此否定危险或要求所有缓解失效' in request.system_prompt
    assert '失效到所列运动仍需独立证据' in request.system_prompt
    assert '当前模型已施加的制动、保持或目标反应必须保留' in request.system_prompt
    assert ('driver_reuse_comparison' in json.loads(_scenario_json(mf(),child))) is comparison
    current_key=exact_task_key(agent,mf(),child)
    agent.prompt_version='scenario-feasibility-v32-source-linked-fault-premise'
    assert exact_task_key(agent,mf(),child)!=current_key


@pytest.mark.parametrize('comparison',[False,True])
def test_applied_stopping_stage_still_rejects_contact_claim(method,comparison):
    child=instantiate(method,raw_plan(initial=7,acceleration=-8),make_candidate(7))[0]
    if comparison:child.analysis_instance['driver_reuse_comparison']={'scenario_ids':['outside']}
    assert child.facts['motion_model_outcome']['status']=='STOPPED_BEFORE_CONTACT'
    payload=claim(child)
    payload['causal_chain']['i_to_h']['evidence_refs']=['DERIVED.motion_model_outcome']
    payload['physical_interaction']={'target':'SCENARIO_OBJECT','object_id':child.facts['object_id'],
        'object_position':child.facts['object_position'],'relation':'CONTACT','assessment_reference':'MODEL_TRAJECTORY'}
    assert 'NO_CONTACT_TRAJECTORY_CONTRADICTS_CONTACT_CLAIM' in causal_physics_conflicts(child,payload)
    with pytest.raises(ScenarioSemanticConsistencyError):
        ScenarioFeasibilityAgent._parse(mf(),payload,scenario=child)
