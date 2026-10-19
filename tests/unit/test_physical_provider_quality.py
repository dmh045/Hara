"""Frozen, unedited real proposals; source generation uses ordinary entry points."""
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace

from hara_agent.application import HARAApplication
from hara_agent.config import RunConfig
from hara_agent.infrastructure.llm import LLMResponse
from hara_agent.models import ItemDefinitionFacts, MalfunctionCandidate, ReviewStatus, SourceRef
from hara_agent.services.analysis import MethodScenarioCandidateService
from hara_agent.services.analysis.malfunction_situation_selection import selection_function
from hara_agent.services.analysis.scenario_equivalence import signature
from hara_agent.services.semantic.physical_scenario_agent import PhysicalScenarioAgent, expand_physical_prompt_view
from hara_agent.workflow import HARAState
from hara_agent.workflow.physical_analysis import construct_physical_scenarios
from test_reuse_reporting_acceptance import active_method


def test_real_three_models_keep_geometry_and_reduce_only_redundant_sampling(active_method,tmp_path):
    fixture=json.loads((Path(__file__).parents[1]/'fixtures/physical_provider_quality_20261019.json').read_text())
    state=HARAState.read_committed(fixture['state'])
    project=ItemDefinitionFacts.from_dict(state.item_definition['typed'])
    service=MethodScenarioCandidateService(active_method)
    app=HARAApplication(RunConfig(Path('input/ItemDef.docx'),None,tmp_path/'unused.xlsx',tmp_path,
        operating_mode='parking_vehicle_control',allow_draft=True),None)
    parents,_=app.prepare_scenario_candidates(state,service)
    mfs=[MalfunctionCandidate(**{**m,'status':ReviewStatus(m['status']),'sources':[SourceRef(**s) for s in m['sources']]}) for m in state.malfunctions]
    candidates={}
    for mf in mfs:
        function=next(f for f in state.functions if f['function_id']==mf.function_id)
        generated,_=service.generate_for_malfunction(mf,parents,project_facts=project,function=selection_function(function))
        candidates[mf.malfunction_id]=[c for c in generated if c.scenario_id in fixture['selected_ids']]
    before=deepcopy(candidates)
    class Frozen:
        config=SimpleNamespace(provider='OFFLINE_FROZEN_REAL',model='recorded')
        calls=0
        def complete_json(self,request):
            self.calls+=1
            payload=json.loads(request.user_prompt)
            context=expand_physical_prompt_view(payload['context'],payload['source_texts'])
            match=next(p for p in fixture['plans'] if p['malfunction_id']==context['malfunction']['malfunction_id']
                and signature(p['conditions'])==signature(context['conditions']))
            return LLMResponse(**deepcopy(match['response']))
    client=Frozen()
    generated=construct_physical_scenarios(state,deepcopy(candidates),mfs,method=active_method,
        agent=PhysicalScenarioAgent(client),project_facts=project)
    assert candidates==before and client.calls==6
    metrics=state.item_definition['physical_analysis_metrics']
    assert metrics['diagnostics']==[] and metrics['physical_plan_reuse']==6
    children=[c for group in generated.values() for c in group]
    assert len(children)==24  # same real replies formerly produced 32
    physical=[c for c in children if c.facts.get('physical_analysis')]
    assert len(physical)==22
    assert Counter(c.facts['fault_motion_model'] for c in physical)=={
        'CONSTANT_ACCELERATION_TO_IMPACT_DISTANCE':12,
        'CONSTANT_LATERAL_ACCELERATION_TO_CONTACT':6,
        'GRADE_ROLLAWAY_TO_IMPACT_DISTANCE':4}
    for c in physical:
        info=c.facts['physical_analysis']
        assert info['engineering_approval']=='NOT_GRANTED'
        assert info['sensitivity']['input_completeness']['classification']=='COMPUTABLE_REPRESENTATIVE'
        assert c.facts['motion_model_outcome']['status']=='CONTACT'
        assert c.facts['relative_speed_kph']>0
        if 'LATERAL' in c.facts['fault_motion_model']:
            assert 'ttc_s' not in c.facts and c.facts['first_contact_time_s']>0
        else:
            assert c.facts['ttc_s']>0
        if 'ROLLAWAY' in c.facts['fault_motion_model']:
            assert c.facts['ego_speed_kph']==0 and c.facts['slope_percent']==15
            assert c.facts['downhill_body_direction']=='FRONT'
    ranges=[c.facts['physical_analysis']['sample_parameters']['longitudinal_acceleration_mps2']
        for c in physical if c.source_scenario_id=='SCN-ANALYTICAL-553F4E2BB72FF4A2']
    assert set(ranges)=={.5,1,2}
