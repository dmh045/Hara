"""R1–R7 acceptance controls: observe requests, identities and unmodified values."""
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace
import time

import pytest

from hara_agent.infrastructure.llm import LLMResponse, LLMRequest
from hara_agent.infrastructure.llm.response_checkpoint import ResponseCheckpoint
from hara_agent.services.analysis.scenario_equivalence import exact_task_key, scenario_conditions, signature
from hara_agent.services.reporting.quantity_format import quantity, quantity_range
from hara_agent.services.reporting import HARAReportProjectionService, load_report_schema
from hara_agent.services.semantic import ScenarioFeasibilityAgent
from hara_agent.workflow.nodes.scenarios import _assess_instantiated_scenarios
from test_p6_scenario_pipeline import parent, malfunction, causal_payload, SOURCE
from test_reviewer_report_closure import _p6_report_state, _method, _pending


class CountingClient:
    config = SimpleNamespace(provider="OFFLINE", model="frozen", base_url="local", max_tokens=8192)
    def __init__(self, candidates, mode="exact"):
        self.by_id = {c.scenario_id: c for c in candidates}
        self.calls = []
        self.mode = mode

    def complete_json(self, request):
        self.calls.append(request)
        raw, _ = json.JSONDecoder().raw_decode(request.user_prompt.split("Scenarios=", 1)[1])
        items = raw['scenarios'] if isinstance(raw, dict) else raw
        outputs = []
        for item in items:
            candidate = self.by_id[item['scenario_id']]
            data = causal_payload(candidate, driver_relevant=self.mode == "driver_relevant")
            data['physical_interaction'] = {'target': 'OTHER', 'object_id': candidate.facts.get('object_id', candidate.scenario_id + ':target')}
            data['dependency_metadata'] = {'complete': False}
            if self.mode in {"local", "driver_relevant"}:
                data['dependency_metadata']['driver_noninterference'] = {
                    aspect: {'independent': self.mode == 'local', 'reason': 'Fixture explicit two-position control policy supports this step',
                             'evidence_refs': ['SCN.driver_control_policy']}
                    for aspect in ('enablement','trigger','interaction','hazard')}
            if self.mode == 'pending':
                data.update(causally_relevant=False, breakpoint='B_TO_I', hazardous_event='')
                data['causal_chain']['b_to_i'] = {'claim':'No interaction inputs', 'basis_type':'ASSUMPTION','evidence_refs':[]}
                data['causal_chain'].pop('i_to_h')
            outputs.append(data)
        return LLMResponse({'assessments':outputs}, 'frozen', usage={'offline_scripted':True})


def pair():
    a = deepcopy(parent('A'))
    a.analysis_instance['p6_generation'] = True
    a.facts.update(driver_position='in_driver_seat', driver_control_policy='Autonomous motion enabled in both positions; trigger/output/interaction do not depend on driver action')
    a.fact_provenance['driver_control_policy'] = deepcopy(a.fact_provenance['vehicle_state'])
    b = deepcopy(a)
    b.scenario_id = 'B'
    b.facts['driver_position'] = 'outside_driver_seat'
    return a, b


def run_candidates(candidates, mode='exact', cache=None):
    client = CountingClient(candidates, mode)
    agent = ScenarioFeasibilityAgent(client, batch_max_items=1, response_checkpoint_dir=cache)
    results, audit = _assess_instantiated_scenarios(agent, malfunction(), candidates, {}, None)
    return client, results, audit


@pytest.mark.parametrize('mode', ['exact','pending'])
def test_T04_T06_T12_exact_and_pending_tasks_reuse_without_global_proof(mode):
    a = parent('A')
    b = deepcopy(a); b.scenario_id='B'; b.situational_description='Another presentation only'
    before = deepcopy([a,b])
    client, results, audit = run_candidates([a,b],mode)
    assert len(client.calls) == 1
    assert audit['causal_reuse']['exact_input_reuse'] == 1
    assert [a,b] == before
    assert [r.scenario_id for r in results] == ['A','B']
    assert results[1].causal_assessment.scenario_id == 'B'
    assert results[1].physical_interaction['object_id'] == 'B:target'
    assert results[1].dependency_metadata['complete'] is False
    assert results[1].retain is (mode != 'pending')


def test_T05_inflight_response_checkpoint_miss_is_shared(tmp_path):
    class Client:
        config=CountingClient.config
        calls=0
        def complete_json(self, request):
            self.calls += 1
            time.sleep(.03)
            return LLMResponse({'result':'pending'},'frozen')
    client=Client(); request=LLMRequest('task','system','same inputs','schema','v1')
    with ThreadPoolExecutor(max_workers=4) as pool:
        results=list(pool.map(lambda _:ResponseCheckpoint(tmp_path).complete(client,request),range(4)))
    assert client.calls==1
    assert sum(bool(result.usage.get('response_checkpoint_hit')) for result in results)==3


@pytest.mark.parametrize('change', ['precision','grade','object','phase','source'])
def test_T07_T08_changed_engineering_inputs_are_not_exact_tasks(change):
    a=parent('A'); b=deepcopy(a); b.scenario_id='B'
    if change=='precision':
        a.facts['ttc_s']=3.9999; b.facts['ttc_s']=4.0001
        assert quantity(a.facts['ttc_s'],'s')==quantity(b.facts['ttc_s'],'s')
    elif change=='grade':
        a.facts['ego_heading_relative_to_downhill']='UPHILL';b.facts['ego_heading_relative_to_downhill']='DOWNHILL'
    elif change=='object':
        a.facts['object_id']='actual-target-A';b.facts['object_id']='actual-target-B'
    elif change=='phase':
        a.facts['impact_time_id']='pre-fault';b.facts['impact_time_id']='post-fault'
    else:
        b.fact_provenance['vehicle_state']['source_refs'][0]['excerpt']='A different condition'
    agent=ScenarioFeasibilityAgent(CountingClient([a,b]))
    assert exact_task_key(agent,malfunction(),a)!=exact_task_key(agent,malfunction(),b)
    client, _, audit=run_candidates([a,b])
    assert len(client.calls)==2
    assert audit['causal_reuse']['exact_input_reuse']==0


@pytest.mark.parametrize('mode,calls,reuse', [('local',1,1),('driver_relevant',2,0)])
def test_T09_T10_T11_paired_mode_executes_local_proof_without_complete(mode,calls,reuse):
    a,b=pair()
    client, results, audit=run_candidates([a,b],mode)
    assert ScenarioFeasibilityAgent.paired_driver_requests
    assert len(client.calls)==calls
    assert audit['causal_reuse']['complete_causal_reuse']==reuse
    assert results[1].causal_assessment.scenario_id=='B'
    assert all(r.dependency_metadata['complete'] is False for r in results)
    assert len(results)==2  # both downstream risk inputs survive


@pytest.mark.parametrize('mode,calls,reuse', [('local',1,1),('driver_relevant',2,0)])
def test_generated_contact_identity_does_not_block_driver_reuse(mode, calls, reuse):
    a, b = pair()
    for candidate in (a, b):
        phase = 'CONTACT-' + candidate.scenario_id
        candidate.facts['impact_time_id'] = phase
        candidate.fact_provenance['vehicle_state']['input_fact_metadata'] = [
            {'impact_time_id': phase, 'time_role': 'FIRST_CONTACT'}]
    before = deepcopy([a, b])
    client, results, audit = run_candidates([a, b], mode)
    assert len(client.calls) == calls
    assert audit['causal_reuse']['complete_causal_reuse'] == reuse
    assert [r.scenario_id for r in results] == ['A', 'B']
    assert [a, b] == before
    # Aliasing is local to a signature; two references to different stages
    # must not become references to the same stage.
    a.facts['phase_links'] = [{'impact_time_id': 'CONTACT-A'}]
    b.facts['phase_links'] = [{'impact_time_id': 'CONTACT-B-later'}]
    assert scenario_conditions(a, without_driver=True) != scenario_conditions(b, without_driver=True)


def test_T13_T14_changed_interaction_reuses_only_upstream_step():
    a,b=pair(); b.facts['object_position']='REAR'; a.facts['object_position']='FRONT'
    b.facts['longitudinal_acceleration_mps2']=2.5
    client, results, audit=run_candidates([a,b])
    assert len(client.calls)==2
    assert audit['causal_reuse']['complete_causal_reuse']==0
    assert audit['causal_reuse']['partial_causal_reuse']==1
    assert 'shared_causal_evidence' in client.calls[1].user_prompt
    assert results[1].physical_interaction['object_id']=='B:target'


def test_T15_T16_resume_and_display_changes_retain_request_identity(tmp_path):
    a,b=pair()
    first,_,_=run_candidates([a,b],'local',tmp_path)
    a.situational_description='Reformatted display';b.situational_description='Another display'
    second,_,audit=run_candidates([a,b],'local',tmp_path)
    assert len(first.calls)==1 and len(second.calls)==0
    assert audit['response_checkpoint_hits']==1
    b.facts['vehicle_state']='standby'
    third,_,_=run_candidates([a,b],'local',tmp_path)
    assert len(third.calls)==1


@pytest.mark.parametrize('value,unit,expected', [(10.182338,'km/h','10.2'),(.154286,'s','.15'),(1.454723,'m/s²','1.45'),
    (0,'s','0'),(.0001,'s','< 0.01'),(-.0001,'m/s²','−0.01 < 值 < 0'),(None,'s','待确定'),(float('inf'),'s','无有限值')])
def test_T23_T24_quantity_display(value,unit,expected):
    assert quantity(value,unit)==('0'+expected if expected.startswith('.') else expected)


def test_T25_rounded_boundary_keeps_inequality_and_range():
    assert quantity(4.004,'s',boundary=4)=='> 4'
    assert quantity(3.999,'s',boundary=4)=='< 4'
    assert '端点舍入后相同' in quantity_range(4.001,4.004,'s',lower_inclusive=False)


def test_T17_T19_T20_pending_review_group_preserves_identities_and_combinations():
    state=_p6_report_state()
    for r in state.risk_results: r.asil=_pending()
    state.scenarios[0].facts.update(object_position='FRONT',ego_longitudinal_direction='FORWARD')
    for c in state.scenarios[1:]:c.facts.update(object_position='REAR',ego_longitudinal_direction='REVERSE')
    before=deepcopy(state)
    view=HARAReportProjectionService(load_report_schema()).project(state,_method())
    assert len(view.rows)==2 and len(view.scenario_details)==3
    groups=view.projection_metrics['grouping_map']
    assert {tuple(g['combinations']) for g in groups}=={('前进＋前方目标',),('倒车＋后方目标',)}
    assert state==before
    assert all(d.asil=='待评定' for d in view.scenario_details)
    assert {sid for g in groups for sid in g['scenario_ids']}=={'SCN-1','SCN-2','SCN-3'}


def test_T18_highest_asil_uses_its_entire_tuple_and_handles_ties():
    state=_p6_report_state()
    a,b=state.risk_results[:2]
    a.branch_evaluations=[{'S':'S3','E':'E1','C':'C1','ASIL':'A'}]
    b.branch_evaluations=[{'S':'S2','E':'E4','C':'C3','ASIL':'C'}]
    entries=[{'risk':r,'trace':{}} for r in (a,b)]
    values,_=HARAReportProjectionService._highest_known_tuple(entries,['QM','A','B','C','D'])
    assert values==('S2','E4','C3','C')
    a.branch_evaluations[0].update(ASIL='C')
    values,_=HARAReportProjectionService._highest_known_tuple(entries,['QM','A','B','C','D'])
    assert values==('S3','E1','C1','C')


def test_T15_exact_task_checkpoint_rebinds_renamed_ids_and_detects_conflict():
    a=parent('A'); b=deepcopy(a);b.scenario_id='B'
    _,results,audit=run_candidates([a,b])
    prior={'assessments':[r.to_dict() for r in results],'audit':audit}
    c=deepcopy(a);c.scenario_id='C'
    client=CountingClient([c]);agent=ScenarioFeasibilityAgent(client,batch_max_items=1)
    restored,newaudit=_assess_instantiated_scenarios(agent,malfunction(),[c],{},None,prior_batch=prior)
    assert not client.calls
    assert restored[0].scenario_id==restored[0].causal_assessment.scenario_id=='C'
    assert restored[0].physical_interaction['object_id']=='C:target'
    assert newaudit['causal_reuse']['exact_task_checkpoint_hits']==1
    prior['assessments'][1]['causal_chain']['m_to_b']['claim']='conflicting historical claim'
    _,conflict=_assess_instantiated_scenarios(agent,malfunction(),[c],{},None,prior_batch=prior)
    assert len(client.calls)==1
    assert any(r['reason']=='HISTORICAL_EXACT_RESULT_CONFLICT' for r in conflict['causal_reuse']['records'])


def test_T21_T22_goal_scope_uses_all_associations_and_keeps_original():
    from hara_agent.services.reporting.engineering_text_mapper import EngineeringReportTextMapper
    mapper=EngineeringReportTextMapper()
    original='避免某一地下停车场的指定场景发生事件'
    display=mapper.safety_intent(['驻车保持请求丢失','驻车保持请求丢失'],original)
    assert display=='防止驻车保持请求丢失。'
    assert '地下停车场' not in display
    display=mapper.safety_intent(['保持请求丢失','制动输出过量'],original)
    assert '保持请求丢失' in display and '制动输出过量' in display
    assert '停车' not in display and '毫秒' not in display


@pytest.fixture(scope='module')
def active_method():
    from hara_agent.method_sources import MethodSourceResolver
    return MethodSourceResolver().resolve(template_path=None,
        baseline_manifest_path='method_assets/fusa_baseline_v1/manifest.yaml',
        report_template_path='references/HARA_Template_AI_20260327.xlsx').method


def test_T18_production_projection_never_pairs_max_s_with_max_asil(active_method):
    state=_p6_report_state()
    state.risk_results[0].branch_evaluations=[
        {'branch_id':'more-severe','S':'S3','E':'E1','C':'C1','ASIL':'A'},
        {'branch_id':'highest-asil','S':'S2','E':'E4','C':'C3','ASIL':'C',
         'relative_speed_kph':10.182338, 'ttc_s':.154286}]
    before=deepcopy(state)
    view=HARAReportProjectionService(load_report_schema()).project(state,active_method)
    assert (view.rows[0].severity,view.rows[0].exposure,view.rows[0].controllability,view.rows[0].asil)==('S2','E4','C3','C')
    assert 'highest-asil' not in view.rows[0].scenario_detail
    assert 'highest-asil' in view.scenario_details[0].branch_results
    assert view.scenario_details[0].severity=='S2'
    assert 'more-severe' in view.scenario_details[0].branch_results
    assert state==before


def test_T25_projection_explains_rounding_against_compiled_boundaries(active_method):
    state=_p6_report_state()
    bound=next(b.upper_ttc_s for b in active_method.structured_risk_method.controllability_profile.bands if b.upper_ttc_s and b.upper_ttc_s>0)
    state.scenarios[0].facts['ttc_s']=bound+.0001
    view=HARAReportProjectionService(load_report_schema()).project(state,active_method)
    assert f'TTC > {bound:g} s' in view.scenario_details[0].analysis_basis
    assert '未舍入' in view.scenario_details[0].analysis_basis


def test_T28_conditional_qm_and_pending_goal_retain_independent_status(active_method):
    from hara_agent.models import EvidenceValue, ReviewStatus, SafetyGoal
    state=_p6_report_state()
    state.risk_results[0].asil=EvidenceValue('QM',ReviewStatus.PENDING,review_reason='条件性假设')
    for r in state.risk_results:r.safety_goal_id='SG-1'
    state.safety_goals=[SafetyGoal('SG-1','历史原文','待确定','QM',['SCN-1','SCN-2','SCN-3'],ReviewStatus.PENDING)]
    state.item_definition['safety_goal_catalog']={'SG-1':{'original_safety_goal':'原始安全意图文本'}}
    view=HARAReportProjectionService(load_report_schema()).project(state,active_method)
    assert view.safety_goals[0].max_asil=='QM'
    assert view.safety_goals[0].status=='条件性草稿'
    assert view.safety_goals[0].original_text=='原始安全意图文本'
    assert '未计算成员' in view.projection_metrics['safety_goal_scope_note']
    assert state.risk_results[0].asil.status is ReviewStatus.PENDING


def test_T29_pending_rationale_does_not_dump_raw_model_fields():
    from hara_agent.services.reporting.engineering_text_mapper import EngineeringReportTextMapper
    text=EngineeringReportTextMapper().controllability_rationale(_pending(), {
        'input_diagnostic':{'reason':'MODEL_NOT_APPLICABLE'},
        'known_physical_inputs':{'ego_speed_kph':10.182338, 'object_position':'LEFT',
            'fault_motion_model':'CONSTANT_LATERAL_ACCELERATION_TO_CONTACT'}})
    assert '10.2 km/h' in text and '左侧' in text and '侧向交互' in text
    assert 'fault_motion_model' not in text and 'CONSTANT_' not in text and '10.182338' not in text
