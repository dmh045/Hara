"""Optional control proposals flow through existing physical/SG/report services."""
from copy import deepcopy
from dataclasses import asdict
import json

import pytest

from test_autonomous_physics import method, raw_plan, make_candidate, mf, score
from hara_agent.contracts.physical_scenario_plan import PhysicalScenarioPlan
from hara_agent.infrastructure.llm import LLMResponse
from hara_agent.models import ReviewStatus
from hara_agent.services.analysis import MethodSafetyGoalService, StructuredRiskScoringService
from hara_agent.services.analysis.asil_service import MethodContractASILService
from hara_agent.services.reporting import HARAReportProjectionService, load_report_schema
from hara_agent.services.semantic.physical_scenario_agent import PhysicalScenarioAgent
from hara_agent.services.semantic.scenario_agent import ScenarioFeasibilityAgent
from hara_agent.workflow import HARAState
from hara_agent.workflow.nodes.safety_goals import aggregate_safety_goals
from hara_agent.workflow.nodes.scoring import score_structured_scenarios
from hara_agent.workflow.physical_analysis import construct_physical_scenarios


def proposal_plan():
    raw = raw_plan('CONSTANT_LATERAL_ACCELERATION_TO_CONTACT')
    h = raw['hypotheses'][0]
    h['remaining_controls'] = ['braking', 'parking_hold']
    h['intervention_assumptions'] = {field: {
        'value': False, 'rationale': 'bounded offline case with no remote control or external escape',
        'basis_refs': ['SCENARIO.conditions'], 'conditions': ['no remote intervention; enclosed target vehicle'],
    } for field in ('remote_intervention_available', 'other_road_user_avoidance_possible')}
    h['safe_state_proposal'] = {
        'description': '车辆停止并保持静止', 'required_controls': ['braking', 'parking_hold'],
        'basis_refs': ['PROJECT.operating_context'], 'conditions': ['独立制动与驻车通道保持可用'],
    }
    return raw


@pytest.mark.parametrize('inside,expected_c', [(False, 'C3'), (True, '')])
def test_workflow_proposals_reach_existing_c_and_sg_with_resume(method, tmp_path, inside, expected_c):
    class Frozen:
        calls = 0
        def complete_json(self, request):
            self.calls += 1
            assert 'safe_state_proposal' in request.system_prompt
            return LLMResponse(proposal_plan(), 'OFFLINE_SYNTHETIC')
    client = Frozen()
    agent = PhysicalScenarioAgent(client, response_checkpoint_dir=tmp_path)
    parent = make_candidate(3.6)
    parent.facts.update(driver_in_vehicle=inside,
                        driver_position='in_driver_seat' if inside else 'outside_driver_seat')
    # The proposal produces these conditions; they are not prefilled C inputs.
    for field in ('remote_intervention_available', 'other_road_user_avoidance_possible'):
        parent.facts.pop(field, None)
        parent.fact_provenance.pop(field, None)
    fault = mf()
    fault.description = 'steering command exceeds demand'
    fault.functional_effect = 'unintended lateral deviation'
    state = HARAState('control-proposal', functions=[{'function_id': 'F', 'name': '转向控制'}],
                      malfunctions=[asdict(fault)])
    def generate():
        return construct_physical_scenarios(state, {'MF': [parent]}, [fault],
            method=method, agent=agent, project_facts=None)['MF'][0]
    child = generate()
    result = score(method, child)
    assert result['controllability']['controllability_score'] == expected_c
    assert result['controllability']['decision_status'] == ('MODEL_NOT_APPLICABLE' if inside else 'OVERRIDE_MATCH')
    assert result['controllability']['engineering_status'] == 'PENDING'
    assert 'ttc_s' not in child.facts  # lateral first contact never becomes longitudinal TTC
    assert child.fact_provenance['remote_intervention_available']['approval'] == 'PENDING'
    state.scenarios = [child]
    causal_response = dict(scenario_id=child.scenario_id, physically_feasible=True, functionally_relevant=True,
        causally_relevant=True, breakpoint='NONE', hazardous_event='Conditional steering deviation contacts the side target',
        causal_chain={
            'm_to_b': dict(claim='excessive steering causes lateral deviation', basis_type='DIRECT_FACT', evidence_refs=['MF.description']),
            'b_to_i': dict(claim='trajectory reaches the side target', basis_type='DERIVED_PHYSICS', evidence_refs=['DERIVED.relative_speed_kph']),
            'i_to_h': dict(claim='contact threatens occupants', basis_type='DERIVED_PHYSICS', evidence_refs=['DERIVED.relative_speed_kph'])},
        risk_dimension_changes=[], confidence=.9, status='FINALIZED')
    assessment = ScenarioFeasibilityAgent._parse(fault, causal_response, scenario=child)
    state.item_definition['scenario_assessments'] = [assessment.to_dict()]
    score_structured_scenarios(state, StructuredRiskScoringService(method), MethodContractASILService(method))
    assert len(state.risk_results) == 1
    assert (state.risk_results[0].controllability.value or '') == expected_c
    aggregate_safety_goals(state, MethodSafetyGoalService(method))
    assert state.safety_goals[0].safe_state == '车辆停止并保持静止'
    assert state.safety_goals[0].status is ReviewStatus.PENDING
    view = HARAReportProjectionService(load_report_schema()).project(state, method)
    assert '候选：车辆停止并保持静止' in view.safety_goals[0].safe_state
    assert view.projection_metrics['safe_state_evidence'][state.safety_goals[0].sg_id]['candidates']
    restored = HARAState.read_committed(json.loads(json.dumps(state.to_dict())))
    assert restored.safety_goals == state.safety_goals
    generate()
    assert client.calls == 1


def test_proposed_intervention_cannot_override_an_existing_source_condition(method):
    from hara_agent.services.analysis.physical_hypothesis_instantiation_service import PhysicalHypothesisInstantiationService
    parent = make_candidate(3.6)
    raw = proposal_plan()
    raw['hypotheses'][0]['intervention_assumptions']['other_road_user_avoidance_possible']['value'] = True
    plan = PhysicalScenarioPlan.from_dict(raw, evidence_ids={
        'MF.description', 'SCENARIO.conditions', 'PROJECT.operating_context'})
    children, errors = PhysicalHypothesisInstantiationService(method).instantiate(parent, mf(), plan,
        plan_key='conflict', response_model='OFFLINE')
    assert not children and errors
    assert 'other_road_user_avoidance_possible' in str(errors)
    assert parent.facts['other_road_user_avoidance_possible'] is False


@pytest.mark.parametrize('new_value', [True, None])
def test_new_target_does_not_inherit_old_target_evasion(method, new_value):
    from hara_agent.services.analysis.physical_hypothesis_instantiation_service import PhysicalHypothesisInstantiationService
    parent = make_candidate(3.6)
    raw = proposal_plan(); h = raw['hypotheses'][0]
    h['condition_changes'] = dict(target=True, initial_state=False, rationale='different independent target condition')
    h['geometry']['object_id'] = 'new-target'
    if new_value is None:
        del h['intervention_assumptions']['other_road_user_avoidance_possible']
    else:
        h['intervention_assumptions']['other_road_user_avoidance_possible']['value'] = new_value
    plan = PhysicalScenarioPlan.from_dict(raw, evidence_ids={
        'MF.description', 'SCENARIO.conditions', 'PROJECT.operating_context'})
    children, errors = PhysicalHypothesisInstantiationService(method).instantiate(parent, mf(), plan,
        plan_key='new-target', response_model='OFFLINE')
    assert children and not errors
    assert children[0].facts.get('other_road_user_avoidance_possible') is new_value
    assert parent.facts['other_road_user_avoidance_possible'] is False


@pytest.mark.parametrize('mutation', ['failed', 'unknown_ref', 'rating', 'remote', 'missing_value'])
def test_proposals_do_not_invent_capabilities_or_approval(mutation):
    raw = proposal_plan(); h = raw['hypotheses'][0]
    if mutation == 'failed': h['safe_state_proposal']['required_controls'] = ['steering']
    elif mutation == 'unknown_ref': h['safe_state_proposal']['basis_refs'] = ['invented']
    elif mutation == 'rating': h['safe_state_proposal']['status'] = 'FINALIZED'
    elif mutation == 'remote': h['intervention_assumptions']['remote_intervention_available']['value'] = True
    else: del h['intervention_assumptions']['remote_intervention_available']['value']
    with pytest.raises(ValueError):
        PhysicalScenarioPlan.from_dict(raw, evidence_ids={'MF.description', 'SCENARIO.conditions', 'PROJECT.operating_context'})


def test_legacy_plan_still_reads_and_does_not_fabricate_state_or_intervention():
    raw = raw_plan()
    plan = PhysicalScenarioPlan.from_dict(raw, evidence_ids={'MF.description'})
    assert plan.to_dict() == raw
    assert 'safe_state_proposal' not in plan.hypotheses[0]


def test_driver_only_safe_state_is_not_applied_to_outside_driver_child(method):
    from test_report_review_corrections import mixed_pending_group
    state = mixed_pending_group()
    for child in state.scenarios:
        child.facts.update(driver_in_vehicle=False, physical_analysis={
            'plan_key': 'offline', 'hypothesis_id': 'driver-only',
            'hypothesis': {'safe_state_proposal': {
                'description': '驾驶员制动后停车', 'required_controls': ['driver_braking'],
                'basis_refs': ['PROJECT.operating_context'], 'conditions': ['驾驶员在车内'],
            }},
        })
    aggregate_safety_goals(state, MethodSafetyGoalService(method))
    assert all(g.safe_state == '待确定' for g in state.safety_goals)
    assert all(not g['safe_state_candidates'] for g in state.item_definition['safety_goal_catalog'].values())
