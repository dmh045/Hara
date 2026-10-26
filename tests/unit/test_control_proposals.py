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


def control_only_plan(*, braking=True):
    return {"hypotheses": [], "uncovered_conditions": [], "control_assessment": {
        "rationale": "命令异常与执行器能力分开；独立人工通道为条件性假设",
        "basis_refs": ["MF.description"], "conditions": ["独立人工制动通道未受自动指令故障影响"],
        "uncovered_conditions": ["未证明车辆实测能力"],
        "failed_controls": ["automatic_braking_command"] if braking else ["braking"],
        "remaining_controls": ["driver_braking"] if braking else [],
        "intervention_assumptions": {field: {
            "value": False, "rationale": "限定为无可用远程路径、目标无规避空间的代表工况",
            "basis_refs": ["SCENARIO.conditions"], "conditions": ["无远程干预；目标无规避空间"]}
            for field in ("remote_intervention_available", "other_road_user_avoidance_possible")}}}


def existing_motion(inside=True):
    from test_p7_closure import source
    parent = make_candidate(7)
    parent.facts.update(driver_in_vehicle=inside, relative_speed_kph=7., relative_distance_m=12.,
        ttc_s=12 / (7 / 3.6), collision_type="FRONTAL", object_position="FRONT",
        collision_branches=[{"branch_id": "existing-contact", "mechanism_id": "MF:CONTACT",
                             "physically_possible": True}])
    parent.fact_provenance.update({field: source() for field in parent.facts})
    return parent


@pytest.mark.parametrize('inside,braking,expected', [(True, True, 'C0'), (False, True, 'C3'), (True, False, '')])
def test_existing_motion_gets_only_missing_controls_then_scores_and_resumes(method, tmp_path, inside, braking, expected):
    from hara_agent.services.analysis.physical_hypothesis_instantiation_service import PhysicalHypothesisInstantiationService
    class Frozen:
        calls = 0
        def complete_json(self, request):
            self.calls += 1
            content = json.loads(request.user_prompt)
            assert content['context']['task_scope'] == 'CONTROL_INPUTS_ONLY'
            assert not content['model_catalog']
            return LLMResponse(control_only_plan(braking=braking), 'OFFLINE_SYNTHETIC')
    client = Frozen()
    parent = existing_motion(inside)
    # No hidden prefilled control facts: the bounded agent must produce them.
    for field in ('remote_intervention_available', 'other_road_user_avoidance_possible'):
        parent.facts.pop(field); parent.fact_provenance.pop(field)
    original = deepcopy(asdict(parent))
    state = HARAState('control-only', malfunctions=[asdict(mf())])
    agent = PhysicalScenarioAgent(client, response_checkpoint_dir=tmp_path)
    def run(candidates):
        return construct_physical_scenarios(state, {'MF': candidates}, [mf()], method=method,
            agent=agent, project_facts=None)['MF']
    child = run([parent])[0]
    assert child.scenario_id != parent.scenario_id
    assert child.status is ReviewStatus.PENDING
    assert asdict(parent) == original
    for field in ('relative_speed_kph', 'relative_distance_m', 'ttc_s', 'collision_branches'):
        assert child.facts[field] == parent.facts[field]
        assert child.fact_provenance[field] == parent.fact_provenance[field]
    before, after = score(method, parent), score(method, child)
    assert after['controllability']['controllability_score'] == expected
    assert before['severity'] == after['severity']
    assert before['exposure'] == after['exposure']
    assert after['controllability']['engineering_status'] == 'PENDING'
    assert 'physical_analysis' not in child.facts
    assert child.fact_provenance['emergency_braking_available']['approval'] == 'PENDING'
    assert state.item_definition['physical_analysis_metrics']['physical_solve_points'] == 0
    assert state.item_definition['physical_analysis_metrics']['control_only_instances'] == 1
    assert run([parent])[0] == child  # response/plan recovery
    assert run([child])[0] == child  # an assessed gap is not re-requested
    assert client.calls == 1
    assert PhysicalHypothesisInstantiationService(method).task_scope(child, 'MF') == ''
    state.scenarios = [child]
    restored = HARAState.read_committed(json.loads(json.dumps(state.to_dict())))
    assert restored.scenarios == state.scenarios


def test_control_only_preserves_conflicting_project_and_old_hypothesis(method):
    from test_autonomous_physics import instantiate
    from test_p7_closure import source
    from hara_agent.services.analysis.physical_hypothesis_instantiation_service import PhysicalHypothesisInstantiationService
    producer = PhysicalHypothesisInstantiationService(method)
    plan = PhysicalScenarioPlan.from_dict(control_only_plan(), evidence_ids={'MF.description', 'SCENARIO.conditions'})
    parents = [existing_motion(), instantiate(method)[0], existing_motion()]
    parents[0].facts['emergency_braking_available'] = False
    parents[0].fact_provenance['emergency_braking_available'] = source()
    parents[1].facts['physical_analysis']['hypothesis']['failed_controls'] = ['braking']
    parents[2].facts['collision_branches'][0].update(emergency_braking_available=False,
        _fact_provenance={'emergency_braking_available': source()})
    for parent in parents:
        before = deepcopy(asdict(parent))
        children, errors = producer.instantiate(parent, mf(), plan, plan_key='controls', response_model='OFFLINE')
        assert not children and errors[0]['reason_code'] == 'CONFLICT_WITH_SELECTED_CONDITION'
        assert asdict(parent) == before


def test_control_only_contract_rejects_motion_and_rating_payloads():
    for key, value in [('parameters', []), ('controllability', 'C1'), ('approved', True)]:
        raw = control_only_plan(); raw['control_assessment'][key] = value
        with pytest.raises(ValueError):
            PhysicalScenarioPlan.from_dict(raw, evidence_ids={'MF.description', 'SCENARIO.conditions'})
    raw = control_only_plan(); raw['hypotheses'] = raw_plan()['hypotheses']
    with pytest.raises(ValueError, match='preserve existing motion'):
        PhysicalScenarioPlan.from_dict(raw, evidence_ids={'MF.description', 'SCENARIO.conditions'})


def test_existing_motion_two_drivers_share_controls_without_reusing_causal_identity(method):
    from hara_agent.services.analysis.scenario_equivalence import exact_task_key
    class Frozen:
        calls = 0
        def complete_json(self, request):
            self.calls += 1
            return LLMResponse(control_only_plan(), 'OFFLINE_SYNTHETIC')
    parents = [existing_motion(True), existing_motion(False)]
    parents[1].scenario_id = 'SC-OUTSIDE'
    for parent in parents:
        for field in ('remote_intervention_available', 'other_road_user_avoidance_possible'):
            parent.facts.pop(field); parent.fact_provenance.pop(field)
    client = Frozen()
    state = HARAState('control-sharing', malfunctions=[asdict(mf())])
    children = construct_physical_scenarios(state, {'MF': parents}, [mf()], method=method,
        agent=PhysicalScenarioAgent(client), project_facts=None)['MF']
    assert client.calls == 1 and len(children) == 2
    assert len({c.scenario_id for c in children}) == 2
    assert len({exact_task_key(ScenarioFeasibilityAgent(client), mf(), c) for c in children}) == 2
    assert {c.facts['driver_in_vehicle']: score(method, c)['controllability']['controllability_score']
            for c in children} == {True: 'C0', False: 'C3'}
    assert state.item_definition['physical_analysis_metrics']['control_plan_reuse'] == 1
    assert state.item_definition['physical_analysis_metrics']['physical_plan_reuse'] == 0


def test_existing_physical_proposal_does_not_fan_out_unresolved_controls(method):
    from test_autonomous_physics import instantiate
    from hara_agent.services.analysis.physical_hypothesis_instantiation_service import PhysicalHypothesisInstantiationService
    child = instantiate(method)[0]  # Valid motion, no declared remaining braking capability.
    assert not score(method, child)['controllability']['controllability_score']
    assert PhysicalHypothesisInstantiationService(method).task_scope(child, 'MF') == ''


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


@pytest.mark.parametrize('inside,remote,expected', [(True, None, 'C0'), (False, None, ''), (False, False, 'C3')])
def test_generated_braking_capability_and_driver_access_are_separate(method, tmp_path, inside, remote, expected):
    raw = raw_plan(initial=7, acceleration=0)
    h = raw['hypotheses'][0]
    h['parameters'][1]['value'] = 12
    h['remaining_controls'] = ['braking']
    if remote is not None:
        h['intervention_assumptions'] = {'remote_intervention_available': {
            'value': remote, 'rationale': 'offline scope explicitly has no remote brake',
            'basis_refs': ['SCENARIO.conditions'], 'conditions': ['no remote intervention']}}
    class Frozen:
        calls = 0
        def complete_json(self, request):
            self.calls += 1
            assert 'emergency_braking_available' in request.user_prompt
            return LLMResponse(deepcopy(raw), 'OFFLINE_SYNTHETIC')
    client = Frozen()
    parent = make_candidate(7)
    parent.facts['driver_in_vehicle'] = inside
    parent.facts.pop('remote_intervention_available')
    parent.fact_provenance.pop('remote_intervention_available', None)
    state = HARAState('braking-access')
    agent = PhysicalScenarioAgent(client, response_checkpoint_dir=tmp_path)
    child = construct_physical_scenarios(state, {'MF': [parent]}, [mf()], method=method,
        agent=agent, project_facts=None)['MF'][0]
    assert child.facts['emergency_braking_available'] is True
    assert child.fact_provenance['emergency_braking_available']['approval'] == 'PENDING'
    result = score(method, child)
    assert result['controllability']['controllability_score'] == expected
    assert result['severity']['severity_score'] == 'S1'
    assert result['controllability']['engineering_status'] == 'PENDING'
    construct_physical_scenarios(state, {'MF': [parent]}, [mf()], method=method,
        agent=agent, project_facts=None)
    assert client.calls == 1


def test_legacy_remaining_control_projection_preserves_checkpoint_and_provenance(method):
    from test_autonomous_physics import instantiate
    from hara_agent.services.analysis.hazardous_event_risk_context_service import HazardousEventRiskContextService
    raw = raw_plan()
    raw['hypotheses'][0]['remaining_controls'] = ['braking']
    child = instantiate(method, raw)[0]
    # Serialized historical records have the proposal, but no flat capability.
    child.facts.pop('emergency_braking_available')
    child.fact_provenance.pop('emergency_braking_available')
    before = deepcopy(child)
    assert score(method, child)['controllability']['controllability_score'] == 'C3'
    assert child == before
    child.fact_provenance['physical_analysis']['validation_status'] = 'UNVALIDATED'
    result = score(method, child)
    assert not result['controllability']['controllability_score']
    assert result['controllability']['decision_status'] == 'PROFILE_APPLICABILITY_UNRESOLVED'


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


@pytest.mark.parametrize('outcome', ['available', 'unresolved', 'omitted_again'])
def test_motion_control_omission_corrects_once_per_plan_not_per_point_or_driver(method, tmp_path, outcome):
    from hara_agent.services.reporting.engineering_text_mapper import EngineeringReportTextMapper
    raw = raw_plan()
    raw['hypotheses'][0]['parameters'][-1].update(kind='RANGE', lower=.5, upper=3)
    class Frozen:
        calls = 0
        def complete_json(self, request):
            self.calls += 1
            context = json.loads(request.user_prompt)['context']
            reply = deepcopy(raw)
            if self.calls == 2:
                assert {d['reason'] for d in context['correction_diagnostics']} == {'CONTROL_CAPABILITY_NOT_ASSESSED'}
                if outcome == 'available':
                    reply['hypotheses'][0]['remaining_controls'] = ['driver_braking']
                    reply['hypotheses'][0]['conditions'].append('Independent driver braking remains outside the drive fault')
                elif outcome == 'unresolved':
                    reply['hypotheses'][0]['unresolved_controls'] = {'braking': '故障是否同时影响人工制动通道尚无相容的分析依据'}
            return LLMResponse(reply, 'OFFLINE_SYNTHETIC')
    client = Frozen()
    parents = [make_candidate(), make_candidate()]
    parents[1].scenario_id = 'OUTSIDE'
    for p, inside in zip(parents, (True, False)):
        p.facts.update(driver_in_vehicle=inside, driver_position='in_driver_seat' if inside else 'outside_driver_seat')
    before = deepcopy([asdict(p) for p in parents])
    state = HARAState('control-completeness', malfunctions=[asdict(mf())])
    agent = PhysicalScenarioAgent(client, response_checkpoint_dir=tmp_path)
    def run():
        return construct_physical_scenarios(state, {'MF': parents}, [mf()], method=method,
            agent=agent, project_facts=None)['MF']
    children = run()
    assert len(children) == 6 and client.calls == 2
    assert [asdict(p) for p in parents] == before
    entry = next(iter(state.item_definition['physical_analysis_plans'].values()))
    assert entry['responses'][0]['response']['data'] == raw
    assert len(entry['responses']) == 2 and entry['semantic_correction_completed']
    assert len(entry['correction_diagnostics']) == 1  # no point-wise duplicate feedback
    inside = [c for c in children if c.facts['driver_in_vehicle']]
    assert all(bool(score(method, c)['controllability']['controllability_score']) == (outcome == 'available') for c in inside)
    if outcome == 'unresolved':
        assert '制动能力未决依据：故障是否同时影响人工制动' in EngineeringReportTextMapper.physical_inputs(inside[0], {})
    elif outcome == 'available':
        assert 'Independent driver braking remains outside the drive fault' in EngineeringReportTextMapper.physical_inputs(inside[0], {})
    if outcome == 'omitted_again':
        assert inside[0].analysis_instance['physical_plan_diagnostics'][0]['reason'] == 'CONTROL_CAPABILITY_NOT_ASSESSED'
    # Numerical motion remains available even if the one correction fails.
    assert all(c.facts['ego_impact_speed_kph'] > 0 for c in children)
    assert all(c.status is ReviewStatus.PENDING for c in children)
    state = HARAState.read_committed(json.loads(json.dumps(state.to_dict())))
    assert run() == children and client.calls == 2


def test_manual_path_gap_gets_one_review_and_other_assessed_inputs_need_none(method):
    from test_p7_closure import source
    class Frozen:
        calls = 0
        def complete_json(self, request):
            self.calls += 1
            raw = raw_plan(assessed=assessed)
            if manual:
                raw['hypotheses'][0]['unresolved_controls']['driver_braking'] = 'Source has a pedal but independence is unverified'
            return LLMResponse(raw, 'OFFLINE_SYNTHETIC')
    for assessed, manual in ((False, False), (True, False), (True, True)):
        parent = make_candidate()
        if not assessed:
            parent.facts['emergency_braking_available'] = True
            parent.fact_provenance['emergency_braking_available'] = source()
        client = Frozen()
        state = HARAState('bounded-review')
        agent = PhysicalScenarioAgent(client)
        result = construct_physical_scenarios(state, {'MF': [parent]}, [mf()],
            method=method, agent=agent, project_facts=None)
        assert client.calls == (2 if manual else 1)
        if assessed:
            assert all(c.facts.get('emergency_braking_available') is None for c in result['MF'])
        construct_physical_scenarios(state, {'MF': [parent]}, [mf()],
            method=method, agent=agent, project_facts=None)
        assert client.calls == (2 if manual else 1)


def test_unresolved_controls_are_bounded_and_cannot_contradict_declared_capability():
    for unresolved, remaining in [({'unrecognised': 'reason'}, []), ({'braking': ''}, []),
                                  ({'braking': 'unknown'}, ['driver_braking'])]:
        raw = raw_plan()
        raw['hypotheses'][0].update(unresolved_controls=unresolved, remaining_controls=remaining)
        with pytest.raises(ValueError):
            PhysicalScenarioPlan.from_dict(raw, evidence_ids={'MF.description'})


def test_unresolved_evasion_preserves_reason_without_inventing_a_boolean():
    raw = raw_plan()
    h = raw['hypotheses'][0]
    h['unresolved_controls'] = {'other_road_user_avoidance_possible': 'Target recognition and escape space unresolved'}
    plan = PhysicalScenarioPlan.from_dict(raw, evidence_ids={'MF.description'})
    assert plan.hypotheses[0]['unresolved_controls'] == h['unresolved_controls']
    assert not plan.hypotheses[0].get('intervention_assumptions')
    h['intervention_assumptions'] = {'other_road_user_avoidance_possible': {
        'value': False, 'rationale': 'contradictory', 'basis_refs': ['MF.description'], 'conditions': ['test']}}
    with pytest.raises(ValueError, match='unresolved intervention'):
        PhysicalScenarioPlan.from_dict(raw, evidence_ids={'MF.description'})


def test_budget_stop_during_control_correction_saves_valid_motion_and_resumes(method, tmp_path):
    from hara_agent.infrastructure.llm.provider_budget import ProviderAttemptBudgetExceeded
    from hara_agent.workflow import CheckpointRepository
    class Frozen:
        calls = 0
        stopped = False
        def complete_json(self, request):
            self.calls += 1
            context = json.loads(request.user_prompt)['context']
            raw = raw_plan()
            if context.get('correction_diagnostics'):
                if not self.stopped:
                    self.stopped = True
                    raise ProviderAttemptBudgetExceeded(limit=1, attempts=1, ledger_path=tmp_path/'synthetic')
                raw['hypotheses'][0]['remaining_controls'] = ['driver_braking']
            return LLMResponse(raw, 'OFFLINE_SYNTHETIC')
    client = Frozen()
    state = HARAState('control-budget')
    repo = CheckpointRepository(tmp_path/'state')
    agent = PhysicalScenarioAgent(client, response_checkpoint_dir=tmp_path/'responses')
    def run():
        return construct_physical_scenarios(state, {'MF': [make_candidate()]}, [mf()],
            method=method, agent=agent, project_facts=None, checkpoint=repo.save)['MF']
    with pytest.raises(ProviderAttemptBudgetExceeded):
        run()
    state = repo.load(state.run_id)
    entry = next(iter(state.item_definition['physical_analysis_plans'].values()))
    assert entry['status'] == 'CORRECTION_REQUESTED' and len(entry['responses']) == 1
    initial = next(iter(entry['instantiations'].values()))['generated_scenarios'][0]
    assert initial['facts']['ego_impact_speed_kph'] > 0
    assert initial['analysis_instance'].get('physical_plan_diagnostics') is None
    children = run()
    assert children[0].facts['ego_impact_speed_kph'] == initial['facts']['ego_impact_speed_kph']
    assert children[0].scenario_id != initial['scenario_id']  # New control assumptions require causal validation.
    assert client.calls == 3 and len(entry['responses']) == 2
    assert run() == children and client.calls == 3


def test_independent_evasion_rule_avoids_unnecessary_braking_request(method):
    from test_p7_closure import source
    class Frozen:
        calls = 0
        def complete_json(self, request):
            self.calls += 1
            return LLMResponse(raw_plan(), 'OFFLINE_SYNTHETIC')
    parent = make_candidate()
    parent.facts['other_road_user_avoidance_possible'] = True
    parent.fact_provenance['other_road_user_avoidance_possible'] = source()
    client = Frozen()
    children = construct_physical_scenarios(HARAState('independent-evasion'), {'MF': [parent]}, [mf()],
        method=method, agent=PhysicalScenarioAgent(client), project_facts=None)['MF']
    assert client.calls == 1
    assert score(method, children[0])['controllability']['controllability_score'] == 'C1'
