"""Five production-generated families; preview by default, no transport without --live.

Source extraction is reused, but no historical collision/causal/risk result is
seeded. --offline exercises the same HARAApplication with labelled scripted JSON.
A future authorized --live run shares one durable budget across every stage.
"""
import argparse
from collections import defaultdict
from dataclasses import asdict, replace
import hashlib
import json
import os
from pathlib import Path
import time
from datetime import datetime, timezone

from validate_autonomous_physics_offline import ROOT, source_state, OfflinePhysicalClient
from hara_agent.application import HARAApplication
from hara_agent.config import RunConfig, load_local_env
from hara_agent.infrastructure.llm.provider_budget import ProviderAttemptBudget
from hara_agent.method_sources import MethodSourceResolver
from hara_agent.models import ItemDefinitionFacts, MalfunctionCandidate, ReviewStatus, SourceRef
from hara_agent.services.analysis import MethodScenarioCandidateService
from hara_agent.services.analysis.malfunction_situation_selection import selection_function
from hara_agent.services.analysis.scenario_equivalence import signature, scenario_conditions
from hara_agent.workflow import CheckpointRepository


FAMILIES = ('braking', 'drive', 'uphill', 'downhill', 'lateral')


class SavedPhysicalProposalClient(OfflinePhysicalClient):
    """Unmodified historical physical replies plus explicitly scripted causality.

    This tests downstream plumbing, never the new prompt/model behavior. Missing
    saved replies stop locally; there is no network client or fallback.
    """
    def __init__(self, output, checkpoint):
        super().__init__(output)
        self.config.provider = 'OFFLINE_SAVED_PHYSICAL_SCRIPTED_CAUSAL'
        self.records = {}
        self.next_index = defaultdict(int)
        for entry in json.loads(checkpoint.read_text())['item_definition']['physical_analysis_plans'].values():
            context = entry['context']
            key = (context['malfunction']['malfunction_id'], signature(context['conditions']))
            self.records[key] = entry['responses']

    def proposal(self, request):
        from hara_agent.services.semantic.physical_scenario_agent import expand_physical_prompt_view
        payload = json.loads(request.user_prompt)
        context = expand_physical_prompt_view(payload['context'], payload.get('source_texts', {}))
        key = (context['malfunction']['malfunction_id'], signature(context['conditions']))
        index = self.next_index[key]
        records = self.records.get(key, [])
        if index >= len(records):
            raise AssertionError('No saved physical response for this exact source condition; no transport allowed')
        record = records[index]
        self.next_index[key] += 1
        with (self.output/'saved_physical_replies.jsonl').open('a') as stream:
            stream.write(json.dumps({'current_plan_key': context['plan_key'], 'original_record': record}, ensure_ascii=False)+'\n')
        return record['response']['data']


def prepare(source, output, run_id, limit, *, families=None, driver_position=None, numeric_representative=False):
    selected = set(FAMILIES if families is None else families)
    if not selected or selected - set(FAMILIES):
        raise ValueError('Select one or more available source families')
    saved = json.loads(source.read_text())
    resolution = MethodSourceResolver().resolve(template_path=None,
        baseline_manifest_path=ROOT/'method_assets/fusa_baseline_v1/manifest.yaml',
        report_template_path=ROOT/'references/HARA_Template_AI_20260327.xlsx')
    method = resolution.method
    state = source_state(saved, method, ROOT/'input/ItemDef.docx', run_id)
    if families is not None:
        function_ids = {dict(braking='F01', drive='F02', lateral='F03', uphill='F08', downhill='F08')[label] for label in selected}
        state.functions = [f for f in state.functions if f['function_id'] in function_ids]
        state.malfunctions = [m for m in state.malfunctions if m['function_id'] in function_ids]
    state.method_contract.update(method_source_hash=method.metadata['method_source_hash'], report_template_hash=resolution.report_template_hash)
    config = RunConfig(ROOT/'input/ItemDef.docx', None, output/'HARA_Physical_Followup.xlsx', output/'checkpoints',
        run_id=run_id, resume=True, allow_draft=True, operating_mode='parking_vehicle_control', max_workers=1)
    service = MethodScenarioCandidateService(method)
    parents, _ = HARAApplication(config, None).prepare_scenario_candidates(state, service)
    project = ItemDefinitionFacts.from_dict(state.item_definition['typed'])
    cases=[]
    for raw in state.malfunctions:
        mf=MalfunctionCandidate(**{**raw,'status':ReviewStatus(raw['status']),'sources':[SourceRef(**ref) for ref in raw['sources']]})
        function=next(f for f in state.functions if f['function_id']==mf.function_id)
        generated,_=service.generate_for_malfunction(mf,parents,project_facts=project,function=selection_function(function))
        groups=defaultdict(list)
        for candidate in generated:
            groups[signature(scenario_conditions(candidate,without_driver=True))].append(candidate)
        labels=['uphill','downhill'] if mf.function_id=='F08' else [{'F01':'braking','F02':'drive','F03':'lateral'}[mf.function_id]]
        for label in labels:
            if label not in selected:
                continue
            choices=list(groups.values())
            if label in {'uphill','downhill'}:
                tag='上坡' if label=='uphill' else '下坡'
                choices=[group for group in choices if group[0].facts.get('source_operating_conditions',{}).get('source_slope_direction')==tag or group[0].facts.get('source_slope_direction')==tag]
            if not choices:
                raise ValueError('Requested source family unavailable: '+label)
            family=choices[0]
            if driver_position:
                family=[c for c in family if c.facts.get('driver_position') == driver_position]
                if not family:
                    raise ValueError('Requested driver position unavailable: '+driver_position)
            cases.append({'label':label,'malfunction_id':mf.malfunction_id,
                'scenario_ids':[c.scenario_id for c in family], 'conditions':scenario_conditions(family[0],without_driver=True)})
    ids=tuple(identity for case in cases for identity in case['scenario_ids'])
    config=replace(config, sample_function_limit=len(state.functions),sample_malfunction_limit=len(state.malfunctions),
        sample_parent_scenario_limit=len(parents),sample_generated_scenario_ids=ids,
        sample_scenario_pair_limit=len(ids),provider_attempt_limit=limit)
    plan={'version':'physical-followup-five-families-v1','run_id':run_id,'cases':cases,'input_identities':len(ids),
        'source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),'method_hash':method.metadata['method_source_hash'],
        'expected_first_physical_proposals':4,'range_policy':'One representative hypothesis first; symbolic inputs remain symbolic. No range Cartesian expansion.',
        'budget_limit':limit,'live_requests':0,'budget_authorization':'Not granted by this plan; obtain a current explicit user budget before --live.'}
    if families is not None:
        from hara_agent.services.semantic.physical_scenario_agent import PhysicalScenarioAgent
        plan.update(version='physical-followup-selected-families-v2', selected_families=sorted(selected),
            physical_prompt_version=PhysicalScenarioAgent.PROMPT_VERSION,
            expected_first_physical_proposals=sum(case['label'] != 'braking' for case in cases))
    if driver_position:
        plan['driver_position']=driver_position
    from hara_agent.services.semantic.physical_scenario_agent import PhysicalScenarioAgent
    plan['range_policy'] = PhysicalScenarioAgent.ANALYSIS_STRATEGY
    # Retain the CLI flag for documented invocations; it no longer changes
    # production semantics or inserts a sample-only prompt into the checkpoint.
    plan['numeric_representative_requested'] = numeric_representative
    return state,config,plan


def main():
    started=time.monotonic()
    started_at=datetime.now(timezone.utc).isoformat()
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-context',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--max-provider-attempts',type=int,default=0)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--live',action='store_true');mode.add_argument('--offline',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--symbolic-drive',action='store_true',help='Offline fixture only; exercises symbolic proposal through reporting')
    parser.add_argument('--families',nargs='+',choices=FAMILIES,help='Narrow existing production sample selection; e.g. drive selects its two driver identities')
    parser.add_argument('--driver-position',choices=('in_driver_seat','outside_driver_seat'),help='Select an existing generated driver identity, without changing its facts')
    parser.add_argument('--numeric-representative',action='store_true',help='Compatibility flag: ordinary production now uses the same justified representative strategy')
    parser.add_argument('--replay-physical-checkpoint', type=Path, help='Offline only: unmodified saved physical replies for exactly matching source conditions; causal/fact replies remain scripted')
    args=parser.parse_args()
    if args.symbolic_drive and not args.offline:
        parser.error('--symbolic-drive is an offline fixture option')
    if args.replay_physical_checkpoint and (not args.offline or args.symbolic_drive):
        parser.error('--replay-physical-checkpoint requires --offline and cannot edit saved replies via --symbolic-drive')
    if args.live and args.max_provider_attempts<1:
        parser.error('--live requires the explicitly authorized positive request budget')
    # Preview/offline uses a non-transport budget object for the production
    # bounded-sample contract; no transport or environment is constructed.
    state,config,plan=prepare(args.source_context,args.output,args.run_id,args.max_provider_attempts if args.live else 1,
        families=args.families,driver_position=args.driver_position,numeric_representative=args.numeric_representative)
    plan['offline_symbolic_drive']=args.symbolic_drive
    if args.replay_physical_checkpoint:
        plan['saved_physical_checkpoint_sha256'] = hashlib.sha256(args.replay_physical_checkpoint.read_bytes()).hexdigest()
    plan['budget_limit']=args.max_provider_attempts if args.live else 0
    args.output.mkdir(parents=True,exist_ok=args.resume)
    plan_path=args.output/'sample_plan.json'
    if args.resume:
        if json.loads(plan_path.read_text())!=plan:
            raise ValueError('Resume must retain the original sample and budget')
    else:
        plan_path.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    if not(args.live or args.offline):
        print(json.dumps(plan,ensure_ascii=False));return
    repo=CheckpointRepository(config.run_dir)
    os.environ['HARA_REVIEW_ARTIFACT_DIR']=str(args.output/'review')
    if args.live:
        load_local_env(); app=HARAApplication.from_env(config)
    else:
        client=(SavedPhysicalProposalClient(args.output,args.replay_physical_checkpoint) if args.replay_physical_checkpoint
                else OfflinePhysicalClient(args.output,symbolic_drive=args.symbolic_drive))
        client.attempt_budget=ProviderAttemptBudget(config.run_dir/f'{config.run_id}.provider-attempts.jsonl',run_id=config.run_id,limit=1)
        app=HARAApplication(config,client)
    if not args.resume:
        state.record('offline_source_context_replay' if args.offline else 'source_context_replay', source=str(args.source_context), sha256=plan['source_sha256'])
        state.record('bounded_sample_configured',scope=config.sample_scope(),source_context=app.bounded_source_context(state.method_contract))
        repo.save(state)
    try:
        result=app.run()
        if result.reason=='bounded_sample_scope_preview':
            result=app.run()
        summary={'reason':result.reason,'scenarios':len(result.state.scenarios),'risk_results':len(result.state.risk_results),
            'physical_metrics':result.state.item_definition.get('physical_analysis_metrics'),
            'live_requests':app.llm_client.attempt_budget.attempts,
            'mode':'LIVE' if args.live else 'OFFLINE_SAVED_PHYSICAL_SCRIPTED_CAUSAL' if args.replay_physical_checkpoint else 'OFFLINE_SCRIPTED',
            'scripted_calls':dict(app.llm_client.calls) if args.offline else {}}
        (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(summary,ensure_ascii=False))
    finally:
        # The workflow persists every first response, parsed plan and completed
        # causal batch, including budget/transport exits, in config.run_dir.
        print('Checkpoint:',config.run_dir)
        timing={'started_at':started_at,'finished_at':datetime.now(timezone.utc).isoformat(),
                'elapsed_seconds':time.monotonic()-started,'mode':'LIVE' if args.live else 'OFFLINE'}
        with (args.output/'execution_timing.jsonl').open('a') as stream:
            stream.write(json.dumps(timing)+'\n')


if __name__=='__main__':
    main()
