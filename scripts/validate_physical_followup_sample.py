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


def prepare(source, output, run_id, limit):
    saved = json.loads(source.read_text())
    resolution = MethodSourceResolver().resolve(template_path=None,
        baseline_manifest_path=ROOT/'method_assets/fusa_baseline_v1/manifest.yaml',
        report_template_path=ROOT/'references/HARA_Template_AI_20260327.xlsx')
    method = resolution.method
    state = source_state(saved, method, ROOT/'input/ItemDef.docx', run_id)
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
            choices=list(groups.values())
            if label in {'uphill','downhill'}:
                tag='上坡' if label=='uphill' else '下坡'
                choices=[group for group in choices if group[0].facts.get('source_operating_conditions',{}).get('source_slope_direction')==tag or group[0].facts.get('source_slope_direction')==tag]
            if not choices:
                raise ValueError('Requested source family unavailable: '+label)
            family=choices[0]
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
    state.item_definition['physical_validation_policy']=plan['range_policy']
    return state,config,plan


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-context',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--run-id',required=True)
    parser.add_argument('--max-provider-attempts',type=int,default=0)
    mode=parser.add_mutually_exclusive_group();mode.add_argument('--live',action='store_true');mode.add_argument('--offline',action='store_true')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--symbolic-drive',action='store_true',help='Offline fixture only; exercises symbolic proposal through reporting')
    args=parser.parse_args()
    if args.symbolic_drive and not args.offline:
        parser.error('--symbolic-drive is an offline fixture option')
    if args.live and args.max_provider_attempts<1:
        parser.error('--live requires the explicitly authorized positive request budget')
    # Preview/offline uses a non-transport budget object for the production
    # bounded-sample contract; no transport or environment is constructed.
    state,config,plan=prepare(args.source_context,args.output,args.run_id,args.max_provider_attempts if args.live else 1)
    plan['offline_symbolic_drive']=args.symbolic_drive
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
        client=OfflinePhysicalClient(args.output,symbolic_drive=args.symbolic_drive)
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
            'live_requests':app.llm_client.attempt_budget.attempts,'mode':'LIVE' if args.live else 'OFFLINE_SCRIPTED',
            'scripted_calls':dict(app.llm_client.calls) if args.offline else {}}
        (args.output/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(summary,ensure_ascii=False))
    finally:
        # The workflow persists every first response, parsed plan and completed
        # causal batch, including budget/transport exits, in config.run_dir.
        print('Checkpoint:',config.run_dir)


if __name__=='__main__':
    main()
