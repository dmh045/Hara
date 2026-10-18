"""Frozen source populations and responses; no Provider, no scoring-input injection."""
import argparse
from collections import defaultdict
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

from validate_autonomous_physics_offline import OfflinePhysicalClient, ROOT
from hara_agent.models import ItemDefinitionFacts, MalfunctionCandidate, SourceRef, ReviewStatus
from hara_agent.method_sources import MethodSourceResolver
from hara_agent.services.semantic import ScenarioFeasibilityAgent, build_project_evidence_registry
from hara_agent.workflow.state import HARAState
from hara_agent.workflow.nodes.scenarios import _assess_instantiated_scenarios


def run(source, output, baseline_agent=None):
    output.mkdir(parents=True, exist_ok=False)
    spec=importlib.util.spec_from_file_location('before',ROOT/'tests/fixtures/reuse_reporting/scheduler_before.py')
    before=importlib.util.module_from_spec(spec); spec.loader.exec_module(before)
    method=MethodSourceResolver().resolve(template_path=None,
        baseline_manifest_path=ROOT/'method_assets/fusa_baseline_v1/manifest.yaml',
        report_template_path=ROOT/'references/HARA_Template_AI_20260327.xlsx').method
    original_agent = (baseline_agent.read_text() if baseline_agent else subprocess.check_output(
        ['git','show','1e150b298d252adea73fa54b99a027e4ddceb2b8:src/hara_agent/services/semantic/scenario_agent.py'],text=True))
    baseline_path=output/'baseline_scenario_agent.py'
    baseline_path.write_text(original_agent)
    spec=importlib.util.spec_from_file_location('hara_agent.services.semantic._frozen_benchmark_agent',baseline_path)
    before_agent=importlib.util.module_from_spec(spec);spec.loader.exec_module(before_agent)
    results=[]
    for kind in ('point','range','symbolic'):
        checkpoint=next((source/kind/'checkpoints').glob('*.checkpoint.json'))
        frozen=checkpoint.read_bytes(); state=HARAState.read_committed(json.loads(frozen))
        registry=build_project_evidence_registry(ItemDefinitionFacts.from_dict(state.item_definition['typed']),method)
        by_id={c.scenario_id:c for c in state.scenarios}
        families=defaultdict(list)
        for risk in state.risk_results:families[risk.malfunction_id].append(by_id[risk.scenario_id])
        input_hash=hashlib.sha256(json.dumps({mf:[asdict(c) for c in members] for mf,members in families.items()},sort_keys=True).encode()).hexdigest()
        for label,scheduler in [('before',before._assess_instantiated_scenarios),('after',_assess_instantiated_scenarios)]:
            folder=output/kind/label;folder.mkdir(parents=True)
            client=OfflinePhysicalClient(folder)
            agent=(before_agent.ScenarioFeasibilityAgent if label=="before" else ScenarioFeasibilityAgent)(client)
            agent.method_execution_identity={'method_hash':method.metadata['method_source_hash'],'compiler_version':method.compiler_version}
            audits=[];out=[]
            for raw in state.malfunctions:
                mf=MalfunctionCandidate(**{**raw,'status':ReviewStatus(raw['status']),'sources':[SourceRef(**r) for r in raw['sources']]})
                assessed,audit=scheduler(agent,mf,deepcopy(families[mf.malfunction_id]),{},registry)
                audits.append(audit);out.extend(r.to_dict() for r in assessed)
            (folder/'audits.json').write_text(json.dumps(audits,ensure_ascii=False,indent=2))
            (folder/'assessments.json').write_text(json.dumps(out,ensure_ascii=False,indent=2))
            requests=[json.loads(line)['request'] for line in (folder/'scripted_requests_responses.jsonl').read_text().splitlines()]
            metrics={'sample':kind,'scheduler':label,'frozen_input_sha256':input_hash,
                'input_count':sum(map(len,families.values())),'output_count':len(out),
                'mock_requests':sum(client.calls.values()),'real_requests':0,
                'prompt_chars':sum(len(r['system_prompt'])+len(r['user_prompt']) for r in requests),
                'tokens':'unmeasured: scripted client is not a tokenizer or Provider',
                'independent_judgments':sum(a.get('causal_reuse',{}).get('independently_assessed_count',0) for a in audits),
                'exact_reuse':sum(a.get('causal_reuse',{}).get('exact_input_reuse',0) for a in audits),
                'complete_reuse':sum(a.get('causal_reuse',{}).get('complete_causal_reuse',0) for a in audits),
                'partial_reuse':sum(sum(bool(r['reused']) for r in a.get('partial_evidence_reuse_records',[])) for a in audits),
                'shared_batches':sum(r['metadata'].get('scenario_count')==2 and '"shared_context"' in r['user_prompt'] for r in requests)}
            results.append(metrics);print(metrics,flush=True)
        assert hashlib.sha256(checkpoint.read_bytes()).digest()==hashlib.sha256(frozen).digest()
        left=json.loads((output/kind/'before/assessments.json').read_text());right=json.loads((output/kind/'after/assessments.json').read_text())
        # Reuse does not alter independently checked causal decisions or original identity population.
        normalized=lambda rows:{r['scenario_id']:{k:r[k] for k in ('malfunction_id','causally_relevant','physically_feasible','functionally_relevant','breakpoint','hazardous_event','physical_interaction')} for r in rows}
        assert normalized(left)==normalized(right)
    (output/'comparison.json').write_text(json.dumps(results,ensure_ascii=False,indent=2)+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--source',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    p.add_argument('--baseline-agent',type=Path,help='Frozen baseline agent source for reproducing outside Git')
    args=p.parse_args();run(args.source,args.output,args.baseline_agent)
