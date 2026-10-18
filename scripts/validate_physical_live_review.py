"""Zero-network comparison of the frozen real proposals against production code.

No saved response is edited. Historical inputs/ledger stay read-only. This does
not measure new prompt tokens or validate the new prompt against a Provider.
"""
import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType

from validate_physical_followup_sample import prepare
from validate_autonomous_physics_offline import ROOT
from hara_agent.application import HARAApplication
from hara_agent.contracts.physical_scenario_plan import PhysicalScenarioPlan
from hara_agent.models import ItemDefinitionFacts, MalfunctionCandidate, ReviewStatus, SourceRef
from hara_agent.services.analysis import MethodScenarioCandidateService
from hara_agent.services.analysis.malfunction_situation_selection import selection_function
from hara_agent.services.analysis.physical_hypothesis_instantiation_service import PhysicalHypothesisInstantiationService
from hara_agent.services.analysis.scenario_equivalence import signature, scenario_conditions
from hara_agent.services.semantic.physical_scenario_agent import physical_prompt_view, expand_physical_prompt_view
from hara_agent.method_sources import MethodSourceResolver


def baseline_module(ref, stem):
    name = 'hara_agent.services.analysis._review_baseline_' + stem
    source = subprocess.check_output(['git', 'show', f'{ref}:src/hara_agent/services/analysis/{stem}.py'], cwd=ROOT, text=True)
    module = ModuleType(name)
    sys.modules[name] = module
    exec(compile(source, f'{ref}:{stem}.py', 'exec'), module.__dict__)
    return module


def dump(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source-context', type=Path, required=True)
    parser.add_argument('--live-checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--baseline-ref', default='1a0458de128db02e853da537975406af2a91c187')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    protected = [args.source_context, *[p for p in args.live_checkpoint.parent.rglob('*') if p.is_file()]]
    hashes = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in protected}
    saved = json.loads(args.live_checkpoint.read_text())
    plans = saved['item_definition']['physical_analysis_plans']
    prompt_rows = []
    for key, entry in plans.items():
        for index, record in enumerate(entry['responses']):
            original = json.loads(record['request']['user_prompt'])
            context, texts = physical_prompt_view(original['context'])
            assert expand_physical_prompt_view(context, texts) == original['context']
            projected = {**original, 'context': context, 'source_texts': texts}
            prompt_rows.append(dict(plan_key=key, response_index=index, model=record['response']['model'],
                request_id=record['response'].get('request_id'), original_chars=len(record['request']['user_prompt']),
                projected_chars=len(json.dumps(projected, ensure_ascii=False, default=str)),
                unique_interned_source_texts=len(texts), round_trip_equal=True,
                historical_prompt_tokens=record['response']['usage'].get('prompt_tokens'), new_prompt_tokens=None))
            # This projection is an audit of the old prompt, not a new request/cache hit.
            dump(args.output / f'{key[:12]}_{index}_original.json', record)
            dump(args.output / f'{key[:12]}_{index}_projected.json', projected)

    state, config, _ = prepare(args.source_context, args.output / 'unused_preview', 'physical-review-frozen', 1)
    method = MethodSourceResolver().resolve(template_path=None,
        baseline_manifest_path=ROOT/'method_assets/fusa_baseline_v1/manifest.yaml',
        report_template_path=ROOT/'references/HARA_Template_AI_20260327.xlsx').method
    service = MethodScenarioCandidateService(method)
    parents, _ = HARAApplication(config, None).prepare_scenario_candidates(state, service)
    project = ItemDefinitionFacts.from_dict(state.item_definition['typed'])
    old = baseline_module(args.baseline_ref, 'physical_hypothesis_instantiation_service')
    old.derive_scenario_physics = baseline_module(args.baseline_ref, 'scenario_physics').derive_scenario_physics
    comparisons = []
    for raw_mf in state.malfunctions:
        mf = MalfunctionCandidate(**{**raw_mf, 'status': ReviewStatus(raw_mf['status']),
            'sources': [SourceRef(**ref) for ref in raw_mf['sources']]})
        function = next(f for f in state.functions if f['function_id'] == mf.function_id)
        generated, _ = service.generate_for_malfunction(mf, parents, project_facts=project, function=selection_function(function))
        for key, entry in plans.items():
            context = entry['context']
            if context['malfunction']['malfunction_id'] != mf.malfunction_id or 'v3-' not in context['prompt_version']:
                continue
            # Re-run the unmodified FIRST real proposal, including conflicts.
            response = entry['responses'][0]['response']
            plan = PhysicalScenarioPlan.from_dict(response['data'], evidence_ids=set(context['evidence']))
            family = [c for c in generated if signature(scenario_conditions(c, without_driver=True)) == signature(context['conditions'])]
            assert len(family) == 2, (key, len(family))
            row = dict(plan_key=key, malfunction_id=mf.malfunction_id, original_response_id=response.get('request_id'),
                source_ids=[c.scenario_id for c in family], before=[], after=[])
            for label, cls in [('before', old.PhysicalHypothesisInstantiationService), ('after', PhysicalHypothesisInstantiationService)]:
                for candidate in family:
                    children, failures = cls(method).instantiate(candidate, mf, plan, plan_key=key,
                        response_model=response['model'], project_context=asdict(project), function=function)
                    row[label].append(dict(parent_id=candidate.scenario_id, children=[asdict(c) for c in children], failures=failures))
            comparisons.append(row)
    assert len(comparisons) == 4
    unchanged = all(hashlib.sha256(Path(p).read_bytes()).hexdigest() == digest for p, digest in hashes.items())
    assert unchanged
    dump(args.output/'before_after.json', comparisons)
    dump(args.output/'prompt_projection.json', prompt_rows)
    dump(args.output/'historical_integrity.json', dict(unchanged=unchanged, sha256=hashes))
    summary = dict(mode='OFFLINE_UNMODIFIED_REAL_PROPOSAL_REPLAY', live_requests=0,
        first_proposals=4, source_identities=sum(len(row['source_ids']) for row in comparisons),
        produced_before=sum(len(p['children']) for r in comparisons for p in r['before']),
        produced_after=sum(len(p['children']) for r in comparisons for p in r['after']),
        original_prompt_chars=sum(r['original_chars'] for r in prompt_rows),
        projected_prompt_chars=sum(r['projected_chars'] for r in prompt_rows), new_prompt_tokens=None,
        historical_files_unchanged=unchanged, new_prompt_live_validated=False, end_to_end_live_validated=False)
    dump(args.output/'summary.json', summary)
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == '__main__':
    main()
