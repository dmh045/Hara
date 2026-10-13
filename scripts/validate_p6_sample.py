"""Bounded P6 source-context sample through production causality and scoring.

No Provider is constructed without --live. All attempts, including retries,
consume the existing durable transport budget. This is a validation harness;
it does not implement alternate scenario selection, risk rules or rendering.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path

from hara_agent.config import LLMConfig, load_local_env
from hara_agent.infrastructure.llm import create_llm_client
from hara_agent.infrastructure.llm.provider_budget import ProviderAttemptBudget
from hara_agent.method_sources import MethodSourceResolver
from hara_agent.models import ItemDefinitionFacts, MalfunctionCandidate, ReviewStatus, SourceRef, ScenarioCandidate
from hara_agent.services.analysis import (
    ExposureInputReadinessService, MethodContractASILService,
    MethodScenarioCandidateService, ProjectFactResolver, StructuredRiskScoringService,
)
from hara_agent.services.analysis.malfunction_situation_selection import selection_function
from hara_agent.services.analysis.scenario_equivalence import driver_families
from hara_agent.services.semantic import ScenarioFeasibilityAgent, build_project_evidence_registry
from hara_agent.workflow import HARAState, WorkflowStage
from hara_agent.workflow.nodes import score_structured_scenarios
from hara_agent.workflow.nodes.scenarios import _assess_instantiated_scenarios


class RecordingClient:
    """Persist returned payloads before parsing/repair can exhaust the budget."""

    def __init__(self, client, output, stop_after=None):
        self.client, self.output, self.config = client, output, client.config
        self.stop_after, self.returned = stop_after, 0

    def complete_json(self, request):
        if self.stop_after is not None and self.returned >= self.stop_after:
            raise RuntimeError('PLANNED_RESUME_VALIDATION_STOP')
        response = self.client.complete_json(request)
        self.returned += 1
        with (self.output / 'provider_responses.jsonl').open('a') as stream:
            stream.write(json.dumps({'task': request.task, 'metadata': request.metadata,
                                     'response': asdict(response)}, ensure_ascii=False) + '\n')
        return response


class RecordingAgent(ScenarioFeasibilityAgent):
    """Preserve each completed scheduling wave using the production agent."""

    def __init__(self, client, output):
        super().__init__(client, response_checkpoint_dir=output / 'scenario-responses')
        self.output = output

    def assess(self, *args, **kwargs):
        assessments, audit = super().assess(*args, **kwargs)
        with (self.output / 'completed_waves.jsonl').open('a') as stream:
            stream.write(json.dumps({'assessments': [a.to_dict() for a in assessments],
                                     'audit': audit}, ensure_ascii=False, default=str) + '\n')
        return assessments, audit


def run_plan(args):
    """Run the offline-reviewed sample through production causal/scoring/render services."""
    from hara_agent.workflow.checkpoints import CheckpointRepository
    from hara_agent.workflow import ReviewArtifactWriter
    from hara_agent.services.analysis import MethodRiskFactBindingService
    from hara_agent.services.reporting.excel_renderer import HARAExcelRenderer
    from hara_agent.services.reporting.report_schema import load_report_schema

    args.output.mkdir(parents=True, exist_ok=True)
    plan = json.loads(args.plan.read_text())
    resolution = MethodSourceResolver().resolve(template_path=None,
        baseline_manifest_path=args.method_baseline, report_template_path=args.report_template)
    method = resolution.method
    assert plan['method_hash'] == method.metadata['method_source_hash']
    identity = hashlib.sha256((args.plan.read_text() + ScenarioFeasibilityAgent.PROMPT_VERSION).encode()).hexdigest()
    repository = CheckpointRepository(args.output / 'checkpoints')
    if args.resume:
        state = repository.load(args.run_id)
        assert state.item_definition['sample_identity'] == identity
    else:
        if repository.path_for(args.run_id).exists() or (args.output / 'provider_attempts.jsonl').exists():
            raise ValueError('Sample already exists; use --resume with the original budget and plan')
        candidates = [ScenarioCandidate(**{**c, 'sources': [SourceRef(**s) for s in c['sources']],
                                            'status': ReviewStatus(c['status'])})
                      for case in plan['cases'] for c in case['candidates']]
        state = HARAState(args.run_id, stage=WorkflowStage.SCENARIOS, scenarios=candidates,
            item_definition={'typed': plan['project'], 'sample_identity': identity,
                             'scenario_assessments': [], 'completed_sample_cases': [], 'sample_audits': []},
            functions=list({c['function']['function_id']: c['function'] for c in plan['cases']}.values()),
            malfunctions=[c['malfunction'] for c in plan['cases']],
            method_contract={'template_hash': plan['method_hash'], 'method_source_hash': plan['method_hash']})
        state.record('bounded_sample_configured', scope={'source': str(args.plan),
                     'scenario_pair_limit': len(candidates), 'provider_attempt_limit': args.max_provider_attempts})
        repository.save(state)
    result = {'run_id': args.run_id, 'sample_count': len(state.scenarios), 'status': 'PREPARED',
              'source_checkpoint_sha256': plan['source_checkpoint_sha256'], 'method_hash': plan['method_hash']}
    if not args.live:
        return result
    load_local_env()
    config = LLMConfig.from_env()
    budget = ProviderAttemptBudget(args.output / 'provider_attempts.jsonl', run_id=args.run_id, limit=args.max_provider_attempts)
    start_attempts = budget.attempts
    client = RecordingClient(create_llm_client(config, attempt_budget=budget), args.output, args.stop_after_responses)
    agent = RecordingAgent(client, args.output)
    project = ItemDefinitionFacts.from_dict(plan['project'])
    registry = build_project_evidence_registry(project, method)
    try:
        for case in plan['cases']:
            raw = case['malfunction']
            if raw['malfunction_id'] in state.item_definition['completed_sample_cases']:
                continue
            mf = MalfunctionCandidate(**{**raw, 'status': ReviewStatus(raw['status']),
                                         'sources': [SourceRef(**s) for s in raw['sources']]})
            ids = {c['scenario_id'] for c in case['candidates']}
            selected = [c for c in state.scenarios if c.scenario_id in ids]
            assessments, audit = _assess_instantiated_scenarios(agent, mf, selected, case['generation'], registry)
            state.item_definition['scenario_assessments'].extend(a.to_dict() for a in assessments)
            state.item_definition['sample_audits'].append(audit)
            state.item_definition['completed_sample_cases'].append(mf.malfunction_id)
            repository.save(state)
        writer = ReviewArtifactWriter(args.run_id, os.getenv('HARA_REVIEW_ARTIFACT_DIR', 'runtime/review'))
        for function in state.functions:
            writer.record_function(function)
        for mf in state.malfunctions:
            writer.record_malfunction(mf)
        for case in plan['cases']:
            ids = {c['scenario_id'] for c in case['candidates']}
            for candidate in state.scenarios:
                if candidate.scenario_id in ids:
                    writer.record_scenario_candidate(candidate, generated_for_malfunction_ids=[case['malfunction']['malfunction_id']])
        for assessment in state.item_definition['scenario_assessments']:
            writer.record_scenario_feasibility(assessment)
        if not state.risk_results:
            # Preserve the completed causal stage before scoring attaches harm
            # nodes. A report-only resume must not score that graph twice.
            (args.output / 'causal_state.json').write_text(json.dumps(state.to_dict(), ensure_ascii=False, indent=2))
            state = score_structured_scenarios(state, StructuredRiskScoringService(method), MethodContractASILService(method),
                                               risk_fact_binding=MethodRiskFactBindingService(method), review_artifact_writer=writer)
        writer.write_summary(state)
        repository.save(state)
        report = HARAExcelRenderer(method.report_contract, template_hash=resolution.report_template_hash,
                                   report_schema=load_report_schema(), method_contract=method).render(
            state, args.report_template, args.output / f'{args.run_id}.xlsx', draft=True)
        audits = state.item_definition['sample_audits']
        result.update(status='COMPLETE', report=str(report), risk_records=len(state.risk_results),
            causal_statuses=dict(Counter(a['risk_eligibility_status'] for a in state.item_definition['scenario_assessments'])),
            reused_count=sum(a['causal_reuse']['reused_count'] for a in audits),
            independently_assessed_count=sum(a['causal_reuse']['independently_assessed_count'] for a in audits),
            response_checkpoint_hits=sum(a.get('response_checkpoint_hits', 0) for a in audits),
            reuse_reasons=dict(Counter(r['reason'] for a in audits for r in a['causal_reuse']['records'])),
            ratings={field: dict(Counter(getattr(r, field).status.value for r in state.risk_results))
                     for field in ('severity', 'exposure', 'controllability', 'asil')})
    except Exception as exc:
        message = str(exc).replace(config.api_key, '[redacted]').replace(config.base_url, '[endpoint]')
        result.update(status='STOPPED', error_type=type(exc).__name__, error=message[:1500])
    result.update(provider_attempts=budget.attempts, new_provider_attempts=budget.attempts-start_attempts,
                  completed_cases=len(state.item_definition['completed_sample_cases']),
                  persisted_responses=len(list((args.output / 'scenario-responses').glob('*.json'))))
    (args.output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def run(args):
    args.output.mkdir(parents=True, exist_ok=True)
    saved = json.loads(args.checkpoint.read_text())
    method = MethodSourceResolver().resolve(
        template_path=None, baseline_manifest_path=args.method_baseline,
        report_template_path=args.report_template,
    ).method
    assert method.metadata['method_source_hash'] == saved['method_contract']['method_source_hash']
    typed = saved['item_definition']['typed']
    typed['risk_facts'] = [f for f in typed['risk_facts'] if 'scenario_id' not in f.get('context', {})]
    fact_ids = {f['fact_id'] for f in typed['risk_facts']}
    typed['method_risk_fact_bindings'] = [b for b in typed['method_risk_fact_bindings'] if b['source_fact_id'] in fact_ids]
    project = ItemDefinitionFacts.from_dict(typed)
    raw = next(m for m in saved['malfunctions'] if m['malfunction_id'] == args.malfunction_id)
    mf = MalfunctionCandidate(**{**raw, 'status': ReviewStatus(raw['status']),
                                 'sources': [SourceRef(**s) for s in raw['sources']]})
    functions = {f['function_id']: selection_function(f) for f in saved['functions']}
    mode = next(s['operating_mode'] for s in saved['scenarios'] if s.get('operating_mode'))
    del saved
    service = MethodScenarioCandidateService(method)
    parents, _ = service.generate(project_facts=project, operating_mode=mode,
        speed_resolution=ProjectFactResolver().resolve_speed_context(project, mode), functions=list(functions.values()))
    candidates, generation = service.generate_for_malfunction(mf, parents, project_facts=project, function=functions[mf.function_id])
    families = driver_families(candidates)
    # These four explicit source contexts are the requested regression sample,
    # not a runtime domain profile or a selection based on the resulting grade.
    targets = [('室外停车场', ''), ('室内停车场', ''), ('室外停车场', '上坡15%坡度'), ('室外停车场', '下坡15%坡度')]
    selected = []
    for place, road in targets:
        matches = [f for f in families if f[0].facts.get('source_operating_conditions', {}).get('operating_scenario') == place
                   and f[0].facts.get('source_operating_conditions', {}).get('road_surface_conditions', '') == road]
        if not matches:
            raise ValueError(f'Missing requested source sample: {place}, {road}')
        selected.extend(matches[0])
    assert len(selected) == 8 and len({c.scenario_id for c in selected}) == 8
    readiness = ExposureInputReadinessService(method)
    samples = [{'candidate': c.to_dict(), 'e_preview': readiness.assess({**c.facts, 'component_category': mf.component_category})} for c in selected]
    (args.output / 'sample_inputs.json').write_text(json.dumps(samples, ensure_ascii=False, indent=2))
    result = {'run_id': args.run_id, 'malfunction_id': mf.malfunction_id, 'sample_count': len(selected),
              'context_count': len(targets), 'provider_attempt_limit': args.max_provider_attempts,
              'provider_attempts': 0, 'status': 'PREPARED', 'method_hash': method.metadata['method_source_hash']}
    if args.live:
        load_local_env()
        config = LLMConfig.from_env()
        budget = ProviderAttemptBudget(args.output / 'provider_attempts.jsonl', run_id=args.run_id, limit=args.max_provider_attempts)
        client = RecordingClient(create_llm_client(config, attempt_budget=budget), args.output)
        try:
            assessments, audit = _assess_instantiated_scenarios(
                RecordingAgent(client, args.output), mf, selected, generation,
                build_project_evidence_registry(project, method),
            )
            (args.output / 'causal_audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2, default=str))
            state = HARAState(args.run_id, stage=WorkflowStage.SCENARIOS, scenarios=selected,
                item_definition={'typed': asdict(project), 'scenario_assessments': [a.to_dict() for a in assessments]},
                functions=[asdict(functions[mf.function_id])], malfunctions=[asdict(mf)])
            scored = score_structured_scenarios(state, StructuredRiskScoringService(method), MethodContractASILService(method))
            (args.output / 'sample_results.json').write_text(json.dumps({
                'assessments': [a.to_dict() for a in assessments],
                'risk_results': [asdict(r) for r in scored.risk_results],
                'audit_trail': scored.audit_trail,
            }, ensure_ascii=False, indent=2, default=str))
            result.update(status='COMPLETE', causal_retained=sum(a.retain for a in assessments),
                          causal_statuses=dict(Counter(a.to_dict()['risk_eligibility_status'] for a in assessments)),
                          risk_records=len(scored.risk_results), reuse=audit['causal_reuse'],
                          ratings={field: dict(Counter(getattr(r, field).status.value for r in scored.risk_results))
                                   for field in ('severity', 'exposure', 'controllability', 'asil')})
        except Exception as exc:
            message = str(exc).replace(config.api_key, '[redacted]').replace(config.base_url, '[endpoint]')
            result.update(status='STOPPED', error_type=type(exc).__name__, error=message[:1500])
        finally:
            result['provider_attempts'] = budget.attempts
    (args.output / 'summary.json').write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + '\n')
    print(json.dumps(result, ensure_ascii=False, indent=2, default=str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path)
    parser.add_argument('--plan', type=Path)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--stop-after-responses', type=int)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method-baseline', type=Path, default=Path('method_assets/fusa_baseline_v1/manifest.yaml'))
    parser.add_argument('--report-template', type=Path, default=Path('references/HARA_Template_AI_20260327.xlsx'))
    parser.add_argument('--malfunction-id', default='MF-F01-001')
    parser.add_argument('--run-id', default='p6-review-sample-20261012-001')
    parser.add_argument('--max-provider-attempts', type=int, default=8)
    parser.add_argument('--live', action='store_true')
    args = parser.parse_args()
    if args.plan:
        result = run_plan(args)
        raise SystemExit(0 if result['status'] in {'PREPARED', 'COMPLETE'} else 2)
    if not args.checkpoint:
        parser.error('--checkpoint or --plan is required')
    run(args)
