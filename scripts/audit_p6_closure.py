"""Audit the fixed P6 population and prepare a representative live sample, offline."""
import argparse
from collections import Counter
from dataclasses import asdict
import hashlib
import json
from pathlib import Path

from hara_agent.method_sources import MethodSourceResolver
from hara_agent.models import ItemDefinitionFacts, MalfunctionCandidate, ReviewStatus, SourceRef
from hara_agent.services.analysis import (
    ExposureInputReadinessService, ExposureDimensionCoverageService, MethodScenarioCandidateService, ProjectFactResolver,
    StructuredRiskScoringService,
)
from hara_agent.services.analysis.malfunction_situation_selection import selection_function
from hara_agent.services.analysis.scenario_equivalence import driver_families, signature


def run(checkpoint, output):
    output.mkdir(parents=True, exist_ok=True)
    with checkpoint.open('rb') as stream:
        source_hash = hashlib.file_digest(stream, 'sha256').hexdigest()
    saved = json.loads(checkpoint.read_text())
    method = MethodSourceResolver().resolve(template_path=None,
        baseline_manifest_path=Path('method_assets/fusa_baseline_v1/manifest.yaml'),
        report_template_path=Path('references/HARA_Template_AI_20260327.xlsx')).method
    assert method.metadata['method_source_hash'] == saved['method_contract']['method_source_hash']
    typed = saved['item_definition']['typed']
    typed['risk_facts'] = [f for f in typed['risk_facts'] if 'scenario_id' not in f.get('context', {})]
    ids = {f['fact_id'] for f in typed['risk_facts']}
    typed['method_risk_fact_bindings'] = [b for b in typed['method_risk_fact_bindings'] if b['source_fact_id'] in ids]
    project = ItemDefinitionFacts.from_dict(typed)
    functions = {f['function_id']: selection_function(f) for f in saved['functions']}
    mode = next(s['operating_mode'] for s in saved['scenarios'] if s.get('operating_mode'))
    service = MethodScenarioCandidateService(method)
    parents, _ = service.generate(project_facts=project, operating_mode=mode,
        speed_resolution=ProjectFactResolver().resolve_speed_context(project, mode), functions=list(functions.values()))
    readiness, scorer = ExposureInputReadinessService(method), StructuredRiskScoringService(method)
    catalog = {a['atom_id']: a for a in method.metadata['scenario_atom_catalog']}
    cases, coverage, pending, checks, groups, sample = [], [], [], Counter(), Counter(), []
    # Regression selection covers environments, slope signs, control faults,
    # driver-triggered transitions and both outstanding E diagnostic classes.
    targets = {
        'MF-F01-001': [('室外停车场', ''), ('室内停车场', ''), ('室外停车场', '上坡'), ('室外停车场', '下坡')],
        'MF-F03-004': [('室外停车场', '上坡')],
        'MF-F06-002': [('室外停车场', '上坡')],
        'MF-F06-007': [('室外停车场', '下坡')],
        'MF-F07-002': [('室内停车场', '')],
        'MF-F08-002': [('室外停车场', '')],
    }
    for raw in saved['malfunctions']:
        mf = MalfunctionCandidate(**{**raw, 'status': ReviewStatus(raw['status']),
                                     'sources': [SourceRef(**s) for s in raw['sources']]})
        children, audit = service.generate_for_malfunction(mf, parents, project_facts=project,
                                                          function=functions[mf.function_id])
        results = {}
        for child in children:
            facts = {**child.facts, 'component_category': mf.component_category}
            facts['_exposure_dimension_coverage_decision'] = ExposureDimensionCoverageService(method).decide(
                assessment_key=f'{mf.malfunction_id}::{child.scenario_id}', function=functions[mf.function_id],
                operating_mode=mode)
            ready = readiness.assess(facts)
            scored = scorer.score(facts, mf.vehicle_level_hazard)['exposure']
            atoms = facts['scenario_atom_ids']
            bindings = facts['method_scenario_dimensions']
            assert set(atoms) == {b['atom_id'] for b in bindings.values() if b.get('resolution_status') == 'RESOLVED'}
            assert not ready['binding_issues']
            preview = ready['baseline_exposure']
            if ready['status'].startswith('READY'):
                assert scored['exposure_score'] == preview['value'] and scored['engineering_status'] == 'FINALIZED'
            else:
                assert scored['engineering_status'] != 'FINALIZED'
            assert [d['atom_id'] for d in preview['atom_details']] == atoms
            requested = [rule.domain.value for rule in readiness.exposure.domain_rules
                         if mf.component_category in rule.component_categories]
            assert requested == [preview['requested_domain']]
            checks['instances'] += 1
            checks['ready' if ready['status'].startswith('READY') else 'pending'] += 1
            groups['domain:' + preview['requested_domain'] + '->' + preview['actual_domain']] += 1
            groups['aggregation:' + preview['aggregation_rule']] += 1
            for d in preview['atom_details']:
                a = readiness.by_id[d['atom_id']]
                expected = a.duration_level if d['requested_domain'] == 'Z' else a.frequency_level
                if d['used']:
                    assert d['e_rank'] == expected
                checks['atom_records_checked'] += 1
            labels = ' '.join(catalog[a]['label'] for a in atoms).casefold()
            assert not any(word in labels for word in ('highway', 'motorway', 'expressway'))
            results[child.scenario_id] = ready
            if not ready['status'].startswith('READY'):
                details = []
                for d in ready['dimension_assessments']:
                    if not d['could_change_exposure']:
                        continue
                    details.append({**d, 'candidate_labels': {a: catalog.get(a, {}).get('label', '') for a in d['candidate_atom_ids']}})
                pending.append({'malfunction_id': mf.malfunction_id, 'scenario_id': child.scenario_id,
                    'source_conditions': child.facts['source_operating_conditions'], 'atoms': atoms,
                    'classification': 'ENGINEERING_RELATION_UNRESOLVED', 'binding_issues': ready['binding_issues'],
                    'baseline_e': preview, 'dimensions': details,
                    'existing_selected_strong_couplings': [list(pair) for pair in readiness.exposure.strong_couplings if set(pair) <= set(atoms)]})
        families = driver_families(children)
        checks['driver_families'] += len(families)
        for family in families:
            c = family[0]
            physical = {key: c.facts.get(key) for key in (
                'object_type', 'object_position', 'object_speed_kph', 'relative_distance_m',
                'collision_type', 'ego_longitudinal_direction', 'ego_speed_constraint')}
            coverage.append({'malfunction_id': mf.malfunction_id, 'hazard_claim': mf.vehicle_level_hazard,
                'functional_effect': mf.functional_effect,
                'source_template_option': c.analysis_instance.get('base_analysis_instance', {}).get('source_option_id', ''),
                'planned_mechanism': c.analysis_instance.get('coverage_plan', {}).get('active_mechanism', ''),
                'physical_conditions': physical, 'atoms': c.facts['scenario_atom_ids'],
                'source_conditions': c.facts['source_operating_conditions'],
                'driver_scenario_ids': [s.scenario_id for s in family],
                'causal_reuse_status': 'STRUCTURALLY_ELIGIBLE_REQUIRES_DEPENDENCY_PROOF',
                'coverage_status': 'CANDIDATE_ONLY_NOT_VERIFIED_HAZARDOUS_EVENT'})
        selected = []
        for place, slope in targets.get(mf.malfunction_id, []):
            matches = [family for family in families if
                family[0].facts['source_operating_conditions']['operating_scenario'] == place and
                family[0].facts['source_operating_conditions'].get('source_slope_direction', '') == slope]
            assert matches, (mf.malfunction_id, place, slope)
            # Include an unresolved E case where available to exercise Pending
            # propagation; never select a sample by the resulting risk grade.
            matches.sort(key=lambda family: results[family[0].scenario_id]['status'].startswith('READY'))
            selected.extend(matches[0])
        if selected:
            sample.append({'malfunction': asdict(mf), 'function': asdict(functions[mf.function_id]),
                           'candidates': [asdict(c) for c in selected], 'generation': audit,
                           'e_previews': {c.scenario_id: results[c.scenario_id] for c in selected}})
    source_map = [
        {'source_term': term, 'source_field': field, 'candidate_family_count': sum(
            (c['source_conditions']['operating_scenario'] == term) if field == 'odd_locations' else
            term in c['source_conditions']['permitted_road_types'] for c in coverage),
         'status': 'REPRESENTATIVE_CONDITIONS_PRESENT' if field == 'odd_locations' else 'ALLOWED_SCOPE_ONLY_GEOMETRY_COVERAGE_UNPROVEN'}
        for field, terms in [('odd_locations', project.odd_locations), ('odd_road_types', project.odd_road_types)] for term in terms
    ]
    summary = {'checks': dict(checks), 'exposure': dict(groups), 'source_checkpoint_sha256': source_hash,
               'method_hash': method.metadata['method_source_hash'], 'provider_calls': 0,
               'pending_by_dimension': dict(Counter(d['dimension'] for p in pending for d in p['dimensions'])),
               'sample_instances': sum(len(s['candidates']) for s in sample),
               'independent_hazards': 'NOT_ESTABLISHED_WITHOUT_CAUSAL_VALIDATION',
               'minimum_independent_causal_candidates': checks['driver_families'],
               'maximum_independent_causal_candidates': checks['instances'],
               'proven_driver_reuse': 0}
    for name, data in [('summary', summary), ('coverage_matrix', coverage), ('source_mapping', source_map),
                       ('pending_e_cases', pending), ('sample_plan', {'project': asdict(project), 'cases': sample,
                        'source_checkpoint_sha256': source_hash, 'method_hash': method.metadata['method_source_hash']})]:
        (output / f'{name}.json').write_text(json.dumps(data, ensure_ascii=False, indent=2, default=str) + '\n')
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    run(args.checkpoint, args.output)
