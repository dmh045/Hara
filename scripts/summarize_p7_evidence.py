"""Read-only P7 evidence index; no selection, scoring or Provider code."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n')


def run(root):
    summary = read(root / 'fixed_population_summary.json')
    old = read('runtime/agent/full-p6-20261012-002.checkpoint.json')
    new = read(root / 'checkpoints' / (summary['run_id'] + '.checkpoint.json'))
    trace = read(root / 'review' / summary['run_id'] / 'risk_execution_trace.json')['assessments']
    mapping = read(root / 'report_identity_map.json')
    reverse = {tuple(pair): key for key, pair in mapping.items()}
    by = {(r['malfunction_id'], r['scenario_id']): r for r in trace}
    sc = {c['scenario_id']: c for c in new['scenarios']}
    old_sc = {c['scenario_id']: c for c in old['scenarios']}
    fields = ('road_user_type', 'collision_type', 'ego_speed_kph', 'object_speed_kph', 'relative_speed_kph', 'relative_distance_m', 'ttc_s')
    diagnostic = []
    for old_risk in old['risk_results']:
        key = (old_risk['malfunction_id'], old_risk['scenario_id'])
        current = by[key]
        context = current.get('hazardous_event_risk_context', {})
        diagnostic.append({'historical_hara_id': reverse.get(key), 'malfunction_id': key[0], 'scenario_id': key[1],
            'risk_eligibility': current['risk_eligibility'], 'prior_s_pending': not old_risk['severity']['value'],
            'inputs': {field: context.get(field, {}) for field in fields},
            'ttc_diagnostic': context.get('ttc_s', {}).get('reason', 'CAUSAL_REVALIDATION_REQUIRED'),
            'new_scores': {dim: current.get(dim, {}).get('result', '') for dim in ('severity','exposure','controllability','asil')},
            'ego_direction_before': old_sc[key[1]]['facts'].get('ego_longitudinal_direction'),
            'ego_direction_after': sc[key[1]]['facts'].get('ego_longitudinal_direction')})
    write(root / 'physical_input_diagnostics.json', diagnostic)
    e_pending = []
    for row in trace:
        if row['risk_scoring_invoked'] and not row['exposure']['result']:
            key = (row['malfunction_id'], row['scenario_id'])
            e_pending.append({'historical_hara_id': reverse.get(key), 'malfunction_id': key[0], 'scenario_id': key[1],
                'hazardous_event': next(a['hazardous_event'] for a in new['item_definition']['scenario_assessments'] if a['scenario_id']==key[1]),
                'trace': row['exposure'],
                'disposition': 'LOCAL_APPLICABILITY_UNRESOLVED: optional traffic candidates have no proven role in this mechanism; conditional arithmetic is not applicability evidence.'})
    write(root / 'six_e_pending_disposition.json', e_pending)
    binding = read(root / 'input_consumption.json')['bindings']
    bindings_by_key = {(row['malfunction_id'],row['scenario_id']):row for row in binding}
    fact_consumption = []
    for fact in old['item_definition']['typed']['risk_facts']:
        context = fact.get('context', {})
        if not context.get('scenario_id'):
            continue
        key = (context.get('malfunction_id'),context['scenario_id'])
        current = by.get(key, {})
        audit = bindings_by_key.get(key, {})
        parameter = fact['parameter']
        state = ('NOT_SCORED_CAUSAL_PENDING_OR_EXCLUDED' if not current.get('risk_scoring_invoked')
                 else 'BOUND_OR_COMPATIBLE_EXISTING_FACT' if parameter in audit.get('bound_fact_types', [])
                 else 'REQUIRES_BINDING_REVIEW')
        fact_consumption.append({'fact_id':fact['fact_id'], 'parameter':parameter, 'context':context,
            'status':state, 'source_conflicts':audit.get('source_conflicts',[])})
    write(root/'scoped_fact_consumption.json', {'counts':dict(Counter(row['status'] for row in fact_consumption)), 'facts':fact_consumption})
    write(root / 'input_diagnostics_summary.json', {
        'legacy_s_pending': sum(r['prior_s_pending'] for r in diagnostic),
        'ttc_diagnostics': dict(Counter(r['ttc_diagnostic'] for r in diagnostic)),
        'restored_ego_directions': sum(not r['ego_direction_before'] and bool(r['ego_direction_after']) for r in diagnostic),
        'remaining_missing_ego_direction': sum(not r['ego_direction_after'] for r in diagnostic),
        'field_diagnostics': {f: dict(Counter(r['inputs'][f].get('reason', 'CAUSAL_REVALIDATION_REQUIRED') for r in diagnostic)) for f in fields},
        'binding_conflicts': [r for r in binding if r['source_conflicts']],
        'existing_sg_links_restored': sum(bool(r.get('safety_goal_id')) for r in new['risk_results']),
        'historical_non_qm_sg_links': sum(r['asil']['value'] not in ('', 'QM') and bool(r.get('safety_goal_id')) for r in old['risk_results'])})
    # Keep source text and units intact; a capability is not a fault trajectory.
    lines = (root / 'item_source_text.txt').read_text().splitlines()
    rows = []
    for line in lines:
        if line.startswith(('t10/r','t12/r','t14/r15','t13/r')) and any(token in line for token in ('速度','车速','km/h','m/s','ms','坡度','deg','°/s')):
            kind = 'ACTUATOR_CAPABILITY' if line.startswith('t12/') else 'NORMAL_OPERATION_OR_PERFORMANCE'
            use = 'CONSTRAINT_ONLY; requires fault applicability and a sourced motion model'
            if '能响应的减速度范围' in line:
                use = 'UNIT_UNRESOLVED: original m/s is not acceleration; not consumed as m/s²'
            rows.append({'source': 'input/ItemDef.docx', 'location': line.split(':',1)[0], 'original_text': line.split(':',1)[1].strip(),
                'semantic': kind, 'consumption': use})
    rows.append({'source':'input/ItemDef.docx','location':'table[14].row[15]', 'original_text':'控车范围 0–7 km/h；搜索车位 0–30 km/h',
                 'semantic':'NORMAL_OPERATION', 'consumption':'parking_vehicle_control selects 7 km/h; explicit standing selects initial 0; neither automatically impact speed'})
    write(root / 'motion_parameter_index.json', {'item_sha256': hashlib.sha256(Path('input/ItemDef.docx').read_bytes()).hexdigest(),
        'parameters': rows, 'fault_trajectory_missing': ['scoped post-fault acceleration or a bounded profile', 'model applicability after this malfunction', 'travel distance/time to impact', 'object motion at the same impact time'],
        'available_model': 'CONSTANT_ACCELERATION_TO_IMPACT_DISTANCE; opt-in sourced applicability, finite signed acceleration, positive feasible radicand; no arbitrary speed cap'})
    target = {c['scenario_id'] for case in read(root/'sample_plan.json')['cases'] for c in case['candidates']}
    def responses(directory, selected=None):
        rows = []
        for path in Path(directory).glob('*.json'):
            value = read(path); payload = value['response']; ids = {a.get('scenario_id') for a in payload.get('data',{}).get('assessments',[]) if isinstance(a,dict)}
            if selected is not None and not ids & selected:
                continue
            rows.append({'file':str(path), 'scenario_ids':sorted(ids), 'outside_sample_ids': sorted(ids - selected) if selected is not None else [],
                         'metadata': value['metadata'], 'usage': payload.get('usage',{})})
        return rows
    old_responses = responses('runtime/agent/full-p6-20261012-002.scenario-responses', target)
    new_responses = responses(root/'live/scenario-responses', target)
    def metrics(rows):
        return {'returned_request_count':len(rows), 'mixed_scope_requests':sum(bool(r['outside_sample_ids']) for r in rows),
                **{field:sum(r['usage'].get(field,0) for r in rows) for field in ('prompt_tokens','completion_tokens','latency_seconds','transport_attempts')},
                'repair_requests':sum(bool(r['metadata'].get('repair_attempt')) or 'I' in r['metadata'].get('split_path','') for r in rows)}
    final = read(root/'live_final/summary.json')
    live = read(root/'live/summary.json')
    write(root/'live_comparison.json', {'sample_ids': sorted(target), 'historical':metrics(old_responses), 'p7_fixed_sample':metrics(new_responses),
        'limitations': 'Same historical identities, updated policy and prompt; engineering conditions changed as documented. Counts include returned requests only; durable HTTP ledger is authoritative. No cross-model speed guarantee.',
        'old_requests':old_responses, 'new_requests':new_responses,
        'live_runs': [live, final], 'total_provider_attempts': live['provider_attempts']+final['provider_attempts'],
        'approved_budget':20, 'pause':read(root/'live/pause_summary.json'),
        'remaining_unexecuted_budget':20-live['provider_attempts']-final['provider_attempts']})
    print(json.dumps({'physical_cases':len(diagnostic), 'e_pending':len(e_pending), 'live_attempts':live['provider_attempts']+final['provider_attempts']},ensure_ascii=False))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,default=Path('output/p7_closure'))
    run(parser.parse_args().output)
