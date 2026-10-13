"""Audit saved P6 previews and merge mappings; never calls a Provider."""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path


def audit(before: Path, after: Path) -> dict:
    old = json.loads((before / 'generation_audit.json').read_text())
    previous_groups, previous_removed, cross_parent_removed = 0, 0, 0
    for mf in old:
        origins = {sid: parent['parent_scenario_id'] for parent in mf['generation']['parents']
                   for sid in parent['scenario_ids']}
        for group in mf['generation']['equivalence_mapping']:
            members = group['scenario_ids']
            previous_groups += 1
            previous_removed += len(members) - 1
            if len({origins[sid] for sid in members}) > 1:
                cross_parent_removed += len(members) - 1
    rows = json.loads((after / 'candidate_previews.json').read_text())
    scopes = Counter()
    road_atoms = Counter()
    violations = []
    pending = Counter()
    for row in rows:
        scope = row['source_operating_conditions']
        dims = row['method_conditions']
        scopes[scope.get('operating_scenario', '')] += 1
        road_atoms[dims['ROAD'].get('atom_id')] += 1
        labels = ' '.join(v.get('atom_label') or '' for v in dims.values()).casefold()
        # These search terms are independent review assertions for this AVP
        # replay, not business rules used by the runtime selector.
        if any(word in labels for word in ('motorway', 'expressway', 'highway')):
            violations.append({'scenario_id': row['scenario_id'], 'reason': 'HIGHWAY_CONTEXT'})
        if 'FA037' in row['scenario_atom_ids'] or 'PU008' in row['scenario_atom_ids']:
            violations.append({'scenario_id': row['scenario_id'], 'reason': 'REPORTED_CONFLICTING_ATOM'})
        if scope.get('operating_scenario') == '室外停车场' and 'CN_parking_multistory' in row['scenario_atom_ids']:
            violations.append({'scenario_id': row['scenario_id'], 'reason': 'OUTDOOR_GARAGE_CONFLICT'})
        if scope.get('source_slope_pct') == 15 and dims['ROAD'].get('atom_id') != 'FB003':
            violations.append({'scenario_id': row['scenario_id'], 'reason': 'SOURCE_SLOPE_NOT_PRESERVED'})
        if not row['e_preview']['status'].startswith('READY'):
            pending[row['e_preview']['reason_code']] += 1
    result = {
        'previous_merge_groups': previous_groups, 'previous_removed_candidates': previous_removed,
        'previous_cross_parent_removed_candidates': cross_parent_removed,
        'previous_merges_status': 'WITHDRAWN: original equivalence omitted source operating conditions',
        'new_candidates': len(rows), 'source_location_count': len(scopes), 'source_locations': dict(scopes),
        'road_atoms': dict(road_atoms), 'semantic_conflict_count': len(violations),
        'semantic_conflicts': violations, 'pending_e_reasons': dict(pending),
        'coverage_limit': 'One-axis representatives; not exhaustive joint ODD coverage or new causal validation.',
    }
    summary = json.loads((after / 'summary.json').read_text())
    result['permitted_road_types'] = summary['source_coverage']['permitted_road_types']
    result['road_type_coverage_status'] = summary['source_coverage']['road_type_coverage_status']
    assert not violations, violations[:5]
    assert set(scopes) == {'室外停车场', '室内停车场', 'AVP可用的停车场自构图区域内'}
    assert all(row['source_operating_conditions']['permitted_road_types'] == result['permitted_road_types'] for row in rows)
    (after / 'source_coverage_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n')
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--before', type=Path, default=Path('output/p6'))
    parser.add_argument('--after', type=Path, default=Path('output/p6_review'))
    args = parser.parse_args()
    print(json.dumps(audit(args.before, args.after), ensure_ascii=False, indent=2))
