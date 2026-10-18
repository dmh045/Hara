"""Save actual offline request counts and identity proofs from acceptance fixtures."""
import argparse
from copy import deepcopy
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'tests/unit'))
from test_reuse_reporting_acceptance import CountingClient, pair, parent, malfunction
from hara_agent.services.semantic import ScenarioFeasibilityAgent
from hara_agent.workflow.nodes.scenarios import _assess_instantiated_scenarios


def run(output):
    output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location('baseline', ROOT / 'tests/fixtures/reuse_reporting/scheduler_before.py')
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    records = []
    for name, mode in [('exact', 'exact'), ('pending_exact', 'pending'),
                       ('driver_noninterference', 'local'), ('driver_enablement', 'driver_relevant'),
                       ('changed_interaction', 'exact')]:
        a, b = pair()
        if name in {'exact', 'pending_exact'}:
            b = deepcopy(a)
            b.scenario_id = 'B'
        if name == 'changed_interaction':
            a.facts['object_position'] = 'FRONT'
            b.facts['object_position'] = 'REAR'
        for version, scheduler in [('before', baseline._assess_instantiated_scenarios),
                                   ('after', _assess_instantiated_scenarios)]:
            client = CountingClient([a, b], mode)
            agent = ScenarioFeasibilityAgent(client, batch_max_items=1)
            results, audit = scheduler(agent, malfunction(), deepcopy([a, b]), {}, None)
            record = {'case': name, 'version': version, 'mock_requests': len(client.calls),
                      'real_requests': 0, 'token_usage': 'not measured by scripted client',
                      'identity_count': len(results), 'audit': audit,
                      'results': [r.to_dict() for r in results]}
            records.append(record)
    expected = {'exact': 1, 'pending_exact': 1, 'driver_noninterference': 1,
                'driver_enablement': 2, 'changed_interaction': 2}
    assert all(r['mock_requests'] == (2 if r['version'] == 'before' else expected[r['case']]) for r in records)
    (output / 'comparison.json').write_text(json.dumps(records, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps([{k: r[k] for k in ('case', 'version', 'mock_requests', 'identity_count')} for r in records], indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, required=True)
    run(parser.parse_args().output)
