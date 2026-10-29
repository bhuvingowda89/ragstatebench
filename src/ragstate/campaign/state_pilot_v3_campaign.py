"""Versioned validity-control harness for the unchanged Track-A runtime.

Scientific rows from v2 are never read here. Component effectiveness is
checked against observed replica/cache state, separately from outcome metrics.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from ragstate.replicas.scientific import ReplicaAuthorizationPlane, ReplicaContentPlane
from .state_pilot_execute import _atomic_json, _verify_freeze, consolidate, freeze_execution
from .state_pilot_manifest import _load_frozen, digest
from .state_pilot_v2_campaign import (execute_recorded, _jsonl, _scenario_subset,
    _stable_fingerprint)
from .state_pilot_clean import clean_rows
from .state_pilot_validity_v3 import (POLICY_VERSION, COMPOUND_RULES, ISOLATED_RULES,
    compound_matrix, isolated_matrix, validate_components)


CONFIG = Path('configs/state-isolated-pilot-v3.json')
ROOT = Path('results/state-isolated-pilot-v3')
PREFLIGHT = Path('results/state-isolated-pilot-v3-preflight')


@contextmanager
def capture_replica_snapshots():
    """Observe state_hash calls without changing a frozen replica operation."""
    original_content = ReplicaContentPlane.state_hash
    original_auth = ReplicaAuthorizationPlane.state_hash
    snapshots = {'chunk': {}, 'parent': {}, 'auth': {}}

    def content_hash(self):
        value = original_content(self)
        plane = 'chunk' if self.chunked else 'parent'
        snapshots[plane][value] = {
            lid: tuple((part.chunk_id, part.version) for part in self.chunks.get(lid, ()))
            if self.chunked else ((doc.document_id, doc.version),)
            for lid, doc in self.documents.items()
        }
        return value

    def auth_hash(self, tick):
        value = original_auth(self, tick)
        snapshots['auth'][value] = {
            'overrides': tuple(sorted((principal, chunk_id, allow)
                for (principal, chunk_id), allow in self.chunk_overrides.items())),
            'epoch': self.epoch,
        }
        return value

    ReplicaContentPlane.state_hash = content_hash
    ReplicaAuthorizationPlane.state_hash = auth_hash
    try:
        yield snapshots
    finally:
        ReplicaContentPlane.state_hash = original_content
        ReplicaAuthorizationPlane.state_hash = original_auth


def recorded_with_state(scenario, world, metadata, query, config):
    with capture_replica_snapshots() as snapshots:
        rows, accesses, cache_ops = execute_recorded(scenario, world, metadata, query, config)
    return rows, accesses, cache_ops, snapshots


def validate_scenario(scenario, rows, accesses, cache_ops, snapshots,
                      query, metadata, world, config):
    reasons = []
    if [r['logical_time'] for r in rows] != config['observation_relative_ticks']:
        reasons.append('observation_grid_mismatch')
    if any(r['scenario_id'] != scenario['scenario_id'] or
           r['campaign_id'] != config['campaign_version'] for r in rows):
        reasons.append('provenance_mismatch')
    if any(not event['allowed'] for event in accesses):
        reasons.append('forbidden_canonical_access')
    if any(not event['valid'] for op in cache_ops for event in op['stage_events']):
        reasons.append('cache_stage_violation')
    pre, post = rows[0], rows[1]
    compound, arm = scenario.get('compound'), scenario.get('component')
    if compound == 'CF5':
        source_expected = arm in ('F1', 'F1_F2')
        auth_expected = arm in ('F2', 'F1_F2')
    else:
        source_expected = scenario['transition'] in ('UPDATE', 'SUPERSEDE', 'DELETE')
        auth_expected = scenario['transition'] in ('GRANT', 'REVOKE')
    if (pre['canonical_source_state_hash'] != post['canonical_source_state_hash']) != source_expected:
        reasons.append('canonical_source_transition_mismatch')
    if (pre['canonical_authorization_state_hash'] != post['canonical_authorization_state_hash']) != auth_expected:
        reasons.append('canonical_authorization_transition_mismatch')
    if scenario['transition'] in ('GRANT', 'REVOKE') and compound != 'CF5':
        if any(r['chunk_index_hash'] != pre['chunk_index_hash'] or
               r['parent_index_hash'] != pre['parent_index_hash'] for r in rows):
            reasons.append('authorization_fault_changed_content_replica')
    components = validate_components(scenario, rows, cache_ops, snapshots,
                                     query, metadata, world)
    for component in components:
        if not component['effective']:
            reasons.append('component_not_effective:' + component['component'])
    return {'scenario_id': scenario['scenario_id'], 'block': scenario['block'],
            'query_id': scenario['query_id'], 'transition': scenario['transition'],
            'fault': scenario['fault'], 'severity': scenario['severity'],
            'architecture': scenario['architecture'],
            'validity': 'invalid' if reasons else 'valid',
            'censored': int(rows[-1]['convergence_status'] == 'censored'),
            'reasons': ';'.join(reasons), 'component_evidence': components,
            'access_count': len(accesses), 'cache_operation_count': len(cache_ops),
            'forbidden_access_count': sum(not e['allowed'] for e in accesses),
            'cache_stage_violation_count': sum(not e['valid'] for op in cache_ops
                                               for e in op['stage_events'])}


def freeze_v3():
    """Clone only campaign identity and execution-control metadata from v2.

    The frozen scenario fields and IDs are compared explicitly before writing.
    The v2 result directory is read only for manifests and clean preflight.
    """
    if ROOT.exists() or PREFLIGHT.exists():
        raise ValueError('v3 output already exists; refusing to overwrite')
    old_config = json.loads(Path('configs/state-isolated-pilot-v2.json').read_text())
    config = dict(old_config)
    config['campaign_version'] = 'state-isolated-pilot-v3'
    config['validity_policy_version'] = POLICY_VERSION
    CONFIG.write_text(json.dumps(config, indent=2, sort_keys=True) + '\n')
    ROOT.mkdir(parents=True)
    PREFLIGHT.mkdir(parents=True)
    for name in ('scenario_manifest.json', 'component_factorial_manifest.json'):
        old = json.loads((Path('results/state-isolated-pilot-v2') / name).read_text())
        scenarios = []
        for item in old['scenarios']:
            new = dict(item)
            new['campaign_version'] = config['campaign_version']
            new['validity_policy_version'] = POLICY_VERSION
            scenarios.append(new)
        output = {**{key: value for key, value in old.items() if key not in ('scenarios', 'sha256')},
                  'scenarios': scenarios, 'sha256': digest(scenarios),
                  'validity_policy_version': POLICY_VERSION}
        _atomic_json(ROOT / name, output)
    old_clean = Path('results/state-isolated-pilot-v2')
    for name in ('ta0_clean_observations.jsonl', 'ta0_clean_observations.csv',
                 'ta0_clean_summary.json'):
        (ROOT / name).write_bytes((old_clean / name).read_bytes())
    policy_dir = Path('data/state-isolated-track-v3')
    policy_dir.mkdir(parents=True, exist_ok=True)
    (policy_dir / 'analysis_policy_v2.json').write_bytes(
        Path('data/state-isolated-track-v2/analysis_policy_v2.json').read_bytes())
    freeze = freeze_execution(CONFIG)
    control = {
        'campaign_id': config['campaign_version'], 'validity_policy_version': POLICY_VERSION,
        'scenario_manifest_sha256': json.loads((ROOT / 'scenario_manifest.json').read_text())['sha256'],
        'reproduction_subset': _scenario_subset(json.loads((ROOT / 'scenario_manifest.json').read_text())['scenarios']),
        'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'policy_sha256': hashlib.sha256(Path(__file__).with_name('state_pilot_validity_v3.py').read_bytes()).hexdigest(),
        'pre_execution_freeze_sha256': freeze['sha256'],
    }
    _atomic_json(ROOT / 'execution_control.json', control)
    return control


def _verify_control():
    _verify_freeze(CONFIG)
    control = json.loads((ROOT / 'execution_control.json').read_text())
    if control['harness_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
        raise ValueError('v3 execution harness changed after freeze')
    if control['policy_sha256'] != hashlib.sha256(Path(__file__).with_name('state_pilot_validity_v3.py').read_bytes()).hexdigest():
        raise ValueError('v3 validity policy changed after freeze')
    manifest = json.loads((ROOT / 'scenario_manifest.json').read_text())
    if control['scenario_manifest_sha256'] != digest(manifest['scenarios']):
        raise ValueError('v3 manifest changed after freeze')
    return control, manifest


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def _scenario_by_id(manifest, scenario_id):
    return next(s for s in manifest['scenarios'] if s['scenario_id'] == scenario_id)


def case_fingerprint(scenario_id):
    _, manifest = _verify_control()
    config = json.loads(CONFIG.read_text())
    world, metadata, queries, _ = _load_frozen(config)
    scenario = _scenario_by_id(manifest, scenario_id)
    query = next(q for q in queries if q.query_id == scenario['query_id'])
    rows, accesses, cache_ops, snapshots = recorded_with_state(
        scenario, world, metadata, query, config)
    validity = validate_scenario(scenario, rows, accesses, cache_ops,
                                 snapshots, query, metadata, world, config)
    return {'scenario_id': scenario_id, 'validity': validity,
            'fingerprint': _stable_fingerprint(rows, accesses, cache_ops),
            'trace': [{'logical_time': row['logical_time'],
                       'canonical_source_hash': row['canonical_source_state_hash'],
                       'canonical_auth_hash': row['canonical_authorization_state_hash'],
                       'chunk_hash': row['chunk_index_hash'],
                       'parent_hash': row['parent_index_hash'],
                       'acl_hash': row['auth_metadata_hash'],
                       'cache_hash': row['cache_hash'],
                       'candidates': row['candidate_chunk_ids'],
                       'post_filter': row['post_filter_chunk_ids'],
                       'context': row['context_chunk_ids'],
                       'scrs': row['scrs'], 'dep': row['dep'],
                       'ucer_context': row['ucer_context'],
                       'cache_hit': row['cache_hit']} for row in rows],
            'access_count': len(accesses),
            'forbidden_access_count': sum(not e['allowed'] for e in accesses),
            'cache_operations': len(cache_ops),
            'cache_stage_violations': sum(not e['valid'] for op in cache_ops
                                          for e in op['stage_events'])}


def _write_csv(path, records):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)


def preflight():
    control, manifest = _verify_control()
    if (PREFLIGHT / 'preflight_gate_summary.json').exists():
        raise ValueError('v3 preflight already exists; refusing to overwrite')
    config = json.loads(CONFIG.read_text())
    scenario_list = manifest['scenarios']
    # Freeze declarative validity semantics independently of campaign rows.
    _atomic_json(PREFLIGHT / 'validity_policy.json', {
        'version': POLICY_VERSION, 'scientific_scenario_sha256': control['scenario_manifest_sha256'],
        'compound_rules': compound_matrix(), 'isolated_rules': isolated_matrix(),
        'validation_basis': 'replica and cache state plus recorded event operations, never final outcomes',
        'v2_scientific_rows_reused': False})
    _write_csv(PREFLIGHT / 'compound_validity_matrix.csv', compound_matrix())
    _write_csv(PREFLIGHT / 'isolated_fault_validity_matrix.csv', isolated_matrix())
    env = dict(os.environ)
    env['PYTHONPATH'] = '/Users/nischithacn/Library/Python/3.9/lib/python/site-packages' + \
        os.pathsep + str(Path('src').resolve())
    suite = subprocess.run([sys.executable, '-m', 'pytest', '-q'],
                           capture_output=True, text=True, env=env)
    _atomic_json(PREFLIGHT / 'regression_summary.json', {
        'returncode': suite.returncode, 'stdout': suite.stdout,
        'stderr': suite.stderr, 'tests_passed': suite.returncode == 0})
    gates = {'V0': 'PASS' if suite.returncode == 0 else 'FAIL'}
    if suite.returncode != 0:
        _atomic_json(PREFLIGHT / 'preflight_gate_summary.json', gates)
        return gates
    world, metadata, queries, _ = _load_frozen(config)
    by_id = {q.query_id: q for q in queries}
    # Representative frozen scenario per declared arm/fault and architecture.
    chosen = {}
    for scenario in scenario_list:
        key = ((scenario['compound'], scenario['component'], scenario['architecture'])
               if scenario['compound'] else
               (scenario['fault'], scenario['transition'], scenario['architecture']))
        chosen.setdefault(key, scenario)
    results = []
    total_accesses = total_cache_ops = forbidden = stage_violations = 0
    for scenario in chosen.values():
        query = by_id[scenario['query_id']]
        rows, accesses, cache_ops, snapshots = recorded_with_state(
            scenario, world, metadata, query, config)
        validity = validate_scenario(scenario, rows, accesses, cache_ops,
                                     snapshots, query, metadata, world, config)
        results.append(validity)
        total_accesses += len(accesses)
        total_cache_ops += len(cache_ops)
        forbidden += validity['forbidden_access_count']
        stage_violations += validity['cache_stage_violation_count']
    compounds = [r for r in results if r['fault'] == 'COMPOUND']
    isolated = [r for r in results if r['fault'] != 'COMPOUND']
    gates['V1'] = 'PASS' if all(r['validity'] == 'valid' for r in compounds
                               if 'CF6' in r['scenario_id']) else 'FAIL'
    gates['V2'] = 'PASS' if all(r['validity'] == 'valid' for r in compounds) else 'FAIL'
    gates['V3'] = 'PASS' if all(r['validity'] == 'valid' for r in isolated) else 'FAIL'
    gates['V4'] = 'PASS' if forbidden == 0 and stage_violations == 0 else 'FAIL'
    _atomic_json(PREFLIGHT / 'access_audit.json', {
        'total_events': total_accesses, 'forbidden': forbidden,
        'sample_count': len(results), 'status': gates['V4']})
    _atomic_json(PREFLIGHT / 'cache_stage_audit.json', {
        'total_operations': total_cache_ops, 'violations': stage_violations,
        'sample_count': len(results), 'status': gates['V4']})
    _atomic_json(PREFLIGHT / 'validity_smoke_summary.json', {
        'sample_count': len(results), 'compound_count': len(compounds),
        'isolated_count': len(isolated),
        'invalid': [{'scenario_id': r['scenario_id'], 'reasons': r['reasons']}
                    for r in results if r['validity'] != 'valid']})
    target = 'TA5__v3-q-engineering-00__REVOKE__S2__A0__COMPOUND__F2__CF6'
    local = case_fingerprint(target)
    _atomic_json(PREFLIGHT / 'cf6_regression_trace.json', local)
    fresh = subprocess.run([sys.executable, '-m',
        'ragstate.campaign.state_pilot_v3_campaign', 'case', '--scenario-id', target],
        capture_output=True, text=True, env=env)
    fresh_result = json.loads(fresh.stdout) if fresh.returncode == 0 else None
    gates['V5'] = ('PASS' if local['validity']['validity'] == 'valid' and
                   fresh_result is not None and
                   local['fingerprint'] == fresh_result['fingerprint'] and
                   local['validity'] == fresh_result['validity'] else 'FAIL')
    _atomic_json(PREFLIGHT / 'fresh_process_reproduction.json', {
        'scenario_id': target, 'fresh_process_exit': fresh.returncode,
        'local_fingerprint': local['fingerprint'],
        'fresh_fingerprint': fresh_result['fingerprint'] if fresh_result else None,
        'identical': gates['V5'] == 'PASS', 'stderr': fresh.stderr})
    _atomic_json(PREFLIGHT / 'preflight_gate_summary.json', gates)
    (PREFLIGHT / 'summary.md').write_text(
        '# Track-A v3 validity preflight\n\n' +
        '\n'.join(f'- {key}: {value}' for key, value in gates.items()) +
        f'\n\nAudited {len(results)} distinct scenario structures; '
        f'{forbidden} forbidden authoritative accesses and {stage_violations} cache-stage violations.\n'
        'The frozen scientific scenario fields, runtime, metrics, corpus, and Track-B manifest were not changed.\n')
    return gates


def run():
    control, manifest = _verify_control()
    gates = json.loads((PREFLIGHT / 'preflight_gate_summary.json').read_text())
    if any(gates.get('V' + str(i)) != 'PASS' for i in range(6)):
        raise ValueError('v3 preflight gates are not all PASS')
    config = json.loads(CONFIG.read_text())
    clean = clean_rows(config)
    old_clean = _read_jsonl(ROOT / 'ta0_clean_observations.jsonl')
    if clean != old_clean:
        raise ValueError('Track-A clean rows differ from frozen preflight')
    world, metadata, queries, _ = _load_frozen(config)
    query_by_id = {q.query_id: q for q in queries}
    scenarios = manifest['scenarios']
    for folder in ('scenario_rows', 'canonical_access_rows', 'cache_stage_rows'):
        (ROOT / folder).mkdir(exist_ok=True)
    (ROOT / 'validity_rows').mkdir(exist_ok=True)
    status_path = ROOT / 'scenario_status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {
        s['scenario_id']: {'status': 'pending', 'reason': None} for s in scenarios}
    started = time.perf_counter()
    for index, scenario in enumerate(scenarios):
        sid = scenario['scenario_id']
        if status[sid]['status'] in ('valid', 'censored'):
            continue
        status[sid] = {'status': 'running', 'reason': None}
        _atomic_json(status_path, status)
        try:
            query = query_by_id[scenario['query_id']]
            rows, accesses, cache_ops, snapshots = recorded_with_state(
                scenario, world, metadata, query, config)
            validity = validate_scenario(scenario, rows, accesses, cache_ops,
                                         snapshots, query, metadata, world, config)
            for folder, records in (('scenario_rows', rows),
                                    ('canonical_access_rows', accesses),
                                    ('cache_stage_rows', cache_ops)):
                _jsonl(ROOT / folder / (sid + '.jsonl'), records)
            _atomic_json(ROOT / 'validity_rows' / (sid + '.json'), validity)
            if validity['validity'] != 'valid':
                raise ValueError('Scenario validity failure: ' + validity['reasons'])
            status[sid] = {'status': 'censored' if validity['censored'] else 'valid',
                           'reason': None}
        except Exception as exc:
            status[sid] = {'status': 'invalid', 'reason': f'{type(exc).__name__}: {exc}'}
            _atomic_json(status_path, status)
            _atomic_json(ROOT / 'campaign_invalid.json', {'status': 'INVALID',
                'scenario_id': sid, 'reason': status[sid]['reason'], 'hard_stop': True})
            raise
        _atomic_json(status_path, status)
        if (index + 1) % 100 == 0:
            print(f'Track A v3: {index + 1}/{len(scenarios)} scenarios', flush=True)
    count = consolidate(ROOT, scenarios)
    validity = [json.loads((ROOT / 'validity_rows' / (s['scenario_id'] + '.json')).read_text())
                for s in scenarios]
    with (ROOT / 'scenario_validity.csv').open('w', newline='') as stream:
        fields = [key for key in validity[0] if key != 'component_evidence']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in validity)
    summary = {'scenario_count': len(scenarios), 'observation_rows': count,
               'status_counts': dict(Counter(v['status'] for v in status.values())),
               'runtime_seconds': time.perf_counter() - started,
               'scenario_manifest_sha256': control['scenario_manifest_sha256']}
    _atomic_json(ROOT / 'execution_summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'preflight', 'run', 'case'))
    parser.add_argument('--scenario-id')
    args = parser.parse_args()
    if args.action == 'freeze':
        result = freeze_v3()
    elif args.action == 'preflight':
        result = preflight()
    elif args.action == 'case':
        if not args.scenario_id:
            parser.error('case requires --scenario-id')
        result = case_fingerprint(args.scenario_id)
    else:
        result = run()
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
