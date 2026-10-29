"""Fail-fast, audited orchestration around the frozen v2 transition runtime.

This module does not change retrieval, event delivery, metrics, or scenarios.
It records every authoritative access and C1 operation, validates each
trajectory, and stops immediately on a hard boundary failure.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

from ragstate.replicas.scientific import (ReplicaAuthorizationPlane,
    ReplicaContentPlane, ReplicaEvidenceCache)
from .state_capabilities import AccessAudit, ForbiddenCanonicalAccess
from .state_pilot_clean import clean_rows
from .state_pilot_execute import (_atomic_json, _verify_freeze, _write_scenario,
                                   consolidate)
from .state_pilot_manifest import _load_frozen, digest
from .state_pilot_runtime import execute_track_a


ROOT = Path('results/state-isolated-pilot-v2')
CONFIG = Path('configs/state-isolated-pilot-v2.json')
DETERMINISTIC_ROW_FIELDS = (
    'scenario_id', 'query_id', 'logical_time', 'transition', 'fault',
    'severity', 'architecture', 'oracle_logical_evidence_ids',
    'replica_resolved_evidence_ids', 'replica_resolved_versions',
    'candidate_chunk_ids', 'post_filter_chunk_ids', 'context_chunk_ids',
    'canonical_source_state_hash', 'canonical_authorization_state_hash',
    'chunk_index_hash', 'parent_index_hash', 'auth_metadata_hash',
    'cache_hash', 'context_hash', 'scrs', 'cbr', 'all_claims_success',
    'srr', 'dep', 'mvcr', 'ucer_candidate', 'ucer_post_filter',
    'ucer_context', 'ardr', 'cter', 'serr',
    'invalid_evidence_absence_success', 'cache_hit',
    'cache_invalidations', 'replica_refreshes', 'auth_validation_count',
    'convergence_status', 't_rsc', 'state_correct',
    'chunk_epoch', 'parent_epoch', 'auth_epoch',
)


def _jsonl(path, records):
    path.parent.mkdir(exist_ok=True, parents=True)
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        for record in records:
            stream.write(json.dumps(record, sort_keys=True) + '\n')
    temporary.replace(path)


def _scenario_subset(scenarios):
    """Structural selection fixed before observing any v2 scientific outcome."""
    requirements = (
        ('content', lambda s: s['block'] == 'TA1' and s['transition'] == 'UPDATE'
         and s['severity'] == 'S2'),
        ('authorization', lambda s: s['block'] == 'TA1' and s['transition'] == 'REVOKE'
         and s['severity'] == 'S2'),
        ('cache', lambda s: s['block'] == 'TA4' and s['transition'] == 'SUPERSEDE'
         and s['architecture'] == 'A0'),
        ('partial', lambda s: s['block'] == 'TA2' and s['fault'] == 'PARTIAL'
         and s['transition'] == 'UPDATE' and s['partial_fraction'] == 0.5),
        ('mitigation', lambda s: s['block'] == 'TA3' and s['transition'] == 'REVOKE'
         and s['architecture'] == 'A4' and s['severity'] == 'S2'),
        ('compound', lambda s: s['block'] == 'TA5' and s['compound'] == 'CF4'
         and s['component'] == 'F1_F2' and s['architecture'] == 'A0'),
    )
    selected = []
    for label, match in requirements:
        matches = sorted((s for s in scenarios if match(s)),
                         key=lambda s: s['scenario_id'])
        if not matches:
            raise ValueError('Frozen manifest lacks reproduction stratum: ' + label)
        selected.append({'stratum': label, 'scenario_id': matches[0]['scenario_id']})
    return selected


def freeze_control(config_path=CONFIG, root=ROOT):
    """Freeze only orchestration/audit identity; scientific manifest is read-only."""
    config_path, root = Path(config_path), Path(root)
    _verify_freeze(config_path)
    manifest = json.loads((root / 'scenario_manifest.json').read_text())
    if digest(manifest['scenarios']) != manifest['sha256']:
        raise ValueError('Frozen scenario hash mismatch')
    output = root / 'execution_control.json'
    if output.exists():
        current = json.loads(output.read_text())
        if current['harness_sha256'] != hashlib.sha256(Path(__file__).read_bytes()).hexdigest():
            raise ValueError('Execution harness changed after control freeze')
        return current
    control = {
        'campaign_id': 'state-isolated-pilot-v2',
        'scenario_manifest_sha256': manifest['sha256'],
        'frozen_runtime_sha256': hashlib.sha256((root / 'pre_execution_freeze.json').read_bytes()).hexdigest(),
        'harness_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'reproduction_subset': _scenario_subset(manifest['scenarios']),
        'audits_enabled': ['canonical_access', 'cache_stage'],
        'hard_stop_on': ['forbidden_canonical_access', 'cache_stage_violation',
                         'ta0_regression', 'unrelated_plane_corruption',
                         'frozen_provenance_mismatch'],
        'source_outcomes_consulted_for_selection': False,
    }
    _atomic_json(output, control)
    return control


def reconfirm_ta0(config_path=CONFIG, root=ROOT):
    config = json.loads(Path(config_path).read_text())
    expected = [json.loads(line) for line in
                (Path(root) / 'ta0_clean_observations.jsonl').read_text().splitlines()]
    observed = clean_rows(config)
    if observed != expected:
        raise ValueError('TA0 differs from the frozen v2 preflight')
    answerable = [r for r in observed if r['answerable']]
    negative = [r for r in observed if not r['answerable']]
    good = sum(r['valid'] for r in answerable)
    if good / len(answerable) < config['pilot_gates']['D1']['state_isolated_clean_answerable_min']:
        raise ValueError('TA0 answerable validity below frozen D1 threshold')
    if sum(r['valid'] for r in negative) != len(negative):
        raise ValueError('TA0 negative validity regression')
    if any(r['ucer_context'] or r['dep'] or r['srr'] or r['cter'] or r['mvcr']
           for r in observed):
        raise ValueError('TA0 clean integrity contamination')
    return {'answerable_valid': good, 'answerable_total': len(answerable),
            'negative_valid': len(negative), 'negative_total': len(negative),
            'rows_identical_to_preflight': True, 'status': 'PASS'}


def _cache_entry(entry):
    if entry is None:
        return None
    return {'cache_type': entry.cache_type, 'pipeline_stage': entry.pipeline_stage,
            'principal_id': entry.principal_id, 'tenant_id': entry.tenant_id,
            'evidence_ids': list(entry.evidence_ids), 'versions': list(entry.versions),
            'scores_ordered': [score for _, score, _ in entry.evidence],
            'components_ordered': [parts for _, _, parts in entry.evidence],
            'creation_time': entry.creation_time,
            'dependency_epochs': {'chunk': entry.chunk_epoch,
                                  'parent': entry.parent_epoch,
                                  'auth': entry.auth_epoch},
            'producer_architecture': entry.producer_architecture,
            'source_state_hashes': dict(entry.source_state_hashes)}


@contextmanager
def capture_scenario(scenario_id):
    """Observe frozen methods without changing their results or stage order."""
    accesses, cache_ops = [], []
    hashes = {'chunk': {}, 'parent': {}, 'auth': {}}
    original_record = AccessAudit.record
    original_put = ReplicaEvidenceCache.put
    original_get = ReplicaEvidenceCache.get
    original_invalidate = ReplicaEvidenceCache.invalidate
    original_content_hash = ReplicaContentPlane.state_hash
    original_auth_hash = ReplicaAuthorizationPlane.state_hash

    def record(self, component, plane, logical_time, reason):
        before = len(self.events)
        try:
            return original_record(self, component, plane, logical_time, reason)
        finally:
            if len(self.events) > before:
                accesses.append({'scenario_id': scenario_id, **self.events[-1]})

    def put(self, key, *args, **kwargs):
        before = len(self.stage_audit)
        try:
            return original_put(self, key, *args, **kwargs)
        finally:
            cache_ops.append({'scenario_id': scenario_id, 'operation': 'put',
                              'key': key, 'entry': _cache_entry(self.entries.get(key)),
                              'stage_events': self.stage_audit[before:]})

    def get(self, key, *args, **kwargs):
        before = len(self.stage_audit)
        hit = None
        try:
            hit = original_get(self, key, *args, **kwargs)
            return hit
        finally:
            cache_ops.append({'scenario_id': scenario_id, 'operation': 'get',
                              'key': key, 'hit': hit is not None,
                              'entry': _cache_entry(self.entries.get(key)),
                              'consumer_stage': kwargs.get('consumer_stage', 'candidate_retrieval'),
                              'logical_time': kwargs.get('logical_time'),
                              'stage_events': self.stage_audit[before:]})

    def invalidate(self, key=None):
        affected = list(self.entries) if key is None else ([key] if key in self.entries else [])
        try:
            return original_invalidate(self, key)
        finally:
            cache_ops.append({'scenario_id': scenario_id, 'operation': 'invalidate',
                              'key': key, 'affected_keys': affected,
                              'entry': None, 'stage_events': []})

    def content_hash(self):
        value = original_content_hash(self)
        hashes['chunk' if self.chunked else 'parent'][value] = self.epoch
        return value

    def auth_hash(self, tick):
        value = original_auth_hash(self, tick)
        hashes['auth'][value] = self.epoch
        return value

    AccessAudit.record = record
    ReplicaEvidenceCache.put = put
    ReplicaEvidenceCache.get = get
    ReplicaEvidenceCache.invalidate = invalidate
    ReplicaContentPlane.state_hash = content_hash
    ReplicaAuthorizationPlane.state_hash = auth_hash
    try:
        yield accesses, cache_ops, hashes
    finally:
        AccessAudit.record = original_record
        ReplicaEvidenceCache.put = original_put
        ReplicaEvidenceCache.get = original_get
        ReplicaEvidenceCache.invalidate = original_invalidate
        ReplicaContentPlane.state_hash = original_content_hash
        ReplicaAuthorizationPlane.state_hash = original_auth_hash


def execute_recorded(scenario, world, metadata, query, config):
    with capture_scenario(scenario['scenario_id']) as (accesses, cache_ops, hashes):
        rows = execute_track_a(scenario, world, metadata, query, config)
    for row in rows:
        row['chunk_epoch'] = hashes['chunk'].get(row['chunk_index_hash'])
        row['parent_epoch'] = hashes['parent'].get(row['parent_index_hash'])
        row['auth_epoch'] = hashes['auth'].get(row['auth_metadata_hash'])
        if None in (row['chunk_epoch'], row['parent_epoch'], row['auth_epoch']):
            raise ValueError('Missing local epoch for recorded state hash')
    return rows, accesses, cache_ops


def scenario_validity(scenario, rows, accesses, cache_ops, config):
    reasons = []
    ticks = config['observation_relative_ticks']
    if [row['logical_time'] for row in rows] != ticks:
        reasons.append('observation_grid_mismatch')
    if any(row['scenario_id'] != scenario['scenario_id'] or
           row['campaign_id'] != config['campaign_version'] for row in rows):
        reasons.append('provenance_mismatch')
    if any(not event['allowed'] for event in accesses):
        reasons.append('forbidden_canonical_access')
    if any(not stage['valid'] for op in cache_ops for stage in op['stage_events']):
        reasons.append('cache_stage_violation')
    pre = rows[0]
    post = next(row for row in rows if row['logical_time'] == 0)
    source_changed = pre['canonical_source_state_hash'] != post['canonical_source_state_hash']
    auth_changed = pre['canonical_authorization_state_hash'] != post['canonical_authorization_state_hash']
    compound = scenario.get('compound')
    component = scenario.get('component')
    if compound == 'CF5':
        expected_source = component in ('F1', 'F1_F2')
        expected_auth = component in ('F2', 'F1_F2')
    else:
        expected_source = scenario['transition'] in ('UPDATE', 'SUPERSEDE', 'DELETE')
        expected_auth = scenario['transition'] in ('GRANT', 'REVOKE')
    if source_changed != expected_source:
        reasons.append('unintended_or_missing_canonical_source_transition')
    if auth_changed != expected_auth:
        reasons.append('unintended_or_missing_canonical_auth_transition')
    if scenario['transition'] in ('GRANT', 'REVOKE') and compound != 'CF5':
        if any(row['chunk_index_hash'] != pre['chunk_index_hash'] or
               row['parent_index_hash'] != pre['parent_index_hash'] for row in rows):
            reasons.append('authorization_transition_changed_content_replica')
    if scenario['fault'] == 'DROPPED' and any(row['replica_refreshes'] for row in rows):
        reasons.append('dropped_event_was_delivered')
    if scenario['fault'] == 'OUT_OF_ORDER' and not any(row['serr'] for row in rows):
        reasons.append('out_of_order_fault_not_applied')
    if scenario['fault'] == 'PARTIAL' and not any(row['replica_refreshes'] >= 2 for row in rows):
        reasons.append('partial_remainder_not_delivered')
    if scenario['fault'] == 'STALE_CACHE' and not any(row['cache_hit'] for row in rows):
        reasons.append('cache_fault_not_exercised')
    if scenario['fault'] == 'COMPOUND' and compound in ('CF1', 'CF2', 'CF6') \
            and component in ('F2', 'F1_F2') and not any(row['cache_hit'] for row in rows):
        reasons.append('compound_cache_component_not_exercised')
    censored = rows[-1]['convergence_status'] == 'censored'
    return {'scenario_id': scenario['scenario_id'], 'block': scenario['block'],
            'query_id': scenario['query_id'], 'transition': scenario['transition'],
            'fault': scenario['fault'], 'severity': scenario['severity'],
            'architecture': scenario['architecture'],
            'validity': 'invalid' if reasons else 'valid',
            'censored': int(censored), 'reasons': ';'.join(reasons),
            'access_count': len(accesses), 'cache_operation_count': len(cache_ops),
            'forbidden_access_count': sum(not event['allowed'] for event in accesses),
            'cache_stage_violation_count': sum(not stage['valid']
                for op in cache_ops for stage in op['stage_events'])}


def _stable_fingerprint(rows, accesses, cache_ops):
    payload = {
        'rows': [{field: row[field] for field in DETERMINISTIC_ROW_FIELDS}
                 for row in rows],
        'accesses': accesses,
        'cache_ops': cache_ops,
    }
    return digest(payload)


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def reproduce(config_path=CONFIG, root=ROOT):
    _verify_freeze(config_path)
    control = freeze_control(config_path, root)
    config = json.loads(Path(config_path).read_text())
    world, metadata, queries, _ = _load_frozen(config)
    q_by_id = {query.query_id: query for query in queries}
    manifest = json.loads((Path(root) / 'scenario_manifest.json').read_text())
    scenarios = {s['scenario_id']: s for s in manifest['scenarios']}
    output = []
    for selected in control['reproduction_subset']:
        scenario = scenarios[selected['scenario_id']]
        rows, accesses, cache_ops = execute_recorded(
            scenario, world, metadata, q_by_id[scenario['query_id']], config)
        output.append({'stratum': selected['stratum'],
                       'scenario_id': scenario['scenario_id'],
                       'fingerprint': _stable_fingerprint(rows, accesses, cache_ops)})
    return output


def run(config_path=CONFIG, root=ROOT):
    config_path, root = Path(config_path), Path(root)
    _verify_freeze(config_path)
    control = freeze_control(config_path, root)
    config = json.loads(config_path.read_text())
    ta0 = reconfirm_ta0(config_path, root)
    _atomic_json(root / 'ta0_reconfirmation.json', ta0)
    manifest = json.loads((root / 'scenario_manifest.json').read_text())
    if digest(manifest['scenarios']) != control['scenario_manifest_sha256']:
        raise ValueError('Manifest differs from frozen execution control')
    scenarios = manifest['scenarios']
    world, metadata, queries, _ = _load_frozen(config)
    q_by_id = {query.query_id: query for query in queries}
    row_dir = root / 'scenario_rows'
    access_dir = root / 'canonical_access_rows'
    cache_dir = root / 'cache_stage_rows'
    for path in (row_dir, access_dir, cache_dir):
        path.mkdir(exist_ok=True)
    status_path = root / 'scenario_status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {
        scenario['scenario_id']: {'status': 'pending', 'reason': None}
        for scenario in scenarios}
    started = time.perf_counter()
    completed_this_invocation = 0
    for scenario in scenarios:
        scenario_id = scenario['scenario_id']
        paths = [folder / (scenario_id + '.jsonl') for folder in
                 (row_dir, access_dir, cache_dir)]
        if status.get(scenario_id, {}).get('status') in ('valid', 'censored') \
                and all(path.exists() for path in paths):
            continue
        status[scenario_id] = {'status': 'running', 'reason': None}
        _atomic_json(status_path, status)
        try:
            rows, accesses, cache_ops = execute_recorded(
                scenario, world, metadata, q_by_id[scenario['query_id']], config)
            validity = scenario_validity(scenario, rows, accesses, cache_ops, config)
            _jsonl(paths[0], rows)
            _jsonl(paths[1], accesses)
            _jsonl(paths[2], cache_ops)
            if validity['validity'] != 'valid':
                status[scenario_id] = {'status': 'invalid', 'reason': validity['reasons']}
                _atomic_json(status_path, status)
                _atomic_json(root / 'campaign_invalid.json', {
                    'status': 'INVALID', 'scenario_id': scenario_id,
                    'reason': validity['reasons'], 'hard_stop': True})
                raise RuntimeError('Scenario validity failure: ' + validity['reasons'])
            status[scenario_id] = {'status': 'censored' if validity['censored'] else 'valid',
                                   'reason': None}
        except (ForbiddenCanonicalAccess, ValueError) as exc:
            status[scenario_id] = {'status': 'invalid', 'reason': str(exc)}
            _atomic_json(status_path, status)
            _atomic_json(root / 'campaign_invalid.json', {
                'status': 'INVALID', 'scenario_id': scenario_id,
                'reason': str(exc), 'hard_stop': True})
            raise
        except RuntimeError:
            raise
        except Exception as exc:
            status[scenario_id] = {'status': 'execution_failed',
                                   'reason': f'{type(exc).__name__}: {exc}'}
            _atomic_json(status_path, status)
            _atomic_json(root / 'campaign_invalid.json', {
                'status': 'INVALID', 'scenario_id': scenario_id,
                'reason': status[scenario_id]['reason'], 'hard_stop': True})
            raise
        _atomic_json(status_path, status)
        completed_this_invocation += 1
        if completed_this_invocation % 50 == 0:
            print(f'Track A v2: {completed_this_invocation} completed this invocation', flush=True)
    observations = consolidate(root, scenarios)
    validity_rows = []
    for scenario in scenarios:
        sid = scenario['scenario_id']
        rows = _read_jsonl(row_dir / (sid + '.jsonl'))
        accesses = _read_jsonl(access_dir / (sid + '.jsonl'))
        cache_ops = _read_jsonl(cache_dir / (sid + '.jsonl'))
        validity_rows.append(scenario_validity(scenario, rows, accesses, cache_ops, config))
    with (root / 'scenario_validity.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(validity_rows[0]))
        writer.writeheader()
        writer.writerows(validity_rows)
    summary = {'campaign_id': config['campaign_version'],
               'scenario_count': len(scenarios), 'observation_rows': observations,
               'status_counts': dict(Counter(row['status'] for row in status.values())),
               'censored_scenarios': sum(row['censored'] for row in validity_rows),
               'runtime_seconds_this_invocation': time.perf_counter() - started,
               'manifest_sha256': manifest['sha256'],
               'harness_sha256': control['harness_sha256'],
               'python': sys.version.split()[0], 'platform': platform.platform(),
               'ta0': ta0}
    _atomic_json(root / 'execution_summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'run', 'reproduce'))
    parser.add_argument('--config', default=str(CONFIG))
    parser.add_argument('--root', default=str(ROOT))
    args = parser.parse_args()
    if args.action == 'freeze':
        result = freeze_control(args.config, args.root)
    elif args.action == 'run':
        result = run(args.config, args.root)
    else:
        result = reproduce(args.config, args.root)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
