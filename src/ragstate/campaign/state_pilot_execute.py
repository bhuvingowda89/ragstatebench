"""Resumable execution of the predeclared two-track transition pilot."""
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
from pathlib import Path

from .phase7b import _json
from .state_pilot_manifest import _load_frozen, digest
from .state_pilot_runtime import execute_track_a


SOURCE_FILES = (
    'src/ragstate/campaign/state_pilot_execute.py',
    'src/ragstate/campaign/state_pilot_runtime.py',
    'src/ragstate/replicas/scientific.py',
    'src/ragstate/retrieval/state_isolated.py',
    'src/ragstate/retrieval/replica_pa2.py',
    'src/ragstate/campaign/state_capabilities.py',
    'src/ragstate/campaign/state_pilot_manifest.py',
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def freeze_execution(config_path='configs/state-isolated-pilot-v1.json'):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    root = Path('results') / config['campaign_version']
    destination = root / 'pre_execution_freeze.json'
    if destination.exists():
        raise ValueError('Pre-execution freeze already exists')
    manifest = json.loads((root / 'scenario_manifest.json').read_text())
    transfer = json.loads((Path('results') / config['transfer_version'] /
                           'transfer_manifest.json').read_text())
    component = json.loads((root / 'component_factorial_manifest.json').read_text())
    if any(digest(obj['scenarios']) != obj['sha256'] for obj in (manifest, transfer, component)):
        raise ValueError('Manifest content hash mismatch')
    clean = json.loads((root / 'ta0_clean_summary.json').read_text())
    if clean['D1'] != 'PASS':
        raise ValueError('Track-A clean validity gate failed')
    policy = {
        'version': f"{config['campaign_version']}-preexecution",
        'runtime_model_version': config.get('runtime_model_version'),
        'cache_semantic_version': config.get('cache_semantic_version'),
        'capability_policy_version': config.get('capability_policy_version'),
        'frozen_before_scientific_transition_outcomes': True,
        'corpus_sha256': config['corpus_sha256'],
        'track_b_cohort_sha256': config['track_b_cohort_sha256'],
        'scenario_sha256': manifest['sha256'],
        'transfer_sha256': transfer['sha256'],
        'component_factorial_sha256': component['sha256'],
        'analysis_policy_sha256': _sha(Path('data') / f"state-isolated-track-{config['campaign_version'].rsplit('-',1)[-1]}" / 'analysis_policy_v2.json'),
        'clean_validation_sha256': _sha(root / 'ta0_clean_observations.jsonl'),
        'config_sha256': _sha(config_path),
        'source_sha256': {name: _sha(name) for name in SOURCE_FILES},
        'replica_ordering': config['state_isolated_ordering'],
        'grant_prestate': 'same-tenant direct deny at absolute tick 9; grant at tick 10',
        'event_delivery': {
            'A0_A3_A4': 'severity delay from absolute tick 10',
            'A1': 'first 4-tick periodic boundary at or after delayed arrival',
            'A2_A5': 'successful event payload delivered at tick 10; dropped events suppressed',
            'PARTIAL': 'fraction of real chunk or ACL units at delivery, remainder 2 ticks later',
            'OUT_OF_ORDER': 'old content or allow replay 2 ticks after delivery; recovery 4 ticks after delivery',
            'DROPPED': 'never delivered within fixed observation horizon; right-censored',
        },
        'cache_semantics': 'C1 candidate cache only; every hit re-enters indexed authorization and live context validation; epochs use local replica state',
        'compound_components': {
            'CF1': 'F1 supersession delay; F2 stale derived cache',
            'CF2': 'F1 revocation delay; F2 stale derived cache',
            'CF3': 'F1 revocation delay; F2 partial per-chunk ACL propagation',
            'CF4': 'F1 delete delay; F2 stale ingestion and ACL replay',
            'CF5': 'F1 content update; F2 permission revoke',
            'CF6': 'common immediate revoke; F1 cross-tenant same-topic candidate cache; F2 stale ACL allow for that candidate',
        },
        'compound_arms': ['NONE', 'F1', 'F2', 'F1_F2'],
        'parent_without_chunk': 'parent ranks a source without corresponding chunk; no evidence emitted',
        'chunk_without_parent': 'candidate chunk absent from parent index is dropped before PA2 fusion',
        'canonical_access': 'oracle relevance/evaluation; A4/A5 live authorization only; no canonical content in retrieval',
        'logical_observation_ticks': config['observation_relative_ticks'],
        'right_censoring': 'no stable correctness through final tick => censored, T_RSC null',
    }
    destination.write_text(_json(policy) + '\n')
    return {'sha256': _sha(destination), 'scenarios_a': len(manifest['scenarios']),
            'scenarios_b': len(transfer['scenarios'])}


def _verify_freeze(config_path):
    config = json.loads(Path(config_path).read_text())
    root = Path('results') / config['campaign_version']
    freeze = json.loads((root / 'pre_execution_freeze.json').read_text())
    checks = {'config_sha256': _sha(config_path),
              'scenario_sha256': json.loads((root / 'scenario_manifest.json').read_text())['sha256'],
              'transfer_sha256': json.loads((Path('results') / config['transfer_version'] /
                                             'transfer_manifest.json').read_text())['sha256'],
              'component_factorial_sha256': json.loads((root / 'component_factorial_manifest.json').read_text())['sha256']}
    for key, value in checks.items():
        if freeze[key] != value:
            raise ValueError('Pre-execution freeze changed: ' + key)
    for name, expected in freeze['source_sha256'].items():
        if _sha(name) != expected:
            raise ValueError('Executable source changed after freeze: ' + name)
    return freeze


def _atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(_json(value) + '\n')
    temporary.replace(path)


def _write_scenario(path, rows):
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('w') as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True) + '\n')
    temporary.replace(path)


def consolidate(output, scenarios):
    output = Path(output)
    rows_dir = output / 'scenario_rows'
    jsonl = output / 'observations.jsonl'
    csv_path = output / 'observations.csv'
    temp_jsonl = jsonl.with_suffix('.jsonl.tmp')
    temp_csv = csv_path.with_suffix('.csv.tmp')
    fieldnames = None
    count = 0
    with temp_jsonl.open('w') as jstream, temp_csv.open('w', newline='') as cstream:
        writer = None
        for scenario in scenarios:
            path = rows_dir / (scenario['scenario_id'] + '.jsonl')
            if not path.exists():
                continue
            for line in path.read_text().splitlines():
                row = json.loads(line)
                if writer is None:
                    fieldnames = list(row)
                    writer = csv.DictWriter(cstream, fieldnames=fieldnames)
                    writer.writeheader()
                writer.writerow({key: json.dumps(value, sort_keys=True)
                                 if isinstance(value, (list, dict)) else value
                                 for key, value in row.items()})
                jstream.write(line + '\n')
                count += 1
    temp_jsonl.replace(jsonl)
    temp_csv.replace(csv_path)
    return count


def run(track, config_path='configs/state-isolated-pilot-v1.json', *,
        blocks=None, limit=None, consolidate_now=True):
    _verify_freeze(config_path)
    config = json.loads(Path(config_path).read_text())
    world, metadata, queries, cohort = _load_frozen(config)
    query_by_id = {q.query_id: q for q in queries}
    if track == 'A':
        output = Path('results') / config['campaign_version']
        manifest = json.loads((output / 'scenario_manifest.json').read_text())
    else:
        output = Path('results') / config['transfer_version']
        manifest = json.loads((output / 'transfer_manifest.json').read_text())
    scenarios = manifest['scenarios']
    if digest(scenarios) != manifest['sha256']:
        raise ValueError('Scenario manifest content changed')
    if track == 'B' and any(s['query_id'] not in cohort for s in scenarios):
        raise ValueError('Track-B scenario outside frozen cohort-v2')
    rows_dir = output / 'scenario_rows'
    rows_dir.mkdir(exist_ok=True)
    status_path = output / 'scenario_status.json'
    status = json.loads(status_path.read_text()) if status_path.exists() else {
        s['scenario_id']: {'status': 'pending', 'reason': None} for s in scenarios}
    adapter = None
    if track == 'B':
        os.environ['HF_HUB_OFFLINE'] = '1'
        os.environ['TRANSFORMERS_OFFLINE'] = '1'
        from ragstate.retrieval.replica_pa2 import PA2ReplicaRetriever
        adapter = PA2ReplicaRetriever(config)
    attempted = 0
    started = time.perf_counter()
    for scenario in scenarios:
        if blocks and scenario['block'] not in blocks:
            continue
        scenario_id = scenario['scenario_id']
        path = rows_dir / (scenario_id + '.jsonl')
        if status.get(scenario_id, {}).get('status') == 'complete' and path.exists():
            continue
        if limit is not None and attempted >= limit:
            break
        attempted += 1
        status[scenario_id] = {'status': 'running', 'reason': None}
        _atomic_json(status_path, status)
        try:
            rows = execute_track_a(scenario, world, metadata,
                                   query_by_id[scenario['query_id']], config,
                                   pa2_retriever=adapter)
            if len(rows) != len(config['observation_relative_ticks']):
                raise ValueError('Incomplete observation trajectory')
            _write_scenario(path, rows)
            status[scenario_id] = {'status': 'complete', 'reason': None}
        except ValueError as exc:
            status[scenario_id] = {'status': 'invalid', 'reason': str(exc)}
        except Exception as exc:
            status[scenario_id] = {'status': 'failed',
                                   'reason': f'{type(exc).__name__}: {exc}'}
        _atomic_json(status_path, status)
        if attempted % 50 == 0:
            print(f'{track}: {attempted} attempted; {sum(v["status"] == "complete" for v in status.values())} complete',
                  flush=True)
    count = consolidate(output, scenarios) if consolidate_now else None
    summary = {'track': track, 'attempted_this_invocation': attempted,
               'scenario_status': dict(Counter(v['status'] for v in status.values())),
               'observation_rows': count,
               'runtime_seconds_this_invocation': time.perf_counter() - started,
               'python': sys.version.split()[0], 'platform': platform.platform(),
               'corpus_sha256': config['corpus_sha256'],
               'cohort_sha256': config['track_b_cohort_sha256'] if track == 'B' else None,
               'manifest_sha256': manifest['sha256']}
    _atomic_json(output / 'execution_summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'run'))
    parser.add_argument('--track', choices=('A', 'B'), default='A')
    parser.add_argument('--config', default='configs/state-isolated-pilot-v1.json')
    parser.add_argument('--block', action='append')
    parser.add_argument('--limit', type=int)
    args = parser.parse_args()
    result = (freeze_execution(args.config) if args.action == 'freeze' else
              run(args.track, args.config, blocks=set(args.block) if args.block else None,
                  limit=args.limit))
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
