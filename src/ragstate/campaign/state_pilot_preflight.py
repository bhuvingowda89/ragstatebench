"""Boundary-first v2 preflight; never executes the scientific scenario matrix."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import site
import subprocess
import sys
from collections import Counter
from pathlib import Path

from .state_pilot_clean import run as run_clean
from .state_pilot_execute import _verify_freeze, freeze_execution
from .state_pilot_manifest import build
from .state_pilot_runtime import execute_track_a


SMOKE_CASES = (
    ('UPDATE', 'A0', 'DELAY'),
    ('DELETE', 'A0', 'DELAY'),
    ('DELETE', 'A4', 'DELAY'),
    ('REVOKE', 'A0', 'DELAY'),
    ('REVOKE', 'A4', 'DELAY'),
    ('REVOKE', 'A3', 'STALE_CACHE'),
    ('SUPERSEDE', 'A3', 'STALE_CACHE'),
)


def _scenario(transition, architecture, fault, index):
    return {'scenario_id': f'v2-preflight-{index}-{transition}-{architecture}-{fault}',
            'block': 'V2_PREFLIGHT', 'query_id': 'v3-q-engineering-01',
            'transition': transition, 'severity': 'S2',
            'architecture': architecture, 'fault': fault,
            'component': None, 'compound': None,
            'component_tuple': None, 'partial_fraction': None}


def _run_cases(config, built, cases, *, detail=True):
    query = built['queries']['v3-q-engineering-01']
    return {case: execute_track_a(_scenario(*case, index), built['world'],
                                  built['metadata'], query, config,
                                  audit_detail=detail)
            for index, case in enumerate(cases)}


def _row(rows, tick):
    return next(row for row in rows if row['logical_time'] == tick)


def _fingerprint(rows_by_case):
    fields = ('logical_time', 'candidate_chunk_ids', 'post_filter_chunk_ids',
              'context_chunk_ids', 'context_hash', 'scrs', 'srr', 'dep',
              'ucer_candidate', 'ucer_post_filter', 'ucer_context', 'ardr',
              'cache_hit', 'cache_type', 'cache_stage_violation_count',
              'canonical_forbidden_access_count', 'authoritative_access_events',
              'cache_stage_events', 't_rsc')
    payload = [(case, [tuple(row[field] for field in fields) for row in rows])
               for case, rows in rows_by_case.items()]
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _pytest_env():
    env = os.environ.copy()
    base = Path(sys.base_prefix) / 'lib' / f'python{sys.version_info.major}.{sys.version_info.minor}' / 'site-packages'
    env['PYTHONPATH'] = os.pathsep.join(part for part in
        (str(base), 'src', env.get('PYTHONPATH', '')) if part)
    env['HF_HUB_OFFLINE'] = '1'
    env['TRANSFORMERS_OFFLINE'] = '1'
    return env


def fingerprint_process(config_path):
    config = json.loads(Path(config_path).read_text())
    built = build(config)
    rows = _run_cases(config, built, SMOKE_CASES, detail=True)
    return _fingerprint(rows)


def _write(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')


def _boundary_spec(config):
    return f'''# Runtime boundary specification — {config['campaign_version']}

| Plane | Owner | Allowed runtime use |
|---|---|---|
| Canonical source/content | scheduler, oracle, evaluator | Transition setup and correctness comparison only. No retrieval, cache, indexed ACL, or A4/A5 lookup. |
| Canonical authorization | oracle, evaluator, A4/A5 live-auth view | Principal/tenant/group/direct-rule validation only. Its view has no document lifecycle API. |
| Replica chunks and parents | content replicas | Determine physically visible chunks, versions, and PA2 parent ranks. |
| Replica ACL metadata | indexed authorization view | A0–A3 filtering; document deletion alone leaves ACL policy unchanged. |
| C1 candidate cache | candidate retrieval boundary | Stores ordered hits and provenance; every hit re-enters indexed authorization and any live validation. |
| Context | post-filter/live validation | Holds only evidence that survived the architecture's declared boundaries. |

`authorized` is an ACL decision. `exists/current/deleted` is a content decision.
The evaluator may consider both, but A4/A5 live authorization cannot infer
content lifecycle through its capability. C2 context and answer caches are
absent from this repaired pilot. A3 validates C1 against locally visible
chunk, parent, and ACL epochs; no canonical epoch is consulted.

Runtime model: `{config['runtime_model_version']}`. Cache semantics:
`{config['cache_semantic_version']}`. Capability policy:
`{config['capability_policy_version']}`.
'''


def run(config_path, output):
    config_path, output = Path(config_path), Path(output)
    config = json.loads(config_path.read_text())
    if config['campaign_version'] != 'state-isolated-pilot-v2':
        raise ValueError('This preflight runs only campaign v2')
    if output.exists() and any(output.iterdir()):
        raise ValueError('Preflight output already exists; refusing overwrite')
    output.mkdir(parents=True)
    gates = {f'R{i}': {'status': 'NOT_RUN'} for i in range(6)}
    (output / 'runtime_boundary_spec.md').write_text(_boundary_spec(config))
    _write(output / 'cache_semantics.json', {
        'version': config['cache_semantic_version'],
        'C0': 'no cache',
        'C1': {'cache_type': 'candidate', 'created_stage': 'candidate_retrieval',
               'consumed_stage': 'candidate_retrieval',
               'replay_rule': 'always rerun indexed ACL and live authorization if enabled',
               'epoch_dependencies': ['replica_chunk', 'replica_parent', 'replica_acl'],
               'identity': ['principal_id', 'tenant_id'],
               'provenance': ['chunk_ids', 'versions', 'scores', 'rank_order',
                              'creation_time', 'producer_architecture', 'state_hashes']},
        'C2': 'not implemented in v2',
        'answer_cache': 'not implemented; no LLM generation',
    })
    def finish():
        _write(output / 'preflight_gate_summary.json', gates)
        status = 'PASS' if all(item['status'] == 'PASS' for item in gates.values()) else 'STOP'
        summary = {'status': status, 'campaign_version': config['campaign_version'],
                   'gates': gates, 'no_scientific_scenarios_executed': True,
                   'track_b_scenarios_executed': 0}
        (output / 'summary.md').write_text(
            f"# State-isolated pilot v2 preflight: {status}\n\n" +
            '\n'.join(f"- {name}: {value['status']} — {value.get('detail', '')}"
                      for name, value in gates.items()) + '\n\n' +
            'Only targeted boundary audits and the seven-case implementation smoke were run. '
            'No scientific scenario matrix or Track B run was executed.\n')
        return summary
    # R0: full tests in the CPU environment, including regression cases.
    proc = subprocess.run([sys.executable, '-m', 'pytest', '-q'],
                          env=_pytest_env(), text=True, capture_output=True,
                          check=False)
    match = re.search(r'(\d+) passed', proc.stdout)
    test_summary = {'returncode': proc.returncode,
                    'passed': int(match.group(1)) if match else None,
                    'stdout_tail': proc.stdout[-1600:],
                    'stderr_tail': proc.stderr[-800:],
                    'interpreter': sys.executable}
    _write(output / 'regression_test_summary.json', test_summary)
    gates['R0'] = {'status': 'PASS' if proc.returncode == 0 else 'FAIL',
                   'detail': f"{test_summary['passed']} tests passed"}
    if proc.returncode:
        return finish()
    built = build(config)
    # R1: every A0–A5 architecture audited on the same REVOKE boundary.
    r1_cases = [('REVOKE', architecture, 'DELAY')
                for architecture in config['architecture_components']]
    r1 = _run_cases(config, built, r1_cases)
    accesses = [event for rows in r1.values() for row in rows
                for event in row['authoritative_access_events']]
    forbidden = [event for event in accesses if not event['allowed']]
    live_content = [event for event in accesses
                    if event['component'] == 'live_authorization'
                    and event['accessed_plane'] != 'canonical_authorization']
    access_audit = {'architectures': list(config['architecture_components']),
                    'accesses': accesses, 'forbidden': forbidden,
                    'live_authorization_non_auth_accesses': live_content,
                    'access_count_by_component_plane': dict(Counter(
                        f"{event['component']}:{event['accessed_plane']}" for event in accesses))}
    _write(output / 'canonical_access_audit.json', access_audit)
    gates['R1'] = {'status': 'PASS' if not forbidden and not live_content else 'FAIL',
                   'detail': f'{len(accesses)} audited accesses, {len(forbidden)} forbidden'}
    if gates['R1']['status'] != 'PASS':
        return finish()
    # R2: cache creation/consumption must remain at candidate stage.
    r2_cases = [('REVOKE', arch, 'STALE_CACHE') for arch in ('A0', 'A3', 'A5')]
    r2_cases += [('DELETE', arch, 'STALE_CACHE') for arch in ('A0', 'A3', 'A5')]
    r2 = _run_cases(config, built, r2_cases)
    stage_events = [event for rows in r2.values() for row in rows
                    for event in row['cache_stage_events']]
    bad_stage = [event for event in stage_events if not event['valid']
                 or event['created_stage'] != 'candidate_retrieval'
                 or event['consumed_stage'] not in (None, 'candidate_retrieval')]
    _write(output / 'cache_stage_audit.json', {
        'events': stage_events, 'violations': bad_stage,
        'revoke_A3_after_index_convergence_ucer': _row(
            r2[('REVOKE', 'A3', 'STALE_CACHE')], 6)['ucer_context'],
    })
    a3_late = _row(r2[('REVOKE', 'A3', 'STALE_CACHE')], 6)
    gates['R2'] = {'status': 'PASS' if not bad_stage and
                   a3_late['cache_hit'] and a3_late['ucer_context'] == 0 else 'FAIL',
                   'detail': f'{len(stage_events)} cache events, {len(bad_stage)} stage violations'}
    if gates['R2']['status'] != 'PASS':
        return finish()
    # R3: metrics distinguish the five transition families.
    r3 = _run_cases(config, built,
                    [(transition, 'A0', 'DELAY') for transition in
                     config['transition_types']])
    samples = {transition: _row(rows, 1)
               for (transition, _, _), rows in r3.items()}
    checks = {
        'DELETE_DEP_without_UCER': samples['DELETE']['dep'] == 1
                                   and samples['DELETE']['ucer_context'] == 0,
        'REVOKE_UCER_without_DEP': samples['REVOKE']['ucer_context'] > 0
                                     and samples['REVOKE']['dep'] == 0,
        'SUPERSEDE_SRR_without_DEP': samples['SUPERSEDE']['srr'] > 0
                                       and samples['SUPERSEDE']['dep'] == 0,
        'GRANT_ARDR_without_UCER': samples['GRANT']['ardr'] == 1
                                    and samples['GRANT']['ucer_context'] == 0,
        'UPDATE_SCRS_without_auth_error': samples['UPDATE']['scrs'] == 0
                                          and samples['UPDATE']['ucer_context'] == 0,
    }
    _write(output / 'metric_orthogonality.json', {'checks': checks,
           'sample_query_id': 'v3-q-engineering-01',
           'sample_tick': 1,
           'metrics': {key: {field: row[field] for field in
                       ('scrs', 'srr', 'dep', 'ucer_context', 'ardr')}
                       for key, row in samples.items()}})
    gates['R3'] = {'status': 'PASS' if all(checks.values()) else 'FAIL',
                   'detail': f'{sum(checks.values())}/{len(checks)} metric invariants'}
    if gates['R3']['status'] != 'PASS':
        return finish()
    # R4: repeat the frozen Track-A clean resolver gate on v2 runtime metadata.
    clean_root = Path('results') / config['campaign_version']
    clean_summary_path = clean_root / 'ta0_clean_summary.json'
    clean = (json.loads(clean_summary_path.read_text()) if clean_summary_path.exists()
             else run_clean(config_path, clean_root))
    gates['R4'] = {'status': 'PASS' if clean['D1'] == 'PASS' else 'FAIL',
                   'detail': f"{clean['answerable_valid']}/{clean['answerable_total']} answerable; "
                             f"{clean['negative_valid']}/{clean['negative_total']} negative"}
    if gates['R4']['status'] != 'PASS':
        return finish()
    # Runtime/config/source hashes are frozen after R0–R4 and before R5 smoke.
    freeze_path = clean_root / 'pre_execution_freeze.json'
    freeze_info = ({'sha256': hashlib.sha256(freeze_path.read_bytes()).hexdigest()}
                   if freeze_path.exists() else freeze_execution(config_path))
    _verify_freeze(config_path)
    # R5: exactly seven declared smoke cases, detailed pipeline trace.
    smoke = _run_cases(config, built, SMOKE_CASES)
    with (output / 'smoke_traces.jsonl').open('w') as stream:
        for rows in smoke.values():
            for row in rows:
                stream.write(json.dumps(row, sort_keys=True) + '\n')
    delete_a4 = _row(smoke[('DELETE', 'A4', 'DELAY')], 1)
    revoke_a4 = _row(smoke[('REVOKE', 'A4', 'DELAY')], 1)
    revoke_a3 = _row(smoke[('REVOKE', 'A3', 'STALE_CACHE')], 6)
    update_a0 = _row(smoke[('UPDATE', 'A0', 'DELAY')], 1)
    supersede_a3 = _row(smoke[('SUPERSEDE', 'A3', 'STALE_CACHE')], 1)
    smoke_checks = {
        'A4_delete_keeps_deleted_content': delete_a4['dep'] == 1
            and delete_a4['ucer_context'] == 0 and bool(delete_a4['context_chunk_ids']),
        'A4_revoke_blocks_context': revoke_a4['ucer_candidate'] > 0
            and revoke_a4['ucer_context'] == 0,
        'A3_revoke_cache_filters_after_convergence': revoke_a3['cache_hit'] == 1
            and revoke_a3['ucer_context'] == 0 and not revoke_a3['context_chunk_ids'],
        'UPDATE_stale_content_visible': update_a0['scrs'] == 0
            and bool(update_a0['context_chunk_ids']),
        'SUPERSEDE_cache_stale_evidence': supersede_a3['srr'] > 0
            and supersede_a3['cache_hit'] == 1,
        'zero_forbidden_accesses': all(row['canonical_forbidden_access_count'] == 0
            for rows in smoke.values() for row in rows),
        'zero_cache_stage_violations': all(row['cache_stage_violation_count'] == 0
            for rows in smoke.values() for row in rows),
    }
    # Two independent subprocesses: no in-process second-pass shortcut.
    command = [sys.executable, '-m', 'ragstate.campaign.state_pilot_preflight',
               '--fingerprint', '--config', str(config_path)]
    child = [subprocess.run(command, env=_pytest_env(), capture_output=True,
                            text=True, check=True).stdout.strip() for _ in range(2)]
    expected = _fingerprint(smoke)
    smoke_checks['fresh_process_reproduction'] = child == [expected, expected]
    _write(output / 'fresh_process_audit.json', {
        'fingerprints': child, 'expected_fingerprint': expected,
        'identical': smoke_checks['fresh_process_reproduction'],
        'compared_fields': [
            'authoritative_access_events', 'cache_stage_events',
            'context_chunk_ids', 'context_hash', 'metrics', 'rankings'],
        'preflight_source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        'command': command})
    _write(output / 'smoke_validation.json', {'checks': smoke_checks,
           'scenario_count': len(SMOKE_CASES), 'observation_rows': sum(map(len, smoke.values())),
           'pre_execution_freeze_sha256': freeze_info['sha256']})
    gates['R5'] = {'status': 'PASS' if all(smoke_checks.values()) else 'FAIL',
                   'detail': f'{sum(smoke_checks.values())}/{len(smoke_checks)} smoke and fresh-process checks'}
    return finish()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/state-isolated-pilot-v2.json')
    parser.add_argument('--output', default='results/state-isolated-pilot-v2-preflight')
    parser.add_argument('--fingerprint', action='store_true')
    args = parser.parse_args()
    if args.fingerprint:
        print(fingerprint_process(args.config))
    else:
        print(json.dumps(run(args.config, args.output), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
