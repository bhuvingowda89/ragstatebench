"""Frozen Track-A v4 wiring audit and fail-fast scientific execution.

The scientific manifest is derived only from scenario-applicability-v2. The
runner never regenerates applicability or chooses partial replica units.
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
from pathlib import Path

from .scenario_applicability import _transition_facts
from .scenario_applicability_v2 import _partial_units
from .state_pilot_clean import clean_rows
from .state_pilot_execute import _atomic_json, consolidate
from .state_pilot_manifest import _load_frozen, digest
from .state_pilot_runtime_v4 import execute_track_a
from .state_pilot_v2_campaign import capture_scenario, _jsonl
from .state_pilot_v3_campaign import capture_replica_snapshots, validate_scenario as v3_validity


CAMPAIGN = 'state-isolated-pilot-v4'
ROOT = Path('results') / CAMPAIGN
PREFLIGHT = Path('results/state-isolated-pilot-v4-preflight')
CONFIG = Path('configs/state-isolated-pilot-v4.json')
APPLICABILITY = Path('results/scenario-applicability-v2/applicability_manifest.json')
H4_POLICY = Path('results/scenario-applicability-v2/h4_analysis_policy.json')
RUNTIME_POLICY = 'state_planes_topology_partial_v4'
VALIDITY_POLICY = 'selected_unit_agreement_v4'
SOURCE_FILES = (
    'src/ragstate/campaign/state_pilot_runtime_v4.py',
    'src/ragstate/campaign/topology_partial_runtime.py',
    'src/ragstate/campaign/state_pilot_v4_campaign.py',
    'src/ragstate/campaign/state_pilot_validity_v3.py',
    'src/ragstate/campaign/state_capabilities.py',
    'src/ragstate/replicas/scientific.py',
)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _load_source():
    source = json.loads(APPLICABILITY.read_text())
    if source['executable_manifest_sha256'] != 'c8df463c1d26b08e64afbc96596e42aa6a69be8649c84e8fa60eb036ec71fa8f':
        raise ValueError('Applicability-v2 executable hash differs from authorized source')
    if digest(source['executable_scenarios']) != source['executable_manifest_sha256']:
        raise ValueError('Applicability-v2 executable content hash mismatch')
    gates = json.loads(Path('results/scenario-applicability-v2/gate_summary.json').read_text())
    if any(gates.get('AP' + str(i)) != 'PASS' for i in range(1, 6)):
        raise ValueError('Applicability-v2 gates are not all PASS')
    return source


def _enrich_partial(scenario, facts):
    result = dict(scenario)
    plane, old, new, identities = _partial_units(facts, scenario['transition'])
    selected = set(scenario['selected_unit_ids'])
    if plane == 'content_chunk' and scenario['transition'] in ('UPDATE', 'SUPERSEDE'):
        if len(old) != len(new):
            raise ValueError('Compiler emitted unaligned version chunks')
        stale = [old[index].chunk_id for index, item in enumerate(new)
                 if item.chunk_id not in selected]
    else:
        stale = [identity for identity in identities if identity not in selected]
    if len(stale) != scenario['total_unit_count'] - scenario['updated_unit_count']:
        raise ValueError('Compiler emitted incomplete stale-unit plan')
    result['selected_stale_unit_ids'] = stale
    result['selected_updated_ids_hash'] = digest(list(scenario['selected_unit_ids']))
    result['selected_unit_plan_hash'] = digest({
        'plane': plane, 'unit_type': scenario['replica_unit_type'],
        'total': scenario['total_unit_count'],
        'updated': list(scenario['selected_unit_ids']), 'stale': stale,
        'topology': scenario['partial_topology'],
        'realized_fraction': scenario['realized_fraction'],
        'selection_seed': scenario['partial_selection_seed']})
    return result


def freeze():
    if ROOT.exists() or PREFLIGHT.exists() or CONFIG.exists():
        raise ValueError('v4 campaign or preflight already exists; refusing overwrite')
    source = _load_source()
    config = json.loads(Path('configs/state-isolated-pilot-v2.json').read_text())
    config.update(campaign_version=CAMPAIGN,
                  runtime_model_version=RUNTIME_POLICY,
                  validity_policy_version=VALIDITY_POLICY,
                  applicability_version='scenario-applicability-v2')
    world, metadata, queries, _ = _load_frozen(config)
    query_by_id = {query.query_id: query for query in queries}
    facts = {}
    scenarios = []
    for item in source['executable_scenarios']:
        scenario = dict(item)
        scenario.update(campaign_version=CAMPAIGN,
                        runtime_model_version=RUNTIME_POLICY,
                        validity_policy_version=VALIDITY_POLICY,
                        source_applicability_scenario_sha256=digest(item))
        if scenario.get('partial_topology'):
            key = scenario['query_id'], scenario['transition']
            if key not in facts:
                facts[key] = _transition_facts(world, metadata,
                    query_by_id[scenario['query_id']], scenario['transition'])
            scenario = _enrich_partial(scenario, facts[key])
        scenarios.append(scenario)
    if len(scenarios) != 4242 or len({s['scenario_id'] for s in scenarios}) != len(scenarios):
        raise ValueError('v4 manifest differs from frozen applicability cardinality')
    CONFIG.write_text(json.dumps(config, indent=2, sort_keys=True) + '\n')
    ROOT.mkdir(parents=True)
    PREFLIGHT.mkdir(parents=True)
    plan_hash = digest([{
        'scenario_id': s['scenario_id'],
        'plan_hash': s['selected_unit_plan_hash']}
        for s in scenarios if s.get('partial_topology')])
    manifest = {'campaign_id': CAMPAIGN, 'scenarios': scenarios,
                'scientific_manifest_sha256': digest(scenarios),
                'source_applicability_manifest_sha256': source['executable_manifest_sha256'],
                'selected_unit_plan_sha256': plan_hash,
                'runtime_policy_version': RUNTIME_POLICY,
                'validity_policy_version': VALIDITY_POLICY,
                'capability_policy_version': config['capability_policy_version'],
                'cache_semantics_version': config['cache_semantic_version'],
                'h4_analysis_policy_sha256': _sha(H4_POLICY)}
    _atomic_json(PREFLIGHT / 'scientific_manifest.json', manifest)
    _atomic_json(ROOT / 'manifest.json', {key: value for key, value in manifest.items()
                                         if key != 'scenarios'})
    provenance = {'campaign_id': CAMPAIGN, 'config_sha256': _sha(CONFIG),
                  'source_applicability_file_sha256': _sha(APPLICABILITY),
                  'source_applicability_manifest_sha256': source['executable_manifest_sha256'],
                  'scientific_manifest_sha256': manifest['scientific_manifest_sha256'],
                  'selected_unit_plan_sha256': plan_hash,
                  'corpus_sha256': config['corpus_sha256'],
                  'track_b_cohort_sha256': config['track_b_cohort_sha256'],
                  'h4_analysis_policy_sha256': manifest['h4_analysis_policy_sha256'],
                  'source_sha256': {name: _sha(name) for name in SOURCE_FILES},
                  'runtime_policy_version': RUNTIME_POLICY,
                  'capability_policy_version': config['capability_policy_version'],
                  'cache_semantics_version': config['cache_semantic_version'],
                  'validity_policy_version': VALIDITY_POLICY,
                  'no_v2_v3_observations_reused': True}
    _atomic_json(PREFLIGHT / 'provenance.json', provenance)
    return {'scenario_count': len(scenarios),
            'manifest_sha256': manifest['scientific_manifest_sha256'],
            'selected_unit_plan_sha256': plan_hash}


def verify_freeze():
    source = _load_source()
    manifest = json.loads((PREFLIGHT / 'scientific_manifest.json').read_text())
    provenance = json.loads((PREFLIGHT / 'provenance.json').read_text())
    config = json.loads(CONFIG.read_text())
    if digest(manifest['scenarios']) != manifest['scientific_manifest_sha256'] or \
       manifest['scientific_manifest_sha256'] != provenance['scientific_manifest_sha256']:
        raise ValueError('Scientific manifest changed after freeze')
    if manifest['source_applicability_manifest_sha256'] != source['executable_manifest_sha256']:
        raise ValueError('Scientific manifest is not sourced from applicability-v2')
    if _sha(CONFIG) != provenance['config_sha256'] or _sha(APPLICABILITY) != provenance['source_applicability_file_sha256']:
        raise ValueError('Frozen config or applicability source changed')
    if _sha(H4_POLICY) != provenance['h4_analysis_policy_sha256']:
        raise ValueError('Frozen H4 policy changed')
    if any(_sha(name) != expected for name, expected in provenance['source_sha256'].items()):
        raise ValueError('v4 executable source changed after freeze')
    source_by_id = {s['scenario_id']: s for s in source['executable_scenarios']}
    if set(source_by_id) != {s['scenario_id'] for s in manifest['scenarios']}:
        raise ValueError('Scientific scenario set differs from applicability-v2')
    for scenario in manifest['scenarios']:
        original = source_by_id[scenario['scenario_id']]
        if scenario['source_applicability_scenario_sha256'] != digest(original):
            raise ValueError('Scientific scenario changed beyond permitted identity/plan enrichment')
        for key, value in original.items():
            if key not in ('campaign_version', 'runtime_model_version', 'validity_policy_version') \
                    and scenario[key] != value:
                raise ValueError('Scientific scenario semantics changed: ' + key)
    plans = [{'scenario_id': s['scenario_id'], 'plan_hash': s['selected_unit_plan_hash']}
             for s in manifest['scenarios'] if s.get('partial_topology')]
    if digest(plans) != manifest['selected_unit_plan_sha256']:
        raise ValueError('Selected-unit plan hash mismatch')
    return config, manifest, provenance


def execute_recorded(scenario, world, metadata, query, config):
    with capture_replica_snapshots() as snapshots:
        with capture_scenario(scenario['scenario_id']) as (accesses, cache_ops, hashes):
            rows = execute_track_a(scenario, world, metadata, query, config)
    for row in rows:
        row['chunk_epoch'] = hashes['chunk'].get(row['chunk_index_hash'])
        row['parent_epoch'] = hashes['parent'].get(row['parent_index_hash'])
        row['auth_epoch'] = hashes['auth'].get(row['auth_metadata_hash'])
        if None in (row['chunk_epoch'], row['parent_epoch'], row['auth_epoch']):
            raise ValueError('Missing local epoch for state hash')
    return rows, accesses, cache_ops, snapshots


def validate_scenario(scenario, rows, accesses, cache_ops, snapshots,
                      query, metadata, world, config):
    validity = v3_validity(scenario, rows, accesses, cache_ops, snapshots,
                           query, metadata, world, config)
    reasons = [reason for reason in validity['reasons'].split(';') if reason]
    if scenario.get('partial_topology'):
        reasons = [reason for reason in reasons if not reason.startswith('component_not_effective:PARTIAL')]
        delivery_events = [event for row in rows for event in row['event_order']
                           if event['stage'] == 'replica_delivery' and event['partial']]
        delivery_ticks = sorted(set(event['tick'] for event in delivery_events))
        if len(delivery_ticks) != 1:
            reasons.append('compiler_runtime_mismatch:partial_delivery_count')
        else:
            tick = delivery_ticks[0]
            row = next(item for item in rows if item['logical_time'] == tick)
            actual = row['partial_event_trace']
            if actual is None:
                reasons.append('compiler_runtime_mismatch:missing_partial_trace')
            else:
                checks = {
                    'updated_ids': actual['actual_updated_unit_ids'] == scenario['selected_unit_ids'],
                    'stale_ids': actual['actual_stale_unit_ids'] == scenario['selected_stale_unit_ids'],
                    'unit_count': actual['updated_unit_count'] + actual['stale_unit_count'] == scenario['total_unit_count'],
                    'updated_hash': actual['selected_updated_ids_hash'] == scenario['selected_updated_ids_hash'],
                    'chunk_hash': actual['chunk_index_hash'] == row['chunk_index_hash'],
                    'parent_hash': actual['parent_index_hash'] == row['parent_index_hash'],
                    'auth_hash': actual['auth_metadata_hash'] == row['auth_metadata_hash'],
                    'target_plane': actual['target_plane'] == scenario['replica_unit_type'],
                    'fraction': row['partial_realized_fraction'] == scenario['realized_fraction'],
                }
                if not all(checks.values()):
                    reasons.append('compiler_runtime_mismatch:' + ','.join(
                        key for key, passed in checks.items() if not passed))
                validity['component_evidence'] = [{
                    'component': 'PARTIAL', 'target_plane': actual['target_plane'],
                    'predicate': 'exact_selected_unit_plan',
                    'effective': all(checks.values()), 'evidence': checks}]
    validity['validity'] = 'invalid' if reasons else 'valid'
    validity['reasons'] = ';'.join(reasons)
    validity['compiler_runtime_agreement'] = not any(
        reason.startswith('compiler_runtime_mismatch') for reason in reasons)
    return validity


def deterministic_fingerprint(rows, accesses, cache_ops, validity):
    stable_rows = [{key: value for key, value in row.items()
                    if key not in ('retrieval_latency_ms', 'parent_latency_ms')}
                   for row in rows]
    return digest({'rows': stable_rows, 'accesses': accesses,
                   'cache_ops': cache_ops, 'validity': validity})


def _sample_scenarios(scenarios):
    """Frozen structural sample, chosen without faulted outcomes."""
    def select(label, predicate):
        found = sorted((s for s in scenarios if predicate(s)),
                       key=lambda s: s['scenario_id'])
        if not found:
            raise ValueError('v4 wiring sample lacks stratum: ' + label)
        return {'stratum': label, 'scenario_id': found[0]['scenario_id']}
    return [
        select('N2_LOW_UPDATE_engineering', lambda s: s.get('partial_topology') == 'P_LOW'
               and s['total_unit_count'] == 2 and s['transition'] == 'UPDATE'
               and 'engineering' in s['query_id']),
        select('N3_LOW_SUPERSEDE_finance', lambda s: s.get('partial_topology') == 'P_LOW'
               and s['total_unit_count'] == 3 and s['transition'] == 'SUPERSEDE'
               and 'finance' in s['query_id']),
        select('N3_HIGH_DELETE_hr', lambda s: s.get('partial_topology') == 'P_HIGH'
               and s['total_unit_count'] == 3 and s['transition'] == 'DELETE'
               and 'hr' in s['query_id']),
        select('N4_LOW_REVOKE_legal', lambda s: s.get('partial_topology') == 'P_LOW'
               and s['total_unit_count'] >= 4 and s['transition'] == 'REVOKE'
               and 'legal' in s['query_id']),
        select('N4_MID_UPDATE_sales', lambda s: s.get('partial_topology') == 'P_MID'
               and s['total_unit_count'] >= 4 and s['transition'] == 'UPDATE'
               and 'sales' in s['query_id']),
        select('N4_HIGH_SUPERSEDE_security', lambda s: s.get('partial_topology') == 'P_HIGH'
               and s['total_unit_count'] >= 4 and s['transition'] == 'SUPERSEDE'
               and 'security' in s['query_id']),
        select('candidate_cache', lambda s: s['block'] == 'TA4' and s['architecture'] == 'A5'
               and s['transition'] == 'DELETE'),
        select('out_of_order', lambda s: s['fault'] == 'OUT_OF_ORDER'
               and s['transition'] == 'UPDATE'),
        select('CF3', lambda s: s.get('compound') == 'CF3'
               and s['component'] == 'F1_F2' and s['architecture'] == 'A0'),
        select('CF6', lambda s: s.get('compound') == 'CF6'
               and s['component'] == 'F1_F2' and s['architecture'] == 'A0'),
        select('live_auth_A4', lambda s: s['block'] == 'TA3'
               and s['transition'] == 'REVOKE' and s['architecture'] == 'A4'
               and s['severity'] == 'S2'),
        select('combined_A5', lambda s: s['block'] == 'TA3'
               and s['transition'] == 'DELETE' and s['architecture'] == 'A5'
               and s['severity'] == 'S2'),
    ]


def _write_csv(path, records):
    path = Path(path)
    with path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        for record in records:
            writer.writerow({key: json.dumps(value, sort_keys=True)
                             if isinstance(value, (dict, list)) else value
                             for key, value in record.items()})


def execute_wiring_sample():
    config, manifest, _ = verify_freeze()
    world, metadata, queries, _ = _load_frozen(config)
    query_by_id = {query.query_id: query for query in queries}
    scenario_by_id = {s['scenario_id']: s for s in manifest['scenarios']}
    selected = _sample_scenarios(manifest['scenarios'])
    output = []
    for chosen in selected:
        scenario = scenario_by_id[chosen['scenario_id']]
        query = query_by_id[scenario['query_id']]
        rows, accesses, cache_ops, snapshots = execute_recorded(
            scenario, world, metadata, query, config)
        validity = validate_scenario(scenario, rows, accesses, cache_ops,
                                     snapshots, query, metadata, world, config)
        partial = next((row['partial_event_trace'] for row in rows
                        if row['partial_event_trace'] is not None), None)
        output.append({'stratum': chosen['stratum'],
                       'scenario_id': scenario['scenario_id'],
                       'transition': scenario['transition'],
                       'severity': scenario['severity'],
                       'architecture': scenario['architecture'],
                       'query_id': scenario['query_id'],
                       'planned_updated_unit_ids': scenario.get('selected_unit_ids'),
                       'planned_stale_unit_ids': scenario.get('selected_stale_unit_ids'),
                       'actual_updated_unit_ids': partial['actual_updated_unit_ids'] if partial else None,
                       'actual_stale_unit_ids': partial['actual_stale_unit_ids'] if partial else None,
                       'partial_event_trace': partial,
                       'validity': validity,
                       'forbidden_access_count': sum(not event['allowed'] for event in accesses),
                       'canonical_access_count': len(accesses),
                       'cache_stage_violation_count': sum(not stage['valid']
                           for op in cache_ops for stage in op['stage_events']),
                       'cache_operation_count': len(cache_ops),
                       'fingerprint': deterministic_fingerprint(rows, accesses, cache_ops, validity)})
    return output


def preflight():
    config, manifest, provenance = verify_freeze()
    if (PREFLIGHT / 'wiring_gate_summary.json').exists():
        raise ValueError('v4 wiring preflight already exists; refusing overwrite')
    source = _load_source()
    w1 = (manifest['source_applicability_manifest_sha256'] ==
          source['executable_manifest_sha256'] and
          len(manifest['scenarios']) == len(source['executable_scenarios']))
    plans = [s for s in manifest['scenarios'] if s.get('partial_topology')]
    plan_rows = [{
        'scenario_id': s['scenario_id'], 'query_id': s['query_id'],
        'transition': s['transition'], 'topology': s['partial_topology'],
        'target_plane': s['replica_unit_type'], 'total_units': s['total_unit_count'],
        'updated_units': s['updated_unit_count'],
        'stale_units': s['total_unit_count'] - s['updated_unit_count'],
        'selected_updated_unit_ids': s['selected_unit_ids'],
        'selected_stale_unit_ids': s['selected_stale_unit_ids'],
        'realized_fraction': s['realized_fraction'],
        'selection_seed': s['partial_selection_seed'],
        'plan_hash': s['selected_unit_plan_hash']}
        for s in plans]
    _write_csv(PREFLIGHT / 'selected_unit_audit.csv', plan_rows)
    local = execute_wiring_sample()
    agreement_rows = [{key: value for key, value in item.items()
                       if key not in ('validity', 'partial_event_trace')}
                      | {'validity': item['validity']['validity'],
                         'validity_reason': item['validity']['reasons']}
                      for item in local]
    _write_csv(PREFLIGHT / 'runtime_plan_agreement.csv', agreement_rows)
    partial_cases = [item for item in local if item['planned_updated_unit_ids'] is not None]
    w2 = all(item['planned_updated_unit_ids'] == item['actual_updated_unit_ids']
             and item['planned_stale_unit_ids'] == item['actual_stale_unit_ids']
             for item in partial_cases) and len(partial_cases) >= 6
    w3 = all(item['validity']['validity'] == 'valid' and
             item['validity']['compiler_runtime_agreement'] for item in local)
    forbidden = sum(item['forbidden_access_count'] for item in local)
    stage_violations = sum(item['cache_stage_violation_count'] for item in local)
    w4, w5 = forbidden == 0, stage_violations == 0
    _atomic_json(PREFLIGHT / 'canonical_access_audit.json', {
        'access_events': sum(item['canonical_access_count'] for item in local),
        'forbidden': forbidden, 'scenario_count': len(local)})
    _atomic_json(PREFLIGHT / 'cache_stage_audit.json', {
        'cache_operations': sum(item['cache_operation_count'] for item in local),
        'violations': stage_violations, 'scenario_count': len(local)})
    _atomic_json(PREFLIGHT / 'validity_agreement.json', {
        'scenario_count': len(local), 'valid': sum(item['validity']['validity'] == 'valid'
                                                for item in local),
        'compiler_runtime_mismatches': [item['scenario_id'] for item in local
            if not item['validity']['compiler_runtime_agreement']],
        'invalid': [{'scenario_id': item['scenario_id'],
                     'reason': item['validity']['reasons']} for item in local
                    if item['validity']['validity'] != 'valid']})
    env = dict(os.environ)
    env['PYTHONPATH'] = str(Path('src').resolve()) + os.pathsep + env.get('PYTHONPATH', '')
    commands = [[sys.executable, '-m',
                 'ragstate.campaign.state_pilot_v4_campaign', 'sample'] for _ in range(2)]
    fresh = [subprocess.run(command, capture_output=True, text=True, env=env)
             for command in commands]
    repeats = [json.loads(run.stdout) if run.returncode == 0 else None for run in fresh]
    normalized_local = json.loads(json.dumps(local, sort_keys=True))
    w6 = repeats[0] is not None and repeats[0] == repeats[1] == normalized_local
    _atomic_json(PREFLIGHT / 'fresh_process_reproduction.json', {
        'identical': w6, 'process_exit_codes': [run.returncode for run in fresh],
        'sample_scenario_ids': [item['scenario_id'] for item in local],
        'fingerprints': [[item['fingerprint'] for item in repeat]
                         if repeat else None for repeat in repeats],
        'stderr': [run.stderr for run in fresh]})
    gates = {f'W{i}': 'PASS' if result else 'FAIL' for i, result in
             enumerate((w1, w2, w3, w4, w5, w6), 1)}
    gates['sample_count'] = len(local)
    gates['partial_sample_count'] = len(partial_cases)
    gates['scientific_manifest_sha256'] = manifest['scientific_manifest_sha256']
    _atomic_json(PREFLIGHT / 'wiring_gate_summary.json', gates)
    (PREFLIGHT / 'summary.md').write_text(
        '# Track-A v4 wiring preflight\n\n' +
        '\n'.join(f'- W{i}: {gates[f"W{i}"]}' for i in range(1, 7)) +
        f'\n\n{len(local)} full-trajectory wiring cases, including {len(partial_cases)} '
        'selected-unit PARTIAL cases. Scientific campaign observations have not been collected.\n')
    return gates


def _read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def run():
    config, manifest, provenance = verify_freeze()
    gates = json.loads((PREFLIGHT / 'wiring_gate_summary.json').read_text())
    if any(gates.get(f'W{i}') != 'PASS' for i in range(1, 7)):
        raise ValueError('Track-A v4 wiring preflight has a failed hard gate')
    if (ROOT / 'campaign_invalid.json').exists():
        raise ValueError('Track-A v4 was marked INVALID; refusing to resume')
    world, metadata, queries, _ = _load_frozen(config)
    query_by_id = {q.query_id: q for q in queries}
    clean = clean_rows(config)
    answerable = [row for row in clean if row['answerable']]
    negative = [row for row in clean if not row['answerable']]
    ta0 = {'answerable_valid': sum(row['valid'] for row in answerable),
           'answerable_total': len(answerable),
           'negative_valid': sum(row['valid'] for row in negative),
           'negative_total': len(negative)}
    ta0['D1'] = ('PASS' if ta0['answerable_valid'] / len(answerable) >=
                 config['pilot_gates']['D1']['state_isolated_clean_answerable_min']
                 and ta0['negative_valid'] == len(negative)
                 and not any(row['ucer_context'] or row['dep'] or row['srr'] or
                             row['cter'] or row['mvcr'] for row in clean) else 'FAIL')
    _atomic_json(ROOT / 'ta0_clean_summary.json', ta0)
    if ta0['D1'] != 'PASS':
        _atomic_json(ROOT / 'campaign_invalid.json', {
            'status': 'INVALID', 'reason': 'D1 clean invariance failed'})
        raise ValueError('D1 clean invariance failed')
    scenarios = manifest['scenarios']
    for folder in ('scenario_rows', 'canonical_access_rows', 'cache_stage_rows',
                   'validity_rows'):
        (ROOT / folder).mkdir(exist_ok=True)
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
            rows, accesses, cache_ops, snapshots = execute_recorded(
                scenario, world, metadata, query, config)
            validity = validate_scenario(scenario, rows, accesses, cache_ops,
                                         snapshots, query, metadata, world, config)
            for folder, records in (('scenario_rows', rows),
                                    ('canonical_access_rows', accesses),
                                    ('cache_stage_rows', cache_ops)):
                _jsonl(ROOT / folder / (sid + '.jsonl'), records)
            _atomic_json(ROOT / 'validity_rows' / (sid + '.json'), validity)
            if validity['validity'] != 'valid':
                category = ('compiler_runtime_mismatch' if not validity['compiler_runtime_agreement']
                            else 'scenario_validity_failure')
                raise ValueError(category + ': ' + validity['reasons'])
            status[sid] = {'status': 'censored' if validity['censored'] else 'valid',
                           'reason': None}
        except Exception as exc:
            reason = f'{type(exc).__name__}: {exc}'
            status[sid] = {'status': 'invalid', 'reason': reason}
            _atomic_json(status_path, status)
            _atomic_json(ROOT / 'campaign_invalid.json', {
                'status': 'INVALID', 'scenario_id': sid,
                'reason': reason, 'hard_stop': True})
            raise
        _atomic_json(status_path, status)
        if (index + 1) % 100 == 0:
            print(f'Track A v4: {index + 1}/{len(scenarios)} scenarios', flush=True)
    count = consolidate(ROOT, scenarios)
    validity = [json.loads((ROOT / 'validity_rows' /
                (scenario['scenario_id'] + '.json')).read_text())
                for scenario in scenarios]
    _write_csv(ROOT / 'scenario_status.csv', [
        {'scenario_id': sid, **state} for sid, state in status.items()])
    _write_csv(ROOT / 'scenario_validity.csv', [
        {key: value for key, value in row.items() if key != 'component_evidence'}
        for row in validity])
    access_total = forbidden_total = cache_total = stage_total = 0
    for scenario in scenarios:
        row = next(v for v in validity if v['scenario_id'] == scenario['scenario_id'])
        access_total += row['access_count']
        forbidden_total += row['forbidden_access_count']
        cache_total += row['cache_operation_count']
        stage_total += row['cache_stage_violation_count']
    _atomic_json(ROOT / 'canonical_access_audit.json', {
        'access_events': access_total, 'forbidden': forbidden_total})
    _atomic_json(ROOT / 'cache_stage_audit.json', {
        'cache_operations': cache_total, 'violations': stage_total})
    _atomic_json(ROOT / 'compiler_runtime_agreement.json', {
        'scenario_count': len(validity),
        'agreements': sum(v['compiler_runtime_agreement'] for v in validity),
        'mismatches': [v['scenario_id'] for v in validity
                       if not v['compiler_runtime_agreement']]})
    summary = {'campaign_id': CAMPAIGN, 'scenario_count': len(scenarios),
               'observation_rows': count,
               'status_counts': dict(Counter(state['status'] for state in status.values())),
               'runtime_seconds': time.perf_counter() - started,
               'scientific_manifest_sha256': manifest['scientific_manifest_sha256'],
               'ta0': ta0}
    _atomic_json(ROOT / 'execution_summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'preflight', 'sample', 'run'))
    args = parser.parse_args()
    if args.action == 'freeze':
        result = freeze()
    elif args.action == 'preflight':
        result = preflight()
    elif args.action == 'sample':
        result = execute_wiring_sample()
    else:
        result = run()
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
