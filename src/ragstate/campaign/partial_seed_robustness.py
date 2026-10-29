"""Prospectively frozen, relevance-blind H4 selected-unit sensitivity study."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .h4_partial_campaign import _audit_arm, _audit_pair, _balance, _execute
from .h4_partial_analysis import _pair_measure
from .state_pilot_manifest import _load_frozen, digest

ROOT = Path('results/partial-seed-robustness-v1')
SOURCE = Path('results/h4-partial-control-pilot-v1/manifest.json')
SEEDS = {'seed_1': 1171, 'seed_2': 2377, 'seed_3': 3559, 'seed_4': 4793}
TRANSITIONS = ('UPDATE', 'SUPERSEDE')


def write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def write_csv(path, rows):
    rows = list(rows)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (list, dict)) else value
                             for key, value in row.items()})


def select_units(unit_ids, k, seed):
    """Only the seed and stable unit IDs influence ranking."""
    ids = tuple(unit_ids)
    if len(ids) != len(set(ids)) or not 1 <= k < len(ids):
        raise ValueError('Invalid strict subset topology')
    ranked = sorted(ids, key=lambda unit: (hashlib.sha256(f'{seed}|{unit}'.encode()).hexdigest(), unit))
    selected = set(ranked[:k])
    return [unit for unit in ids if unit in selected]


def source_pairs():
    source = json.loads(SOURCE.read_text())
    pairs = [source['scenarios'][i:i+2] for i in range(0, len(source['scenarios']), 2)]
    pairs = [pair for pair in pairs if pair[1]['transition'] in TRANSITIONS
             and pair[1]['architecture'] == 'A0' and pair[1]['severity'] == 'S2']
    if len(pairs) != 402 or any(not _balance(pair)['balanced'] for pair in pairs):
        raise ValueError('Validated H4 source pair population changed')
    return source, pairs


def unit_universe(partial):
    chosen = partial['selected_unit_ids']
    stale = partial['selected_stale_unit_ids']
    new_prefix = chosen[0].rsplit(':c', 1)[0]
    old_prefix = stale[0].rsplit(':c', 1)[0]
    n = partial['total_unit_count']
    return ([f'{new_prefix}:c{i}' for i in range(n)],
            [f'{old_prefix}:c{i}' for i in range(n)])


def build_pair(source_pair, seed_label, seed):
    original = source_pair[1]
    new_ids, old_ids = unit_universe(original)
    selected = select_units(new_ids, original['updated_unit_count'], seed)
    selected_positions = {int(unit.rsplit(':c', 1)[1]) for unit in selected}
    stale = [old_ids[i] for i in range(len(old_ids)) if i not in selected_positions]
    pair_id = f"PS__{seed_label}__{original['h4_pair_id']}"
    result = []
    for arm in ('U', 'P'):
        scenario = dict(source_pair[0 if arm == 'U' else 1])
        scenario.update(scenario_id=f'{pair_id}__{arm}', h4_pair_id=pair_id,
                        campaign_version='partial-seed-robustness-v1', block='PS',
                        selected_unit_ids=selected, selected_stale_unit_ids=stale,
                        selected_updated_ids_hash=digest(selected),
                        selected_unit_plan_hash=digest({'selected': selected, 'stale': stale}),
                        partial_selection_seed=seed,
                        h4_pre_unit_ids=sorted(set(selected) | set(stale)),
                        h4_source_plan_hash=original['h4_source_plan_hash'],
                        ps_seed=seed_label, ps_source_pair_id=original['h4_pair_id'])
        result.append(scenario)
    return result


def freeze():
    if ROOT.exists():
        raise ValueError('Supplement already exists')
    source, pairs = source_pairs()
    ROOT.mkdir(parents=True)
    write_json(ROOT/'seed_policy.json', {'seed_0': 'historical frozen H4 plan, not rerun',
                                       'additional_seeds': SEEDS,
                                       'selector': 'sha256(seed|stable_unit_id); first k ranks; source order',
                                       'outcome_blind': True})
    all_pairs = [build_pair(pair, label, seed) for label, seed in SEEDS.items() for pair in pairs]
    scenarios = [scenario for pair in all_pairs for scenario in pair]
    plan = []
    for pair in all_pairs:
        s = pair[1]
        plan.append({'pair_id': s['h4_pair_id'], 'source_pair_id': s['ps_source_pair_id'],
                     'seed': s['ps_seed'], 'query_id': s['query_id'], 'transition': s['transition'],
                     'domain': s['query_id'].split('-')[2], 'topology': s['partial_topology'],
                     'N': s['total_unit_count'], 'k': s['updated_unit_count'],
                     'selected_unit_ids': s['selected_unit_ids'], 'stale_unit_ids': s['selected_stale_unit_ids'],
                     'subset_hash': digest(s['selected_unit_ids']),
                     'realized_fraction': s['realized_fraction']})
    write_csv(ROOT/'pair_seed_plan.csv', plan)
    manifest = {'campaign_id': 'partial-seed-robustness-v1', 'source_h4_manifest_sha256':
                hashlib.sha256(SOURCE.read_bytes()).hexdigest(), 'source_pair_count': len(pairs),
                'source_pair_ids': [pair[1]['h4_pair_id'] for pair in pairs],
                'seed_policy_sha256': hashlib.sha256((ROOT/'seed_policy.json').read_bytes()).hexdigest(),
                'scenarios': scenarios, 'scenario_sha256': digest(scenarios),
                'config': dict(source['config'], campaign_version='partial-seed-robustness-v1'),
                'schedule': {'t_partial': 2, 'T': 5, 'mixed_window': [2, 3, 4]}}
    write_json(ROOT/'manifest.json', manifest)
    return {'pairs': len(all_pairs), 'scenarios': len(scenarios), 'manifest_sha256': hashlib.sha256((ROOT/'manifest.json').read_bytes()).hexdigest()}


def verify():
    manifest = json.loads((ROOT/'manifest.json').read_text())
    source, source_set = source_pairs()
    if manifest['source_h4_manifest_sha256'] != hashlib.sha256(SOURCE.read_bytes()).hexdigest() or \
       manifest['seed_policy_sha256'] != hashlib.sha256((ROOT/'seed_policy.json').read_bytes()).hexdigest() or \
       manifest['source_pair_ids'] != [pair[1]['h4_pair_id'] for pair in source_set] or \
       manifest['scenario_sha256'] != digest(manifest['scenarios']):
        raise ValueError('Frozen provenance mismatch')
    expected = [s for label, seed in SEEDS.items() for pair in source_set for s in build_pair(pair, label, seed)]
    if expected != manifest['scenarios']:
        raise ValueError('Prospective seed plans changed')
    return manifest


def stratified_pairs(manifest):
    pairs = [manifest['scenarios'][i:i+2] for i in range(0, len(manifest['scenarios']), 2)]
    result = []
    for label in SEEDS:
        for transition in TRANSITIONS:
            for n in (2, 3, 4):
                match = next((pair for pair in pairs if pair[1]['ps_seed'] == label and
                              pair[1]['transition'] == transition and pair[1]['total_unit_count'] == n), None)
                if match is not None and match not in result:
                    result.append(match)
    return result


def sample():
    manifest = verify()
    world, metadata, queries, _ = _load_frozen(manifest['config'])
    by_query = {query.query_id: query for query in queries}
    result = []
    for pair in stratified_pairs(manifest):
        for scenario in pair:
            rows, accesses, cache_ops, _ = _execute(scenario, world, metadata,
                                                    by_query[scenario['query_id']], manifest['config'])
            audit = _audit_arm(scenario, rows, accesses, cache_ops)
            stable = [{key: value for key, value in row.items()
                       if key not in ('retrieval_latency_ms', 'parent_latency_ms')} for row in rows]
            result.append({'scenario_id': scenario['scenario_id'], 'hash': digest(stable), 'audit': audit})
    return result


def preflight():
    manifest = verify()
    if (ROOT/'observations.jsonl').exists():
        raise ValueError('Science already exists')
    pairs = [manifest['scenarios'][i:i+2] for i in range(0, len(manifest['scenarios']), 2)]
    source, source_set = source_pairs()
    source_by_id = {pair[1]['h4_pair_id']: pair[1] for pair in source_set}
    p1 = len(pairs) == 4*len(source_set) and all(pair[1]['ps_source_pair_id'] in source_by_id for pair in pairs)
    p2 = all(pair[1]['total_unit_count'] == source_by_id[pair[1]['ps_source_pair_id']]['total_unit_count'] and
             pair[1]['updated_unit_count'] == source_by_id[pair[1]['ps_source_pair_id']]['updated_unit_count'] and
             pair[1]['partial_topology'] == source_by_id[pair[1]['ps_source_pair_id']]['partial_topology']
             for pair in pairs)
    p3 = all(select_units(unit_universe(pair[1])[0], pair[1]['updated_unit_count'],
                          SEEDS[pair[1]['ps_seed']]) == pair[1]['selected_unit_ids'] for pair in pairs)
    p4 = all(_balance(pair)['balanced'] for pair in pairs)
    p5 = all(pair[0]['h4_final_tick'] == pair[1]['h4_final_tick'] == 5 and
             pair[0]['h4_partial_tick'] == pair[1]['h4_partial_tick'] == 2 for pair in pairs)
    local = sample()
    env = dict(os.environ, PYTHONPATH=str(Path('src').resolve()) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    fresh = subprocess.run([sys.executable, '-m', 'ragstate.campaign.partial_seed_robustness', 'sample'],
                           capture_output=True, text=True, env=env)
    reproduced = fresh.returncode == 0 and json.loads(fresh.stdout) == json.loads(json.dumps(local))
    p6 = all(item['audit']['valid'] for item in local)
    p7 = all(item['audit']['forbidden_accesses'] == item['audit']['cache_stage_violations'] == 0 for item in local)
    gates = dict(zip((f'PS-P{i}' for i in range(1, 9)),
                     (p1, p2, p3, p4, p5, p6, p7, reproduced)))
    write_json(ROOT/'preflight_gate_summary.json', {'gates': gates, 'sample_count': len(local),
                                                   'fresh_exit': fresh.returncode, 'fresh_stderr': fresh.stderr})
    write_json(ROOT/'reproduction.json', {'preflight_identical': reproduced, 'sample_count': len(local),
                                         'fresh_exit': fresh.returncode})
    if not all(gates.values()):
        write_json(ROOT/'invalid.json', {'status': 'INVALID', 'reason': 'preflight'})
    return gates


def run():
    manifest = verify()
    if not all(json.loads((ROOT/'preflight_gate_summary.json').read_text())['gates'].values()) or \
       (ROOT/'invalid.json').exists() or (ROOT/'observations.jsonl').exists():
        raise ValueError('Preflight not passed or science already exists')
    world, metadata, queries, _ = _load_frozen(manifest['config'])
    by_query = {query.query_id: query for query in queries}
    scenarios = manifest['scenarios']
    audits = []
    count = 0
    with (ROOT/'observations.jsonl').open('w') as jsonl, (ROOT/'observations.csv').open('w', newline='') as csvfile:
        writer = None
        for i in range(0, len(scenarios), 2):
            pair = scenarios[i:i+2]
            data = {}
            for scenario in pair:
                rows, accesses, cache_ops, _ = _execute(scenario, world, metadata,
                                                        by_query[scenario['query_id']], manifest['config'])
                audit = _audit_arm(scenario, rows, accesses, cache_ops)
                audits.append(audit)
                if not audit['valid']:
                    write_json(ROOT/'invalid.json', {'status': 'INVALID', 'scenario_id': scenario['scenario_id'],
                                                     'reasons': audit['reasons']})
                    raise ValueError('Runtime validity failure')
                data[scenario['scenario_id']] = (rows, None, audit)
                for row in rows:
                    jsonl.write(json.dumps(row, sort_keys=True) + '\n')
                    if writer is None:
                        writer = csv.DictWriter(csvfile, fieldnames=list(row))
                        writer.writeheader()
                    writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (dict, list)) else value
                                     for key, value in row.items()})
                    count += 1
            if not _audit_pair(pair, data)['same_final_evidence']:
                write_json(ROOT/'invalid.json', {'status': 'INVALID', 'pair_id': pair[0]['h4_pair_id'],
                                                 'reason': 'final_evidence_mismatch'})
                raise ValueError('Matched final state mismatch')
            if (i//2 + 1) % 100 == 0:
                print(f'Completed {i//2+1}/{len(scenarios)//2} pairs', flush=True)
    write_csv(ROOT/'runtime_audit.csv', audits)
    summary = {'pairs': len(scenarios)//2, 'scenarios': len(scenarios), 'observations': count,
               'valid_arms': sum(a['valid'] for a in audits)}
    write_json(ROOT/'execution_summary.json', summary)
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('action', choices=('freeze', 'sample', 'preflight', 'run'))
    args = parser.parse_args()
    print(json.dumps({'freeze': freeze, 'sample': sample, 'preflight': preflight,
                      'run': run}[args.action](), sort_keys=True))


if __name__ == '__main__':
    main()
