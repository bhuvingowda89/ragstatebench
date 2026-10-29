"""Matched-horizon H4 supplement; never reads v4 scientific outcomes."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from .h4_partial_runtime import execute_track_a
from .state_pilot_manifest import _load_frozen, digest
from .state_pilot_v2_campaign import capture_scenario, _jsonl
from .state_pilot_v3_campaign import capture_replica_snapshots
from .state_pilot_v4_campaign import verify_freeze as verify_v4


CAMPAIGN = 'h4-partial-control-pilot-v1'
ROOT = Path('results') / CAMPAIGN
V4_SOURCE = Path('results/state-isolated-pilot-v4-preflight/scientific_manifest.json')
TICKS = {'S1': (1, 2), 'S2': (2, 5), 'S3': (5, 10)}
SOURCE_HASH = '4b45ef1c41b7b08ebddee0e250e0e6d1c1446e25292b07a324ed2a20d05dfeed'


def _write_json(path, data):
    Path(path).write_text(json.dumps(data, indent=2, sort_keys=True) + '\n')


def _csv(path, rows):
    rows = list(rows)
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]) if rows else [])
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True) if isinstance(value, (list,dict))
                             else value for key,value in row.items()})


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source():
    config, v4, _ = verify_v4()
    if v4['scientific_manifest_sha256'] != SOURCE_HASH:
        raise ValueError('Frozen v4 scientific manifest changed')
    source = [item for item in v4['scenarios']
              if item['block'] == 'TA2' and item.get('partial_topology')
              and item['architecture'] == 'A0']
    if len(source) != 752:
        raise ValueError('H4 source set differs from frozen applicability-v2 topology')
    return config, source


def _pair(source):
    partial_tick, final_tick = TICKS[source['severity']]
    pair_id = 'H4__' + source['scenario_id']
    result = []
    for arm in ('U','P'):
        scenario = dict(source)
        scenario.update(scenario_id=f'{pair_id}__{arm}', campaign_version=CAMPAIGN,
                        block='H4', h4_pair_id=pair_id, h4_arm=arm,
                        h4_partial_tick=partial_tick, h4_final_tick=final_tick,
                        h4_source_scenario_id=source['scenario_id'],
                        h4_source_plan_hash=source['selected_unit_plan_hash'],
                        h4_pre_unit_ids=sorted(set(source['selected_unit_ids']) |
                                                set(source['selected_stale_unit_ids'])),
                        fault='PARTIAL' if arm == 'P' else 'DELAY')
        result.append(scenario)
    return result


def _balance(pair):
    u,p = pair
    keys = ('query_id','transition','architecture','replica_unit_type',
            'total_unit_count','severity','h4_final_tick','selected_unit_ids',
            'selected_stale_unit_ids','h4_source_plan_hash')
    equal = {key:u[key] == p[key] for key in keys}
    schedule_only = u['h4_arm']=='U' and p['h4_arm']=='P' and \
                    u['h4_partial_tick']==p['h4_partial_tick'] and \
                    u['h4_pair_id']==p['h4_pair_id']
    return {'pair_id':p['h4_pair_id'], 'query_id':p['query_id'],
            'transition':p['transition'], 'architecture':p['architecture'],
            'domain':p['query_id'].split('-')[2], 'target_plane':p['replica_unit_type'],
            'N':p['total_unit_count'], 'severity':p['severity'],
            'T':p['h4_final_tick'], 't_partial':p['h4_partial_tick'],
            'balanced':all(equal.values()) and schedule_only,
            'mismatched_fields':[key for key,ok in equal.items() if not ok]}


def freeze():
    if ROOT.exists():
        raise ValueError('H4 campaign exists; refusing to overwrite')
    config, source = _source()
    ROOT.mkdir(parents=True)
    pairs = [_pair(item) for item in source]
    balance = [_balance(pair) for pair in pairs]
    if not all(row['balanced'] for row in balance):
        raise ValueError('H4 pair balance failed before freeze')
    scenarios = [arm for pair in pairs for arm in pair]
    config = dict(config, campaign_version=CAMPAIGN)
    manifest = {'campaign_id':CAMPAIGN, 'source_v4_manifest_sha256':SOURCE_HASH,
                'source_applicability_v2_sha256':
                    'c8df463c1d26b08e64afbc96596e42aa6a69be8649c84e8fa60eb036ec71fa8f',
                'source_partial_plan_sha256':digest([item['selected_unit_plan_hash'] for item in source]),
                'schedule_policy':{'S1':{'t_partial':1,'T':2},
                                   'S2':{'t_partial':2,'T':5},
                                   'S3':{'t_partial':5,'T':10}},
                'primary_interval':'t_partial <= t < T',
                'config':config,'scenarios':scenarios,
                'scenario_sha256':digest(scenarios),
                'runtime_sha256':_sha('src/ragstate/campaign/h4_partial_runtime.py')}
    _write_json(ROOT/'manifest.json',manifest)
    _csv(ROOT/'h4_pair_balance.csv',balance)
    return {'pairs':len(pairs),'scenarios':len(scenarios),
            'scenario_sha256':manifest['scenario_sha256']}


def verify():
    manifest = json.loads((ROOT/'manifest.json').read_text())
    if manifest['source_v4_manifest_sha256'] != SOURCE_HASH or \
       manifest['scenario_sha256'] != digest(manifest['scenarios']) or \
       manifest['runtime_sha256'] != _sha('src/ragstate/campaign/h4_partial_runtime.py'):
        raise ValueError('H4 frozen provenance changed')
    _, source = _source()
    expected = [_pair(item) for item in source]
    if manifest['scenarios'] != [arm for pair in expected for arm in pair]:
        raise ValueError('H4 scenarios differ from applicability-v2 selected plans')
    return manifest


def _execute(scenario, world, metadata, query, config):
    with capture_replica_snapshots() as snapshots:
        with capture_scenario(scenario['scenario_id']) as (accesses, cache_ops, _):
            rows = execute_track_a(scenario,world,metadata,query,config)
    return rows, accesses, cache_ops, snapshots


def _audit_arm(scenario, rows, accesses, cache_ops):
    n,k = scenario['total_unit_count'],scenario['updated_unit_count']
    tp,T = scenario['h4_partial_tick'],scenario['h4_final_tick']
    failures=[]
    for row in rows:
        tick=row['logical_time']
        expected=0 if tick < (tp if scenario['h4_arm']=='P' else T) else \
                 (k if scenario['h4_arm']=='P' and tick < T else n)
        if row['h4_actual_updated_units'] != expected:
            failures.append(f'unit_count_at_{tick}:{row["h4_actual_updated_units"]}!={expected}')
        if row['h4_actual_stale_units'] != n-expected:
            failures.append(f'stale_count_at_{tick}')
    events=rows[-1]['event_order']
    actual_schedule=[(event['tick'],event['stage']) for event in events
                     if event['stage'] in ('replica_delivery','partial_completion')]
    planned_schedule=[(tp,'replica_delivery'),(T,'partial_completion')] \
                     if scenario['h4_arm']=='P' else [(T,'replica_delivery')]
    if actual_schedule!=planned_schedule:
        failures.append(f'event_schedule:{actual_schedule}!={planned_schedule}')
    if any(not event['allowed'] for event in accesses):
        failures.append('forbidden_canonical_access')
    if any(not stage['valid'] for op in cache_ops for stage in op['stage_events']):
        failures.append('cache_stage_violation')
    return {'scenario_id':scenario['scenario_id'],'pair_id':scenario['h4_pair_id'],
            'arm':scenario['h4_arm'],'transition':scenario['transition'],
            'severity':scenario['severity'],'N':n,'k':k,'t_partial':tp,'T':T,
            'actual_schedule':actual_schedule,'planned_schedule':planned_schedule,
            'forbidden_accesses':sum(not event['allowed'] for event in accesses),
            'cache_stage_violations':sum(not stage['valid'] for op in cache_ops
                                         for stage in op['stage_events']),
            'valid':not failures,'reasons':failures}


def _audit_pair(pair, arm_data):
    u,p=pair
    ru,rp=arm_data[u['scenario_id']][0],arm_data[p['scenario_id']][0]
    T=u['h4_final_tick']
    ufinal=next(row for row in ru if row['logical_time']==T)
    pfinal=next(row for row in rp if row['logical_time']==T)
    # Epoch hashes intentionally differ: the mixed arm applied two events.
    same_final=(ufinal['context_chunk_ids']==pfinal['context_chunk_ids'] and
                ufinal['h4_actual_updated_units']==pfinal['h4_actual_updated_units']==u['total_unit_count'])
    return {'pair_id':u['h4_pair_id'],'same_final_evidence':same_final,
            'uniform_context_at_T':ufinal['context_chunk_ids'],
            'partial_context_at_T':pfinal['context_chunk_ids']}


def _sample_pairs(manifest):
    pairs=[manifest['scenarios'][index:index+2]
           for index in range(0,len(manifest['scenarios']),2)]
    selected=[]
    criteria=(('S2_N2',lambda p:p[1]['severity']=='S2' and p[1]['total_unit_count']==2),
              ('S2_N3',lambda p:p[1]['severity']=='S2' and p[1]['total_unit_count']==3),
              ('S2_N4',lambda p:p[1]['severity']=='S2' and p[1]['total_unit_count']>=4),
              ('UPDATE',lambda p:p[1]['transition']=='UPDATE'),
              ('SUPERSEDE',lambda p:p[1]['transition']=='SUPERSEDE'),
              ('DELETE',lambda p:p[1]['transition']=='DELETE'),
              ('REVOKE',lambda p:p[1]['transition']=='REVOKE'))
    for label,predicate in criteria:
        matches=[pair for pair in pairs if predicate(pair)]
        if not matches:raise ValueError('Missing H4 sample stratum '+label)
        selected.append((label,matches[0]))
    return selected


def sample():
    manifest=verify()
    world,metadata,queries,_=_load_frozen(manifest['config'])
    by_query={query.query_id:query for query in queries}
    results=[]
    for label,pair in _sample_pairs(manifest):
        data={}
        for scenario in pair:
            rows,accesses,cache_ops,_=_execute(scenario,world,metadata,
                by_query[scenario['query_id']],manifest['config'])
            audit=_audit_arm(scenario,rows,accesses,cache_ops)
            stable_rows=[{key:value for key,value in row.items()
                          if key not in ('retrieval_latency_ms','parent_latency_ms')}
                         for row in rows]
            data[scenario['scenario_id']]=(rows,digest({'rows':stable_rows,
                'accesses':accesses,'cache_ops':cache_ops,'audit':audit}),audit)
        pair_audit=_audit_pair(pair,data)
        results.append({'stratum':label,'pair_id':pair[0]['h4_pair_id'],
                        'arm_fingerprints':{key:value[1] for key,value in data.items()},
                        'arm_audits':{key:value[2] for key,value in data.items()},
                        'pair_audit':pair_audit})
    return results


def preflight():
    manifest=verify()
    if (ROOT/'gate_summary.json').exists():
        raise ValueError('H4 science already exists; preflight cannot be replaced')
    balances=[_balance(manifest['scenarios'][index:index+2])
              for index in range(0,len(manifest['scenarios']),2)]
    local=sample()
    env=dict(os.environ,PYTHONPATH=str(Path('src').resolve())+os.pathsep+os.environ.get('PYTHONPATH',''))
    fresh=subprocess.run([sys.executable,'-m','ragstate.campaign.h4_partial_campaign','sample'],
                         capture_output=True,text=True,env=env)
    # JSON transport converts tuples in event schedules to lists; compare the
    # same serialized representation on both sides, as in the frozen v4 audit.
    normalized_local=json.loads(json.dumps(local, sort_keys=True))
    reproduced=json.loads(fresh.stdout)==normalized_local if fresh.returncode==0 else False
    arm_audits=[audit for item in local for audit in item['arm_audits'].values()]
    pair_audits=[item['pair_audit'] for item in local]
    gates={'H4-P1':all(row['balanced'] for row in balances),
           'H4-P2':all(audit['valid'] for audit in arm_audits if audit['arm']=='P'),
           'H4-P3':all(audit['valid'] for audit in arm_audits if audit['arm']=='U'),
           'H4-P4':all(item['same_final_evidence'] for item in pair_audits),
           'H4-P5':all(audit['valid'] for audit in arm_audits),
           'H4-P6':all(not audit['forbidden_accesses'] and not audit['cache_stage_violations']
                       for audit in arm_audits),
           'H4-P7':reproduced}
    _write_json(ROOT/'preflight.json',{'gates':gates,'sample':local,
                                     'fresh_exit':fresh.returncode,'stderr':fresh.stderr})
    if not all(gates.values()):
        _write_json(ROOT/'invalid.json',{'status':'INVALID','reason':'preflight_gate_failure'})
    return gates


def run():
    manifest=verify()
    pre=json.loads((ROOT/'preflight.json').read_text())
    if not all(pre['gates'].values()) or (ROOT/'invalid.json').exists():
        raise ValueError('H4 preflight failed; refusing scientific execution')
    if (ROOT/'observations.jsonl').exists():
        raise ValueError('H4 observations already exist; refusing overwrite')
    world,metadata,queries,_=_load_frozen(manifest['config'])
    by_query={query.query_id:query for query in queries}
    pairs=[manifest['scenarios'][index:index+2]
           for index in range(0,len(manifest['scenarios']),2)]
    audit_rows=[]
    all_rows=[]
    start=time.perf_counter()
    for index,pair in enumerate(pairs):
        data={}
        for scenario in pair:
            rows,accesses,cache_ops,_=_execute(scenario,world,metadata,
                by_query[scenario['query_id']],manifest['config'])
            audit=_audit_arm(scenario,rows,accesses,cache_ops)
            audit_rows.append(audit)
            if not audit['valid']:
                _write_json(ROOT/'invalid.json',{'status':'INVALID','scenario_id':scenario['scenario_id'],
                                                 'reasons':audit['reasons']})
                raise ValueError('H4 runtime schedule mismatch: '+scenario['scenario_id'])
            data[scenario['scenario_id']]=(rows,None,audit)
            all_rows.extend(rows)
        pair_audit=_audit_pair(pair,data)
        if not pair_audit['same_final_evidence']:
            _write_json(ROOT/'invalid.json',{'status':'INVALID','pair_id':pair[0]['h4_pair_id'],
                                             'reason':'final_evidence_mismatch'})
            raise ValueError('H4 pair final evidence differs')
        if (index+1)%100==0:
            print(f'H4 pairs {index+1}/{len(pairs)}',flush=True)
    _jsonl(ROOT/'observations.jsonl',all_rows)
    _csv(ROOT/'observations.csv',all_rows)
    _csv(ROOT/'schedule_audit.csv',audit_rows)
    summary={'pairs':len(pairs),'scenarios':len(manifest['scenarios']),
             'observations':len(all_rows),'runtime_seconds':time.perf_counter()-start,
             'schedule_agreements':sum(audit['valid'] for audit in audit_rows),
             'forbidden_accesses':sum(audit['forbidden_accesses'] for audit in audit_rows),
             'cache_stage_violations':sum(audit['cache_stage_violations'] for audit in audit_rows)}
    _write_json(ROOT/'execution_summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('freeze','sample','preflight','run'))
    args=parser.parse_args()
    print(json.dumps({'freeze':freeze,'sample':sample,'preflight':preflight,
                      'run':run}[args.action](),indent=2,sort_keys=True))


if __name__=='__main__':main()
