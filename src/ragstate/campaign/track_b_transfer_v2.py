"""Execute the frozen 120-scenario Track-B transfer subset with replica PA2."""
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

from ragstate.replicas.scientific import ReplicaAuthorizationPlane, ReplicaContentPlane
from ragstate.retrieval.metrics import (invalid_evidence_absence_success,
    state_consistent_retrieval_success)
from ragstate.retrieval.replica_pa2 import PA2ReplicaRetriever

from .scenario_applicability import _transition_facts, compile_candidate, dry_run
from .state_pilot_manifest import _load_frozen, digest
from .state_pilot_runtime_v4 import _claim_success, execute_track_a
from .state_pilot_v2_campaign import capture_scenario, _jsonl
from .state_pilot_v3_campaign import capture_replica_snapshots, validate_scenario


ROOT=Path('results/realistic-transfer-pilot-v2')
TRANSFER=ROOT/'transfer_manifest.json'
FROZEN=ROOT/'manifest.json'
V4=Path('results/state-isolated-pilot-v4-preflight/scientific_manifest.json')
COHORT='e74d643f6c2eb7c8ed8c52683e0db0152f7d7ddbcf29d86c070a28e22c79ce43'
TRANSFER_SHA='58739ace13241d0a775a23843d6615be6669ded6440a7103ae931f9ccb8ba537'
V4_SHA='4b45ef1c41b7b08ebddee0e250e0e6d1c1446e25292b07a324ed2a20d05dfeed'


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def _csv(path,rows):
    rows=list(rows)
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key:json.dumps(value,sort_keys=True) if isinstance(value,(list,dict))
                             else value for key,value in row.items()})


def frozen_inputs():
    transfer=json.loads(TRANSFER.read_text())
    baseline=json.loads(FROZEN.read_text())
    v4=json.loads(V4.read_text())
    if transfer['sha256']!=TRANSFER_SHA or digest(transfer['scenarios'])!=TRANSFER_SHA:
        raise ValueError('Frozen Track-B transfer manifest changed')
    if baseline['manifest_hashes']['track_b']!=TRANSFER_SHA or baseline['track_b_cohort_sha256']!=COHORT:
        raise ValueError('Track-B source provenance changed')
    if v4['scientific_manifest_sha256']!=V4_SHA or digest(v4['scenarios'])!=V4_SHA:
        raise ValueError('Frozen Track-A v4 manifest changed')
    ids=json.loads(Path(baseline['config']['track_b_cohort_ids']).read_text())+\
        json.loads(Path(baseline['config']['track_b_negative_ids']).read_text())
    if any(s['query_id'] not in ids for s in transfer['scenarios']):
        raise ValueError('Transfer scenario escaped frozen cohort-v2')
    return baseline,transfer,v4


def track_a_link(scenario, track_a_scenarios=None):
    """Frozen structural/query-level link; no outcome field is consulted."""
    block=scenario['block']
    suffix=scenario['scenario_id'].split('__',1)[1]
    if block=='TB1':proposed='TA1__'+suffix
    elif block=='TB2':proposed='TA3__'+suffix
    elif block=='TB3':proposed='TA4__'+suffix
    elif block=='TB4':proposed='TA5__'+suffix
    else:raise ValueError('Unfrozen Track-B block')
    if track_a_scenarios is None:return proposed
    by_id={item['scenario_id']:item for item in track_a_scenarios}
    if proposed in by_id:return proposed,'query_level'
    domain=scenario['query_id'].split('-')[2]
    candidates=[item for item in track_a_scenarios
        if item['block']==proposed.split('__',1)[0]
        and item['transition']==scenario['transition']
        and item['architecture']==scenario['architecture']
        and item['severity']==scenario['severity']
        and item['fault']==scenario['fault']
        and item.get('compound')==scenario.get('compound')
        and item.get('component')==scenario.get('component')
        and item['query_id'].split('-')[2]==domain]
    if not candidates:raise ValueError('No frozen Track-A structural match: '+proposed)
    return min(item['scenario_id'] for item in candidates),'structural_domain'


def classify_direction(a,b,*, tolerance=1e-12):
    if a is None or b is None or abs(a)<=tolerance:return 'not estimable'
    if abs(b)<=tolerance:return 'absent'
    if a*b<0:return 'reversed'
    return 'attenuated but same direction' if abs(b)<abs(a)-tolerance else 'preserved'


def classify_masking(track_a_failure,track_b_failure,clean_b_correct):
    if track_a_failure and track_b_failure:return 'both_present'
    if track_a_failure and not track_b_failure:return 'track_a_present_track_b_absent'
    if not track_a_failure and track_b_failure:
        return 'baseline_retrieval_failure' if not clean_b_correct else 'track_b_only_failure'
    return 'neither'


def d11(effect_rows,domain_rows,*, retrieval_interpretable):
    counted=[row for row in effect_rows if row['track_a_direction'] in ('positive','negative')]
    agreeing=[row for row in counted if row['classification'] in
              ('preserved','attenuated but same direction')]
    reversals=[row for row in counted if row['classification']=='reversed']
    domain_reversal=any(row.get('systematic_reversal') for row in domain_rows)
    return {'status':'PASS' if counted and len(agreeing)>len(counted)/2 and
            not reversals and not domain_reversal and retrieval_interpretable else 'FAIL',
            'agreement_numerator':len(agreeing),'agreement_denominator':len(counted),
            'reversed_effects':[row['effect_id'] for row in reversals],
            'retrieval_interpretable':retrieval_interpretable}


def _policy():
    # Every effect class and constituent contrast is declared before Track B.
    return {'version':'transfer_effect_policy_v2','primary_role':'secondary_transfer',
        'Track_A_source_manifest_sha256':V4_SHA,
        'effect_classes':[
            {'id':'T1','members':['UPDATE','SUPERSEDE','DELETE','GRANT','REVOKE'],
             'contrast':'A0 S2 t=1 state failure minus t=-1 clean failure'},
            {'id':'T2','members':['A4','A5'],
             'contrast':'REVOKE S2 t=1 UCER_context mitigation minus A0'},
            {'id':'T3','members':['SUPERSEDE:A3','SUPERSEDE:A5','DELETE:A3','DELETE:A5'],
             'contrast':'cache incorrect-after-index-converged rate minus A0'},
            {'id':'T4','members':['UPDATE','SUPERSEDE','DELETE','GRANT','REVOKE'],
             'contrast':'A0 S2 stable convergence occurred by t=14'},
            {'id':'T5','members':['CF2','CF4','CF5'],
             'contrast':'A0 compound t=5 failure minus matched no-fault arm'},
        ],'direction_classes':['preserved','attenuated but same direction',
                               'absent','reversed','not estimable'],
        'matching':'query-level when exact TA1/TA3/TA4/TA5 link exists',
        'H4_excluded_from_transfer':True}


def freeze_control():
    if (ROOT/'provenance.json').exists():
        raise ValueError('Track-B execution control already frozen')
    baseline,transfer,v4=frozen_inputs()
    links=[]
    for scenario in transfer['scenarios']:
        aid,level=track_a_link(scenario,v4['scenarios'])
        links.append({'track_b_scenario_id':scenario['scenario_id'],
                      'track_a_scenario_id':aid,'track_a_comparison_id':scenario['block'].replace('TB','TA'),
                      'query_id':scenario['query_id'],'transition':scenario['transition'],
                      'architecture':scenario['architecture'],'severity':scenario['severity'],
                      'fault':scenario['fault'],'compound':scenario['compound'],
                      'matching_level':level})
    policy=_policy()
    provenance={'campaign_id':'realistic-transfer-pilot-v2',
        'transfer_manifest_sha256':TRANSFER_SHA,'track_a_v4_manifest_sha256':V4_SHA,
        'cohort_v2_sha256':COHORT,'corpus_sha256':baseline['corpus_sha256'],
        'transfer_file_sha256':_sha(TRANSFER),'baseline_manifest_file_sha256':_sha(FROZEN),
        'v4_manifest_file_sha256':_sha(V4),
        'runtime_sha256':_sha('src/ragstate/campaign/state_pilot_runtime_v4.py'),
        'replica_pa2_sha256':_sha('src/ragstate/retrieval/replica_pa2.py'),
        'link_sha256':digest(links),'effect_policy_sha256':digest(policy),
        'embedding_model':baseline['config']['embedding_model'],
        'embedding_revision':baseline['config']['embedding_revision'],
        'pa2':{'chunk_words':160,'overlap_words':0,'candidate_k':30,'top_k':10,
               'parent_chunk_rrf_k':60,'hybrid':baseline['config']['hybrid']}}
    _json(ROOT/'transfer_effect_policy.json',policy)
    _csv(ROOT/'trackA_trackB_links.csv',links)
    _json(ROOT/'provenance.json',provenance)
    return {'scenario_count':len(transfer['scenarios']),'linked':len(links),
            'transfer_manifest_sha256':TRANSFER_SHA,'effect_policy_sha256':digest(policy)}


def verify_control():
    baseline,transfer,v4=frozen_inputs()
    provenance=json.loads((ROOT/'provenance.json').read_text())
    for path,key in ((TRANSFER,'transfer_file_sha256'),(FROZEN,'baseline_manifest_file_sha256'),
                     (V4,'v4_manifest_file_sha256'),
                     ('src/ragstate/campaign/state_pilot_runtime_v4.py','runtime_sha256'),
                     ('src/ragstate/retrieval/replica_pa2.py','replica_pa2_sha256')):
        if _sha(path)!=provenance[key]:raise ValueError('Frozen Track-B provenance drift: '+key)
    if digest(json.loads((ROOT/'transfer_effect_policy.json').read_text()))!=provenance['effect_policy_sha256']:
        raise ValueError('Transfer effect policy changed')
    return baseline,transfer,v4,provenance


def _config(baseline):
    return {**baseline['config'],'campaign_version':'realistic-transfer-pilot-v2'}


def _record(scenario,world,metadata,query,config,retriever):
    with capture_replica_snapshots() as snapshots:
        with capture_scenario(scenario['scenario_id']) as (accesses,cache_ops,_):
            rows=execute_track_a(scenario,world,metadata,query,config,
                                 pa2_retriever=retriever)
    validity=validate_scenario(scenario,rows,accesses,cache_ops,snapshots,
                               query,metadata,world,config)
    return rows,accesses,cache_ops,validity


def _fingerprint(rows,accesses,cache_ops,validity):
    stable=[{key:value for key,value in row.items()
             if key not in ('retrieval_latency_ms','parent_latency_ms')}
            for row in rows]
    return digest({'rows':stable,'accesses':accesses,
                   'cache_ops':cache_ops,'validity':validity})


def _sample_scenarios(scenarios):
    checks=(('content',lambda s:s['block']=='TB1' and s['transition']=='UPDATE'),
            ('authorization',lambda s:s['block']=='TB1' and s['transition']=='REVOKE'),
            ('live_auth',lambda s:s['block']=='TB2' and s['architecture']=='A4'),
            ('cache',lambda s:s['block']=='TB3' and s['architecture']=='A3'),
            ('compound_CF2',lambda s:s['block']=='TB4' and s['compound']=='CF2'),
            ('compound_CF4',lambda s:s['block']=='TB4' and s['compound']=='CF4'),
            ('compound_CF5',lambda s:s['block']=='TB4' and s['compound']=='CF5'))
    return [(label,sorted((s for s in scenarios if predicate(s)),
                          key=lambda s:s['scenario_id'])[0])
            for label,predicate in checks]


def sample():
    baseline,transfer,_,_=verify_control()
    config=_config(baseline)
    world,metadata,queries,_=_load_frozen(config)
    by_query={query.query_id:query for query in queries}
    retriever=PA2ReplicaRetriever(config)
    output=[]
    for label,scenario in _sample_scenarios(transfer['scenarios']):
        rows,accesses,cache_ops,validity=_record(scenario,world,metadata,
            by_query[scenario['query_id']],config,retriever)
        output.append({'stratum':label,'scenario_id':scenario['scenario_id'],
                       'validity':validity['validity'],'reason':validity['reasons'],
                       'forbidden':validity['forbidden_access_count'],
                       'cache_violations':validity['cache_stage_violation_count'],
                       'fingerprint':_fingerprint(rows,accesses,cache_ops,validity),
                       'candidate_count_at_1':len(next(row for row in rows
                            if row['logical_time']==1)['candidate_chunk_ids']),
                       'parent_hash_at_1':next(row for row in rows
                            if row['logical_time']==1)['parent_index_hash']})
    return output


def _clean_reproduction(config,world,metadata,queries,cohort_ids,retriever):
    chunks=ReplicaContentPlane(world,9,chunked=True)
    parents=ReplicaContentPlane(world,9,chunked=False)
    auth=ReplicaAuthorizationPlane(world)
    selected=[query for query in queries if query.query_id in cohort_ids]
    actual=[]
    frozen={row['query_id']:row for row in csv.DictReader(
        Path('results/phase7g-parent-aware-v1/per_query_results.csv').open())
        if row['policy']=='PA2'}
    for query in selected:
        candidates=retriever.retrieve(query.text,chunks,parents)
        context=[hit for hit in candidates if auth.allowed(query.principal_id,hit.chunk,9)]
        gold=set(query.required_document_ids) if query.answerable else set()
        scrs=int(state_consistent_retrieval_success(context,gold,query.principal_id,world,9))
        _,claim=_claim_success(world,query,metadata,{},context,9)
        absence=invalid_evidence_absence_success(context,gold,
                                                  set(query.logical_document_ids))
        expected=frozen[query.query_id]
        exact_ids=[hit.chunk.chunk_id for hit in context]==json.loads(expected['context_chunk_ids'])
        correct=(scrs==1 and claim==1) if query.answerable else (absence==1)
        actual.append({'query_id':query.query_id,'domain':query.domain,
                       'answerable':query.answerable,'scrs':scrs,'claim_success':claim,
                       'negative_absence':absence,'clean_correct':correct,
                       'exact_frozen_context':exact_ids})
    return actual


def preflight():
    baseline,transfer,v4,provenance=verify_control()
    if (ROOT/'preflight.json').exists():
        raise ValueError('Track-B preflight already exists')
    config=_config(baseline)
    world,metadata,queries,cohort_ids=_load_frozen(config)
    by_query={query.query_id:query for query in queries}
    facts={}
    applicability=[]
    for scenario in transfer['scenarios']:
        key=scenario['query_id'],scenario['transition']
        if key not in facts:
            facts[key]=_transition_facts(world,metadata,by_query[key[0]],key[1])
        compiled=compile_candidate(scenario,by_query[key[0]],facts[key],config)
        dry=dry_run(scenario,by_query[key[0]],facts[key],compiled)
        applicability.append({'scenario_id':scenario['scenario_id'],
            'applicable':compiled.applicable,'reasons':list(compiled.reasons),
            'realizable':dry['realizable'],'target_planes':list(compiled.expected_target_planes)})
    _csv(ROOT/'applicability_audit.csv',applicability)
    retriever=PA2ReplicaRetriever(config)
    clean=_clean_reproduction(config,world,metadata,queries,cohort_ids,retriever)
    _csv(ROOT/'clean_cohort_reproduction.csv',clean)
    local=sample()
    env=dict(os.environ,PYTHONPATH=str(Path('src').resolve())+os.pathsep+os.environ.get('PYTHONPATH',''))
    fresh=subprocess.run([sys.executable,'-m','ragstate.campaign.track_b_transfer_v2','sample'],
                         capture_output=True,text=True,env=env)
    observed=json.loads(fresh.stdout) if fresh.returncode==0 else None
    gates={'TB-P1':len(transfer['scenarios'])==120 and all(row['applicable'] and row['realizable']
                                                          for row in applicability),
           'TB-P2':all(item['candidate_count_at_1']>=0 and item['parent_hash_at_1']
                       for item in local),
           'TB-P3':all(item['forbidden']==0 for item in local),
           'TB-P4':all(item['cache_violations']==0 for item in local),
           'TB-P5':all(item['validity']=='valid' for item in local),
           'TB-P6':len(clean)==197 and all(item['clean_correct'] and item['exact_frozen_context']
                                          for item in clean),
           'TB-P7':observed==local}
    result={'gates':gates,'applicable':sum(r['applicable'] for r in applicability),
            'realizable':sum(r['realizable'] for r in applicability),
            'clean_count':len(clean),'clean_correct':sum(r['clean_correct'] for r in clean),
            'exact_frozen_context':sum(r['exact_frozen_context'] for r in clean),
            'sample':local,'fresh_process_exit':fresh.returncode,'fresh_stderr':fresh.stderr}
    _json(ROOT/'preflight.json',result)
    if not all(gates.values()):
        _json(ROOT/'invalid.json',{'status':'INVALID','stage':'preflight','failed_gates':
                                  [key for key,value in gates.items() if not value]})
    return {key:'PASS' if value else 'FAIL' for key,value in gates.items()}


def run():
    baseline,transfer,_,_=verify_control()
    pre=json.loads((ROOT/'preflight.json').read_text())
    if not all(pre['gates'].values()) or (ROOT/'invalid.json').exists():
        raise ValueError('Track-B preflight failed; refusing scientific execution')
    if (ROOT/'observations.jsonl').exists():
        raise ValueError('Track-B observations already exist; refusing overwrite')
    config=_config(baseline)
    world,metadata,queries,_=_load_frozen(config)
    by_query={query.query_id:query for query in queries}
    retriever=PA2ReplicaRetriever(config)
    all_rows=[];status=[];audits=[]
    started=time.perf_counter()
    for index,scenario in enumerate(transfer['scenarios']):
        try:
            rows,accesses,cache_ops,validity=_record(scenario,world,metadata,
                by_query[scenario['query_id']],config,retriever)
            audits.append(validity)
            if validity['validity']!='valid':
                raise ValueError('compiler_runtime_mismatch:'+validity['reasons'])
            all_rows.extend(rows)
            status.append({'scenario_id':scenario['scenario_id'],
                           'status':'censored' if validity['censored'] else 'valid',
                           'reason':''})
        except Exception as exc:
            status.append({'scenario_id':scenario['scenario_id'],'status':'invalid',
                           'reason':f'{type(exc).__name__}:{exc}'})
            _csv(ROOT/'scenario_status.csv',status)
            _json(ROOT/'invalid.json',{'status':'INVALID','stage':'execution',
                  'scenario_id':scenario['scenario_id'],'reason':status[-1]['reason']})
            raise
        if (index+1)%20==0:print(f'Track B {index+1}/{len(transfer["scenarios"])}',flush=True)
    _jsonl(ROOT/'observations.jsonl',all_rows)
    _csv(ROOT/'observations.csv',all_rows)
    _csv(ROOT/'scenario_status.csv',status)
    _csv(ROOT/'scenario_validity.csv',audits)
    summary={'scenarios':len(transfer['scenarios']),'observations':len(all_rows),
             'status_counts':dict(Counter(row['status'] for row in status)),
             'forbidden_accesses':sum(row['forbidden_access_count'] for row in audits),
             'cache_stage_violations':sum(row['cache_stage_violation_count'] for row in audits),
             'runtime_seconds':time.perf_counter()-started}
    _json(ROOT/'execution_summary.json',summary)
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('freeze','sample','preflight','run'))
    args=parser.parse_args()
    print(json.dumps({'freeze':freeze_control,'sample':sample,'preflight':preflight,
                      'run':run}[args.action](),indent=2,sort_keys=True))


if __name__=='__main__':main()
