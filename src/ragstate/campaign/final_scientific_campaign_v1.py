"""Frozen, resumable final state campaign using the validated v4 instrument."""
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

from .scenario_applicability import _transition_facts, compile_candidate, dry_run
from .state_pilot_manifest import _load_frozen, _scenario, digest
from .state_pilot_v4_campaign import (deterministic_fingerprint, execute_recorded,
    validate_scenario)

ROOT=Path('results/final-scientific-campaign-v1')
CONFIG=Path('configs/final-scientific-campaign-v1.json')
POLICY=ROOT/'analysis_policy.json'
SOURCE=Path('results/state-isolated-pilot-v4-preflight/scientific_manifest.json')
INSTRUMENT=Path('configs/state-isolated-pilot-v4.json')
BLOCK_MAP={'TA1':'FA1','TA2':'FA3','TA3':'FA5','TA4':'FA4','TA5':'FA7','TA6':'FA6'}
SOURCE_FILES=('src/ragstate/campaign/state_pilot_runtime_v4.py',
    'src/ragstate/campaign/topology_partial_runtime.py',
    'src/ragstate/campaign/state_pilot_validity_v3.py',
    'src/ragstate/campaign/state_capabilities.py',
    'src/ragstate/replicas/scientific.py')


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')
    os.replace(temporary,path)


def _csv(path,rows):
    rows=list(rows)
    fields=list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader()
        for row in rows:writer.writerow({k:json.dumps(v,sort_keys=True)
            if isinstance(v,(dict,list)) else v for k,v in row.items()})


def _instrument_config():
    config=json.loads(INSTRUMENT.read_text())
    return {**config,'campaign_version':'final-scientific-campaign-v1'}


def _expanded_scenarios(source_scenarios,world,metadata,queries,config):
    by_query={q.query_id:q for q in queries}
    base=[s for s in source_scenarios if s['block']=='TA1' and
          s['severity']=='S2' and s['architecture']=='A0']
    by_transition=defaultdict(list)
    for s in base:by_transition[s['transition']].append(s)
    existing={s['scenario_id'] for s in source_scenarios}
    def semantic_key(s):
        return (s['query_id'],s['transition'],s['severity'],s['architecture'],
                s['fault'],s.get('compound'),s.get('component'),
                tuple(s.get('component_tuple') or ()),s.get('partial_topology'))
    existing_semantics={semantic_key(s) for s in source_scenarios}
    facts={};added=[];audit=[]
    def add(scenario):
        key_semantic=semantic_key(scenario)
        if scenario['scenario_id'] in existing or key_semantic in existing_semantics:return
        key=scenario['query_id'],scenario['transition']
        if key not in facts:
            facts[key]=_transition_facts(world,metadata,by_query[key[0]],key[1])
        result=compile_candidate(scenario,by_query[key[0]],facts[key],config)
        dry=dry_run(scenario,by_query[key[0]],facts[key],result)
        accepted=result.applicable and dry['realizable']
        audit.append({'scenario_id':scenario['scenario_id'],
            'final_block':scenario['final_block'],'applicable':result.applicable,
            'realizable':dry['realizable'],'included':accepted,
            'reasons':';'.join(result.reasons)})
        if accepted:
            existing.add(scenario['scenario_id']);existing_semantics.add(key_semantic)
            added.append(scenario)
    for transition in ('UPDATE','SUPERSEDE','DELETE','REVOKE'):
        for seed in sorted(by_transition[transition],key=lambda s:s['query_id']):
            for architecture in ('A0','A3','A5'):
                scenario=_scenario('FA4',seed['query_id'],transition,'S2',architecture,
                                   'STALE_CACHE')
                scenario.update(final_block='FA4',campaign_version=config['campaign_version'])
                add(scenario)
    for transition in ('SUPERSEDE','DELETE','GRANT','REVOKE'):
        by_domain=defaultdict(list)
        for seed in by_transition[transition]:
            by_domain[seed['query_id'].split('-')[2]].append(seed['query_id'])
        for domain in sorted(by_domain):
            for qid in sorted(set(by_domain[domain]))[:3]:
                for architecture in ('A0','A1','A2','A3','A4','A5'):
                    scenario=_scenario('FA5',qid,transition,'S2',architecture,'DELAY')
                    scenario.update(final_block='FA5',campaign_version=config['campaign_version'])
                    add(scenario)
    return added,audit


def freeze():
    if (ROOT/'campaign_manifest.json').exists():
        raise ValueError('Final campaign already frozen')
    frozen=json.loads(CONFIG.read_text())
    policy=json.loads(POLICY.read_text())
    if frozen['analysis_policy_version']!=policy['version']:
        raise ValueError('Policy version mismatch')
    source=json.loads(SOURCE.read_text())
    if digest(source['scenarios'])!=source['scientific_manifest_sha256']:
        raise ValueError('Source v4 manifest changed')
    config=_instrument_config()
    world,metadata,queries,_=_load_frozen(config)
    inherited=[{**s,'final_block':BLOCK_MAP[s['block']]} for s in source['scenarios']]
    added,audit=_expanded_scenarios(inherited,world,metadata,queries,config)
    scenarios=inherited+added
    if len({s['scenario_id'] for s in scenarios})!=len(scenarios):
        raise ValueError('Duplicate final scenario ID')
    manifest={'campaign_id':config['campaign_version'],'scenarios':scenarios,
        'scenario_sha256':digest(scenarios),'policy_sha256':_sha(POLICY),
        'source_v4_sha256':source['scientific_manifest_sha256'],
        'scenario_count':len(scenarios),
        'block_counts':dict(Counter(s['final_block'] for s in scenarios)),
        'FA2_analytic_view_of':'FA1'}
    _json(ROOT/'campaign_manifest.json',manifest)
    _csv(ROOT/'candidate_applicability_audit.csv',audit)
    _json(ROOT/'provenance.json',{'campaign_id':config['campaign_version'],
        'policy_sha256':_sha(POLICY),'config_sha256':_sha(CONFIG),
        'instrument_config_sha256':_sha(INSTRUMENT),
        'source_manifest_file_sha256':_sha(SOURCE),
        'source_v4_sha256':source['scientific_manifest_sha256'],
        'manifest_sha256':manifest['scenario_sha256'],
        'runtime_source_hashes':{path:_sha(path) for path in SOURCE_FILES},
        'corpus_sha256':config['corpus_sha256'],
        'cohort_v2_sha256':config['track_b_cohort_sha256']})
    _json(ROOT/'block_status.json',{block:{'status':'PENDING','completed':0,
        'expected':count} for block,count in manifest['block_counts'].items()} |
        {'FA2':{'status':'PENDING','analytic_view_of':'FA1'}})
    return {'manifest_sha256':manifest['scenario_sha256'],
            'policy_sha256':manifest['policy_sha256'],
            'scenario_count':len(scenarios),'block_counts':manifest['block_counts'],
            'expanded_candidates':len(audit),'expanded_included':len(added)}


def verify_freeze():
    manifest=json.loads((ROOT/'campaign_manifest.json').read_text())
    provenance=json.loads((ROOT/'provenance.json').read_text())
    for path,key in ((POLICY,'policy_sha256'),(CONFIG,'config_sha256'),
        (INSTRUMENT,'instrument_config_sha256'),(SOURCE,'source_manifest_file_sha256')):
        if _sha(path)!=provenance[key]:raise ValueError('Frozen input changed: '+str(path))
    if digest(manifest['scenarios'])!=manifest['scenario_sha256'] or \
       manifest['scenario_sha256']!=provenance['manifest_sha256']:
        raise ValueError('Final manifest changed')
    for path,expected in provenance['runtime_source_hashes'].items():
        if _sha(path)!=expected:raise ValueError('Shared runtime source changed: '+path)
    return _instrument_config(),manifest,provenance


def _sample(scenarios):
    predicates=[('UPDATE',lambda s:s['final_block']=='FA1' and s['transition']=='UPDATE'),
        ('SUPERSEDE',lambda s:s['final_block']=='FA1' and s['transition']=='SUPERSEDE'),
        ('DELETE',lambda s:s['final_block']=='FA1' and s['transition']=='DELETE'),
        ('GRANT',lambda s:s['final_block']=='FA1' and s['transition']=='GRANT'),
        ('REVOKE',lambda s:s['final_block']=='FA1' and s['transition']=='REVOKE'),
        ('cache',lambda s:s['final_block']=='FA4' and s['architecture']=='A5'),
        ('partial',lambda s:s.get('partial_topology')=='P_LOW'),
        ('out_of_order',lambda s:s['fault']=='OUT_OF_ORDER'),
        ('mitigation',lambda s:s['final_block']=='FA5' and s['architecture']=='A4'),
        ('compound',lambda s:s['final_block']=='FA7' and s.get('compound')=='CF4')]
    return [(name,min((s for s in scenarios if predicate(s)),
                     key=lambda s:s['scenario_id'])) for name,predicate in predicates]


def sample():
    config,manifest,_=verify_freeze()
    world,metadata,queries,_=_load_frozen(config)
    by_query={q.query_id:q for q in queries}
    result=[]
    for stratum,scenario in _sample(manifest['scenarios']):
        query=by_query[scenario['query_id']]
        rows,accesses,cache_ops,snapshots=execute_recorded(
            scenario,world,metadata,query,config)
        validity=validate_scenario(scenario,rows,accesses,cache_ops,
                                   snapshots,query,metadata,world,config)
        result.append({'stratum':stratum,'scenario_id':scenario['scenario_id'],
            'validity':validity['validity'],
            'fingerprint':deterministic_fingerprint(rows,accesses,cache_ops,validity),
            'forbidden':validity['forbidden_access_count'],
            'cache_violations':validity['cache_stage_violation_count']})
    return result


def preflight():
    config,manifest,_=verify_freeze()
    local=sample()
    env=dict(os.environ,PYTHONPATH=str(Path('src').resolve())+os.pathsep+
             os.environ.get('PYTHONPATH',''))
    fresh=subprocess.run([sys.executable,'-m',
        'ragstate.campaign.final_scientific_campaign_v1','sample'],
        capture_output=True,text=True,env=env)
    observed=json.loads(fresh.stdout) if fresh.returncode==0 else None
    gates={'manifest_provenance':True,
        'sample_runtime_valid':all(r['validity']=='valid' for r in local),
        'canonical_access':all(r['forbidden']==0 for r in local),
        'cache_stage':all(r['cache_violations']==0 for r in local),
        'fresh_process':observed==local}
    _json(ROOT/'preflight_gate_summary.json',{'gates':{
        k:'PASS' if v else 'FAIL' for k,v in gates.items()},
        'sample':local,'fresh_exit':fresh.returncode,'fresh_stderr':fresh.stderr})
    return {k:'PASS' if v else 'FAIL' for k,v in gates.items()}


def _checkpoint_paths(sid):
    return (ROOT/'scenario_rows'/f'{sid}.jsonl',
            ROOT/'validity_audits'/f'{sid}.json',
            ROOT/'scenario_checkpoints'/f'{sid}.json')


def _completed(scenario):
    rows_path,validity_path,checkpoint_path=_checkpoint_paths(scenario['scenario_id'])
    if not all(path.exists() for path in (rows_path,validity_path,checkpoint_path)):
        return None
    checkpoint=json.loads(checkpoint_path.read_text())
    if checkpoint['scenario_sha256']!=digest(scenario) or \
       checkpoint['rows_sha256']!=_sha(rows_path) or \
       checkpoint['validity_sha256']!=_sha(validity_path) or \
       checkpoint['observation_count']!=15 or \
       checkpoint['status'] not in ('valid','censored'):
        raise ValueError('Completed checkpoint integrity failed: '+scenario['scenario_id'])
    return checkpoint


def run():
    config,manifest,_=verify_freeze()
    pre=json.loads((ROOT/'preflight_gate_summary.json').read_text())
    if any(value!='PASS' for value in pre['gates'].values()):
        raise ValueError('Final preflight did not pass')
    world,metadata,queries,_=_load_frozen(config)
    by_query={q.query_id:q for q in queries}
    for folder in ('scenario_rows','validity_audits','scenario_checkpoints'):
        (ROOT/folder).mkdir(exist_ok=True)
    status=[]
    block_states={block:{'status':'PENDING','completed':0,'expected':count,
                         'invalid':0,'failed':0}
                  for block,count in manifest['block_counts'].items()}
    start=time.perf_counter()
    for index,scenario in enumerate(manifest['scenarios'],1):
        sid=scenario['scenario_id'];block=scenario['final_block']
        if block_states[block]['status']=='INVALID':
            status.append({'scenario_id':sid,'final_block':block,'status':'PENDING',
                           'reason':'block_invalid'});continue
        checkpoint=_completed(scenario)
        if checkpoint:
            state=checkpoint['status']
        else:
            rows_path,validity_path,checkpoint_path=_checkpoint_paths(sid)
            try:
                query=by_query[scenario['query_id']]
                rows,accesses,cache_ops,snapshots=execute_recorded(
                    scenario,world,metadata,query,config)
                validity=validate_scenario(scenario,rows,accesses,cache_ops,
                                           snapshots,query,metadata,world,config)
                if validity['validity']!='valid':
                    raise ValueError('invalid_scenario:'+validity['reasons'])
                if validity['forbidden_access_count'] or validity['cache_stage_violation_count'] \
                   or not validity['compiler_runtime_agreement']:
                    raise ValueError('shared_runtime_integrity_failure')
                for row in rows:row['final_block']=block
                temporary=rows_path.with_suffix('.jsonl.tmp')
                with temporary.open('w') as stream:
                    for row in rows:stream.write(json.dumps(row,sort_keys=True)+'\n')
                os.replace(temporary,rows_path)
                _json(validity_path,validity)
                state='censored' if validity['censored'] else 'valid'
                _json(checkpoint_path,{'scenario_id':sid,'scenario_sha256':digest(scenario),
                    'rows_sha256':_sha(rows_path),'validity_sha256':_sha(validity_path),
                    'observation_count':len(rows),'status':state,
                    'fingerprint':deterministic_fingerprint(rows,accesses,cache_ops,validity)})
            except Exception as exc:
                reason=f'{type(exc).__name__}:{exc}'
                block_states[block]['status']='INVALID'
                block_states[block]['invalid']+=1
                _json(ROOT/'block_status.json',block_states)
                _json(ROOT/'execution_error.json',{'scenario_id':sid,'block':block,
                    'reason':reason,'shared_runtime_defect':'shared_runtime_integrity_failure' in reason})
                status.append({'scenario_id':sid,'final_block':block,
                               'status':'INVALID','reason':reason})
                if 'shared_runtime_integrity_failure' in reason:
                    raise
                continue
        status.append({'scenario_id':sid,'final_block':block,'status':state,'reason':''})
        block_states[block]['completed']+=1
        if index%100==0:
            _json(ROOT/'block_status.json',block_states)
            print(f'Final Track A {index}/{len(manifest["scenarios"])}',flush=True)
    for block,item in block_states.items():
        if item['status']!='INVALID':
            item['status']='VALID' if item['completed']==item['expected'] else 'FAILED'
    block_states['FA2']={'status':block_states['FA1']['status'],
                         'analytic_view_of':'FA1'}
    _json(ROOT/'block_status.json',block_states)
    _csv(ROOT/'scenario_status.csv',status)
    if any(item['status']!='VALID' for key,item in block_states.items() if key!='FA2'):
        return {'status':'BLOCK_INVALID','block_states':block_states}
    rows_file=ROOT/'observations.jsonl';csv_file=ROOT/'observations.csv'
    first=True;count=0
    with rows_file.open('w') as json_stream,csv_file.open('w',newline='') as csv_stream:
        writer=None
        for scenario in manifest['scenarios']:
            path,_,_=_checkpoint_paths(scenario['scenario_id'])
            with path.open() as source:
                for line in source:
                    row=json.loads(line)
                    if first:
                        writer=csv.DictWriter(csv_stream,fieldnames=list(row))
                        writer.writeheader();first=False
                    writer.writerow({k:json.dumps(v,sort_keys=True)
                        if isinstance(v,(dict,list)) else v for k,v in row.items()})
                    json_stream.write(line);count+=1
    summary={'status':'VALID','scenario_count':len(manifest['scenarios']),
        'observation_count':count,'manifest_sha256':manifest['scenario_sha256'],
        'block_states':block_states,'runtime_seconds':time.perf_counter()-start,
        'converged':sum(r['status']=='valid' for r in status),
        'censored':sum(r['status']=='censored' for r in status)}
    _json(ROOT/'execution_summary.json',summary)
    return summary


def confirm_track_b():
    """Repeat only the frozen 120-case PA2 transfer set under final identity."""
    execution=json.loads((ROOT/'execution_summary.json').read_text())
    if execution['status']!='VALID':raise ValueError('Track A must be valid first')
    from .track_b_transfer_v2 import _record,verify_control as verify_tb
    from ragstate.retrieval.replica_pa2 import PA2ReplicaRetriever
    baseline,transfer,_,tb_provenance=verify_tb()
    config={**baseline['config'],'campaign_version':'final-scientific-campaign-v1-track-b'}
    world,metadata,queries,_=_load_frozen(config)
    by_query={q.query_id:q for q in queries}
    retriever=PA2ReplicaRetriever(config)
    folder=ROOT/'track_b_scenario_rows';folder.mkdir(exist_ok=True)
    statuses=[];summaries=[]
    for index,scenario in enumerate(transfer['scenarios'],1):
        path=folder/f'{scenario["scenario_id"]}.jsonl'
        if path.exists():
            records=[json.loads(line) for line in path.open()]
            if len(records)!=15 or any(r['scenario_id']!=scenario['scenario_id'] for r in records):
                raise ValueError('Track-B checkpoint corrupt')
        else:
            records,accesses,cache_ops,validity=_record(scenario,world,metadata,
                by_query[scenario['query_id']],config,retriever)
            if validity['validity']!='valid' or validity['forbidden_access_count'] or \
               validity['cache_stage_violation_count']:
                _json(ROOT/'track_b_invalid.json',{'scenario_id':scenario['scenario_id'],
                    'validity':validity})
                raise ValueError('Final Track-B confirmation invalid')
            temporary=path.with_suffix('.tmp')
            with temporary.open('w') as stream:
                for row in records:stream.write(json.dumps(row,sort_keys=True)+'\n')
            os.replace(temporary,path)
        at1=next(row for row in records if row['logical_time']==1)
        summaries.append({'scenario_id':scenario['scenario_id'],
            'block':scenario['block'],'transition':scenario['transition'],
            'architecture':scenario['architecture'],'compound':scenario.get('compound'),
            'query_id':scenario['query_id'],'domain':at1['domain'],
            'early_failure':1-at1['state_correct'],
            'ucer_context':at1['ucer_context'],
            'cache_hit':at1['cache_hit'],'t_rsc':records[-1]['t_rsc'],
            'censored':records[-1]['t_rsc'] is None,
            'median_retrieval_latency_ms':sorted(r['retrieval_latency_ms'] for r in records)[7]})
        statuses.append({'scenario_id':scenario['scenario_id'],'status':'VALID'})
        if index%20==0:print(f'Final Track B {index}/120',flush=True)
    _csv(ROOT/'trackB_confirmation.csv',summaries)
    _csv(ROOT/'track_b_scenario_status.csv',statuses)
    result={'scenario_count':len(summaries),'observation_count':15*len(summaries),
        'transfer_manifest_sha256':tb_provenance['transfer_manifest_sha256'],
        'all_valid':len(summaries)==120,'separate_population':True}
    _json(ROOT/'track_b_confirmation_summary.json',result)
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('freeze','sample','preflight','run','confirm-track-b'))
    action=parser.parse_args().action
    print(json.dumps({'freeze':freeze,'sample':sample,'preflight':preflight,
                      'run':run,'confirm-track-b':confirm_track_b}[action](),
                     indent=2,sort_keys=True))


if __name__=='__main__':main()
