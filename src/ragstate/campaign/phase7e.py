"""Frozen RA1 candidate ceiling and offline zero-shot reranker comparison."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

from ragstate.retrieval.metrics import (current_evidence_recall,
    invalid_evidence_absence_success,state_consistent_retrieval_success)
from ragstate.retrieval.models import RetrievalHit
from ragstate.retrieval.reranker import LocalCrossEncoder,rerank_candidates
from ragstate.world.scientific_v2 import DOMAINS
from .evidence_spans import mapping_for_queries
from .phase7a import PromptedEncoder
from .phase7a_figures import Canvas,_hist_plot
from .phase7b import _json,_mean,_quantile,_sha,_write_csv
from .phase7c import _load
from .phase7d import _index,_paired
from .runner import _git_commit
from ragstate.retrieval.dense import SentenceTransformerEncoder

CONFIGS=('RR0','RR1','RR2')


def _sources():
    root=Path(__file__).parent.parent
    return [Path(__file__),Path(__file__).parent/'evidence_spans.py',
            root/'retrieval'/'reranker.py',root/'retrieval'/'architecture.py',
            root/'retrieval'/'metrics.py',root/'retrieval'/'hybrid.py',
            root/'retrieval'/'dense.py']


def freeze(config_path,output):
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('Phase 7E output exists')
    config=json.loads(Path(config_path).read_text())
    frozen_path=Path(config['frozen_corpus_manifest'])
    frozen=json.loads(frozen_path.read_text())
    if config['corpus_version']!='v3' or config['architecture']!='RA1' or \
       config['chunk_words']!=160 or config['overlap_words']!=0 or \
       config['first_stage_candidate_k']!=30 or config['final_top_k']!=10 or \
       tuple(config['rerankers'])!=CONFIGS or config['hybrid']!=frozen['config']['hybrid'] or \
       config['embedding_revision']!=frozen['config']['embedding_revision']:
        raise ValueError('Frozen Phase 7E design mismatch')
    manifest={'phase':config['phase'],'config':config,'config_sha256':_sha(config_path),
              'v3_manifest_sha256':_sha(frozen_path),'v3_corpus_sha256':frozen['corpus_sha256'],
              'source_sha256':{p.name:_sha(p) for p in _sources()},
              'frozen_before_retrieval':True,'git_commit':_git_commit()}
    output.mkdir(parents=True)
    (output/'manifest.json').write_text(_json(manifest))
    return manifest


def _load_frozen(config_path,output):
    output=Path(output);manifest=json.loads((output/'manifest.json').read_text())
    config=manifest['config']
    if _sha(config_path)!=manifest['config_sha256']:
        raise ValueError('Predeclared Phase 7E config changed')
    frozen_path=Path(config['frozen_corpus_manifest'])
    if _sha(frozen_path)!=manifest['v3_manifest_sha256']:
        raise ValueError('Frozen v3 manifest changed')
    for path in _sources():
        if _sha(path)!=manifest['source_sha256'][path.name]:
            raise ValueError('Frozen retrieval source changed: '+path.name)
    frozen=json.loads(frozen_path.read_text())
    world,metadata,queries=_load(frozen_path.parent,frozen)
    return manifest,world,metadata,queries


def _oracle(queries,world,metadata,chunks):
    by_query=defaultdict(list)
    for row in mapping_for_queries(world,metadata,queries,chunks):
        by_query[row['query_id']].append(row)
    return by_query


def _evaluate(query,config_id,hits,world,claims,first_stage_ms,rerank_ms,
              candidate_hits):
    allowed=[h for h in hits if world.allowed(query.principal_id,h.chunk.logical_document_id,query.logical_time)]
    allowed_ids={h.chunk.chunk_id for h in allowed}
    gold=set(query.required_document_ids)
    present={h.chunk.document_id for h in allowed}
    claim_present=[bool(allowed_ids.intersection(claim['containing_chunk_ids'])) for claim in claims]
    def ranks(items):
        return {claim['document_id']:next((i for i,h in enumerate(items,1) if h.chunk.document_id==claim['document_id']),None)
                for claim in claims}
    def claim_ranks(items):
        return {claim['document_id']:next((i for i,h in enumerate(items,1) if h.chunk.chunk_id in claim['containing_chunk_ids']),None)
                for claim in claims}
    return {'configuration':config_id,'query_id':query.query_id,'domain':query.domain,
        'query_family':query.family,'principal_id':query.principal_id,
        'tenant_id':query.tenant_id,'answerable':int(query.answerable),
        'required_document_ids':json.dumps(sorted(gold)),
        'required_claim_chunk_ids':json.dumps({c['document_id']:c['containing_chunk_ids'] for c in claims},sort_keys=True),
        'candidate_chunk_ids':json.dumps([h.chunk.chunk_id for h in candidate_hits]),
        'context_chunk_ids':json.dumps([h.chunk.chunk_id for h in allowed]),
        'candidate_document_ranks':json.dumps(ranks(candidate_hits),sort_keys=True),
        'candidate_claim_ranks':json.dumps(claim_ranks(candidate_hits),sort_keys=True),
        'final_document_ranks':json.dumps(ranks(hits),sort_keys=True),
        'final_claim_ranks':json.dumps(claim_ranks(hits),sort_keys=True),
        'cer_at_10':current_evidence_recall(allowed,gold),
        'scrs':int(state_consistent_retrieval_success(allowed,gold,query.principal_id,world,query.logical_time)),
        'cbr_at_10':_mean(claim_present),
        'all_required_claim_success':int(all(claim_present)) if gold else None,
        'all_required_document_success':int(gold.issubset(present)) if gold else None,
        'negative_absence_success':invalid_evidence_absence_success(allowed,gold,set(query.logical_document_ids)),
        'first_stage_ms':first_stage_ms,'reranker_ms':rerank_ms,
        'total_ms':first_stage_ms+rerank_ms,'pairs_scored':len(candidate_hits) if config_id!='RR0' else 0}


def candidate_preflight(config_path,output):
    output=Path(output)
    if (output/'candidate_ceiling.csv').exists():raise ValueError('Candidate preflight already complete')
    manifest,world,metadata,queries=_load_frozen(config_path,output)
    config=manifest['config']
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    import torch
    torch.manual_seed(0);torch.set_num_threads(1)
    start=time.perf_counter()
    encoder=PromptedEncoder(SentenceTransformerEncoder(config['embedding_snapshot'],config['embedding_revision']),
                            config['embedding_query_prefix'],config['embedding_passage_prefix'])
    model_load_ms=(time.perf_counter()-start)*1000
    _,chunks,index,index_build_ms,index_bytes=_index('RA1',world,config,encoder)
    oracle=_oracle(queries,world,metadata,chunks)
    by_id={c.chunk_id:c for c in chunks}
    index['hybrid'].retrieve(queries[0].text,30)
    candidate_rows=[];ceiling_rows=[];rr0_rows=[]
    for query in queries:
        start=time.perf_counter()
        candidates=index['hybrid'].retrieve(query.text,config['first_stage_candidate_k'])
        latency=(time.perf_counter()-start)*1000
        claim_rows=oracle[query.query_id]
        gold=set(query.required_document_ids)
        present={h.chunk.document_id for h in candidates}
        ids={h.chunk.chunk_id for h in candidates}
        doc_success=int(gold.issubset(present)) if gold else None
        claim_success=int(all(ids.intersection(c['containing_chunk_ids']) for c in claim_rows)) if gold else None
        ceiling_rows.append({'query_id':query.query_id,'domain':query.domain,
            'query_family':query.family,'answerable':int(query.answerable),
            'required_count':len(gold),'candidate_count':len(candidates),
            'all_required_documents_at_30':doc_success,
            'all_required_claims_at_30':claim_success,
            'available_claim_spans':int(all(c['complete_span_available'] for c in claim_rows)) if gold else None,
            'required_document_ids':json.dumps(sorted(gold)),
            'candidate_document_ranks':json.dumps({c['document_id']:next((i for i,h in enumerate(candidates,1) if h.chunk.document_id==c['document_id']),None) for c in claim_rows},sort_keys=True),
            'candidate_claim_ranks':json.dumps({c['document_id']:next((i for i,h in enumerate(candidates,1) if h.chunk.chunk_id in c['containing_chunk_ids']),None) for c in claim_rows},sort_keys=True)})
        candidate_rows.append({'query_id':query.query_id,'first_stage_ms':latency,
            'chunk_ids':[h.chunk.chunk_id for h in candidates],
            'scores':[h.score for h in candidates],
            'components':[h.components for h in candidates]})
        rr0_rows.append(_evaluate(query,'RR0',candidates[:10],world,claim_rows,latency,0.0,candidates))
    candidates_file=output/'candidate_pool.jsonl'
    with candidates_file.open('w') as stream:
        for row in candidate_rows:stream.write(json.dumps(row,sort_keys=True)+'\n')
    _write_csv(output/'candidate_ceiling.csv',ceiling_rows)
    _write_csv(output/'rr0_baseline.csv',rr0_rows)
    answerable=[r for r in ceiling_rows if r['answerable']]
    scopes={'all_answerable':answerable,
            'multi_document':[r for r in answerable if r['query_family']=='multi_document'],
            'finance':[r for r in answerable if r['domain']=='finance'],
            'legal':[r for r in answerable if r['domain']=='legal'],
            'identifier_bearing':[r for r in answerable if r['query_family']=='identifier_bearing']}
    summary={'candidate_sha256':_sha(candidates_file),'candidate_k':30,'final_top_k':10,
        'chunk_count':len(chunks),'first_stage_model_load_ms':model_load_ms,
        'first_stage_index_build_ms':index_build_ms,
        'estimated_embedding_bytes':index_bytes,
        'ceiling_by_scope':{scope:{'n':len(group),
            'document_all_required':_mean([r['all_required_documents_at_30'] for r in group]),
            'claim_all_required':_mean([r['all_required_claims_at_30'] for r in group])}
            for scope,group in scopes.items()},
        'rr0_scrs':_mean([r['scrs'] for r in rr0_rows if r['answerable']]),
        'rr0_claim_success':_mean([r['all_required_claim_success'] for r in rr0_rows if r['answerable']]),
        'snapshot_status':'RR1/RR2 unavailable pending local paths'}
    (output/'candidate_summary.json').write_text(_json(summary))
    (output/'status.json').write_text(_json({'stage':'candidate_complete','missing_models':['RR1','RR2']}))
    return summary


def _summaries(rows):
    def group(field):
        grouped=defaultdict(list)
        for row in rows:
            if row['answerable']:grouped[(row['configuration'],row[field])].append(row)
        return [{'configuration':cfg,'scope':scope,'n_queries':len(group),
            'cer_at_10':_mean([r['cer_at_10'] for r in group]),
            'scrs':_mean([r['scrs'] for r in group]),
            'cbr_at_10':_mean([r['cbr_at_10'] for r in group]),
            'all_required_claim_success':_mean([r['all_required_claim_success'] for r in group]),
            'all_required_document_success':_mean([r['all_required_document_success'] for r in group])}
            for (cfg,scope),group in sorted(grouped.items())]
    return group('domain'),group('query_family')


def _figure_set(output,summary,domain_rows,transitions,latency,paired):
    folder=output/'figures';folder.mkdir()
    palette={'RR0':'#235789','RR1':'#c05621','RR2':'#2f855a'}
    def bars(name,title,categories,series):
        canvas=Canvas(title,'category','fraction')
        canvas.legend([(key,palette.get(key,'#6b46c1')) for key in series])
        step=650/max(1,len(categories));width=max(7,int(step/(len(series)+1)))
        for i,label in enumerate(categories):
            for j,(key,values) in enumerate(series.items()):
                x=int(85+i*step+j*width)
                canvas.rect(x,canvas.y(values[i]),x+width-2,430,palette.get(key,'#6b46c1'))
            canvas.label(int(85+i*step+step/2),450,str(label)[:12],size=9)
        canvas.save(folder/name)
    by={r['configuration']:r for r in summary}
    bars('scrs','SCRS by reranker',list(CONFIGS),{'RR0':[by[c]['scrs'] for c in CONFIGS]})
    bars('claim_success','All-required claim success',list(CONFIGS),{'RR1':[by[c]['all_required_claim_success'] for c in CONFIGS]})
    bars('multi_success','Multi-document claim success',list(CONFIGS),{'RR2':[by[c]['multi_claim_success'] for c in CONFIGS]})
    domains=[d.name for d in DOMAINS]
    bars('domain_scrs','Domain SCRS by reranker',domains,
         {c:[next(r['scrs'] for r in domain_rows if r['configuration']==c and r['scope']==d) for d in domains]
          for c in CONFIGS})
    # Candidate rank before/after is summarized by per-query rank transitions.
    rank_deltas=[]
    for row in transitions:
        before=json.loads(row['rr0_candidate_claim_ranks'])
        after=json.loads(row['rr1_final_claim_ranks'])
        for doc,rank in before.items():
            if rank is not None and after.get(doc) is not None:
                rank_deltas.append(after[doc]-rank)
    _hist_plot(folder,'candidate_rank_shift','RR1 claim rank change',rank_deltas)
    rescues=[sum(r[f'{c}_claim_outcome']=='rescued' for r in transitions) for c in ('RR1','RR2')]
    regressions=[sum(r[f'{c}_claim_outcome']=='regressed' for r in transitions) for c in ('RR1','RR2')]
    bars('rescue_regression','Claim rescue and regression',('RR1','RR2'),
         {'RR1':[v/len(transitions) for v in rescues],
          'RR2':[v/len(transitions) for v in regressions]})
    maxlat=max(r['median_total_ms'] for r in latency)
    canvas=Canvas('Latency versus claim success','median total latency (ms)','claim success')
    for c in CONFIGS:
        x=int(90+640*by[c]['median_total_ms']/maxlat)
        y=canvas.y(by[c]['all_required_claim_success'])
        canvas.rect(x-5,y-5,x+6,y+6,palette[c]);canvas.label(x,y-9,c)
    canvas.save(folder/'latency_vs_claim_success')
    bars('paired_improvement','Paired claim success difference vs RR0',('RR1','RR2'),
         {'RR1':[max(0,next(r['paired_difference'] for r in paired if r['configuration']==c and r['endpoint']=='all_required_claim_success')) for c in ('RR1','RR2')]})


def run(config_path,output,rr1_snapshot,rr2_snapshot):
    output=Path(output)
    if (output/'gate_summary.json').exists():raise ValueError('Completed Phase 7E output is immutable')
    manifest,world,metadata,queries=_load_frozen(config_path,output)
    config=manifest['config']
    candidate_summary=json.loads((output/'candidate_summary.json').read_text())
    if _sha(output/'candidate_pool.jsonl')!=candidate_summary['candidate_sha256']:
        raise ValueError('Frozen first-stage candidate pool changed')
    snapshots={'RR1':Path(rr1_snapshot),'RR2':Path(rr2_snapshot)}
    for cfg,path in snapshots.items():
        if not path.is_dir() or path.parent.name!='snapshots':
            raise ValueError(cfg+' local Hugging Face snapshot is unavailable')
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    import torch,sentence_transformers
    torch.manual_seed(0);torch.set_num_threads(1)
    from ragstate.retrieval.architecture import chunks_for_architecture
    _,chunks=chunks_for_architecture(world,9,'RA1')
    by_id={c.chunk_id:c for c in chunks}
    oracle=_oracle(queries,world,metadata,chunks)
    candidate_rows={r['query_id']:r for line in (output/'candidate_pool.jsonl').read_text().splitlines() for r in [json.loads(line)]}
    rr0=list(csv.DictReader((output/'rr0_baseline.csv').open()))
    # Preserve numeric/null semantics when reading the frozen baseline CSV.
    for r in rr0:
        for key in ('answerable','scrs','all_required_claim_success','all_required_document_success','negative_absence_success'):
            r[key]=None if r[key]=='' else int(r[key])
        for key in ('cer_at_10','cbr_at_10','first_stage_ms','reranker_ms','total_ms'):
            r[key]=None if r[key]=='' else float(r[key])
        r['pairs_scored']=0
    outputs={'RR0':rr0};model_meta={};rank_repeats={}
    for cfg,path in snapshots.items():
        revision=path.name
        start=time.perf_counter();scorer=LocalCrossEncoder(path,revision)
        load_ms=(time.perf_counter()-start)*1000
        size_bytes=sum(p.stat().st_size for p in path.rglob('*') if p.is_file())
        model_meta[cfg]={'model_id':config['rerankers'][cfg]['model'],'snapshot_path':str(path),
                         'revision':revision,'model_load_ms':load_ms,'snapshot_bytes':size_bytes}
        model_runs=[];ranking_runs=[]
        for repeat in range(2):
            rows=[];rankings={}
            for query in queries:
                frozen=candidate_rows[query.query_id]
                candidates=[RetrievalHit(by_id[cid],float(score)) for cid,score in zip(frozen['chunk_ids'],frozen['scores'])]
                start=time.perf_counter();hits=rerank_candidates(query.text,candidates,scorer,10)
                rerank_ms=(time.perf_counter()-start)*1000
                rankings[query.query_id]=[h.chunk.chunk_id for h in hits]
                rows.append(_evaluate(query,cfg,hits,world,oracle[query.query_id],
                                      frozen['first_stage_ms'],rerank_ms,candidates))
            model_runs.append(rows);ranking_runs.append(rankings)
        outputs[cfg]=model_runs[0]
        rank_repeats[cfg]=ranking_runs[0]==ranking_runs[1] and all(
            (a['cer_at_10'],a['scrs'],a['cbr_at_10'],a['all_required_claim_success'])==
            (b['cer_at_10'],b['scrs'],b['cbr_at_10'],b['all_required_claim_success'])
            for a,b in zip(*model_runs))
    rows=[r for cfg in CONFIGS for r in outputs[cfg]]
    claim_rows=[]
    for cfg in CONFIGS:
        for query,row in zip(queries,outputs[cfg]):
            final=set(json.loads(row['context_chunk_ids']))
            for claim in oracle[query.query_id]:
                claim_rows.append({'configuration':cfg,'query_id':query.query_id,
                    'domain':query.domain,'query_family':query.family,
                    'document_id':claim['document_id'],'version':claim['version'],
                    'span_sha256':claim['span_sha256'],
                    'containing_chunk_ids':json.dumps(claim['containing_chunk_ids']),
                    'retrieved_claim':int(bool(final.intersection(claim['containing_chunk_ids'])))})
    domain_rows,family_rows=_summaries(rows)
    summary=[];multi=[];latency=[]
    for cfg in CONFIGS:
        answerable=[r for r in outputs[cfg] if r['answerable']]
        negatives=[r for r in outputs[cfg] if not r['answerable']]
        multi_rows=[r for r in answerable if r['query_family']=='multi_document']
        all_rows=outputs[cfg]
        record={'configuration':cfg,'n_answerable':len(answerable),
            'cer_at_10':_mean([r['cer_at_10'] for r in answerable]),
            'scrs':_mean([r['scrs'] for r in answerable]),
            'cbr_at_10':_mean([r['cbr_at_10'] for r in answerable]),
            'all_required_claim_success':_mean([r['all_required_claim_success'] for r in answerable]),
            'multi_document_doc_success':_mean([r['all_required_document_success'] for r in multi_rows]),
            'multi_claim_success':_mean([r['all_required_claim_success'] for r in multi_rows]),
            'negative_absence_success':_mean([r['negative_absence_success'] for r in negatives]),
            'median_first_stage_ms':statistics.median(r['first_stage_ms'] for r in all_rows),
            'p95_first_stage_ms':_quantile([r['first_stage_ms'] for r in all_rows],.95),
            'median_reranker_ms':statistics.median(r['reranker_ms'] for r in all_rows),
            'p95_reranker_ms':_quantile([r['reranker_ms'] for r in all_rows],.95),
            'median_total_ms':statistics.median(r['total_ms'] for r in all_rows),
            'p95_total_ms':_quantile([r['total_ms'] for r in all_rows],.95),
            'median_pairs_scored':statistics.median(r['pairs_scored'] for r in all_rows),
            'model_load_ms':model_meta.get(cfg,{}).get('model_load_ms'),
            'snapshot_bytes':model_meta.get(cfg,{}).get('snapshot_bytes')}
        summary.append(record)
        latency.append({k:record[k] for k in ('configuration','median_first_stage_ms','p95_first_stage_ms',
            'median_reranker_ms','p95_reranker_ms','median_total_ms','p95_total_ms',
            'median_pairs_scored','model_load_ms','snapshot_bytes')})
        multi.append({k:record[k] for k in ('configuration','multi_document_doc_success','multi_claim_success')})
    by_cfg={cfg:{r['query_id']:r for r in outputs[cfg]} for cfg in CONFIGS}
    def outcome(before,after):
        if before and after:return 'preserved'
        if before and not after:return 'regressed'
        if not before and after:return 'rescued'
        return 'unchanged_failure'
    transitions=[];multi_analysis=[]
    for query in queries:
        base=by_cfg['RR0'][query.query_id]
        r1=by_cfg['RR1'][query.query_id];r2=by_cfg['RR2'][query.query_id]
        transitions.append({'query_id':query.query_id,'domain':query.domain,
            'query_family':query.family,'required_document_ids':base['required_document_ids'],
            'required_claim_chunk_ids':base['required_claim_chunk_ids'],
            'rr0_scrs':base['scrs'],'rr1_scrs':r1['scrs'],'rr2_scrs':r2['scrs'],
            'rr0_claim_success':base['all_required_claim_success'],
            'rr1_claim_success':r1['all_required_claim_success'],
            'rr2_claim_success':r2['all_required_claim_success'],
            'RR1_claim_outcome':outcome(base['all_required_claim_success'],r1['all_required_claim_success']) if query.answerable else 'negative',
            'RR2_claim_outcome':outcome(base['all_required_claim_success'],r2['all_required_claim_success']) if query.answerable else 'negative',
            'RR1_scrs_outcome':outcome(base['scrs'],r1['scrs']),
            'RR2_scrs_outcome':outcome(base['scrs'],r2['scrs']),
            'rr0_candidate_document_ranks':base['candidate_document_ranks'],
            'rr0_candidate_claim_ranks':base['candidate_claim_ranks'],
            'rr0_final_document_ranks':base['final_document_ranks'],
            'rr1_final_document_ranks':r1['final_document_ranks'],
            'rr2_final_document_ranks':r2['final_document_ranks'],
            'rr0_final_claim_ranks':base['final_claim_ranks'],
            'rr1_final_claim_ranks':r1['final_claim_ranks'],
            'rr2_final_claim_ranks':r2['final_claim_ranks']})
        if query.family=='multi_document':
            multi_analysis.append({'query_id':query.query_id,'domain':query.domain,
                'rr0_document_success':base['all_required_document_success'],
                'rr0_claim_success':base['all_required_claim_success'],
                'candidate_doc_success':int(all(v is not None for v in json.loads(base['candidate_document_ranks']).values())),
                'candidate_claim_success':int(all(v is not None for v in json.loads(base['candidate_claim_ranks']).values())),
                'required_document_ids':base['required_document_ids'],
                'required_claim_chunk_ids':base['required_claim_chunk_ids'],
                'candidate_document_ranks':base['candidate_document_ranks'],
                'candidate_claim_ranks':base['candidate_claim_ranks'],
                'rr0_final_document_ranks':base['final_document_ranks'],
                'rr1_final_document_ranks':r1['final_document_ranks'],
                'rr2_final_document_ranks':r2['final_document_ranks'],
                'rr0_final_claim_ranks':base['final_claim_ranks'],
                'rr1_final_claim_ranks':r1['final_claim_ranks'],
                'rr2_final_claim_ranks':r2['final_claim_ranks'],
                'rr1_doc_rescue':int(not base['all_required_document_success'] and r1['all_required_document_success']),
                'rr2_doc_rescue':int(not base['all_required_document_success'] and r2['all_required_document_success']),
                'rr1_claim_rescue':int(not base['all_required_claim_success'] and r1['all_required_claim_success']),
                'rr2_claim_rescue':int(not base['all_required_claim_success'] and r2['all_required_claim_success'])})
    paired=[];settings=config['paired_analysis']
    base=sorted((r for r in outputs['RR0'] if r['answerable']),key=lambda r:r['query_id'])
    for cfg in ('RR1','RR2'):
        other=sorted((r for r in outputs[cfg] if r['answerable']),key=lambda r:r['query_id'])
        if [r['query_id'] for r in base]!=[r['query_id'] for r in other]:raise ValueError('Paired query mismatch')
        for endpoint in ('scrs','all_required_claim_success'):
            paired.append({'configuration':cfg,'scope':'all_answerable','endpoint':endpoint,
                **_paired(base,other,endpoint,settings['bootstrap_seed'],settings['bootstrap_replicates'])})
        base_multi=[r for r in base if r['query_family']=='multi_document']
        other_multi=[r for r in other if r['query_family']=='multi_document']
        for endpoint in ('all_required_document_success','all_required_claim_success'):
            paired.append({'configuration':cfg,'scope':'multi_document','endpoint':endpoint,
                **_paired(base_multi,other_multi,endpoint,settings['bootstrap_seed'],settings['bootstrap_replicates'])})
    scores={}
    for cfg in CONFIGS:
        domain=[r for r in domain_rows if r['configuration']==cfg]
        rec=next(r for r in summary if r['configuration']==cfg)
        scores[cfg]={'macro_claim_success':_mean([r['all_required_claim_success'] for r in domain]),
            'multi_claim_success':rec['multi_claim_success'],
            'minimum_domain_scrs':min(r['scrs'] for r in domain),
            'overall_scrs':rec['scrs'],'median_total_ms':rec['median_total_ms']}
    selected=max(CONFIGS,key=lambda cfg:(scores[cfg]['macro_claim_success'],
        scores[cfg]['multi_claim_success'],scores[cfg]['minimum_domain_scrs'],
        scores[cfg]['overall_scrs'],-scores[cfg]['median_total_ms']))
    ceiling=candidate_summary['ceiling_by_scope']['all_answerable']['claim_all_required']
    thresholds=config['gates_frozen_before_execution']
    gates={}
    for cfg in CONFIGS:
        rec=next(r for r in summary if r['configuration']==cfg)
        gates[cfg]={
            'E1':'PASS' if rec['scrs']>=thresholds['E1']['overall_scrs_min'] else 'FAIL',
            'E2':'PASS' if scores[cfg]['minimum_domain_scrs']>=thresholds['E2']['minimum_domain_scrs_min'] else 'FAIL',
            'E3':'PASS' if rec['cbr_at_10']>=thresholds['E3']['cbr_at_10_min'] else 'FAIL',
            'E4':'PASS' if rec['multi_document_doc_success']>=thresholds['E4']['multi_document_doc_success_min'] else 'FAIL',
            'E5':'PASS' if rec['multi_claim_success']>=thresholds['E5']['multi_document_claim_success_min'] else 'FAIL',
            'E6':'PASS' if rec['negative_absence_success']>=thresholds['E6']['negative_absence_min'] else 'FAIL',
            'E7':'PASS' if cfg=='RR0' or rank_repeats[cfg] else 'FAIL',
            'E8':'PASS' if rec['scrs']>=next(r['scrs'] for r in summary if r['configuration']=='RR0')-thresholds['E8']['overall_scrs_max_drop_vs_rr0'] and rec['negative_absence_success']>=thresholds['E8']['negative_absence_min'] else 'FAIL',
            'E9':'PASS' if ceiling>=thresholds['E9']['candidate_all_required_claim_ceiling_min'] else 'FAIL'}
    gate_summary={'selected_configuration':selected,'selection_scores':scores,
        'gates_by_configuration':gates,'selected_passes_all_gates':all(x=='PASS' for x in gates[selected].values()),
        'candidate_ceiling':candidate_summary['ceiling_by_scope'],
        'repeated_rankings_identical':rank_repeats,'models':model_meta,
        'environment':{'python':sys.version.split()[0],'platform':platform.platform(),
            'torch':torch.__version__,'sentence_transformers':sentence_transformers.__version__,
            'device':'cpu','git_commit':_git_commit()},'fault_campaign_executed':False}
    _write_csv(output/'architecture_summary.csv',summary)
    _write_csv(output/'per_query_results.csv',rows)
    _write_csv(output/'claim_results.csv',claim_rows)
    _write_csv(output/'multi_document_analysis.csv',multi_analysis)
    _write_csv(output/'domain_summary.csv',domain_rows)
    _write_csv(output/'query_family_summary.csv',family_rows)
    _write_csv(output/'reranker_transitions.csv',transitions)
    _write_csv(output/'latency_summary.csv',latency)
    _write_csv(output/'statistical_comparisons.csv',paired)
    (output/'model_resolution.json').write_text(_json(model_meta))
    (output/'gate_summary.json').write_text(_json(gate_summary))
    _figure_set(output,summary,domain_rows,transitions,latency,paired)
    lines=['# Phase 7E offline reranker validation','',
        f"Frozen v3 corpus hash: `{manifest['v3_corpus_sha256']}`. Frozen candidate pool SHA-256: `{candidate_summary['candidate_sha256']}`.",'',
        '## Candidate ceiling','',
        *[f"- {scope}: document {v['document_all_required']:.3f}; complete-claim {v['claim_all_required']:.3f} (n={v['n']})."
          for scope,v in candidate_summary['ceiling_by_scope'].items()],
        '', '## Clean results','',
        '| Configuration | SCRS | CBR@10 | Claim success | Multi-doc | Multi-claim | Median total ms |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: |',
        *[f"| {r['configuration']} | {r['scrs']:.3f} | {r['cbr_at_10']:.3f} | "
          f"{r['all_required_claim_success']:.3f} | {r['multi_document_doc_success']:.3f} | "
          f"{r['multi_claim_success']:.3f} | {r['median_total_ms']:.2f} |" for r in summary],
        '',f'Selected by the predeclared rule: `{selected}`.','',
        '## Gates','',*[f"- {cfg}: "+', '.join(f'{k} {v}' for k,v in gates[cfg].items()) for cfg in CONFIGS],
        '', 'No transition faults, fine-tuning, or LLM generation were performed.']
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    (output/'status.json').write_text(_json({'stage':'complete','selected':selected}))
    return gate_summary


def main():
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=('freeze','candidates','run'))
    parser.add_argument('--config',default='configs/phase7e-reranker-v1.json')
    parser.add_argument('--output',default='results/phase7e-reranker-v1')
    parser.add_argument('--rr1-snapshot');parser.add_argument('--rr2-snapshot')
    args=parser.parse_args()
    if args.stage=='freeze':result=freeze(Path(args.config),Path(args.output))
    elif args.stage=='candidates':result=candidate_preflight(Path(args.config),Path(args.output))
    else:
        if not args.rr1_snapshot or not args.rr2_snapshot:
            raise ValueError('Both local reranker snapshots are required')
        result=run(Path(args.config),Path(args.output),args.rr1_snapshot,args.rr2_snapshot)
    print(_json({'stage':args.stage,'result':result if args.stage!='freeze' else {'manifest_created':True}}))


if __name__=='__main__':main()
