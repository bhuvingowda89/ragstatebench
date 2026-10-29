"""Frozen v3 clean retrieval-architecture comparison, without transition faults."""
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
from collections import Counter, defaultdict
from pathlib import Path

from ragstate.retrieval.architecture import HierarchicalRetriever, chunks_for_architecture, retrieval_bundle
from ragstate.retrieval.metrics import (current_evidence_recall,
    invalid_evidence_absence_success, state_consistent_retrieval_success)
from ragstate.world.scientific_v2 import DOMAINS
from .evidence_spans import mapping_for_queries, numeric_pair_boundary
from .phase7a import PromptedEncoder
from .phase7a_figures import Canvas, _hist_plot
from .phase7b import _json, _mean, _quantile, _sha, _write_csv
from .phase7c import _load
from .runner import _git_commit
from ragstate.retrieval.dense import SentenceTransformerEncoder

ARCHITECTURES=('RA0','RA1','RA2','RA3')
METHODS=('bm25','e5','hybrid')


def _source_paths():
    root=Path(__file__).parent.parent
    return [Path(__file__),Path(__file__).parent/'evidence_spans.py',
            root/'retrieval'/'architecture.py',root/'retrieval'/'chunking.py',
            root/'retrieval'/'bm25.py',root/'retrieval'/'dense.py',
            root/'retrieval'/'hybrid.py',root/'retrieval'/'metrics.py']


def freeze(config_path,output):
    output=Path(output)
    if output.exists() and any(output.iterdir()):raise ValueError('Phase 7D output exists; refusing overwrite')
    config=json.loads(Path(config_path).read_text())
    source=Path(config['frozen_v3_data_manifest'])
    corpus_manifest=json.loads(source.read_text())
    if config['corpus_version']!='v3' or config['top_k']!=10 or tuple(config['retrievers'])!=METHODS or \
       tuple(config['architectures'])!=ARCHITECTURES or config['architectures']['RA0']['chunk_words']!=80 or \
       config['architectures']['RA1']['chunk_words']!=160 or config['architectures']['RA2']['maximum_words']!=200 or \
       config['architectures']['RA3']['document_top_n']!=8:
        raise ValueError('Frozen Phase 7D design mismatch')
    if config['embedding_revision']!=corpus_manifest['config']['embedding_revision'] or \
       config['hybrid']!=corpus_manifest['config']['hybrid']:
        raise ValueError('Model/fusion differs from frozen v3 baseline')
    manifest={'phase':'phase7d-retrieval-architecture-v1','config':config,
              'config_sha256':_sha(config_path),'v3_corpus_sha256':corpus_manifest['corpus_sha256'],
              'v3_manifest_sha256':_sha(source),
              'source_sha256':{p.name:_sha(p) for p in _source_paths()},
              'frozen_before_retrieval':True,'git_commit':_git_commit()}
    output.mkdir(parents=True)
    (output/'manifest.json').write_text(_json(manifest))
    return manifest


def _index(arch,world,config,encoder):
    docs,chunks=chunks_for_architecture(world,9,arch)
    start=time.perf_counter()
    if arch=='RA3':
        index=HierarchicalRetriever(docs,chunks,encoder,config['embedding_model'],
                                    config['embedding_revision'],config['hybrid'],8)
        vectors=index.vector_count
    else:
        index=retrieval_bundle(chunks,encoder,config['embedding_model'],
                               config['embedding_revision'],config['hybrid'])
        vectors=len(index['e5'].vectors)
    build_ms=(time.perf_counter()-start)*1000
    dimensions=384 if not chunks else len(index.chunk_indices['e5'].vectors[0] if arch=='RA3' else index['e5'].vectors[0])
    return docs,chunks,index,build_ms,vectors*dimensions*4


def _run_query(index,arch,method,text):
    start=time.perf_counter()
    if arch=='RA3':
        hits=index.retrieve(text,method,10)
        document_ms=index.last_document_ms;chunk_ms=index.last_chunk_ms
        selected_document_ids=index.last_document_ids
    else:
        hits=index[method].retrieve(text,10)
        document_ms=None;chunk_ms=None;selected_document_ids=()
    return hits,(time.perf_counter()-start)*1000,document_ms,chunk_ms,selected_document_ids


def _evaluate_arch(arch,world,queries,oracle,index,timed=True):
    if timed:
        for method in METHODS:_run_query(index,arch,method,queries[0].text)
    rows=[];claim_rows=[];rankings={}
    for q in queries:
        rankings[q.query_id]={}
        required=set(q.required_document_ids)
        claim_oracle=oracle[q.query_id]
        for method in METHODS:
            hits,latency,document_ms,chunk_ms,doc_stage=_run_query(index,arch,method,q.text)
            context=[h for h in hits if world.allowed(q.principal_id,h.chunk.logical_document_id,q.logical_time)]
            context_ids={h.chunk.chunk_id for h in context}
            got={h.chunk.document_id for h in context}
            retrieved_claims=[]
            for claim in claim_oracle:
                present=bool(context_ids.intersection(claim['containing_chunk_ids']))
                retrieved_claims.append(present)
                claim_rows.append({'architecture':arch,'retriever':method,'query_id':q.query_id,
                    'domain':q.domain,'query_family':q.family,'document_id':claim['document_id'],
                    'version':claim['version'],'span_sha256':claim['span_sha256'],
                    'span_start_char':claim['start_char'],'span_end_char':claim['end_char'],
                    'available_chunk_ids':json.dumps(claim['containing_chunk_ids']),
                    'span_chunk_available':int(claim['complete_span_available']),
                    'retrieved_claim':int(present)})
            rankings[q.query_id][method]=[h.chunk.chunk_id for h in hits]
            rows.append({'architecture':arch,'retriever':method,'query_id':q.query_id,
                'domain':q.domain,'query_family':q.family,'principal_id':q.principal_id,
                'tenant_id':q.tenant_id,'answerable':int(q.answerable),
                'required_document_ids':json.dumps(sorted(required)),
                'candidate_chunk_ids':json.dumps([h.chunk.chunk_id for h in hits]),
                'context_chunk_ids':json.dumps([h.chunk.chunk_id for h in context]),
                'document_stage_ids':json.dumps(doc_stage),
                'cer_at_10':current_evidence_recall(context,required),
                'cbr_at_10':_mean(retrieved_claims),
                'claim_bearing_success':int(all(retrieved_claims)) if required else None,
                'document_all_required_success':int(required.issubset(got)) if required else None,
                'scrs':int(state_consistent_retrieval_success(context,required,q.principal_id,world,q.logical_time)),
                'negative_absence_success':invalid_evidence_absence_success(context,required,set(q.logical_document_ids)),
                'latency_ms':latency if timed else None,
                'document_stage_ms':document_ms if timed else None,
                'chunk_stage_ms':chunk_ms if timed else None})
    return rows,claim_rows,rankings


def _group(rows,field):
    grouped=defaultdict(list)
    for row in rows:
        if not row['answerable']:continue
        grouped[(row['architecture'],row['retriever'],row[field])].append(row)
    return [{'architecture':arch,'retriever':method,'scope':scope,'n_queries':len(group),
        'cer_at_10':_mean([r['cer_at_10'] for r in group]),
        'scrs':_mean([r['scrs'] for r in group]),
        'cbr_at_10':_mean([r['cbr_at_10'] for r in group]),
        'claim_bearing_success':_mean([r['claim_bearing_success'] for r in group]),
        'document_all_required_success':_mean([r['document_all_required_success'] for r in group])}
        for (arch,method,scope),group in sorted(grouped.items())]


def _paired(base,other,field,seed,reps):
    a=[int(x[field]) for x in base];b=[int(x[field]) for x in other]
    if len(a)!=len(b):raise ValueError('Unmatched paired observations')
    difference=_mean([y-x for x,y in zip(a,b)])
    rng=random.Random(seed)
    deltas=[]
    for _ in range(reps):
        sample=[rng.randrange(len(a)) for _ in a]
        deltas.append(sum(b[i]-a[i] for i in sample)/len(a))
    n01=sum(x==0 and y==1 for x,y in zip(a,b))
    n10=sum(x==1 and y==0 for x,y in zip(a,b))
    discordant=n01+n10
    if discordant:
        tail=sum(math.comb(discordant,k) for k in range(min(n01,n10)+1))/2**discordant
        p=min(1.0,2*tail)
    else:p=1.0
    return {'paired_difference':difference,'ci95_low':_quantile(deltas,.025),
            'ci95_high':_quantile(deltas,.975),'improved_pairs':n01,
            'worsened_pairs':n10,'exact_mcnemar_p':p,'n_pairs':len(a)}


def _figures(output,summary,domain_rows,family_rows,chunk_rows,statistics_rows):
    folder=output/'figures';folder.mkdir()
    palette={'RA0':'#235789','RA1':'#c05621','RA2':'#2f855a','RA3':'#6b46c1'}
    def bars(name,title,categories,series):
        c=Canvas(title,'architecture or domain','fraction')
        c.legend([(label,palette.get(label,'#235789')) for label in series])
        step=650/max(1,len(categories));width=max(7,int(step/(len(series)+1)))
        for i,label in enumerate(categories):
            for j,(key,values) in enumerate(series.items()):
                x=int(85+i*step+j*width)
                c.rect(x,c.y(values[i]),x+width-2,430,palette.get(key,'#235789'))
            c.label(int(85+i*step+step/2),450,str(label)[:12],size=9)
        c.save(folder/name)
    hybrid={r['architecture']:r for r in summary if r['retriever']=='hybrid'}
    bars('overall_scrs','Hybrid SCRS by architecture',list(ARCHITECTURES),
         {'RA0':[hybrid[a]['scrs'] for a in ARCHITECTURES]})
    bars('claim_bearing_success','Hybrid claim-bearing success',list(ARCHITECTURES),
         {'RA2':[hybrid[a]['claim_bearing_success'] for a in ARCHITECTURES]})
    bars('multi_claim_success','Hybrid multi-document claim success',list(ARCHITECTURES),
         {'RA3':[hybrid[a]['multi_claim_success'] for a in ARCHITECTURES]})
    domains=[d.name for d in DOMAINS]
    bars('domain_hybrid_scrs','Hybrid SCRS across domains',domains,
         {a:[next(r['scrs'] for r in domain_rows if r['architecture']==a and
                 r['retriever']=='hybrid' and r['scope']==d) for d in domains]
            for a in ARCHITECTURES})
    position=['beginning','middle','end']
    bars('evidence_position_recall','Claim recall by declared position',position,
         {a:[next((r['retrieved']/r['count'] for r in family_rows if r['architecture']==a and
                   r['retriever']=='hybrid' and r['scope']==p),0) for p in position]
            for a in ARCHITECTURES})
    # Normalize latency to the chart's 0..1 y-axis; exact times are in latency_summary.csv.
    max_latency=max(hybrid[a]['median_latency_ms'] for a in ARCHITECTURES)
    c=Canvas('Latency versus hybrid claim success','median latency (ms)','claim success')
    for a in ARCHITECTURES:
        x=int(90+640*hybrid[a]['median_latency_ms']/max_latency)
        y=c.y(hybrid[a]['claim_bearing_success'])
        c.rect(x-5,y-5,x+6,y+6,palette[a]);c.label(x,y-9,a)
    c.save(folder/'latency_vs_claim_success')
    _hist_plot(folder,'chunk_count','Chunks per document across architectures',
               [r['chunk_count'] for r in chunk_rows])
    paired=[r for r in statistics_rows if r['retriever']=='hybrid' and r['endpoint']=='claim_bearing_success']
    bars('paired_improvement','Paired hybrid claim improvement vs RA0',
         [r['architecture'] for r in paired],
         {'RA1':[max(0,r['paired_difference']) for r in paired]})


def run(config_path,output):
    output=Path(output)
    manifest=json.loads((output/'manifest.json').read_text())
    if (output/'gate_summary.json').exists():raise ValueError('Completed Phase 7D output is immutable')
    config=manifest['config']
    if _sha(config_path)!=manifest['config_sha256']:
        raise ValueError('Predeclared config changed')
    source=Path(config['frozen_v3_data_manifest'])
    if _sha(source)!=manifest['v3_manifest_sha256']:
        raise ValueError('Frozen v3 manifest changed')
    for path in _source_paths():
        if _sha(path)!=manifest['source_sha256'][path.name]:
            raise ValueError('Frozen retrieval implementation changed: '+path.name)
    frozen=json.loads(source.read_text())
    if frozen['corpus_sha256']!=manifest['v3_corpus_sha256']:
        raise ValueError('Frozen v3 corpus changed')
    world,metadata,queries=_load(source.parent,frozen)
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    import torch,sentence_transformers
    torch.manual_seed(0);torch.set_num_threads(1)
    encoder=PromptedEncoder(SentenceTransformerEncoder(config['embedding_snapshot'],
                            config['embedding_revision']),
                            config['embedding_query_prefix'],config['embedding_passage_prefix'])
    all_rows=[];all_claims=[];all_chunks=[];all_boundaries=[];builds={};repeated_ok={}
    evidence_positions={lid:m['evidence_position'] for lid,m in metadata.items()}
    for arch in ARCHITECTURES:
        docs,chunks,index,build_ms,index_bytes=_index(arch,world,config,encoder)
        by_doc=defaultdict(list)
        for chunk in chunks:by_doc[chunk.document_id].append(chunk)
        oracle_rows=mapping_for_queries(world,metadata,queries,chunks)
        oracle=defaultdict(list)
        for row in oracle_rows:
            oracle[row['query_id']].append(row)
            claim=next(c for q in queries if q.query_id==row['query_id'] for c in q.required_atomic_claims
                       if c['document_id']==row['document_id'])
            document=world.documents[row['document_id']]
            all_boundaries.append({'architecture':arch,'query_id':row['query_id'],
                'document_id':row['document_id'],'version':row['version'],
                'numeric_pair_crosses':int(numeric_pair_boundary(document,by_doc[row['document_id']],
                                              claim['value'],claim['unit'])),
                'complete_span_available':int(row['complete_span_available']),
                'containing_chunk_ids':json.dumps(row['containing_chunk_ids'])})
        for d in docs:
            all_chunks.append({'architecture':arch,'document_id':d.document_id,
                'logical_document_id':d.logical_document_id,'word_count':len(d.content.split()),
                'chunk_count':len(by_doc[d.document_id]),
                'maximum_chunk_words':max(len(c.text.split()) for c in by_doc[d.document_id])})
        rows,claim_rows,ranks=_evaluate_arch(arch,world,queries,oracle,index)
        repeated,_,ranks2=_evaluate_arch(arch,world,queries,oracle,index,timed=False)
        repeated_ok[arch]=ranks==ranks2 and all((a['scrs'],a['cer_at_10'],a['cbr_at_10'],a['claim_bearing_success'])==
            (b['scrs'],b['cer_at_10'],b['cbr_at_10'],b['claim_bearing_success'])
            for a,b in zip(rows,repeated))
        all_rows.extend(rows);all_claims.extend(claim_rows)
        builds[arch]={'chunk_count':len(chunks),'documents':len(docs),
                      'average_chunks_per_document':len(chunks)/len(docs),
                      'index_build_ms':build_ms,'estimated_vector_bytes':index_bytes,
                      'repeated_rankings_identical':repeated_ok[arch]}
    domain_rows=_group(all_rows,'domain')
    family_rows=_group(all_rows,'query_family')
    architecture_rows=[];latency_rows=[];multi_rows=[]
    for arch in ARCHITECTURES:
        for method in METHODS:
            group=[r for r in all_rows if r['architecture']==arch and r['retriever']==method and r['answerable']]
            negative=[r for r in all_rows if r['architecture']==arch and r['retriever']==method and not r['answerable']]
            multi=[r for r in group if r['query_family']=='multi_document']
            latency=[r['latency_ms'] for r in all_rows if r['architecture']==arch and r['retriever']==method]
            doc_lat=[r['document_stage_ms'] for r in all_rows if r['architecture']==arch and r['retriever']==method and r['document_stage_ms'] is not None]
            chunk_lat=[r['chunk_stage_ms'] for r in all_rows if r['architecture']==arch and r['retriever']==method and r['chunk_stage_ms'] is not None]
            base={'architecture':arch,'retriever':method,'n_answerable':len(group),
                'cer_at_10':_mean([r['cer_at_10'] for r in group]),
                'scrs':_mean([r['scrs'] for r in group]),
                'cbr_at_10':_mean([r['cbr_at_10'] for r in group]),
                'claim_bearing_success':_mean([r['claim_bearing_success'] for r in group]),
                'multi_document_success':_mean([r['document_all_required_success'] for r in multi]),
                'multi_claim_success':_mean([r['claim_bearing_success'] for r in multi]),
                'negative_absence_success':_mean([r['negative_absence_success'] for r in negative]),
                'chunk_count':builds[arch]['chunk_count'],
                'average_chunks_per_document':builds[arch]['average_chunks_per_document'],
                'index_build_ms':builds[arch]['index_build_ms'],
                'estimated_vector_bytes':builds[arch]['estimated_vector_bytes'],
                'median_latency_ms':statistics.median(latency),
                'p95_latency_ms':_quantile(latency,.95),
                'median_document_stage_ms':statistics.median(doc_lat) if doc_lat else None,
                'median_chunk_stage_ms':statistics.median(chunk_lat) if chunk_lat else None}
            architecture_rows.append(base)
            latency_rows.append({k:base[k] for k in ('architecture','retriever','index_build_ms',
                'estimated_vector_bytes','median_latency_ms','p95_latency_ms',
                'median_document_stage_ms','median_chunk_stage_ms')})
            multi_rows.append({k:base[k] for k in ('architecture','retriever','multi_document_success','multi_claim_success')})
    # Paired comparisons use the same 204 answerable queries in a stable order.
    comparisons=[];paired_config=config['paired_analysis']
    for method in METHODS:
        base=sorted((r for r in all_rows if r['architecture']=='RA0' and r['retriever']==method and r['answerable']),key=lambda r:r['query_id'])
        for arch in ARCHITECTURES[1:]:
            other=sorted((r for r in all_rows if r['architecture']==arch and r['retriever']==method and r['answerable']),key=lambda r:r['query_id'])
            if [r['query_id'] for r in base]!=[r['query_id'] for r in other]:raise ValueError('Paired query mismatch')
            for field in ('scrs','claim_bearing_success'):
                comparisons.append({'architecture':arch,'retriever':method,'endpoint':field,
                    **_paired(base,other,field,paired_config['bootstrap_seed'],paired_config['bootstrap_replicates'])})
    # Claim recall by declared source-position is a diagnostic, not a gate.
    claim_position=[]
    for arch in ARCHITECTURES:
        for method in METHODS:
            for position in ('beginning','middle','end'):
                group=[r for r in all_claims if r['architecture']==arch and r['retriever']==method and
                       evidence_positions[world.documents[r['document_id']].logical_document_id]==position]
                claim_position.append({'architecture':arch,'retriever':method,'scope':position,
                    'count':len(group),'retrieved':sum(r['retrieved_claim'] for r in group),
                    'claim_recall':_mean([r['retrieved_claim'] for r in group])})
    hybrid={r['architecture']:r for r in architecture_rows if r['retriever']=='hybrid'}
    thresholds=config['gates_frozen_before_execution']
    gates={};scores={}
    for arch in ARCHITECTURES:
        h=hybrid[arch]
        domains=[r for r in domain_rows if r['architecture']==arch and r['retriever']=='hybrid']
        minimum=min(r['scrs'] for r in domains)
        macro_claim=_mean([r['claim_bearing_success'] for r in domains])
        scores[arch]={'macro_hybrid_claim_success':macro_claim,
                      'multi_claim_success':h['multi_claim_success'],
                      'min_domain_scrs':minimum,
                      'median_latency_ms':h['median_latency_ms'],
                      'estimated_vector_bytes':h['estimated_vector_bytes']}
        gates[arch]={
            'A1':'PASS' if h['scrs']>=thresholds['A1']['hybrid_overall_scrs_min'] else 'FAIL',
            'A2':'PASS' if minimum>=thresholds['A2']['hybrid_min_domain_scrs_min'] else 'FAIL',
            'A3':'PASS' if h['cbr_at_10']>=thresholds['A3']['hybrid_claim_bearing_recall_min'] else 'FAIL',
            'A4':'PASS' if h['multi_document_success']>=thresholds['A4']['hybrid_multi_document_doc_success_min'] else 'FAIL',
            'A5':'PASS' if h['multi_claim_success']>=thresholds['A5']['hybrid_multi_document_claim_success_min'] else 'FAIL',
            'A6':'PASS' if h['negative_absence_success']>=thresholds['A6']['negative_absence_success_min'] else 'FAIL',
            'A7':'PASS' if repeated_ok[arch] else 'FAIL'}
    selected=max(ARCHITECTURES,key=lambda a:(scores[a]['macro_hybrid_claim_success'],
        scores[a]['multi_claim_success'],scores[a]['min_domain_scrs'],
        -scores[a]['median_latency_ms'],-scores[a]['estimated_vector_bytes']))
    gate_summary={'phase':'phase7d-retrieval-architecture-v1',
        'v3_corpus_sha256':manifest['v3_corpus_sha256'],
        'gates_by_architecture':gates,'selection_scores':scores,
        'selected_architecture':selected,'selected_passes_all_gates':all(v=='PASS' for v in gates[selected].values()),
        'repeated_rankings_identical':repeated_ok,
        'environment':{'python':sys.version.split()[0],'platform':platform.platform(),
            'torch':torch.__version__,'sentence_transformers':sentence_transformers.__version__,
            'device':'cpu','embedding_model':config['embedding_model'],
            'revision':config['embedding_revision'],'git_commit':_git_commit()},
        'fault_campaign_executed':False}
    _write_csv(output/'architecture_summary.csv',architecture_rows)
    _write_csv(output/'per_query_results.csv',all_rows)
    _write_csv(output/'claim_bearing_results.csv',all_claims)
    _write_csv(output/'domain_summary.csv',domain_rows)
    _write_csv(output/'query_family_summary.csv',family_rows)
    _write_csv(output/'multi_document_summary.csv',multi_rows)
    _write_csv(output/'chunk_statistics.csv',all_chunks)
    _write_csv(output/'boundary_claims.csv',all_boundaries)
    _write_csv(output/'latency_summary.csv',latency_rows)
    _write_csv(output/'statistical_comparisons.csv',comparisons)
    _write_csv(output/'claim_position_recall.csv',claim_position)
    (output/'gate_summary.json').write_text(_json(gate_summary))
    _figures(output,architecture_rows,domain_rows,claim_position,all_chunks,comparisons)
    lines=['# Phase 7D retrieval architecture comparison','',
        f"Frozen v3 corpus hash: `{manifest['v3_corpus_sha256']}`. No transition faults or generation run.",'',
        '## Hybrid clean outcomes','',
        '| Architecture | Chunks | CER@10 | SCRS | CBR@10 | Claim success | Multi-doc success | Multi-claim success | Median ms |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |',
        *[f"| {a} | {hybrid[a]['chunk_count']} | {hybrid[a]['cer_at_10']:.3f} | {hybrid[a]['scrs']:.3f} | "
          f"{hybrid[a]['cbr_at_10']:.3f} | {hybrid[a]['claim_bearing_success']:.3f} | "
          f"{hybrid[a]['multi_document_success']:.3f} | {hybrid[a]['multi_claim_success']:.3f} | "
          f"{hybrid[a]['median_latency_ms']:.2f} |" for a in ARCHITECTURES],
        '', '## Frozen selection','',f'Selected: `{selected}` by macro hybrid claim-bearing success and predeclared tie-breakers.',
        '', '## Gates','',*[f"- {a}: "+', '.join(f'{k} {v}' for k,v in gates[a].items()) for a in ARCHITECTURES],
        '', 'Document-level CER/SCRS retain frozen definitions. Claim-bearing recall uses a complete canonical sentence span and is reported separately.']
    (output/'summary.md').write_text('\n'.join(lines)+'\n')
    return gate_summary


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=('freeze','run'))
    p.add_argument('--config',default='configs/phase7d-architecture-v1.json')
    p.add_argument('--output',default='results/phase7d-retrieval-architecture-v1')
    args=p.parse_args()
    result=(freeze if args.stage=='freeze' else run)(Path(args.config),Path(args.output))
    print(_json({'stage':args.stage,'selected':result.get('selected_architecture'),
                 'gates':result.get('gates_by_architecture')}))


if __name__=='__main__':main()
