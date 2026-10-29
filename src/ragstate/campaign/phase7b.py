"""Offline v2 corpus freeze and clean retrieval validation; no transition faults."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import platform
import statistics
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from ragstate.retrieval.bm25 import BM25Retriever, tokenize
from ragstate.retrieval.dense import DenseRetriever, SentenceTransformerEncoder
from ragstate.retrieval.hybrid import HybridRetriever
from ragstate.retrieval.indexing import build_chunks_from_ledger
from ragstate.retrieval.metrics import (current_evidence_recall,
    invalid_evidence_absence_success, state_consistent_retrieval_success)
from ragstate.world.ledger import WorldLedger
from ragstate.world.models import DocumentVersion, Principal
from ragstate.world.scientific_v2 import (DOMAINS, FAMILY_COUNTS, LOGICAL_TIME,
    V2Query, corpus_hash, generate_v2_queries, generate_v2_world)
from ragstate.world.v2_validation import leakage_rows, repair_lexical_leakage, validate_gold
from .phase7a import PromptedEncoder
from .phase7a_figures import Canvas
from .runner import _git_commit


def _json(value):
    return json.dumps(value, indent=2, sort_keys=True, default=lambda x: sorted(x) if isinstance(x,(set,frozenset)) else x)


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _write_csv(path, rows, fields=None):
    with Path(path).open('w', newline='') as stream:
        writer=csv.DictWriter(stream, fieldnames=fields or list(rows[0]))
        writer.writeheader();writer.writerows(rows)


def _write_jsonl(path, rows):
    with Path(path).open('w') as stream:
        for row in rows:stream.write(json.dumps(row,sort_keys=True,default=lambda x:sorted(x) if isinstance(x,(set,frozenset)) else x)+'\n')


def _quantile(values,p):
    if not values:return None
    ordered=sorted(values)
    return ordered[math.ceil(p*len(ordered))-1]


def _mean(values):
    return sum(values)/len(values) if values else None


def generate(config_path:Path,data:Path,output:Path):
    if (data.exists() and any(data.iterdir())) or (output.exists() and any(output.iterdir())):
        raise ValueError('v2 output exists; frozen artifacts are immutable')
    config=json.loads(config_path.read_text())
    if config['corpus_version']!='v2' or config['top_k']!=10 or config['logical_time']!=LOGICAL_TIME:
        raise ValueError('Unexpected v2 config or benchmark top-k')
    if config['family_counts_per_domain']!=FAMILY_COUNTS:
        raise ValueError('Frozen family allocation differs from generator')
    if config['domains']!=[x.name for x in DOMAINS]:
        raise ValueError('Frozen six domains differ from generator')
    world=generate_v2_world(config['world_seed'])
    queries,lexical=repair_lexical_leakage(world,generate_v2_queries(world,config['query_seed']))
    errors=validate_gold(world,queries)
    if errors:raise ValueError('Invalid canonical gold: '+str(errors[:5]))
    data.mkdir(parents=True);output.mkdir(parents=True)
    _write_jsonl(data/'documents.jsonl',[asdict(world.documents[k]) for k in sorted(world.documents)])
    _write_jsonl(data/'principals.jsonl',[asdict(world.principals[k]) for k in sorted(world.principals)])
    _write_jsonl(data/'queries.jsonl',[asdict(q) for q in queries])
    _write_csv(data/'lexical_leakage.csv',lexical)
    source_files=[Path(__file__),Path(__file__).parent.parent/'world'/'scientific_v2.py',
                  Path(__file__).parent.parent/'world'/'v2_validation.py',
                  Path(__file__).parent.parent/'retrieval'/'metrics.py']
    manifest={'corpus_version':'v2','validation_version':config['validation_version'],
              'config':config,'config_sha256':_sha(config_path),
              'generator_source_sha256':{p.name:_sha(p) for p in source_files},
              'corpus_sha256':corpus_hash(world,queries),
              'data_sha256':{p.name:_sha(p) for p in sorted(data.iterdir())},
              'document_versions':len(world.documents),
              'logical_documents':len({d.logical_document_id for d in world.documents.values()}),
              'queries':len(queries),'gold_validation_errors':errors,
              'pre_retrieval_frozen':True,'git_commit':_git_commit(),
              'created_without_retrieval_scores':True}
    (data/'manifest.json').write_text(_json(manifest))
    (output/'manifest.json').write_text(_json(manifest))
    return manifest


def _load(data:Path,manifest):
    for name,expected in manifest['data_sha256'].items():
        if _sha(data/name)!=expected:raise ValueError('Frozen v2 data modified: '+name)
    world=WorldLedger()
    for line in (data/'principals.jsonl').read_text().splitlines():
        raw=json.loads(line);raw['groups']=frozenset(raw['groups']);world.add_principal(Principal(**raw))
    for line in (data/'documents.jsonl').read_text().splitlines():
        raw=json.loads(line);raw['required_groups']=frozenset(raw['required_groups'])
        world.add_document(DocumentVersion(**raw))
    queries=[]
    for line in (data/'queries.jsonl').read_text().splitlines():
        raw=json.loads(line)
        for key in ('logical_document_ids','required_document_ids','required_atomic_claims'):
            raw[key]=tuple(raw[key])
        queries.append(V2Query(**raw))
    if corpus_hash(world,queries)!=manifest['corpus_sha256']:
        raise ValueError('Frozen v2 corpus hash mismatch')
    errors=validate_gold(world,queries)
    if errors:raise ValueError('Frozen canonical gold invalid: '+str(errors[:5]))
    return world,queries


def _retrievers(chunks,config):
    snapshot=Path(config['embedding_snapshot'])
    if not snapshot.is_dir() or snapshot.name!=config['embedding_revision']:
        raise ValueError('Pinned local E5 snapshot unavailable')
    base=SentenceTransformerEncoder(str(snapshot),snapshot.name)
    encoder=PromptedEncoder(base,config['embedding_query_prefix'],config['embedding_passage_prefix'])
    dense=DenseRetriever(chunks,model_name=config['embedding_model'],
                         model_version=config['embedding_revision'],encoder=encoder)
    bm25=BM25Retriever(chunks)
    hybrid=HybridRetriever(bm25,dense,**config['hybrid'])
    return {'bm25':bm25,'e5':dense,'hybrid':hybrid}


def _evaluate(world,queries,retrievers,top_k,timed=True):
    if timed:
        for r in retrievers.values():r.retrieve(queries[0].text,top_k)
    rows=[];rankings={}
    for q in queries:
        rankings[q.query_id]={}
        for method,retriever in retrievers.items():
            start=time.perf_counter()
            candidate=retriever.retrieve(q.text,top_k)
            latency=(time.perf_counter()-start)*1000
            context=[h for h in candidate if world.allowed(q.principal_id,h.chunk.logical_document_id,q.logical_time)]
            gold=set(q.required_document_ids)
            got={h.chunk.document_id for h in context}
            ranks={doc:next((i for i,h in enumerate(candidate,1) if h.chunk.document_id==doc),None)
                   for doc in sorted(gold)}
            rankings[q.query_id][method]=[h.chunk.chunk_id for h in candidate]
            rows.append({'query_id':q.query_id,'domain':q.domain,'query_family':q.family,
                         'retriever':method,'principal_id':q.principal_id,
                         'tenant_id':q.tenant_id,'logical_time':q.logical_time,
                         'answerable':int(q.answerable),'required_count':len(gold),
                         'required_document_ids':json.dumps(sorted(gold)),
                         'required_claims':json.dumps(q.required_atomic_claims,sort_keys=True),
                         'candidate_chunk_ids':json.dumps([h.chunk.chunk_id for h in candidate]),
                         'context_chunk_ids':json.dumps([h.chunk.chunk_id for h in context]),
                         'gold_ranks':json.dumps(ranks,sort_keys=True),
                         'cer_at_10':current_evidence_recall(context,gold),
                         'scrs':int(state_consistent_retrieval_success(context,gold,q.principal_id,world,q.logical_time)),
                         'all_required_success':int(bool(gold) and gold.issubset(got)) if gold else None,
                         'any_required_success':int(bool(gold & got)) if gold else None,
                         'invalid_evidence_absence_success':invalid_evidence_absence_success(context,gold,set(q.logical_document_ids)),
                         'target_evidence_in_context':int(any(h.chunk.logical_document_id in q.logical_document_ids for h in context)),
                         'latency_ms':latency if timed else None})
    return rows,rankings


def _summarize(rows,field,answerable_only=True):
    grouped=defaultdict(list)
    for row in rows:
        if answerable_only and not row['answerable']:continue
        grouped[(row['retriever'],row[field])].append(row)
    return [{'retriever':r,'scope':scope,'n_queries':len(group),
             'cer_at_10':_mean([x['cer_at_10'] for x in group if x['cer_at_10'] is not None]),
             'scrs':_mean([x['scrs'] for x in group]),
             'all_required_success':_mean([x['all_required_success'] for x in group if x['all_required_success'] is not None]),
             'negative_absence_success':_mean([x['invalid_evidence_absence_success'] for x in group if x['invalid_evidence_absence_success'] is not None]),
             'median_latency_ms':statistics.median(x['latency_ms'] for x in group),
             'p95_latency_ms':_quantile([x['latency_ms'] for x in group],.95)}
            for (r,scope),group in sorted(grouped.items())]


def _figures(output,lexical,family_stats,domain_stats,family_summary,multi_summary):
    output=output/'figures';output.mkdir()
    palette={'bm25':'#235789','e5':'#c05621','hybrid':'#2f855a'}
    def bars(name,title,categories,series,ylabel='fraction'):
        c=Canvas(title,'category',ylabel)
        c.legend([(key,palette.get(key,'#6b46c1')) for key in series])
        width=min(100,600//max(1,len(categories)))
        for i,label in enumerate(categories):
            start=90+i*(650//len(categories))
            for j,(key,values) in enumerate(series.items()):
                x=start+j*(width//max(1,len(series)))
                color=palette.get(key,'#6b46c1')
                c.rect(x,c.y(values[i]),x+max(8,width//max(1,len(series))-3),430,color)
            c.label(start+width//2,450,label[:12],size=9)
        c.save(output/name)
    from .phase7a_figures import _hist_plot
    cats=[r['family'] for r in family_stats]
    bars('query_family_distribution','Query-family distribution',cats,{'share':[r['fraction'] for r in family_stats]})
    valid=[r for r in lexical if r['answerable']]
    _hist_plot(output,'query_gold_lexical_overlap','Query/gold lexical overlap',
               [r['query_gold_overlap'] for r in valid])
    _hist_plot(output,'gold_lexical_advantage','Gold versus nongold lexical advantage',
               [r['gold_lexical_advantage'] for r in valid])
    domains=[d.name for d in DOMAINS]
    bars('cer_by_domain','CER@10 by retriever and domain',domains,
         {m:[next(r['cer_at_10'] for r in domain_stats if r['retriever']==m and r['scope']==d)
             for d in domains] for m in palette})
    fams=[f for f in FAMILY_COUNTS if f!='negative']
    bars('scrs_by_family','SCRS by retriever and query family',fams,
         {m:[next(r['scrs'] for r in family_summary if r['retriever']==m and r['scope']==f)
             for f in fams] for m in palette})
    bars('multi_document_success','Multi-document all-required success',list(palette),
         {'share':[next(r['all_required_success'] for r in multi_summary if r['retriever']==m)
                   for m in palette]})


def validate(config_path:Path,data:Path,output:Path):
    if not (data/'manifest.json').is_file() or not (output/'manifest.json').is_file():
        raise ValueError('Generate and freeze v2 before retrieval validation')
    if (output/'clean_gate_summary.json').exists():
        raise ValueError('Completed v2 validation is immutable')
    manifest=json.loads((data/'manifest.json').read_text())
    if _sha(data/'manifest.json')!=_sha(output/'manifest.json'):
        raise ValueError('Data and result manifests differ')
    if _sha(config_path)!=manifest['config_sha256']:
        raise ValueError('Frozen v2 config changed')
    source_files=[Path(__file__),Path(__file__).parent.parent/'world'/'scientific_v2.py',
                  Path(__file__).parent.parent/'world'/'v2_validation.py',
                  Path(__file__).parent.parent/'retrieval'/'metrics.py']
    for source in source_files:
        if _sha(source)!=manifest['generator_source_sha256'][source.name]:
            raise ValueError('Frozen generator code changed: '+source.name)
    config=manifest['config'];world,queries=_load(data,manifest)
    reg_world=generate_v2_world(config['world_seed'])
    reg_queries,_=repair_lexical_leakage(reg_world,generate_v2_queries(reg_world,config['query_seed']))
    generation_deterministic=corpus_hash(reg_world,reg_queries)==manifest['corpus_sha256']
    if not generation_deterministic:raise ValueError('v2 generator is nondeterministic')
    lexical=leakage_rows(world,queries)
    chunks=build_chunks_from_ledger(world,config['logical_time'])
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    import torch,sentence_transformers
    torch.manual_seed(0);torch.set_num_threads(1)
    start=time.perf_counter();retrievers=_retrievers(chunks,config)
    index_build_ms=(time.perf_counter()-start)*1000
    rows,rankings=_evaluate(world,queries,retrievers,config['top_k'])
    torch.manual_seed(0)
    repeated,_rankings=_evaluate(world,queries,_retrievers(chunks,config),config['top_k'],timed=False)
    ranking_deterministic=rankings==_rankings
    metric_deterministic=all((a['cer_at_10'],a['scrs'],a['all_required_success'])==
                             (b['cer_at_10'],b['scrs'],b['all_required_success'])
                             for a,b in zip(rows,repeated))
    family_counts=Counter(q.family for q in queries)
    family_stats=[{'family':family,'count':family_counts[family],
                   'fraction':family_counts[family]/len(queries)} for family in FAMILY_COUNTS]
    domain_stats=_summarize(rows,'domain')
    family_summary=_summarize(rows,'query_family',False)
    multi_summary=[]
    for method in retrievers:
        group=[r for r in rows if r['retriever']==method and r['query_family']=='multi_document']
        multi_summary.append({'retriever':method,'n_queries':len(group),
                              'cer_at_10':_mean([r['cer_at_10'] for r in group]),
                              'scrs':_mean([r['scrs'] for r in group]),
                              'all_required_success':_mean([r['all_required_success'] for r in group])})
    answerable=[r for r in lexical if r['answerable']]
    non_id=[r for r in answerable if r['family']!='identifier_bearing']
    id_rate=_mean([r['gold_identifier_overlap'] for r in answerable])
    unique_rate=_mean([r['unique_gold_token_leak'] for r in non_id])
    title_rate=_mean([r['exact_title_overlap'] for r in answerable])
    warnings=sum(r['construction_warning'] for r in lexical)
    lengths=[len(d.content.split()) for d in world.documents.values() if d.is_temporally_valid(LOGICAL_TIME)]
    def distribution(values):
        return {'n':len(values),'min':min(values),'median':statistics.median(values),
                'p05':_quantile(values,.05),'p95':_quantile(values,.95),
                'max':max(values),'mean':_mean(values)}
    corpus_stats={'corpus_version':'v2','corpus_sha256':manifest['corpus_sha256'],
                  'logical_documents':manifest['logical_documents'],
                  'document_versions':manifest['document_versions'],
                  'indexed_chunks':len(chunks),'base_questions':len(queries),
                  'answerable_questions':sum(q.answerable for q in queries),
                  'negative_questions':sum(not q.answerable for q in queries),
                  'document_words':distribution(lengths),
                  'identifier_leakage_rate':id_rate,
                  'non_identifier_unique_gold_token_rate':unique_rate,
                  'exact_title_overlap_rate':title_rate,
                  'query_gold_overlap':distribution([r['query_gold_overlap'] for r in answerable]),
                  'query_nongold_overlap':distribution([r['query_best_nongold_overlap'] for r in answerable]),
                  'gold_lexical_advantage':distribution([r['gold_lexical_advantage'] for r in answerable]),
                  'longest_shared_phrase_tokens':distribution([r['longest_shared_phrase_tokens'] for r in answerable]),
                  'construction_warnings':warnings,
                  'family_counts':dict(family_counts)}
    overall={method:{'n':len(group),'cer_at_10':_mean([r['cer_at_10'] for r in group]),
                     'scrs':_mean([r['scrs'] for r in group]),
                     'median_latency_ms':statistics.median(r['latency_ms'] for r in group),
                     'p95_latency_ms':_quantile([r['latency_ms'] for r in group],.95)}
             for method in retrievers
             for group in [[r for r in rows if r['retriever']==method and r['answerable']]]}
    negative={method:_mean([r['invalid_evidence_absence_success'] for r in rows
                             if r['retriever']==method and not r['answerable']]) for method in retrievers}
    threshold=config['gates_frozen_before_retrieval']
    macro=lambda method,field:_mean([r[field] for r in domain_stats if r['retriever']==method])
    hybrid_domains=[r['scrs'] for r in domain_stats if r['retriever']=='hybrid']
    gates={
        'V2-G1':'PASS' if id_rate<=threshold['V2-G1']['identifier_leakage_max'] and
                          unique_rate<=threshold['V2-G1']['non_identifier_unique_gold_token_max'] and
                          warnings<=threshold['V2-G1']['unresolved_construction_warnings_max'] else 'FAIL',
        'V2-G2':'PASS' if macro('bm25','cer_at_10')>=threshold['V2-G2']['bm25_macro_cer_at_10_min'] else 'FAIL',
        'V2-G3':'PASS' if macro('e5','cer_at_10')>=threshold['V2-G3']['e5_macro_cer_at_10_min'] else 'FAIL',
        'V2-G4':'PASS' if macro('hybrid','scrs')>=threshold['V2-G4']['hybrid_macro_scrs_min'] and
                          min(hybrid_domains)>=threshold['V2-G4']['hybrid_min_domain_scrs'] else 'FAIL',
        'V2-G5':'PASS' if next(r['all_required_success'] for r in multi_summary if r['retriever']=='hybrid')>=
                          threshold['V2-G5']['hybrid_multi_document_all_required_min'] else 'FAIL',
        'V2-G6':'PASS' if min(negative.values())>=threshold['V2-G6']['negative_absence_success_min'] else 'FAIL',
        'V2-G7':'PASS' if generation_deterministic and ranking_deterministic and metric_deterministic else 'FAIL'}
    summary={'corpus_version':'v2','validation_version':config['validation_version'],
             'gates':gates,'all_critical_gates_pass':all(x=='PASS' for x in gates.values()),
             'overall':overall,'macro_cer_at_10':{m:macro(m,'cer_at_10') for m in retrievers},
             'macro_scrs':{m:macro(m,'scrs') for m in retrievers},
             'hybrid_min_domain_scrs':min(hybrid_domains),
             'negative_absence_success':negative,'multi_document':multi_summary,
             'corpus_hash':manifest['corpus_sha256'],'generation_deterministic':generation_deterministic,
             'ranking_deterministic':ranking_deterministic,'metric_deterministic':metric_deterministic,
             'index_build_ms':index_build_ms,
             'environment':{'python':sys.version.split()[0],'platform':platform.platform(),
                            'torch':torch.__version__,'sentence_transformers':sentence_transformers.__version__,
                            'device':'cpu','model':config['embedding_model'],
                            'revision':config['embedding_revision'],'git_commit':_git_commit()},
             'fault_campaign_executed':False}
    _write_csv(output/'lexical_leakage.csv',lexical)
    _write_csv(output/'query_family_statistics.csv',family_stats)
    _write_csv(output/'per_query_retrieval.csv',rows)
    _write_csv(output/'domain_summary.csv',domain_stats)
    _write_csv(output/'query_family_summary.csv',family_summary)
    _write_csv(output/'multi_document_summary.csv',multi_summary)
    (output/'corpus_statistics.json').write_text(_json(corpus_stats))
    (output/'clean_gate_summary.json').write_text(_json(summary))
    _figures(output,lexical,family_stats,domain_stats,family_summary,multi_summary)
    md=['# Phase 7B v2 corpus validation','',
        f"Frozen corpus: {manifest['logical_documents']} logical records, {manifest['document_versions']} versions, {len(queries)} questions. No transition faults run.",'',
        '## Construction','',
        f'Identifier leakage: {id_rate:.1%}; non-identifier unique-gold-token rate: {unique_rate:.1%}; exact title overlap: {title_rate:.1%}.',
        f"Query/gold overlap median {corpus_stats['query_gold_overlap']['median']:.3f}; best nongold median {corpus_stats['query_nongold_overlap']['median']:.3f}.",'',
        '## Clean retrieval','',
        '| Retriever | CER@10 | SCRS | Multi all-required | Negative absence |',
        '| --- | ---: | ---: | ---: | ---: |',
        *[f"| {m} | {overall[m]['cer_at_10']:.3f} | {overall[m]['scrs']:.3f} | "
          f"{next(r['all_required_success'] for r in multi_summary if r['retriever']==m):.3f} | {negative[m]:.3f} |"
          for m in retrievers],'', '## Gates','',*[f'- {k}: {v}' for k,v in gates.items()],
        '', 'No retrieval settings, questions, gold labels, or thresholds were changed after evaluation.']
    (output/'summary.md').write_text('\n'.join(md)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('generate','validate'))
    parser.add_argument('--config',default='configs/phase7b-corpus-v2.json')
    parser.add_argument('--data',default='data/generated-v2')
    parser.add_argument('--output',default='results/phase7b-corpus-validation-v2')
    args=parser.parse_args()
    fn=generate if args.stage=='generate' else validate
    result=fn(Path(args.config),Path(args.data),Path(args.output))
    print(_json({'stage':args.stage,'corpus_hash':result.get('corpus_sha256',result.get('corpus_hash')),
                 'gates':result.get('gates')}))


if __name__=='__main__':main()
