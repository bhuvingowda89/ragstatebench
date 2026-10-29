"""Predeclared BGE/E5 CPU screen on the immutable Phase 7 preflight corpus."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import statistics
import time
from collections import defaultdict
from pathlib import Path

from ragstate.retrieval.bm25 import BM25Retriever
from ragstate.retrieval.dense import DenseRetriever, SentenceTransformerEncoder
from ragstate.retrieval.hybrid import HybridRetriever
from ragstate.retrieval.indexing import build_chunks_from_ledger
from ragstate.retrieval.metrics import current_evidence_recall, state_consistent_retrieval_success
from ragstate.world.scientific import generate_scientific_queries, generate_scientific_world
from .phase7a import PromptedEncoder, _availability, _mean, _quantile, _write_csv
from .phase7a_figures import generate_figures


def _evaluate_model(spec,snapshot,world,queries,chunks,manifest,output):
    import torch
    torch.manual_seed(0)
    start=time.perf_counter()
    base=SentenceTransformerEncoder(str(snapshot),snapshot.name)
    load_ms=(time.perf_counter()-start)*1000
    params=sum(p.numel() for p in base.model.parameters())
    encoder=PromptedEncoder(base,spec['query_prefix'],spec['passage_prefix'])
    start=time.perf_counter()
    dense=DenseRetriever(chunks,model_name=spec['id'],model_version=snapshot.name,encoder=encoder)
    encode_ms=(time.perf_counter()-start)*1000
    sparse=BM25Retriever(chunks)
    fusion=manifest['retrieval']['hybrid']
    hybrid=HybridRetriever(sparse,dense,sparse_weight=fusion['sparse_weight'],
                           dense_weight=fusion['dense_weight'],rrf_k=fusion['rrf_k'],
                           candidate_k=fusion['candidate_k'])
    retrievers={'dense':dense,'hybrid':hybrid}
    for retriever in retrievers.values():retriever.retrieve(queries[0].text,10)
    rows=[];timings=defaultdict(list);rankings={}
    for q in queries:
        required={world.document_at(lid,9).document_id for lid in q.logical_document_ids}
        rankings[q.query_id]={}
        for method,retriever in retrievers.items():
            start=time.perf_counter()
            hits=retriever.retrieve(q.text,10)
            latency=(time.perf_counter()-start)*1000
            timings[method].append(latency)
            rankings[q.query_id][method]=[h.chunk.chunk_id for h in hits]
            context=[h for h in hits if world.allowed(q.principal_id,h.chunk.logical_document_id,9)]
            present=required & {h.chunk.document_id for h in context}
            rows.append({'query_id':q.query_id,'domain':q.domain,'family':q.family,
                         'retriever':method,'required_documents':len(required),
                         'cer_at_10':current_evidence_recall(context,required),
                         'scrs':int(state_consistent_retrieval_success(
                             context,required,q.principal_id,world,9)),
                         'all_required_success':int(len(present)==len(required)),
                         'any_required_success':int(bool(present)),
                         'latency_ms':round(latency,3),
                         'ranks':json.dumps({doc:next((i for i,h in enumerate(hits,1)
                             if h.chunk.document_id==doc),None) for doc in sorted(required)},sort_keys=True)})
    # Independent model/index construction and query pass; no cache reuse.
    torch.manual_seed(0)
    base2=SentenceTransformerEncoder(str(snapshot),snapshot.name)
    encoder2=PromptedEncoder(base2,spec['query_prefix'],spec['passage_prefix'])
    dense2=DenseRetriever(chunks,model_name=spec['id'],model_version=snapshot.name,encoder=encoder2)
    hybrid2=HybridRetriever(sparse,dense2,sparse_weight=fusion['sparse_weight'],
                            dense_weight=fusion['dense_weight'],rrf_k=fusion['rrf_k'],
                            candidate_k=fusion['candidate_k'])
    deterministic=all(rankings[q.query_id][method]==[h.chunk.chunk_id for h in ret.retrieve(q.text,10)]
                      for q in queries for method,ret in [('dense',dense2),('hybrid',hybrid2)])
    comparison=[]
    for method in retrievers:
        for scope in ['all_answerable']+[x.name for x in __import__(
                'ragstate.world.scientific',fromlist=['BLUEPRINTS']).BLUEPRINTS]+['multi_document']:
            group=[r for r in rows if r['retriever']==method and
                   (scope=='all_answerable' or r['domain']==scope or
                    (scope=='multi_document' and r['family']=='multi_document'))]
            comparison.append({'model_id':spec['id'],'revision':snapshot.name,
                               'status':'measured','retriever':method,'scope':scope,
                               'n_queries':len(group),'cer_at_10':_mean([r['cer_at_10'] for r in group]),
                               'scrs':_mean([r['scrs'] for r in group]),
                               'multi_document_success':_mean([r['all_required_success'] for r in group]) if scope=='multi_document' else None,
                               'encoding_ms':encode_ms if method=='dense' else None,
                               'median_query_ms':statistics.median(timings[method]),
                               'p95_query_ms':_quantile(timings[method],0.95)})
    destination=output/'models'/spec['id'].replace('/','--')
    destination.mkdir(parents=True,exist_ok=True)
    _write_csv(destination/'per_query.csv',rows)
    model_summary={'model_id':spec['id'],'revision':snapshot.name,'snapshot_path':str(snapshot),
                   'parameter_count':params,'query_prefix':spec['query_prefix'],
                   'passage_prefix':spec['passage_prefix'],
                   'model_load_ms':load_ms,'encoding_ms':encode_ms,
                   'deterministic_ranks':deterministic,'comparison':comparison}
    (destination/'summary.json').write_text(json.dumps(model_summary,indent=2,sort_keys=True))
    return comparison,model_summary


def run_screen(output):
    output=Path(output)
    manifest=json.loads((output/'manifest.json').read_text())
    pre=Path('results/scientific-pilot-v1-preflight')
    for name,key in [('manifest.json','preflight_manifest_sha256'),
                     ('summary.json','preflight_summary_sha256'),
                     ('per_query.csv','preflight_per_query_sha256')]:
        if hashlib.sha256((pre/name).read_bytes()).hexdigest()!=manifest['frozen_corpus'][key]:
            raise ValueError('Frozen preflight changed: '+name)
    os.environ['HF_HUB_OFFLINE']='1';os.environ['TRANSFORMERS_OFFLINE']='1'
    os.environ['HF_DATASETS_OFFLINE']='1';os.environ['TOKENIZERS_PARALLELISM']='false'
    import torch
    torch.manual_seed(0);torch.set_num_threads(1)
    world=generate_scientific_world(manifest['frozen_corpus']['world_seed'])
    queries=[q for q in generate_scientific_queries(world,manifest['frozen_corpus']['query_seed'])
             if all(world.allowed(q.principal_id,lid,9) for lid in q.logical_document_ids)]
    chunks=build_chunks_from_ledger(world,9)
    available=_availability(manifest)
    if any(snapshot is None for _,snapshot in available):
        raise ValueError('All three predeclared local snapshots are required for the frozen screen')
    prior=json.loads((output/'summary.json').read_text())
    if prior.get('selection_status')=='complete':
        raise ValueError('Screen already complete; refusing to overwrite results')
    shutil.copy2(output/'summary.json',output/'summary_minilm.json')
    all_rows=[];model_summaries={}
    for spec,snapshot in available:
        rows,summary=_evaluate_model(spec,snapshot,world,queries,chunks,manifest,output)
        all_rows.extend(rows);model_summaries[spec['id']]=summary
    _write_csv(output/'model_comparison.csv',all_rows,list(all_rows[0]))
    model_ids=[s['id'] for s,_ in available]
    def metric(model,method,scope,field):
        row=next(r for r in all_rows if r['model_id']==model and r['retriever']==method and r['scope']==scope)
        return float(row[field])
    domains=[x.name for x in __import__('ragstate.world.scientific',fromlist=['BLUEPRINTS']).BLUEPRINTS]
    scores={model:{'macro_scrs':_mean([metric(model,'dense',d,'scrs') for d in domains]),
                   'multi_scrs':metric(model,'dense','multi_document','scrs'),
                   'macro_cer':_mean([metric(model,'dense',d,'cer_at_10') for d in domains]),
                   'median_latency_ms':metric(model,'dense','all_answerable','median_query_ms')}
            for model in model_ids}
    selected=max(model_ids,key=lambda m:(scores[m]['macro_scrs'],scores[m]['multi_scrs'],
                                         scores[m]['macro_cer'],-scores[m]['median_latency_ms']))
    hybrid_macro=_mean([metric(selected,'hybrid',d,'scrs') for d in domains])
    hybrid_min=min(metric(selected,'hybrid',d,'scrs') for d in domains)
    dense_adequate=any(scores[m]['macro_cer']>=0.9 and
                       min(metric(m,'dense',d,'cer_at_10') for d in domains)>=0.85
                       for m in model_ids)
    multi=metric(selected,'hybrid','multi_document','scrs')
    deterministic=prior['deterministic_ranks'] and all(x['deterministic_ranks'] for x in model_summaries.values())
    gates={'D1':prior['gates']['D1'],
           'D2':'PASS' if dense_adequate else 'FAIL',
           'D3':'PASS' if hybrid_macro>=0.9 and hybrid_min>=0.85 else 'FAIL',
           'D4':'PASS' if multi>=0.85 else 'FAIL',
           'D5':'PASS' if deterministic else 'FAIL'}
    prior.update({'selection_status':'complete','selected_dense_encoder':selected,
                  'selection_scores':scores,'model_summaries':model_summaries,
                  'model_status':{model:'measured' for model in model_ids},
                  'gates':gates,'selected_hybrid_macro_scrs':hybrid_macro,
                  'selected_hybrid_min_domain_scrs':hybrid_min,
                  'selected_hybrid_multi_document_success':multi,
                  'recommendation':('version and redesign the corpus/query generator' if gates['D1']=='FAIL'
                                    else 'redesign the clean retrieval architecture' if any(gates[g]=='FAIL' for g in ('D2','D3','D4'))
                                    else 'proceed to Phase 7 scientific pilot unchanged'),
                  'stop_reason':'No transition faults run during Phase 7A diagnostic.'})
    (output/'summary.json').write_text(json.dumps(prior,indent=2,sort_keys=True))
    depth=list(csv.DictReader((output/'depth_sensitivity.csv').open()))
    for row in depth:
        row['depth']=int(row['depth'])
        for key in ('cer_at_k','all_required_success','any_required_success','scrs'):
            row[key]=float(row[key])
    per_gold=list(csv.DictReader((output/'failure_diagnostics.csv').open()))
    realism=[]
    # Figure 6 is regenerated from frozen lexical diagnostic rows reconstructed
    # from the per-gold table (one row per query to avoid multi-doc overweighting).
    for qid in sorted({r['query_id'] for r in per_gold}):
        rows=[r for r in per_gold if r['query_id']==qid]
        realism.append({'query_gold_overlap':_mean([float(r['lexical_overlap_query_gold']) for r in rows]),
                        'query_best_nongold_overlap':float(rows[0]['lexical_overlap_query_top_false_positive'])})
    from .phase7a_figures import generate_figures
    generate_figures(output,depth,per_gold,all_rows,realism)
    model_lines=['| Encoder | Dense CER@10 | Dense SCRS | Dense multi SCRS | Hybrid SCRS | Hybrid multi SCRS |',
                 '| --- | ---: | ---: | ---: | ---: | ---: |']
    for model in model_ids:
        model_lines.append(f"| {model} | {metric(model,'dense','all_answerable','cer_at_10'):.3f} | "
                           f"{metric(model,'dense','all_answerable','scrs'):.3f} | "
                           f"{metric(model,'dense','multi_document','scrs'):.3f} | "
                           f"{metric(model,'hybrid','all_answerable','scrs'):.3f} | "
                           f"{metric(model,'hybrid','multi_document','scrs'):.3f} |")
    depth_lines=['| Retriever | k | CER@k | SCRS | Multi-document all-required success |',
                 '| --- | ---: | ---: | ---: | ---: |']
    for method in ('bm25','dense','hybrid'):
        for k in (5,10,20,50):
            overall=next(r for r in depth if r['retriever']==method and r['depth']==k and r['scope']=='all_answerable')
            multi_row=next(r for r in depth if r['retriever']==method and r['depth']==k and r['scope']=='multi_document')
            depth_lines.append(f"| {method} | {k} | {overall['cer_at_k']:.3f} | "
                               f"{overall['scrs']:.3f} | {multi_row['all_required_success']:.3f} |")
    md=['# Phase 7A diagnostic v1','',
        'All three predeclared CPU encoders were screened on the frozen preflight corpus. No transition faults were run.','',
        '## Frozen model selection','',
        'Highest macro clean dense SCRS across six domains; ties: multi-document SCRS, macro CER@10, lower median CPU query latency.','',
        *model_lines,'',
        f"Selected encoder under this rule: `{selected}`. This is a diagnostic selection, not approval to run Phase 7 faults.",'',
        '## MiniLM retrieval depth','',*depth_lines,'',
        '## Corpus realism','',
        f"Exact gold-only identifiers occur in {prior['corpus_realism']['exact_gold_identifier']:.1%} of answerable queries. "
        f"Mean query-token coverage is {prior['corpus_realism']['query_gold_overlap']:.3f} for gold "
        f"and {prior['corpus_realism']['query_best_nongold_overlap']:.3f} for the best BM25 nongold. "
        'D1 fails regardless of model adequacy.','',
        '## Gates','',*[f'- {k}: {v}' for k,v in gates.items()],
        '', '## Recommendation','',prior['recommendation']+'. Preserve v1 preflight and diagnostic artifacts; any redesign requires a new benchmark version.']
    (output/'summary.md').write_text('\n'.join(md)+'\n')
    execution={'manifest_sha256':hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest(),
               'snapshots':{spec['id']:str(snapshot) for spec,snapshot in available},
               'selection_rule':manifest['model_selection_rule'],'selected':selected,
               'model_status':{model:'measured' for model in model_ids}}
    (output/'execution_manifest.json').write_text(json.dumps(execution,indent=2,sort_keys=True))
    return prior


def main():
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--output',default='results/phase7a-diagnostic-v1')
    args=p.parse_args();result=run_screen(args.output)
    print(json.dumps({k:result[k] for k in ('selected_dense_encoder','selection_scores','gates','recommendation')},indent=2))


if __name__=='__main__':main()
