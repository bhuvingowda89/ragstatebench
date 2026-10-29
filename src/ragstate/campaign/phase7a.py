"""Phase 7A diagnostics on the immutable scientific-pilot-v1 preflight corpus."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import re
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from ragstate.retrieval.bm25 import BM25Retriever, tokenize
from ragstate.retrieval.dense import DenseRetriever, SentenceTransformerEncoder
from ragstate.retrieval.hybrid import HybridRetriever
from ragstate.retrieval.indexing import build_chunks_from_ledger
from ragstate.retrieval.metrics import current_evidence_recall, state_consistent_retrieval_success
from ragstate.world.scientific import BLUEPRINTS, generate_scientific_queries, generate_scientific_world


class PromptedEncoder:
    """Predeclared model-card prefixes, with distinct document/query calls."""
    def __init__(self, encoder, query_prefix, passage_prefix):
        self.encoder = encoder
        self.query_prefix = query_prefix
        self.passage_prefix = passage_prefix
        self.cache_identity = (query_prefix, passage_prefix)

    def encode_documents(self, texts, batch_size):
        return self.encoder.encode([self.passage_prefix + t for t in texts], batch_size)

    def encode_query(self, texts, batch_size):
        return self.encoder.encode([self.query_prefix + t for t in texts], batch_size)

    def encode(self, texts, batch_size):
        return self.encoder.encode(texts, batch_size)


def _write_csv(path, rows, fields=None):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields or list(rows[0]), extrasaction='ignore')
        writer.writeheader()
        writer.writerows(rows)


def _rank(hits, document_id):
    return next(((i, hit.score) for i, hit in enumerate(hits, 1)
                 if hit.chunk.document_id == document_id), (None, None))


def _coverage(query, text):
    terms = set(tokenize(query))
    return len(terms & set(tokenize(text))) / len(terms) if terms else 0.0


def _jaccard(a, b):
    x, y = set(tokenize(a)), set(tokenize(b))
    return len(x & y) / len(x | y) if x or y else 0.0


def _doc_type(chunk):
    return chunk.text.split(' record ', 1)[0].lower()


def _competition(gold, rival):
    if rival.logical_document_id == gold.logical_document_id and rival.version != gold.version:
        return 'wrong_version'
    if rival.logical_document_id == gold.logical_document_id:
        return 'same_logical_document'
    gd, rd = gold.logical_document_id.split('-'), rival.logical_document_id.split('-')
    same_domain = gd[0] == rd[0]
    if same_domain and _doc_type(gold) == _doc_type(rival):
        if abs(int(gd[1]) - int(rd[1])) <= 1:
            return 'near_duplicate'
        return 'same_logical_family'
    return 'same_domain' if same_domain else 'unrelated'


def _mean(values):
    return sum(values) / len(values) if values else None


def _quantile(values, p):
    if not values:
        return None
    values = sorted(values)
    return values[math.ceil(p * len(values)) - 1]


def _availability(manifest):
    out = []
    hub = Path('/Users/nischithacn/.cache/huggingface/hub')
    for model in manifest['screened_models']:
        location = hub / ('models--' + model['id'].replace('/', '--')) / 'snapshots'
        snapshots = sorted(p for p in location.glob('*') if p.is_dir()) if location.exists() else []
        chosen = next((p for p in snapshots if p.name == model['snapshot_revision']), None) if model['snapshot_revision'] else (snapshots[0] if len(snapshots) == 1 else None)
        out.append((model, chosen))
    return out


def run_diagnostic(output):
    output = Path(output)
    manifest = json.loads((output / 'manifest.json').read_text())
    preflight = Path('results/scientific-pilot-v1-preflight')
    for name, key in [('manifest.json','preflight_manifest_sha256'),
                      ('summary.json','preflight_summary_sha256'),
                      ('per_query.csv','preflight_per_query_sha256')]:
        observed = hashlib.sha256((preflight / name).read_bytes()).hexdigest()
        if observed != manifest['frozen_corpus'][key]:
            raise ValueError('Frozen preflight artifact changed: ' + name)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    os.environ['HF_DATASETS_OFFLINE'] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    import torch
    torch.manual_seed(0)
    torch.set_num_threads(1)
    world = generate_scientific_world(manifest['frozen_corpus']['world_seed'])
    queries = [q for q in generate_scientific_queries(world, manifest['frozen_corpus']['query_seed'])
               if all(world.allowed(q.principal_id, lid, 9) for lid in q.logical_document_ids)]
    chunks = build_chunks_from_ledger(world, 9)
    by_id = {chunk.document_id: chunk for chunk in chunks}
    availability = _availability(manifest)
    mini_spec, mini_snapshot = availability[0]
    if mini_snapshot is None:
        raise ValueError('Frozen MiniLM snapshot unavailable')
    started = time.perf_counter()
    base_encoder = SentenceTransformerEncoder(str(mini_snapshot), mini_snapshot.name)
    load_ms = (time.perf_counter() - started) * 1000
    encoder = PromptedEncoder(base_encoder, mini_spec['query_prefix'], mini_spec['passage_prefix'])
    started = time.perf_counter()
    dense = DenseRetriever(chunks, model_name=mini_spec['id'], model_version=mini_snapshot.name,
                           encoder=encoder)
    encode_ms = (time.perf_counter() - started) * 1000
    sparse = BM25Retriever(chunks)
    hybrid = HybridRetriever(sparse, dense,
                             sparse_weight=manifest['retrieval']['hybrid']['sparse_weight'],
                             dense_weight=manifest['retrieval']['hybrid']['dense_weight'],
                             rrf_k=manifest['retrieval']['hybrid']['rrf_k'],
                             candidate_k=manifest['retrieval']['hybrid']['candidate_k'])
    retrievers = {'bm25': sparse, 'dense': dense, 'hybrid': hybrid}
    # Warm query paths before latency measurement; no rank or gold is used for tuning.
    for retriever in retrievers.values():
        retriever.retrieve(queries[0].text, 50)
    all_results = {}
    timing = defaultdict(list)
    for q in queries:
        all_results[q.query_id] = {}
        for name, retriever in retrievers.items():
            start = time.perf_counter()
            hits = retriever.retrieve(q.text, len(chunks))
            timing[name].append((time.perf_counter() - start) * 1000)
            all_results[q.query_id][name] = hits
    depths = manifest['retrieval']['diagnostic_depths']
    depth_rows = []
    per_gold = []
    multi_rows = []
    realism_rows = []
    dense_margins = []
    for q in queries:
        gold_ids = {world.document_at(lid, 9).document_id for lid in q.logical_document_ids}
        qresults = all_results[q.query_id]
        gold_chunks = [by_id[x] for x in sorted(gold_ids)]
        dense_hits = qresults['dense']
        nongold_dense = [h for h in dense_hits if h.chunk.document_id not in gold_ids]
        best_nongold = nongold_dense[0]
        bm25_nongold = next((h for h in qresults['bm25'] if h.chunk.document_id not in gold_ids), None)
        unique_ids = re.findall(r'\b(?:HR|FINANCE|ENGINEERING|SECURITY|SALES|LEGAL)-\d{4}\b', q.text)
        lexical_gold = _mean([_coverage(q.text, c.text) for c in gold_chunks])
        lexical_false = _coverage(q.text, bm25_nongold.chunk.text) if bm25_nongold else 0.0
        realism_rows.append({'query_id':q.query_id,'family':q.family,'domain':q.domain,
                             'query_gold_overlap':lexical_gold,
                             'query_best_nongold_overlap':lexical_false,
                             'exact_gold_identifier':int(bool(unique_ids) and all(
                                 any(code in c.text for c in gold_chunks) and
                                 not any(code in c.text for c in chunks if c.document_id not in gold_ids)
                                 for code in unique_ids)),
                             'near_exact_lexical_match':int(bool(unique_ids) and lexical_gold > lexical_false),
                             'distinctive_title_term':int(any(
                                 t in set(tokenize(q.text)) for c in gold_chunks
                                 for t in set(tokenize(_doc_type(c))) if t not in
                                 {'record',c.logical_document_id.split('-')[0].lower()}))})
        for gold_id in sorted(gold_ids):
            gold = by_id[gold_id]
            ranks = {name:_rank(hits,gold_id) for name,hits in qresults.items()}
            top_competitors = []
            for hit in nongold_dense[:5]:
                top_competitors.append({'document_id':hit.chunk.document_id,
                                        'category':_competition(gold,hit.chunk)})
            dense_score = ranks['dense'][1]
            margin = dense_score - best_nongold.score
            if ranks['dense'][0] is None or ranks['dense'][0] > 10:
                dense_margins.append(margin)
            per_gold.append({'query_id':q.query_id,'domain':q.domain,'query_family':q.family,
                             'number_required_documents':len(gold_ids),
                             'gold_document_id':gold_id,
                             'bm25_rank':ranks['bm25'][0], 'dense_rank':ranks['dense'][0],
                             'hybrid_rank':ranks['hybrid'][0],
                             'bm25_score':ranks['bm25'][1],
                             'dense_score':dense_score, 'hybrid_score':ranks['hybrid'][1],
                             **{f'{method}_hit_at_{k}':int(rank is not None and rank <= k)
                                for method,(rank,_) in ranks.items() for k in depths},
                             'top_competing_document_ids':json.dumps([x['document_id'] for x in top_competitors]),
                             'top_competitor_categories':json.dumps(top_competitors),
                             'lexical_overlap_query_gold':_coverage(q.text,gold.text),
                             'lexical_overlap_query_top_false_positive':_coverage(
                                 q.text,best_nongold.chunk.text),
                             'bm25_best_nongold_document_id':bm25_nongold.chunk.document_id if bm25_nongold else None,
                             'lexical_overlap_query_bm25_nongold':lexical_false,
                             'dense_best_gold_similarity':dense_score,
                             'dense_rank1_nongold_similarity':best_nongold.score,
                             'dense_rank5_threshold':dense_hits[4].score,
                             'dense_rank10_threshold':dense_hits[9].score,
                             'gold_margin':margin})
        if q.family == 'multi_document':
            entry={'query_id':q.query_id,'domain':q.domain,
                   'number_required_documents':len(gold_ids),
                   'gold_document_ids':json.dumps(sorted(gold_ids))}
            for name,hits in qresults.items():
                ranks={gold:_rank(hits,gold)[0] for gold in sorted(gold_ids)}
                found=sum(rank is not None and rank<=10 for rank in ranks.values())
                entry[name+'_retrieved_at_10']=found
                entry[name+'_ranks']=json.dumps(ranks,sort_keys=True)
                missing=[gold for gold,rank in ranks.items() if rank is None or rank>10]
                if not missing: reason='none'
                elif len(missing)>1: reason='multiple_missing'
                else:
                    rank=ranks[missing[0]]
                    rival=next((h for h in hits[:10] if h.chunk.document_id not in gold_ids),None)
                    category=_competition(by_id[missing[0]],rival.chunk) if rival else 'none'
                    reason=('one_missing_rank_displacement' if rank is not None and rank<=50
                            else 'one_missing_beyond_50')+';'+category
                entry[name+'_failure_structure']=reason
            multi_rows.append(entry)
        for name,hits in qresults.items():
            for k in depths:
                raw=hits[:k]
                context=[h for h in raw if world.allowed(q.principal_id,h.chunk.logical_document_id,9)]
                present=gold_ids & {h.chunk.document_id for h in context}
                depth_rows.append({'query_id':q.query_id,'domain':q.domain,'family':q.family,
                                   'retriever':name,'depth':k,'cer_at_k':len(present)/len(gold_ids),
                                   'all_required_success':int(len(present)==len(gold_ids)),
                                   'any_required_success':int(bool(present)),
                                   'scrs':int(state_consistent_retrieval_success(
                                       context,gold_ids,q.principal_id,world,9))})
    # Same model, new CPU index and full second query pass: rank and metric reproducibility.
    torch.manual_seed(0)
    second_encoder=PromptedEncoder(SentenceTransformerEncoder(str(mini_snapshot),mini_snapshot.name),
                                   mini_spec['query_prefix'],mini_spec['passage_prefix'])
    second_dense=DenseRetriever(chunks,model_name=mini_spec['id'],model_version=mini_snapshot.name,
                                encoder=second_encoder)
    second_hybrid=HybridRetriever(sparse,second_dense,
                                 sparse_weight=manifest['retrieval']['hybrid']['sparse_weight'],
                                 dense_weight=manifest['retrieval']['hybrid']['dense_weight'],
                                 rrf_k=manifest['retrieval']['hybrid']['rrf_k'],
                                 candidate_k=manifest['retrieval']['hybrid']['candidate_k'])
    deterministic=all(
        [h.chunk.chunk_id for h in all_results[q.query_id][name]]==
        [h.chunk.chunk_id for h in ret.retrieve(q.text,len(chunks))]
        for q in queries for name,ret in [('dense',second_dense),('hybrid',second_hybrid)])
    by_depth=defaultdict(list)
    for row in depth_rows:
        by_depth[(row['retriever'],row['depth'],'all_answerable')].append(row)
        by_depth[(row['retriever'],row['depth'],row['domain'])].append(row)
        if row['family']=='multi_document':
            by_depth[(row['retriever'],row['depth'],'multi_document')].append(row)
    depth_summary=[]
    for (method,k,scope),group in sorted(by_depth.items()):
        depth_summary.append({'retriever':method,'depth':k,'scope':scope,'n_queries':len(group),
                              **{field:_mean([r[field] for r in group]) for field in
                                 ('cer_at_k','all_required_success','any_required_success','scrs')}})
    # Corpus statistics use the frozen text only and do not depend on retrieval outcomes.
    corpus_terms=[set(tokenize(c.text)) for c in chunks]
    pairs=[]; dense_pairs=[]; duplicate_pairs=0
    vectors=np.asarray(dense.vectors,dtype=np.float64)
    norms=np.linalg.norm(vectors,axis=1)
    normalized=vectors/np.maximum(norms[:,None],1e-12)
    for i,a in enumerate(chunks):
        for j in range(i+1,len(chunks)):
            b=chunks[j]
            if a.logical_document_id.split('-')[0]!=b.logical_document_id.split('-')[0]:continue
            overlap=len(corpus_terms[i]&corpus_terms[j])/len(corpus_terms[i]|corpus_terms[j])
            pairs.append(overlap)
            dense_pairs.append(float(normalized[i]@normalized[j]))
            if _doc_type(a)==_doc_type(b) and abs(int(a.logical_document_id.split('-')[1])-int(b.logical_document_id.split('-')[1]))<=1:
                duplicate_pairs+=1
    family_stats=[]
    for family in sorted({x['family'] for x in realism_rows}):
        group=[x for x in realism_rows if x['family']==family]
        family_stats.append({'family':family,'n':len(group),
                             **{field:_mean([x[field] for x in group]) for field in
                                ('query_gold_overlap','query_best_nongold_overlap',
                                 'exact_gold_identifier','near_exact_lexical_match',
                                 'distinctive_title_term')}})
    corpus_realism={'n_answerable_queries':len(queries),'n_documents':len(chunks),
                    'overall':{field:_mean([x[field] for x in realism_rows]) for field in
                               ('query_gold_overlap','query_best_nongold_overlap',
                                'exact_gold_identifier','near_exact_lexical_match',
                                'distinctive_title_term')},
                    'by_query_family':family_stats,
                    'near_duplicate_adjacent_pairs':duplicate_pairs,
                    'same_domain_pairs':len(pairs),
                    'near_duplicate_pair_density':duplicate_pairs/len(pairs),
                    'mean_same_domain_lexical_jaccard':_mean(pairs),
                    'mean_same_domain_minilm_cosine':_mean(dense_pairs),
                    'definitions':{'exact_gold_identifier':'query contains DOMAIN-NNNN identifiers present in required gold chunks and absent from every nongold chunk',
                                   'near_exact_lexical_match':'exact gold identifier plus greater query-token coverage in gold than best BM25 nongold',
                                   'near_duplicate':'adjacent numeric identifiers in the same domain and document type'}}
    margin_summary={'dense_misses_at_10':len(dense_margins),
                    'mean':_mean(dense_margins),'median':statistics.median(dense_margins) if dense_margins else None,
                    'min':min(dense_margins) if dense_margins else None,
                    'max':max(dense_margins) if dense_margins else None,
                    'p05':_quantile(dense_margins,0.05),'p95':_quantile(dense_margins,0.95),
                    'fraction_negative':_mean([x<0 for x in dense_margins])}
    model_rows=[]
    for spec,snapshot in availability:
        if spec is mini_spec:
            for method in ('dense','hybrid'):
                for scope in ['all_answerable']+[b.name for b in BLUEPRINTS]+['multi_document']:
                    result=next(x for x in depth_summary if x['retriever']==method and x['depth']==10 and x['scope']==scope)
                    model_rows.append({'model_id':spec['id'],'revision':snapshot.name,
                                       'status':'measured','retriever':method,'scope':scope,
                                       'n_queries':result['n_queries'],
                                       'cer_at_10':result['cer_at_k'],'scrs':result['scrs'],
                                       'multi_document_success':result['all_required_success'] if scope=='multi_document' else None,
                                       'encoding_ms':encode_ms if method=='dense' else None,
                                       'median_query_ms':statistics.median(timing[method]),
                                       'p95_query_ms':_quantile(timing[method],0.95)})
        else:
            model_rows.append({'model_id':spec['id'],'revision':snapshot.name if snapshot else None,
                               'status':'pending_screen' if snapshot else 'missing_snapshot',
                               'retriever':'dense/hybrid','scope':'all_answerable',
                               'n_queries':None,'cer_at_10':None,'scrs':None,
                               'multi_document_success':None,'encoding_ms':None,
                               'median_query_ms':None,'p95_query_ms':None})
    _write_csv(output/'failure_diagnostics.csv',per_gold)
    _write_csv(output/'multi_document_diagnostic.csv',multi_rows)
    _write_csv(output/'depth_sensitivity.csv',depth_summary)
    _write_csv(output/'model_comparison.csv',model_rows)
    (output/'corpus_realism.json').write_text(json.dumps(corpus_realism,indent=2,sort_keys=True))
    (output/'dense_margin_summary.json').write_text(json.dumps(margin_summary,indent=2,sort_keys=True))
    from .phase7a_figures import generate_figures
    generate_figures(output,depth_summary,per_gold,model_rows,realism_rows)
    overall10={name:next(x for x in depth_summary if x['retriever']==name and x['depth']==10 and x['scope']=='all_answerable') for name in retrievers}
    multi10={name:next(x for x in depth_summary if x['retriever']==name and x['depth']==10 and x['scope']=='multi_document') for name in retrievers}
    summary={'diagnostic_id':manifest['diagnostic_id'],'manifest_sha256':hashlib.sha256((output/'manifest.json').read_bytes()).hexdigest(),
             'miniLM_snapshot':mini_snapshot.name,'model_load_ms':load_ms,'encoding_ms':encode_ms,
             'n_answerable_queries':len(queries),'n_gold_document_pairs':len(per_gold),
             'deterministic_ranks':deterministic,'depth_10':overall10,'multi_document_10':multi10,
             'corpus_realism':corpus_realism['overall'],'margin_summary':margin_summary,
             'model_status':{r['model_id']:r['status'] for r in model_rows},
             'selection_status':'pending_predeclared_model_screen',
             'gates':{'D1':'FAIL','D2':'CONDITIONAL','D3':'CONDITIONAL','D4':'CONDITIONAL','D5':'PASS' if deterministic else 'FAIL'},
             'recommendation':'pending full predeclared model screen',
             'stop_reason':'Exact unique gold identifiers in the frozen queries create lexical leakage; no transition faults run.'}
    (output/'summary.json').write_text(json.dumps(summary,indent=2,sort_keys=True))
    md=['# Phase 7A diagnostic v1','',
        'The immutable Phase 7 preflight remains unchanged. BGE and E5 screening is pending in the predeclared three-model diagnostic.','',
        '## Decision','',
        f"D1 FAIL: {corpus_realism['overall']['exact_gold_identifier']:.1%} of answerable queries contain an exact gold identifier. BM25 CER@10 is {overall10['bm25']['cer_at_k']:.3f}.",
        f"MiniLM CER@10 is {overall10['dense']['cer_at_k']:.3f}; hybrid CER@10 is {overall10['hybrid']['cer_at_k']:.3f}.",
        f"Multi-document all-required success at 10: dense {multi10['dense']['all_required_success']:.3f}, hybrid {multi10['hybrid']['all_required_success']:.3f}.",
        'D1 indicates a corpus/query design problem. The final recommendation follows the frozen three-model screen. Do not change v1 questions, gold, top-k, or fusion after observing these results.','',
        '## Model selection','',
        'The frozen selection rule will be applied after the predeclared BGE and E5 screen.','',
        '## Gates','',
        *[f'- {key}: {value}' for key,value in summary['gates'].items()],
        '', '## Provenance','',
        f"Manifest SHA-256: `{summary['manifest_sha256']}`. MiniLM snapshot: `{mini_snapshot.name}`.",
        'All scores come from the frozen preflight world and queries. No transition-fault campaign was run.']
    (output/'summary.md').write_text('\n'.join(md)+'\n')
    return summary


def main():
    import argparse
    p=argparse.ArgumentParser()
    p.add_argument('--output',default='results/phase7a-diagnostic-v1')
    args=p.parse_args()
    result=run_diagnostic(args.output)
    print(json.dumps({key:result[key] for key in ('gates','depth_10','multi_document_10','corpus_realism','recommendation')},indent=2))


if __name__=='__main__':
    main()
