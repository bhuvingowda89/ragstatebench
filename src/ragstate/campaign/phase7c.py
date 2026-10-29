"""Offline v3 corpus freeze and enterprise-realism clean validation."""
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

from ragstate.retrieval.bm25 import BM25Retriever
from ragstate.retrieval.chunking import chunk_document
from ragstate.retrieval.dense import DenseRetriever, SentenceTransformerEncoder
from ragstate.retrieval.hybrid import HybridRetriever
from ragstate.retrieval.indexing import build_chunks_from_ledger
from ragstate.retrieval.metrics import (current_evidence_recall,
    invalid_evidence_absence_success, state_consistent_retrieval_success)
from ragstate.world.ledger import WorldLedger
from ragstate.world.models import DocumentVersion, Principal
from ragstate.world.scientific_v2 import DOMAINS, FAMILY_COUNTS, LOGICAL_TIME
from ragstate.world.scientific_v3 import (ARCHETYPES, V3Query, corpus_hash,
    generate_v3_queries, generate_v3_world)
from ragstate.world.v3_validation import cramers_v, leakage_rows, repair_lexical_leakage, validate_gold
from .phase7a import PromptedEncoder
from .phase7a_figures import Canvas, _hist_plot
from .phase7b import _json, _mean, _quantile, _sha, _write_csv, _write_jsonl
from .runner import _git_commit


def _sources():
    root=Path(__file__).parent.parent
    return [Path(__file__),root/'world'/'scientific_v3.py',root/'world'/'v3_validation.py',
            root/'world'/'scientific_v2.py',root/'retrieval'/'metrics.py',
            Path(__file__).parent/'phase7b.py']


def generate(config_path:Path,data:Path,output:Path):
    if (data.exists() and any(data.iterdir())) or (output.exists() and any(output.iterdir())):
        raise ValueError('v3 output exists; frozen artifacts are immutable')
    config=json.loads(config_path.read_text())
    if config['corpus_version']!='v3' or config['top_k']!=10 or config['logical_time']!=LOGICAL_TIME or \
       config['family_counts_per_domain']!=FAMILY_COUNTS or config['domains']!=[d.name for d in DOMAINS]:
        raise ValueError('v3 frozen config mismatch')
    world,metadata=generate_v3_world(config['world_seed'])
    queries,lexical=repair_lexical_leakage(world,metadata,generate_v3_queries(world,config['query_seed']))
    errors=validate_gold(world,metadata,queries)
    if errors:raise ValueError('Invalid canonical gold: '+str(errors[:5]))
    data.mkdir(parents=True);output.mkdir(parents=True)
    _write_jsonl(data/'documents.jsonl',[asdict(world.documents[k]) for k in sorted(world.documents)])
    _write_jsonl(data/'principals.jsonl',[asdict(world.principals[k]) for k in sorted(world.principals)])
    _write_jsonl(data/'metadata.jsonl',[metadata[k] for k in sorted(metadata)])
    _write_jsonl(data/'queries.jsonl',[asdict(q) for q in queries])
    _write_csv(data/'lexical_leakage.csv',lexical)
    manifest={'corpus_version':'v3','generator_version':config['generator_version'],
              'config':config,'config_sha256':_sha(config_path),
              'generator_source_sha256':{p.name:_sha(p) for p in _sources()},
              'corpus_sha256':corpus_hash(world,metadata,queries),
              'data_sha256':{p.name:_sha(p) for p in sorted(data.iterdir())},
              'logical_documents':len(metadata),'document_versions':len(world.documents),
              'queries':len(queries),'gold_validation_errors':errors,
              'pre_retrieval_frozen':True,'created_without_retrieval_scores':True,
              'git_commit':_git_commit()}
    (data/'manifest.json').write_text(_json(manifest))
    (output/'manifest.json').write_text(_json(manifest))
    return manifest


def _load(data,manifest):
    for name,expected in manifest['data_sha256'].items():
        if _sha(data/name)!=expected:raise ValueError('Frozen v3 data modified: '+name)
    world=WorldLedger()
    for line in (data/'principals.jsonl').read_text().splitlines():
        raw=json.loads(line);raw['groups']=frozenset(raw['groups']);world.add_principal(Principal(**raw))
    for line in (data/'documents.jsonl').read_text().splitlines():
        raw=json.loads(line);raw['required_groups']=frozenset(raw['required_groups'])
        world.add_document(DocumentVersion(**raw))
    metadata={}
    for line in (data/'metadata.jsonl').read_text().splitlines():
        raw=json.loads(line);metadata[raw['logical_document_id']]=raw
    queries=[]
    for line in (data/'queries.jsonl').read_text().splitlines():
        raw=json.loads(line)
        for key in ('logical_document_ids','required_document_ids','required_atomic_claims'):
            raw[key]=tuple(raw[key])
        queries.append(V3Query(**raw))
    if corpus_hash(world,metadata,queries)!=manifest['corpus_sha256']:
        raise ValueError('Frozen v3 corpus hash mismatch')
    errors=validate_gold(world,metadata,queries)
    if errors:raise ValueError('Frozen v3 gold invalid: '+str(errors[:5]))
    return world,metadata,queries


def _retrievers(chunks,config):
    snapshot=Path(config['embedding_snapshot'])
    if not snapshot.is_dir() or snapshot.name!=config['embedding_revision']:
        raise ValueError('Pinned local E5 snapshot unavailable')
    encoder=PromptedEncoder(SentenceTransformerEncoder(str(snapshot),snapshot.name),
                            config['embedding_query_prefix'],config['embedding_passage_prefix'])
    dense=DenseRetriever(chunks,model_name=config['embedding_model'],
                         model_version=config['embedding_revision'],encoder=encoder)
    bm25=BM25Retriever(chunks)
    return {'bm25':bm25,'e5':dense,'hybrid':HybridRetriever(bm25,dense,**config['hybrid'])}


def _evaluate(world,queries,retrievers,top_k,timed=True):
    if timed:
        for ret in retrievers.values():ret.retrieve(queries[0].text,top_k)
    rows=[];rankings={};gold_ranks=[]
    for q in queries:
        rankings[q.query_id]={}
        for name,retriever in retrievers.items():
            start=time.perf_counter();hits=retriever.retrieve(q.text,top_k)
            latency=(time.perf_counter()-start)*1000
            context=[h for h in hits if world.allowed(q.principal_id,h.chunk.logical_document_id,q.logical_time)]
            gold=set(q.required_document_ids);got={h.chunk.document_id for h in context}
            claim_coverage=[];ranks={};claim_ranks={}
            for claim in q.required_atomic_claims:
                doc=claim['document_id'];token=f"{claim['value']} {claim['unit']}"
                rank=next((i for i,h in enumerate(hits,1) if h.chunk.document_id==doc),None)
                claim_rank=next((i for i,h in enumerate(hits,1) if h.chunk.document_id==doc and token in h.chunk.text),None)
                ranks[doc]=rank;claim_ranks[doc]=claim_rank
                claim_coverage.append(any(h.chunk.document_id==doc and token in h.chunk.text for h in context))
                gold_ranks.append({'query_id':q.query_id,'domain':q.domain,'query_family':q.family,
                                   'retriever':name,'gold_document_id':doc,'document_rank_at_10':rank,
                                   'claim_chunk_rank_at_10':claim_rank})
            rankings[q.query_id][name]=[h.chunk.chunk_id for h in hits]
            rows.append({'query_id':q.query_id,'domain':q.domain,'query_family':q.family,
                         'retriever':name,'principal_id':q.principal_id,'tenant_id':q.tenant_id,
                         'logical_time':q.logical_time,'answerable':int(q.answerable),
                         'required_count':len(gold),'required_document_ids':json.dumps(sorted(gold)),
                         'required_claims':json.dumps(q.required_atomic_claims,sort_keys=True),
                         'candidate_chunk_ids':json.dumps([h.chunk.chunk_id for h in hits]),
                         'context_chunk_ids':json.dumps([h.chunk.chunk_id for h in context]),
                         'gold_document_ranks':json.dumps(ranks,sort_keys=True),
                         'gold_claim_chunk_ranks':json.dumps(claim_ranks,sort_keys=True),
                         'cer_at_10':current_evidence_recall(context,gold),
                         'scrs':int(state_consistent_retrieval_success(context,gold,q.principal_id,world,q.logical_time)),
                         'all_required_success':int(gold.issubset(got)) if gold else None,
                         'claim_chunk_all_required':int(all(claim_coverage)) if gold else None,
                         'invalid_evidence_absence_success':invalid_evidence_absence_success(context,gold,set(q.logical_document_ids)),
                         'latency_ms':latency if timed else None})
    return rows,rankings,gold_ranks


def _summary_by(rows,field,include_negative=False):
    grouped=defaultdict(list)
    for row in rows:
        if not include_negative and not row['answerable']:continue
        grouped[(row['retriever'],row[field])].append(row)
    return [{'retriever':r,'scope':scope,'n_queries':len(group),
             'cer_at_10':_mean([x['cer_at_10'] for x in group if x['cer_at_10'] is not None]),
             'scrs':_mean([x['scrs'] for x in group]),
             'all_required_success':_mean([x['all_required_success'] for x in group if x['all_required_success'] is not None]),
             'claim_chunk_all_required':_mean([x['claim_chunk_all_required'] for x in group if x['claim_chunk_all_required'] is not None]),
             'negative_absence_success':_mean([x['invalid_evidence_absence_success'] for x in group if x['invalid_evidence_absence_success'] is not None]),
             'median_latency_ms':statistics.median(x['latency_ms'] for x in group),
             'p95_latency_ms':_quantile([x['latency_ms'] for x in group],.95)}
            for (r,scope),group in sorted(grouped.items())]


def _distribution(values):
    values=list(values)
    return {'n':len(values),'min':min(values),'median':statistics.median(values),
            'p75':_quantile(values,.75),'p90':_quantile(values,.90),
            'p95':_quantile(values,.95),'max':max(values),'mean':_mean(values)}


def _figures(output,length_rows,archetype_rows,lexical,domain_rows,multi_rows,gold_ranks,association):
    figures=output/'figures';figures.mkdir()
    _hist_plot(figures,'document_length','Current document lengths',
               [r['word_count'] for r in length_rows])
    _hist_plot(figures,'chunks_per_document','Chunks per current document',
               [r['chunk_count'] for r in length_rows])
    valid=[r for r in lexical if r['answerable']]
    _hist_plot(figures,'lexical_overlap','Query/gold and best nongold token overlap',
               [r['query_gold_overlap'] for r in valid],
               [r['query_best_nongold_overlap'] for r in valid])
    _hist_plot(figures,'gold_rank','Gold document rank at 10 (misses in last bin)',
               [r['document_rank_at_10'] if r['document_rank_at_10'] is not None else 11 for r in gold_ranks])
    def bars(name,title,categories,series):
        c=Canvas(title,'category','fraction')
        palette={'bm25':'#235789','e5':'#c05621','hybrid':'#2f855a','share':'#6b46c1'}
        c.legend([(label,palette.get(label,'#6b46c1')) for label in series])
        step=650/max(1,len(categories));bar_width=max(7,int(step/(len(series)+1)))
        for i,label in enumerate(categories):
            for j,(key,values) in enumerate(series.items()):
                x=int(85+i*step+j*bar_width)
                c.rect(x,c.y(values[i]),x+bar_width-2,430,palette.get(key,'#6b46c1'))
            c.label(int(85+i*step+step/2),450,str(label)[:13],size=9)
        c.save(figures/name)
    total=sum(r['count'] for r in archetype_rows)
    bars('archetype_distribution','Source archetype distribution',
         [r['archetype'] for r in archetype_rows],{'share':[r['count']/total for r in archetype_rows]})
    domains=[d.name for d in DOMAINS]
    bars('cer_scrs_by_domain','Clean SCRS by retriever and domain',domains,
         {m:[next(r['scrs'] for r in domain_rows if r['retriever']==m and r['scope']==d)
             for d in domains] for m in ('bm25','e5','hybrid')})
    bars('multi_document_success','Multi-document all-required success',
         [r['retriever'] for r in multi_rows],
         {'share':[r['all_required_success'] for r in multi_rows]})
    cats=['axis_one','axis_two']
    bars('tenant_scope_association',"Tenant/scope association (Cramer's V)",cats,
         {'share':[association[c]['cramers_v'] for c in cats]})


def validate(config_path:Path,data:Path,output:Path):
    if not (data/'manifest.json').is_file() or not (output/'manifest.json').is_file():
        raise ValueError('Generate and freeze v3 first')
    if (output/'clean_gate_summary.json').exists():raise ValueError('Completed v3 validation is immutable')
    manifest=json.loads((data/'manifest.json').read_text())
    if _sha(data/'manifest.json')!=_sha(output/'manifest.json') or _sha(config_path)!=manifest['config_sha256']:
        raise ValueError('v3 manifest or frozen config changed')
    for source in _sources():
        if _sha(source)!=manifest['generator_source_sha256'][source.name]:
            raise ValueError('Frozen v3 source changed: '+source.name)
    config=manifest['config'];world,metadata,queries=_load(data,manifest)
    again_world,again_meta=generate_v3_world(config['world_seed'])
    again_queries,_=repair_lexical_leakage(again_world,again_meta,generate_v3_queries(again_world,config['query_seed']))
    generation_deterministic=corpus_hash(again_world,again_meta,again_queries)==manifest['corpus_sha256']
    if not generation_deterministic:raise ValueError('v3 generation nondeterministic')
    lexical=leakage_rows(world,metadata,queries)
    chunks=build_chunks_from_ledger(world,LOGICAL_TIME)
    chunk_map=defaultdict(list)
    for chunk in chunks:chunk_map[chunk.logical_document_id].append(chunk)
    length_rows=[];position_rows=[]
    for lid,meta in sorted(metadata.items()):
        doc=world.document_at(lid,LOGICAL_TIME);own_chunks=chunk_map[lid]
        length_rows.append({'logical_document_id':lid,'domain':meta['domain'],
                            'archetype':meta['archetype'],'kind':meta['kind'],
                            'word_count':len(doc.content.split()),'chunk_count':len(own_chunks),
                            'length_bucket':meta['length_bucket']})
        if meta['kind'] not in ('BASE','EXC'):continue
        position_rows.append({'logical_document_id':lid,'domain':meta['domain'],
                              'evidence_position':meta['evidence_position'],
                              'chunk_count':len(own_chunks),
                              'version_sensitive':int(meta['version_sensitive']),
                              'authorization_sensitive':int(meta['kind']=='EXC')})
    association={axis:cramers_v(list(metadata.values()),axis) for axis in ('axis_one','axis_two')}
    archetypes=Counter(m['archetype'] for m in metadata.values())
    archetype_rows=[{'archetype':a,'count':archetypes[a],
                     'fraction':archetypes[a]/len(metadata)} for a in ARCHETYPES]
    distractors=[];multi_structure=[]
    for q in queries:
        first=metadata[q.logical_document_ids[0]]
        same=[m for m in metadata.values() if m['domain']==q.domain and
              m['topic_index']==first['topic_index'] and m['logical_document_id'] not in q.logical_document_ids]
        distractors.append(len(same))
        if q.family=='multi_document':
            parts=[metadata[lid] for lid in q.logical_document_ids]
            multi_structure.append({'query_id':q.query_id,'domain':q.domain,
                'evidence_relation':q.evidence_relation,
                'required_logical_ids':json.dumps(q.logical_document_ids),
                'source_archetypes':json.dumps([p['archetype'] for p in parts]),
                'different_source_types':int(parts[0]['archetype']!=parts[1]['archetype']),
                'same_logical_family':int(parts[0]['topic_index']==parts[1]['topic_index']),
                'different_logical_documents':int(parts[0]['logical_document_id']!=parts[1]['logical_document_id']),
                'claim_values':json.dumps([c['value'] for c in q.required_atomic_claims])})
    os.environ.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',HF_DATASETS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    import torch,sentence_transformers
    torch.manual_seed(0);torch.set_num_threads(1)
    start=time.perf_counter();retrievers=_retrievers(chunks,config)
    index_build_ms=(time.perf_counter()-start)*1000
    rows,rankings,gold_ranks=_evaluate(world,queries,retrievers,10)
    torch.manual_seed(0)
    repeated,repeat_ranks,_=_evaluate(world,queries,_retrievers(chunks,config),10,timed=False)
    ranking_deterministic=rankings==repeat_ranks
    metric_deterministic=all((a['cer_at_10'],a['scrs'],a['all_required_success'],a['claim_chunk_all_required'])==
                             (b['cer_at_10'],b['scrs'],b['all_required_success'],b['claim_chunk_all_required'])
                             for a,b in zip(rows,repeated))
    domain_rows=_summary_by(rows,'domain')
    family_rows=_summary_by(rows,'query_family',True)
    multi_rows=[]
    for method in retrievers:
        group=[r for r in rows if r['retriever']==method and r['query_family']=='multi_document']
        multi_rows.append({'retriever':method,'n_queries':len(group),
                           'cer_at_10':_mean([r['cer_at_10'] for r in group]),
                           'scrs':_mean([r['scrs'] for r in group]),
                           'all_required_success':_mean([r['all_required_success'] for r in group]),
                           'claim_chunk_all_required':_mean([r['claim_chunk_all_required'] for r in group])})
    overall={method:{'n':len(group),'cer_at_10':_mean([r['cer_at_10'] for r in group]),
                     'scrs':_mean([r['scrs'] for r in group]),
                     'claim_chunk_all_required':_mean([r['claim_chunk_all_required'] for r in group]),
                     'median_latency_ms':statistics.median(r['latency_ms'] for r in group),
                     'p95_latency_ms':_quantile([r['latency_ms'] for r in group],.95)}
             for method in retrievers for group in [[r for r in rows if r['retriever']==method and r['answerable']]]}
    negative={method:_mean([r['invalid_evidence_absence_success'] for r in rows if r['retriever']==method and not r['answerable']])
              for method in retrievers}
    answerable=[r for r in lexical if r['answerable']]
    non_id=[r for r in answerable if r['family']!='identifier_bearing']
    id_rate=_mean([r['gold_identifier_overlap'] for r in answerable])
    unique_rate=_mean([r['unique_gold_token_leak'] for r in non_id])
    title_rate=_mean([r['exact_title_overlap'] for r in answerable])
    words=_distribution(r['word_count'] for r in length_rows)
    chunk_dist=_distribution(r['chunk_count'] for r in length_rows)
    share_multi=_mean([r['chunk_count']>1 for r in length_rows])
    version_multi=sum(m['version_sensitive'] and len(chunk_map[lid])>1 for lid,m in metadata.items())
    auth_multi=sum(m['kind']=='EXC' and len(chunk_map[lid])>1 for lid,m in metadata.items())
    mixed_feasible=version_multi
    corpus_stats={'corpus_version':'v3','corpus_sha256':manifest['corpus_sha256'],
        'logical_documents':len(metadata),'document_versions':len(world.documents),
        'indexed_chunks':len(chunks),'base_questions':len(queries),
        'answerable_questions':sum(q.answerable for q in queries),
        'negative_questions':sum(not q.answerable for q in queries),
        'document_words':words,'chunks_per_document':chunk_dist,
        'chunk_shares':{'one':_mean([r['chunk_count']==1 for r in length_rows]),
                        'two_to_three':_mean([2<=r['chunk_count']<=3 for r in length_rows]),
                        'four_plus':_mean([r['chunk_count']>=4 for r in length_rows]),
                        'more_than_one':share_multi},
        'archetypes':dict(archetypes),
        'identifier_leakage_rate':id_rate,'non_identifier_unique_gold_token_rate':unique_rate,
        'exact_title_overlap_rate':title_rate,
        'query_gold_overlap':_distribution(r['query_gold_overlap'] for r in answerable),
        'query_nongold_overlap':_distribution(r['query_best_nongold_overlap'] for r in answerable),
        'gold_lexical_advantage':_distribution(r['gold_lexical_advantage'] for r in answerable),
        'construction_warnings':sum(r['construction_warning'] for r in lexical),
        'partial_propagation_feasibility':{'version_sensitive_multi_chunk':version_multi,
                                          'authorization_sensitive_multi_chunk':auth_multi,
                                          'mixed_version_candidates':mixed_feasible}}
    thresholds=config['gates_frozen_before_retrieval']
    macro=lambda method,field:_mean([r[field] for r in domain_rows if r['retriever']==method])
    hybrid_domain=[r['scrs'] for r in domain_rows if r['retriever']=='hybrid']
    gates={
        'R1':'PASS' if len([x for x in archetypes.values() if x])>=thresholds['R1']['min_archetypes'] and
                    max(archetypes.values())/len(metadata)<=thresholds['R1']['max_archetype_share'] else 'FAIL',
        'R2':'PASS' if words['median']>=thresholds['R2']['median_words_min'] and words['p90']>=thresholds['R2']['p90_words_min'] else 'FAIL',
        'R3':'PASS' if share_multi>=thresholds['R3']['multi_chunk_share_min'] else 'FAIL',
        'R4':'PASS' if all(association[a]['cramers_v']<thresholds['R4']['tenant_axis_cramers_v_max'] for a in association) else 'FAIL',
        'R5':'PASS' if id_rate<=thresholds['R5']['identifier_leakage_max'] and unique_rate<=thresholds['R5']['non_identifier_unique_gold_token_max'] and title_rate<=thresholds['R5']['exact_title_overlap_max'] and not corpus_stats['construction_warnings'] else 'FAIL',
        'R6':'PASS' if macro('bm25','cer_at_10')>=thresholds['R6']['bm25_macro_cer_at_10_min'] else 'FAIL',
        'R7':'PASS' if macro('e5','cer_at_10')>=thresholds['R7']['e5_macro_cer_at_10_min'] else 'FAIL',
        'R8':'PASS' if macro('hybrid','scrs')>=thresholds['R8']['hybrid_macro_scrs_min'] and min(hybrid_domain)>=thresholds['R8']['hybrid_min_domain_scrs'] else 'FAIL',
        'R9':'PASS' if next(r['all_required_success'] for r in multi_rows if r['retriever']=='hybrid')>=thresholds['R9']['hybrid_multi_document_all_required_min'] else 'FAIL',
        'R10':'PASS' if min(negative.values())>=thresholds['R10']['negative_absence_success_min'] else 'FAIL',
        'R11':'PASS' if generation_deterministic and ranking_deterministic and metric_deterministic else 'FAIL'}
    summary={'corpus_version':'v3','generator_version':config['generator_version'],
        'corpus_hash':manifest['corpus_sha256'],'gates':gates,
        'all_critical_gates_pass':all(x=='PASS' for x in gates.values()),
        'overall':overall,'macro_cer_at_10':{m:macro(m,'cer_at_10') for m in retrievers},
        'macro_scrs':{m:macro(m,'scrs') for m in retrievers},
        'hybrid_min_domain_scrs':min(hybrid_domain),
        'multi_document':multi_rows,'negative_absence_success':negative,
        'multi_document_diagnostic_expectation_met':any(r['all_required_success']<1 for r in multi_rows),
        'generation_deterministic':generation_deterministic,
        'ranking_deterministic':ranking_deterministic,'metric_deterministic':metric_deterministic,
        'index_build_ms':index_build_ms,
        'environment':{'python':sys.version.split()[0],'platform':platform.platform(),
            'torch':torch.__version__,'sentence_transformers':sentence_transformers.__version__,
            'device':'cpu','model':config['embedding_model'],
            'revision':config['embedding_revision'],'git_commit':_git_commit()},
        'fault_campaign_executed':False}
    _write_csv(output/'archetype_distribution.csv',archetype_rows)
    _write_csv(output/'document_length_distribution.csv',length_rows)
    _write_csv(output/'chunk_distribution.csv',length_rows)
    (output/'tenant_scope_association.json').write_text(_json(association))
    (output/'distractor_statistics.json').write_text(_json({'plausible_same_topic_nongold_per_query':_distribution(distractors)}))
    (output/'evidence_position_statistics.json').write_text(_json({
        'declared_positions':dict(Counter(r['evidence_position'] for r in position_rows)),
        'by_domain':{d.name:dict(Counter(r['evidence_position'] for r in position_rows if r['domain']==d.name)) for d in DOMAINS}}))
    _write_csv(output/'multi_document_structure.csv',multi_structure)
    _write_csv(output/'lexical_leakage.csv',lexical)
    _write_csv(output/'clean_retrieval.csv',rows)
    _write_csv(output/'gold_rank_distribution.csv',gold_ranks)
    _write_csv(output/'domain_summary.csv',domain_rows)
    _write_csv(output/'query_family_summary.csv',family_rows)
    (output/'corpus_statistics.json').write_text(_json(corpus_stats))
    (output/'clean_gate_summary.json').write_text(_json(summary))
    _figures(output,length_rows,archetype_rows,lexical,domain_rows,multi_rows,gold_ranks,association)
    report=['# Phase 7C v3 clean corpus validation','',
            f"Corpus hash: `{manifest['corpus_sha256']}`. No transition faults were run.",'',
            '## Corpus structure','',
            f"{len(metadata)} logical documents, {len(world.documents)} stored versions, {len(chunks)} indexed chunks, {len(queries)} questions.",
            f"Median length {words['median']} words; p90 {words['p90']}. Multi-chunk share {share_multi:.1%}.",
            f"Tenant association V: axis one {association['axis_one']['cramers_v']:.3f}, axis two {association['axis_two']['cramers_v']:.3f}.",
            f"Identifier leakage {id_rate:.1%}; non-identifier unique-gold-token rate {unique_rate:.1%}; title overlap {title_rate:.1%}.",'',
            '## Clean retrieval','',
            '| Retriever | CER@10 | SCRS | Claim-bearing evidence | Multi all-required | Negative absence |',
            '| --- | ---: | ---: | ---: | ---: | ---: |',
            *[f"| {m} | {overall[m]['cer_at_10']:.3f} | {overall[m]['scrs']:.3f} | "
              f"{overall[m]['claim_chunk_all_required']:.3f} | "
              f"{next(r['all_required_success'] for r in multi_rows if r['retriever']==m):.3f} | {negative[m]:.3f} |"
              for m in retrievers],
            '', '## Gates','',*[f'- {k}: {v}' for k,v in gates.items()],
            '',f"Predeclared multi-document diagnostic expectation met: {summary['multi_document_diagnostic_expectation_met']}.",
            '', 'Document-level SCRS and CER retain their frozen definitions. Claim-bearing chunk coverage is reported separately because a document may be retrieved without its answer-bearing chunk.']
    (output/'summary.md').write_text('\n'.join(report)+'\n')
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('stage',choices=('generate','validate'))
    parser.add_argument('--config',default='configs/phase7c-enterprise-v3.json')
    parser.add_argument('--data',default='data/generated-v3')
    parser.add_argument('--output',default='results/phase7c-enterprise-realism-v3')
    args=parser.parse_args()
    result=(generate if args.stage=='generate' else validate)(Path(args.config),Path(args.data),Path(args.output))
    print(_json({'stage':args.stage,'corpus_hash':result.get('corpus_sha256',result.get('corpus_hash')),
                 'gates':result.get('gates')}))


if __name__=='__main__':main()
