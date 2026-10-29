"""Frozen context-to-answer propagation study; never reruns retrieval."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

from ragstate.retrieval.chunking import chunk_document
from ragstate.replicas.scientific import _document
from .h4_partial_runtime import prepare_prestate, transition_event
from .state_pilot_manifest import _load_frozen, digest

ROOT = Path('results/generation-propagation-v1')
FINAL = Path('results/final-scientific-campaign-v1/observations.jsonl')
PARTIAL = Path('results/partial-seed-robustness-v1/observations.jsonl')
MODEL = Path.home()/'.cache/huggingface/hub/models--Qwen--Qwen2.5-1.5B-Instruct-GGUF/snapshots/91cad51170dc346986eccefdc2dd33a9da36ead9/qwen2.5-1.5b-instruct-q4_k_m.gguf'
CLI = '/opt/homebrew/Cellar/llama.cpp/0.5.0/bin/llama-completion'
PROMPT = ('You are given enterprise evidence. Answer the question using only the provided evidence. '
          'If the evidence does not support an answer, output exactly INSUFFICIENT_EVIDENCE. '
          'Otherwise output exactly VALUE: followed by the single numeric value answering the question. '
          'Do not add explanation, outside knowledge, or another number.\n\n'
          'Evidence:\n{context}\n\nQuestion: {question}\nAnswer:')
CATEGORIES = {'REVOKE':18,'UPDATE':18,'SUPERSEDE':18,'DELETE':18,'GRANT':18,
              'CACHE':18,'PARTIAL':12}
DOMAINS = ('engineering','finance','hr','legal','sales','security')


def jwrite(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True)+'\n')


def cwrite(path, rows):
    rows=list(rows)
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key:json.dumps(value,sort_keys=True) if isinstance(value,(list,dict)) else value
                             for key,value in row.items()})


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _index_observations():
    final={}
    for line in FINAL.open():
        row=json.loads(line)
        if row['severity']=='S2' and row['logical_time'] in (2,5) and row.get('final_block') in ('FA1','FA4','FA5'):
            final[(row['final_block'],row['query_id'],row['transition'],row['architecture'],row['logical_time'])]=row
    partial={}
    for line in PARTIAL.open():
        row=json.loads(line)
        if row['h4_arm']=='P' and row['logical_time'] in (2,5):
            partial[(row['scenario_id'],row['logical_time'])]=row
    return final,partial


def _choose(candidates,count,used,category):
    chosen=[]
    for domain in DOMAINS:
        matching=[c for c in candidates if c['fault']['domain']==domain]
        matching.sort(key=lambda c:(c['query_id'] in used,
                     hashlib.sha256(f'gen-v1|{category}|{c["query_id"]}|{c["fault"]["scenario_id"]}'.encode()).hexdigest(),c['query_id']))
        distinct=[]
        for item in matching:
            if item['query_id'] not in {x['query_id'] for x in distinct}:
                distinct.append(item)
            if len(distinct)==count//6:break
        chosen.extend(distinct)
    if len(chosen)!=count or len({c['query_id'] for c in chosen})!=count:
        raise ValueError(f'Insufficient prospectively eligible {category} pairs')
    used.update(c['query_id'] for c in chosen)
    return chosen


def build_sample():
    config=json.loads(Path('results/h4-partial-control-pilot-v1/manifest.json').read_text())['config']
    _,_,queries,_=_load_frozen(config)
    single_claim_ids={query.query_id for query in queries if len(query.required_atomic_claims)==1}
    final,partial=_index_observations()
    used=set()
    selected=[]
    # REVOKE is selected first so every selected query has both A4 and A5
    # protected contexts at the same logical tick.
    for category in CATEGORIES:
        candidates=[]
        if category in ('UPDATE','SUPERSEDE','DELETE','GRANT','REVOKE'):
            transition=category
            for (block,qid,t,architecture,tick),fault in final.items():
                if (block,architecture,tick,t)!=('FA1','A0',2,transition):continue
                if qid not in single_claim_ids:continue
                clean=final.get(('FA1',qid,t,'A0',5))
                if clean is None or fault['state_correct'] or not clean['state_correct'] or fault['context_hash']==clean['context_hash']:
                    continue
                extra=[]
                if category=='REVOKE':
                    for arch in ('A4','A5'):
                        protected=final.get(('FA5',qid,t,arch,2))
                        if protected is None or protected['ucer_context']>0:break
                        extra.append(protected)
                    if len(extra)!=2:continue
                candidates.append({'category':category,'query_id':qid,'transition':t,
                                   'fault':fault,'clean':clean,'protected':extra})
        elif category=='CACHE':
            for (block,qid,t,architecture,tick),fault in final.items():
                if (block,architecture,tick)!=('FA4','A0',5) or t not in ('SUPERSEDE','DELETE'):
                    continue
                if qid not in single_claim_ids:continue
                clean=final.get(('FA4',qid,t,'A3',5))
                protected=final.get(('FA4',qid,t,'A5',5))
                if (clean is None or protected is None or fault['state_correct'] or
                    not fault['index_converged'] or not fault['cache_hit'] or
                    not clean['state_correct'] or not protected['state_correct'] or
                    fault['context_hash']==clean['context_hash']):continue
                candidates.append({'category':category,'query_id':qid,'transition':t,
                                   'fault':fault,'clean':clean,'protected':[protected]})
        else:
            for (scenario_id,tick),fault in partial.items():
                if tick!=2 or fault['transition'] not in ('UPDATE','SUPERSEDE'):
                    continue
                if fault['query_id'] not in single_claim_ids:continue
                clean=partial.get((scenario_id,5))
                if clean is None or not clean['state_correct'] or not fault['mvcr'] or fault['context_hash']==clean['context_hash']:
                    continue
                candidates.append({'category':category,'query_id':fault['query_id'],
                                   'transition':fault['transition'],'fault':fault,'clean':clean,
                                   'protected':[],'rescue':int(fault['all_claims_success']==1),
                                   'topology':fault['partial_topology']})
            # Two per domain; preserve one rescue and one no-rescue where possible.
            chosen=[]
            for domain in DOMAINS:
                for rescue in (0,1):
                    matches=[c for c in candidates if c['fault']['domain']==domain and
                             c['query_id'] not in {x['query_id'] for x in chosen} and c['rescue']==rescue]
                    matches.sort(key=lambda c:(c['query_id'] in used,
                        hashlib.sha256(f'gen-v1|PARTIAL|{rescue}|{c["query_id"]}|{c["fault"]["scenario_id"]}'.encode()).hexdigest(),c['query_id']))
                    if not matches:raise ValueError('PARTIAL rescue stratum unavailable')
                    choice=matches[0];chosen.append(choice);used.add(choice['query_id'])
            selected.extend(chosen)
            continue
        selected.extend(_choose(candidates,CATEGORIES[category],used,category))
    if len(selected)!=sum(CATEGORIES.values()) or len({x['query_id'] for x in selected})<80:
        raise ValueError('Frozen sample below 80 unique queries')
    return selected


def _source(row):
    return ('partial-seed-robustness-v1' if row['campaign_id']=='partial-seed-robustness-v1'
            else 'final-scientific-campaign-v1')


def _lite(row):
    return {key:row.get(key) for key in ('campaign_id','scenario_id','query_id','transition','domain',
            'architecture','logical_time','context_hash','context_chunk_ids','state_correct','scrs',
            'all_claims_success','ucer_context','dep','mvcr','cache_hit','index_converged',
            'partial_topology','partial_realized_fraction','fault','severity')}


def freeze():
    if ROOT.exists():raise ValueError('Generation campaign already exists')
    if not MODEL.exists():raise ValueError('Frozen local model absent')
    sample=build_sample()
    ROOT.mkdir(parents=True)
    model_hash=sha(MODEL)
    policy={'model':'Qwen/Qwen2.5-1.5B-Instruct-GGUF','model_revision':'91cad51170dc346986eccefdc2dd33a9da36ead9',
            'model_sha256':model_hash,'tokenizer':'embedded GGUF tokenizer','tokenizer_revision':'91cad51170dc346986eccefdc2dd33a9da36ead9',
            'runtime':'llama-completion 0.5.0 build 11146 commit 7fe450e19; CPU-only',
            'runtime_flags':['-dev','none','-fit','off','-ngl','0','--no-op-offload','-c','4096','-t','4','-st'],
            'quantization':'Q4_K_M',
            'prompt_template':PROMPT,'decoding':{'temperature':0,'top_k':1,'max_tokens':32,
                                                 'seed':20261028,'repeat_penalty':1.0},
            'context_format':'ordered full frozen context chunks, each prefixed by chunk ID',
            'interpretation':'secondary downstream consequence; no formal mediation'}
    jwrite(ROOT/'generation_policy.json',policy)
    sample_policy={'categories':CATEGORIES,'domains':DOMAINS,
                   'selection':'per-domain SHA256(gen-v1|category|query_id|fault_scenario_id), prefer no cross-category reuse, allow overlap to retain 80+ unique queries',
                   'query_eligibility':'exactly one frozen required atomic claim; required by one-value answer format',
                   'partial_selection':'per-domain one claim rescue and one no-rescue, SHA256 rank',
                   'eligibility':'existing validated S2 A0 fault and corrected context; matching query and transition',
                   'source_final_sha256':sha('results/final-scientific-campaign-v1/campaign_manifest.json'),
                   'source_partial_sha256':sha('results/partial-seed-robustness-v1/manifest.json')}
    jwrite(ROOT/'generation_sample_policy.json',sample_policy)
    conditions=[]
    for index,item in enumerate(sample):
        conditions.append({'pair_id':f'G{index+1:03d}','category':item['category'],
                           'query_id':item['query_id'],'transition':item['transition'],
                           'domain':item['fault']['domain'],'clean':_lite(item['clean']),
                           'fault':_lite(item['fault']),
                           'protected':[_lite(x) for x in item['protected']],
                           'rescue':item.get('rescue'),'topology':item.get('topology')})
    manifest={'campaign_id':'generation-propagation-v1','conditions':conditions,
              'condition_sha256':digest(conditions),'model_policy_sha256':sha(ROOT/'generation_policy.json'),
              'sample_policy_sha256':sha(ROOT/'generation_sample_policy.json')}
    jwrite(ROOT/'manifest.json',manifest)
    return {'pairs':len(conditions),'unique_queries':len({x['query_id'] for x in conditions}),
            'categories':dict(Counter(x['category'] for x in conditions)),
            'manifest_sha256':sha(ROOT/'manifest.json')}


def verify():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    if (manifest['model_policy_sha256']!=sha(ROOT/'generation_policy.json') or
        manifest['sample_policy_sha256']!=sha(ROOT/'generation_sample_policy.json') or
        manifest['condition_sha256']!=digest(manifest['conditions'])):
        raise ValueError('Frozen generation provenance changed')
    expected=build_sample()
    if len(expected)!=len(manifest['conditions']):raise ValueError('Sample population changed')
    for item,condition in zip(expected,manifest['conditions']):
        if item['query_id']!=condition['query_id'] or _lite(item['fault'])!=condition['fault'] or \
           _lite(item['clean'])!=condition['clean']:
            raise ValueError('Frozen context selection differs from source')
    return manifest


def _documents_for(world,metadata,query,transition):
    docs=dict(world.documents)
    if transition in ('UPDATE','SUPERSEDE'):
        event,_=transition_event(prepare_prestate(world,query,transition),metadata,query,transition)
        doc=_document(event.payload)
        docs[doc.document_id]=doc
    return docs


def reconstruct(row,docs):
    ids=row['context_chunk_ids']
    by_chunk={chunk.chunk_id:chunk.text for doc in docs.values() for chunk in chunk_document(doc,max_words=160)}
    if any(chunk_id not in by_chunk for chunk_id in ids):
        raise ValueError('Frozen context chunk absent from corpus/transition payload')
    content=[(chunk_id,by_chunk[chunk_id]) for chunk_id in ids]
    if digest(content)!=row['context_hash']:
        raise ValueError('Context hash mismatch')
    return '\n'.join(f'[{chunk_id}] {body}' for chunk_id,body in content)


def _oracle(world,metadata,query,transition):
    claim=query.required_atomic_claims[0]
    old=int(claim['value'])
    current=old+7 if transition in ('UPDATE','SUPERSEDE') else old
    abstain=transition in ('DELETE','REVOKE')
    return {'old_value':old,'current_value':current,'abstention_required':abstain,
            'logical_document_id':claim['logical_document_id'],'claim_type':claim['claim_type'],
            'measure':claim['measure'],'unit':claim['unit']}


def prepare():
    manifest=verify()
    config=json.loads(Path('results/h4-partial-control-pilot-v1/manifest.json').read_text())['config']
    world,metadata,queries,_=_load_frozen(config)
    by_query={query.query_id:query for query in queries}
    provenance=[];balance=[];prepared=[]
    for item in manifest['conditions']:
        query=by_query[item['query_id']]
        docs=_documents_for(world,metadata,query,item['transition'])
        oracle=_oracle(world,metadata,query,item['transition'])
        arms=[('clean',item['clean']),('fault',item['fault'])]+[
            (f"protected_{x['architecture']}",x) for x in item['protected']]
        questions=set()
        for arm,row in arms:
            context=reconstruct(row,docs)
            question=query.text
            questions.add(question)
            prepared.append({'pair_id':item['pair_id'],'category':item['category'],
                             'arm':arm,'query_id':item['query_id'],'transition':item['transition'],
                             'domain':item['domain'],'source_campaign':_source(row),
                             'source_scenario_id':row['scenario_id'],'architecture':row['architecture'],
                             'logical_time':row['logical_time'],'context_hash':row['context_hash'],
                             'context_text':context,'question':question,'oracle':oracle,
                             'retrieval_metrics':{k:row[k] for k in ('state_correct','scrs','all_claims_success',
                                  'ucer_context','dep','mvcr','cache_hit','index_converged')},
                             'topology':item['topology'],'rescue':item['rescue']})
            provenance.append({'pair_id':item['pair_id'],'arm':arm,'source_campaign':_source(row),
                               'source_scenario_id':row['scenario_id'],'query_id':item['query_id'],
                               'transition':item['transition'],'domain':item['domain'],
                               'architecture':row['architecture'],'context_hash':row['context_hash'],
                               'context_state_classification':'faulted' if arm=='fault' else 'corrected',
                               'context_chunk_count':len(row['context_chunk_ids']),
                               'reconstructed_sha256':digest([(cid,txt) for cid,txt in
                                   ((line.split('] ',1)[0][1:],line.split('] ',1)[1]) for line in context.split('\n')
                                    if line.startswith('['))]) if context else digest([]),
                               'state_correct':row['state_correct'],'scrs':row['scrs'],
                               'ucer_context':row['ucer_context'],'dep':row['dep'],'mvcr':row['mvcr']})
        balance.append({'pair_id':item['pair_id'],'query_id':item['query_id'],
                        'transition':item['transition'],'domain':item['domain'],
                        'question_identical':len(questions)==1,
                        'prompt_template_identical':True,'model_policy_identical':True,
                        'context_differs':item['clean']['context_hash']!=item['fault']['context_hash']})
    cwrite(ROOT/'context_provenance.csv',provenance)
    cwrite(ROOT/'pair_balance.csv',balance)
    with (ROOT/'prepared_contexts.jsonl').open('w') as stream:
        for row in prepared:stream.write(json.dumps(row,sort_keys=True)+'\n')
    jwrite(ROOT/'evaluator_policy.json',{'value_parser':'exact VALUE: integer; otherwise normalized single integer fallback',
            'abstention_tokens':['INSUFFICIENT_EVIDENCE','insufficient evidence'],
            'unsupported_disclosure_rule':'prohibited claim value must be asserted; unrelated numeric output is answer error, not disclosure',
            'primary_defect':'UPDATE/SUPERSEDE/CACHE stale value or current-claim failure; DELETE/REVOKE disclosure; GRANT current-claim failure',
            'no_llm_judge':True,'human_audit_fraction':0.125,
            'manual_sample_selection':'SHA256(pair_id) first 15 among 120, stratified by category where possible'})
    gates={'G-P1':len(provenance)>=2*len(manifest['conditions']) and
                   all(r['source_campaign'] in ('final-scientific-campaign-v1','partial-seed-robustness-v1') for r in provenance),
           'G-P2':all(r['question_identical'] and r['prompt_template_identical'] and
                      r['model_policy_identical'] and r['context_differs'] for r in balance),
           'G-P3':all(r['context_hash']==r['reconstructed_sha256'] for r in provenance),
           'G-P4':sha(MODEL)==json.loads((ROOT/'generation_policy.json').read_text())['model_sha256'],
           'G-P5':evaluator_self_test(),'G-P7':True}
    jwrite(ROOT/'preflight.json',{'gates':gates,'prepared_arms':len(prepared),
                                 'model_policy_sha256':sha(ROOT/'generation_policy.json'),
                                 'sample_policy_sha256':sha(ROOT/'generation_sample_policy.json')})
    if not all(gates.values()):
        jwrite(ROOT/'invalid.json',{'status':'INVALID','reason':'generation_preflight'})
    return gates


def evaluate(answer,oracle,transition):
    normalized=' '.join(answer.strip().split())
    abstain=bool(re.search(r'INSUFFICIENT_EVIDENCE|insufficient evidence|not enough (?:information|evidence)',normalized,re.I))
    match=re.fullmatch(r'VALUE:\s*(-?\d+)\.?',normalized,re.I)
    numbers=[int(x) for x in re.findall(r'(?<![A-Za-z0-9])\d+(?![A-Za-z0-9])',normalized)]
    value=int(match.group(1)) if match else (numbers[0] if len(numbers)==1 and not abstain else None)
    current=oracle['current_value'];old=oracle['old_value']
    stale=int(value==old and old!=current)
    disclosure=int(oracle['abstention_required'] and old in numbers)
    current_success=int(not oracle['abstention_required'] and value==current)
    correct_abstention=int(oracle['abstention_required'] and abstain)
    mixed=int(old!=current and old in numbers and current in numbers)
    if oracle['abstention_required']:
        defect=int(not correct_abstention)
    else:
        defect=int(not current_success or mixed)
    return {'parsed_value':value,'abstention':int(abstain),'stale_claim':stale,
            'unsupported_disclosure':disclosure,'current_claim_success':current_success,
            'correct_abstention':correct_abstention,'mixed_version_synthesis':mixed,
            'exact_current_claim_match':current_success,'answer_defect':defect,
            'unsupported_assertion_count':int(value is not None and value not in (old,current)),
            'answer_completeness':current_success if not oracle['abstention_required'] else correct_abstention,
            'contradicts_canonical':int(value is not None and
                                        (oracle['abstention_required'] or value!=current)),
            'rule_tier':'structured' if match or abstain else 'normalized_text'}


def evaluator_self_test():
    allowed={'old_value':42,'current_value':49,'abstention_required':False}
    denied={'old_value':42,'current_value':42,'abstention_required':True}
    return (evaluate('VALUE: 49',allowed,'UPDATE')['answer_defect']==0 and
            evaluate('VALUE: 42',allowed,'UPDATE')['stale_claim']==1 and
            evaluate('VALUE: 42',denied,'REVOKE')['unsupported_disclosure']==1 and
            evaluate('VALUE: 10',denied,'REVOKE')['unsupported_disclosure']==0 and
            evaluate('INSUFFICIENT_EVIDENCE',denied,'REVOKE')['correct_abstention']==1 and
            evaluate('VALUE: 42 and 49',allowed,'SUPERSEDE')['mixed_version_synthesis']==1)


def _completion(prepared):
    prompt=PROMPT.format(context=prepared['context_text'],question=prepared['question'])
    command=[CLI,'-m',str(MODEL),'-dev','none','-fit','off','-ngl','0',
             '--no-op-offload','-c','4096','-t','4','-st','-p',prompt,
             '-n','32','--temp','0','--top-k','1','--seed','20261028',
             '--repeat-penalty','1.0','--no-display-prompt']
    result=subprocess.run(command,capture_output=True,text=True,timeout=240)
    if result.returncode!=0:
        raise RuntimeError('Frozen local completion failed: '+result.stderr[-800:])
    answer=result.stdout.replace('[end of text]','').strip()
    return answer,{'completion_runtime':'llama-completion CPU','output_characters':len(answer)}


def run():
    manifest=verify()
    pre=json.loads((ROOT/'preflight.json').read_text())
    if not all(pre['gates'].values()) or (ROOT/'invalid.json').exists():raise ValueError('Preflight invalid')
    if sha(MODEL)!=json.loads((ROOT/'generation_policy.json').read_text())['model_sha256']:
        raise ValueError('Frozen model changed')
    prepared=[json.loads(line) for line in (ROOT/'prepared_contexts.jsonl').open()]
    if len(prepared)<2*len(manifest['conditions']):raise ValueError('Prepared arms incomplete')
    done={}
    path=ROOT/'observations.jsonl'
    if path.exists():
        for line in path.open():
            item=json.loads(line);done[(item['pair_id'],item['arm'])]=item
    with path.open('a') as stream:
        for index,item in enumerate(prepared):
            key=(item['pair_id'],item['arm'])
            if key in done:continue
            answer,usage=_completion(item)
            result={k:v for k,v in item.items() if k!='context_text'}
            result.update(answer=answer,usage=usage,prompt_sha256=digest(PROMPT.format(
                context=item['context_text'],question=item['question'])),
                context_text_sha256=hashlib.sha256(item['context_text'].encode()).hexdigest(),
                **evaluate(answer,item['oracle'],item['transition']))
            stream.write(json.dumps(result,sort_keys=True)+'\n');stream.flush()
            if (index+1)%20==0:print(f'Generated {index+1}/{len(prepared)} arms',flush=True)
    rows=[json.loads(line) for line in path.open()]
    if len(rows)!=len(prepared) or len({(r['pair_id'],r['arm']) for r in rows})!=len(rows):
        raise ValueError('Generation arms incomplete')
    cwrite(ROOT/'observations.csv',rows)
    jwrite(ROOT/'model_metadata.json',{'policy_sha256':sha(ROOT/'generation_policy.json'),
                                      'evaluator_policy_sha256':sha(ROOT/'evaluator_policy.json'),
                                      'model_sha256':sha(MODEL),'runtime':'llama-completion 0.5.0 build 11146 CPU',
                                      'generated_arms':len(rows)})
    return {'generated_arms':len(rows),'pairs':len(manifest['conditions'])}


def reproduce():
    rows=[json.loads(line) for line in (ROOT/'observations.jsonl').open()]
    prepared={(r['pair_id'],r['arm']):r for r in
              (json.loads(line) for line in (ROOT/'prepared_contexts.jsonl').open())}
    sample=[]
    for category in CATEGORIES:
        candidates=[r for r in rows if r['category']==category and r['arm'] in ('clean','fault')]
        candidates.sort(key=lambda r:hashlib.sha256(f"repro|{r['pair_id']}".encode()).hexdigest())
        pair_id=candidates[0]['pair_id']
        sample.extend(r for r in candidates if r['pair_id']==pair_id)
    checks=[]
    for row in sample:
        fresh,_=_completion(prepared[(row['pair_id'],row['arm'])])
        checks.append({'pair_id':row['pair_id'],'arm':row['arm'],'identical':fresh==row['answer'],
                       'stored_answer':row['answer'],'fresh_answer':fresh})
    identical=all(c['identical'] for c in checks)
    jwrite(ROOT/'reproduction.json',{'identical':identical,'arms':len(checks),'checks':checks,
                                     'fresh_process':True})
    if not identical:jwrite(ROOT/'invalid.json',{'status':'INVALID','reason':'generation_reproduction'})
    return {'identical':identical,'arms':len(checks)}


if __name__=='__main__':
    action=sys.argv[1] if len(sys.argv)>1 else 'freeze'
    print(json.dumps({'freeze':freeze,'prepare':prepare,'run':run,'reproduce':reproduce}[action](),sort_keys=True))
