"""Predeclared paired answer propagation analysis from frozen generation rows."""
from __future__ import annotations

import csv
import hashlib
import json
import math
import re
import statistics
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .generation_propagation import ROOT,CATEGORIES,DOMAINS,cwrite,jwrite,verify,sha


def mean(values):
    values=list(values)
    return sum(values)/len(values) if values else None


def wilson(success,n):
    if n==0:return (None,None)
    z=1.959963984540054;p=success/n
    den=1+z*z/n
    mid=(p+z*z/(2*n))/den
    half=z*math.sqrt(p*(1-p)/n+z*z/(4*n*n))/den
    return (max(0,mid-half),min(1,mid+half))


def paired_effect(pairs,endpoint='answer_defect',seed=1919):
    diffs=np.array([int(f[endpoint])-int(c[endpoint]) for c,f in pairs],dtype=float)
    n=len(diffs)
    if not n:return {'n':0,'clean_risk':None,'fault_risk':None,'risk_difference':None,
                     'ci95_low':None,'ci95_high':None,'mcnemar_p':None}
    rng=np.random.default_rng(seed)
    boot=np.mean(diffs[rng.integers(0,n,size=(3000,n))],axis=1)
    b=sum(c[endpoint]==0 and f[endpoint]==1 for c,f in pairs)
    c=sum(c[endpoint]==1 and f[endpoint]==0 for c,f in pairs)
    discordant=b+c
    p=min(1.0,2*sum(math.comb(discordant,i) for i in range(min(b,c)+1))/2**discordant) if discordant else 1.0
    return {'n':n,'clean_risk':mean(c[endpoint] for c,f in pairs),
            'fault_risk':mean(f[endpoint] for c,f in pairs),
            'risk_difference':float(np.mean(diffs)),
            'ci95_low':float(np.quantile(boot,.025)),
            'ci95_high':float(np.quantile(boot,.975)),
            'mcnemar_p':p,'discordant_clean_only':c,'discordant_fault_only':b}


def _bars(stem,title,labels,values):
    from .state_pilot_v4_analysis import _Canvas
    chart=_Canvas(title)
    vals=[float(x or 0) for x in values]
    ymax=max(vals+[.1])
    chart.line(70,455,850,455,'#293a49')
    chart.line(70,70,70,455,'#293a49')
    slot=760/max(len(vals),1)
    for i,(label,value) in enumerate(zip(labels,vals)):
        x=85+i*slot;height=max(1,int(340*value/ymax))
        chart.rect(x,455-height,max(3,slot*.60),height,chart.COLORS[i%len(chart.COLORS)])
        chart.text(x+slot*.30,480,label[:13],11)
        chart.text(x+slot*.30,445-height,f'{value:.2f}',11)
    chart.save(stem)


def figures(primary,transitions,domains,revoke,cache,partial,supersede_mixed):
    import ragstate.campaign.state_pilot_v4_analysis as charts
    charts.FIGURES=ROOT/'figures';charts.FIGURES.mkdir(exist_ok=True)
    _bars('01_context_to_answer_by_transition','Faulted context to answer defect',
          [r['category'] for r in transitions if r['category']!='OVERALL'],
          [r['propagation_rate'] for r in transitions if r['category']!='OVERALL'])
    _bars('02_clean_fault_risk','Clean and fault answer defect risk',
          ['CLEAN','FAULT'],[primary['clean_risk'],primary['fault_risk']])
    _bars('03_revoke_exposure_disclosure','REVOKE exposure and disclosure',
          ['A0 exposure','A0 disclosure','A4 disclosure','A5 disclosure'],
          [revoke['a0_context_exposure'],revoke['a0_disclosure'],revoke['a4_disclosure'],revoke['a5_disclosure']])
    _bars('04_cache_propagation','Stale cache context and answer harm',
          ['A0 stale context','A0 answer harm','A3 answer harm','A5 answer harm'],
          [cache['a0_stale_context'],cache['a0_answer_harm'],cache['a3_answer_harm'],cache['a5_answer_harm']])
    _bars('05_supersede_mixed','SUPERSEDE mixed-version answer',
          ['mixed context','mixed answer'],
          [1.0,supersede_mixed])
    _bars('06_partial_tradeoff','PARTIAL exploratory answer outcomes',
          ['rescue current','rescue mixed','no rescue current','no rescue mixed'],
          [partial['rescue_current_success'],partial['rescue_mixed_answer'],
           partial['no_rescue_current_success'],partial['no_rescue_mixed_answer']])
    _bars('07_domain_propagation','Answer defect propagation by domain',
          [r['domain'] for r in domains],[r['propagation_rate'] for r in domains])


def analyze():
    manifest=verify()
    if (ROOT/'invalid.json').exists():raise ValueError('Generation campaign invalid')
    pre=json.loads((ROOT/'preflight.json').read_text())
    repro=json.loads((ROOT/'reproduction.json').read_text())
    source_final=json.loads(Path('results/final-scientific-campaign-v1/gate_summary.json').read_text())
    source_partial=json.loads(Path('results/partial-seed-robustness-v1/preflight_gate_summary.json').read_text())
    sample_policy=json.loads((ROOT/'generation_sample_policy.json').read_text())
    if not all(value=='PASS' for value in source_final['gates'].values()) or \
       not all(source_partial['gates'].values()) or \
       sample_policy['source_final_sha256']!=sha('results/final-scientific-campaign-v1/campaign_manifest.json') or \
       sample_policy['source_partial_sha256']!=sha('results/partial-seed-robustness-v1/manifest.json'):
        raise ValueError('Source campaign validity no longer passes')
    if not all(pre['gates'].values()) or not repro['identical'] or repro['arms']<14:
        raise ValueError('Generation validity gates incomplete')
    rows=[json.loads(line) for line in (ROOT/'observations.jsonl').open()]
    tier_counts=Counter(row['rule_tier'] for row in rows)
    unparseable=sum(row['parsed_value'] is None and not row['abstention'] for row in rows)
    if unparseable:
        jwrite(ROOT/'invalid.json',{'status':'INVALID','reason':'unparseable_generated_answers',
                                     'count':unparseable})
        raise ValueError('Deterministic evaluator cannot classify all answers')
    prepared_all=[json.loads(line) for line in (ROOT/'prepared_contexts.jsonl').open()]
    for item in prepared_all:
        if item['category'] not in ('REVOKE','DELETE'):
            continue
        value=str(item['oracle']['old_value'])
        contains=bool(re.search(r'(?<!\d)'+re.escape(value)+r'(?!\d)',item['context_text']))
        if contains!=(item['arm']=='fault'):
            raise ValueError('Prohibited numeric claim not isolated to faulted context')
    pairs={}
    for row in rows:
        pairs.setdefault(row['pair_id'],{})[row['arm']]=row
    if len(pairs)!=len(manifest['conditions']) or any('clean' not in p or 'fault' not in p for p in pairs.values()):
        raise ValueError('Generation pair completeness failed')
    pair_list=[(p['clean'],p['fault']) for p in pairs.values()]
    primary=paired_effect(pair_list)
    answer_metrics=[];propagation=[]
    for clean,fault in pair_list:
        context_defect=int(not fault['retrieval_metrics']['state_correct'])
        answer_defect=int(fault['answer_defect'])
        kind=('context_failure_propagated' if context_defect and answer_defect else
              'context_failure_resisted' if context_defect else
              'answer_only_failure' if answer_defect else 'neither_failure')
        record={'pair_id':fault['pair_id'],'category':fault['category'],'query_id':fault['query_id'],
                'domain':fault['domain'],'transition':fault['transition'],
                'clean_answer_defect':clean['answer_defect'],'fault_answer_defect':answer_defect,
                'context_defect':context_defect,'classification':kind,
                'clean_answer':clean['answer'],'fault_answer':fault['answer'],
                'clean_context_hash':clean['context_hash'],'fault_context_hash':fault['context_hash'],
                'fault_stale_claim':fault['stale_claim'],
                'fault_unsupported_disclosure':fault['unsupported_disclosure'],
                'fault_correct_abstention':fault['correct_abstention'],
                'fault_current_claim_success':fault['current_claim_success'],
                'fault_mixed_version_synthesis':fault['mixed_version_synthesis']}
        answer_metrics.append(record)
    cwrite(ROOT/'answer_metrics.csv',answer_metrics)
    overall_context=sum(not f['retrieval_metrics']['state_correct'] for c,f in pair_list)
    overall_answer=sum(f['answer_defect'] for c,f in pair_list if not f['retrieval_metrics']['state_correct'])
    overall_lo,overall_hi=wilson(overall_answer,overall_context)
    propagation.append({'category':'OVERALL','pairs':len(pair_list),
                        'context_defect_count':overall_context,'answer_defect_count':overall_answer,
                        'propagation_rate':overall_answer/overall_context if overall_context else None,
                        'ci95_low':overall_lo,'ci95_high':overall_hi,
                        'clean_risk':primary['clean_risk'],'fault_risk':primary['fault_risk'],
                        'paired_risk_difference':primary['risk_difference'],
                        'mixed_answer_rate':mean(f['mixed_version_synthesis'] for c,f in pair_list),
                        'context_mixed_rate':mean(f['retrieval_metrics']['mvcr']>0 for c,f in pair_list),
                        'context_defect_rate':overall_context/len(pair_list),
                        'stale_answer_rate':mean(f['stale_claim'] for c,f in pair_list)})
    for category in CATEGORIES:
        subset=[(c,f) for c,f in pair_list if f['category']==category]
        defect_count=sum(f['answer_defect'] for c,f in subset if not f['retrieval_metrics']['state_correct'])
        context_count=sum(not f['retrieval_metrics']['state_correct'] for c,f in subset)
        lo,hi=wilson(defect_count,context_count)
        effect=paired_effect(subset)
        propagation.append({'category':category,'pairs':len(subset),
                            'context_defect_count':context_count,'answer_defect_count':defect_count,
                            'propagation_rate':defect_count/context_count if context_count else None,
                            'ci95_low':lo,'ci95_high':hi,
                            'clean_risk':effect['clean_risk'],'fault_risk':effect['fault_risk'],
                            'paired_risk_difference':effect['risk_difference'],
                            'mixed_answer_rate':mean(f['mixed_version_synthesis'] for c,f in subset),
                            'context_mixed_rate':mean(f['retrieval_metrics']['mvcr']>0 for c,f in subset),
                            'context_defect_rate':context_count/len(subset),
                            'stale_answer_rate':mean(f['stale_claim'] for c,f in subset)})
    cwrite(ROOT/'propagation_rates.csv',propagation)
    cwrite(ROOT/'transition_analysis.csv',propagation)
    effects=[dict(category='OVERALL',**primary)]+[
        dict(category=c,**paired_effect([(a,b) for a,b in pair_list if b['category']==c])) for c in CATEGORIES]
    ranked=sorted(effects[1:],key=lambda r:r['mcnemar_p'])
    adjusted={}
    running=1.0
    for rank in range(len(ranked),0,-1):
        row=ranked[rank-1]
        running=min(running,row['mcnemar_p']*len(ranked)/rank)
        adjusted[row['category']]=running
    for row in effects:row['secondary_bh_fdr_p']=adjusted.get(row['category'])
    cwrite(ROOT/'paired_effects.csv',effects)
    revoke_pairs=[(c,f,pairs[f['pair_id']]) for c,f in pair_list if f['category']=='REVOKE']
    revoke={'n':len(revoke_pairs),'a0_context_exposure':mean(f['retrieval_metrics']['ucer_context']>0 for c,f,p in revoke_pairs),
            'a0_disclosure':mean(f['unsupported_disclosure'] for c,f,p in revoke_pairs),
            'a4_disclosure':mean(p['protected_A4']['unsupported_disclosure'] for c,f,p in revoke_pairs),
            'a5_disclosure':mean(p['protected_A5']['unsupported_disclosure'] for c,f,p in revoke_pairs),
            'a0_answer_defect':mean(f['answer_defect'] for c,f,p in revoke_pairs),
            'a4_answer_defect':mean(p['protected_A4']['answer_defect'] for c,f,p in revoke_pairs),
            'a5_answer_defect':mean(p['protected_A5']['answer_defect'] for c,f,p in revoke_pairs)}
    revoke['a0_disclosure_count']=sum(f['unsupported_disclosure'] for c,f,p in revoke_pairs)
    revoke['a0_disclosure_ci95_low'],revoke['a0_disclosure_ci95_high']=wilson(revoke['a0_disclosure_count'],len(revoke_pairs))
    revoke['a0_vs_a4_paired_difference']=paired_effect([
        ({'answer_defect':p['protected_A4']['unsupported_disclosure']},
         {'answer_defect':f['unsupported_disclosure']}) for c,f,p in revoke_pairs])['risk_difference']
    revoke['a0_vs_a5_paired_difference']=paired_effect([
        ({'answer_defect':p['protected_A5']['unsupported_disclosure']},
         {'answer_defect':f['unsupported_disclosure']}) for c,f,p in revoke_pairs])['risk_difference']
    cwrite(ROOT/'revoke_disclosure.csv',[revoke])
    cache_pairs=[(c,f,pairs[f['pair_id']]) for c,f in pair_list if f['category']=='CACHE']
    cache={'n':len(cache_pairs),'a0_stale_context':mean(not f['retrieval_metrics']['state_correct'] for c,f,p in cache_pairs),
           'a0_stale_answer':mean(f['stale_claim'] for c,f,p in cache_pairs),
           'a3_stale_answer':mean(c['stale_claim'] for c,f,p in cache_pairs),
           'a5_stale_answer':mean(p['protected_A5']['stale_claim'] for c,f,p in cache_pairs),
           'a0_answer_harm':mean(f['stale_claim'] or f['unsupported_disclosure'] for c,f,p in cache_pairs),
           'a3_answer_harm':mean(c['stale_claim'] or c['unsupported_disclosure'] for c,f,p in cache_pairs),
           'a5_answer_harm':mean(p['protected_A5']['stale_claim'] or p['protected_A5']['unsupported_disclosure'] for c,f,p in cache_pairs),
           'a0_answer_defect':mean(f['answer_defect'] for c,f,p in cache_pairs),
           'a3_answer_defect':mean(c['answer_defect'] for c,f,p in cache_pairs),
           'a5_answer_defect':mean(p['protected_A5']['answer_defect'] for c,f,p in cache_pairs)}
    cache['a0_answer_harm_count']=sum(f['stale_claim'] or f['unsupported_disclosure'] for c,f,p in cache_pairs)
    cache['a0_answer_harm_ci95_low'],cache['a0_answer_harm_ci95_high']=wilson(cache['a0_answer_harm_count'],len(cache_pairs))
    cache_rows=[dict(transition='ALL',**cache)]
    for transition in ('SUPERSEDE','DELETE'):
        subset=[(c,f,p) for c,f,p in cache_pairs if f['transition']==transition]
        if subset:
            cache_rows.append({'transition':transition,'n':len(subset),
                               'a0_answer_harm':mean(f['stale_claim'] or f['unsupported_disclosure'] for c,f,p in subset),
                               'a3_answer_harm':mean(c['stale_claim'] or c['unsupported_disclosure'] for c,f,p in subset),
                               'a5_answer_harm':mean(p['protected_A5']['stale_claim'] or
                                                      p['protected_A5']['unsupported_disclosure'] for c,f,p in subset)})
    cwrite(ROOT/'cache_answer_propagation.csv',cache_rows)
    partial_pairs=[(c,f) for c,f in pair_list if f['category']=='PARTIAL']
    rescue=[f for c,f in partial_pairs if f['rescue']==1]
    no_rescue=[f for c,f in partial_pairs if f['rescue']==0]
    partial={'pairs':len(partial_pairs),'rescue_pairs':len(rescue),'no_rescue_pairs':len(no_rescue),
             'rescue_current_success':mean(f['current_claim_success'] for f in rescue),
             'rescue_mixed_answer':mean(f['mixed_version_synthesis'] for f in rescue),
             'no_rescue_current_success':mean(f['current_claim_success'] for f in no_rescue),
             'no_rescue_mixed_answer':mean(f['mixed_version_synthesis'] for f in no_rescue),
             'topologies':sorted({f['topology'] for c,f in partial_pairs})}
    partial_rows=[dict(transition='ALL',**partial)]
    for transition in ('UPDATE','SUPERSEDE'):
        subset=[f for c,f in partial_pairs if f['transition']==transition]
        if subset:
            partial_rows.append({'transition':transition,'pairs':len(subset),
                                 'current_claim_success':mean(f['current_claim_success'] for f in subset),
                                 'mixed_answer_rate':mean(f['mixed_version_synthesis'] for f in subset),
                                 'stale_claim_rate':mean(f['stale_claim'] for f in subset),
                                 'answer_defect_rate':mean(f['answer_defect'] for f in subset)})
    cwrite(ROOT/'partial_exploratory.csv',partial_rows)
    supersede_mixed_faults=[f for c,f in pair_list if f['transition']=='SUPERSEDE'
                            and f['retrieval_metrics']['mvcr']>0]
    supersede_mixed_rate=mean(f['mixed_version_synthesis'] for f in supersede_mixed_faults)
    domains=[]
    for domain in DOMAINS:
        subset=[(c,f) for c,f in pair_list if f['domain']==domain]
        n=sum(not f['retrieval_metrics']['state_correct'] for c,f in subset)
        k=sum(f['answer_defect'] for c,f in subset if not f['retrieval_metrics']['state_correct'])
        lo,hi=wilson(k,n)
        domains.append({'domain':domain,'pairs':len(subset),'context_defects':n,
                        'answer_defects_after_context_defect':k,
                        'propagation_rate':k/n if n else None,'ci95_low':lo,'ci95_high':hi,
                        'paired_risk_difference':paired_effect(subset)['risk_difference']})
    cwrite(ROOT/'domain_analysis.csv',domains)
    # No LLM judge is used. Human verification remains reported separately;
    # deterministic evaluator validity is the G-P5 requirement.
    audit_path=ROOT/'human_audit.csv'
    prepared={(r['pair_id'],r['arm']):r for r in prepared_all}
    # Hash-ranked within each category and answer-outcome stratum where both
    # outcomes occur, with one extra from the remaining pool. This keeps the
    # 12.5% review packet represented across all seven categories.
    audit_ids=[]
    for category in CATEGORIES:
        members=[pair_id for pair_id,pair in pairs.items() if pair['fault']['category']==category]
        members.sort(key=lambda pair_id:hashlib.sha256(pair_id.encode()).hexdigest())
        success=[pair_id for pair_id in members if not pairs[pair_id]['fault']['answer_defect']]
        failure=[pair_id for pair_id in members if pairs[pair_id]['fault']['answer_defect']]
        audit_ids.extend([success[0],failure[0]] if success and failure else members[:2])
    remaining=sorted((pair_id for pair_id in pairs if pair_id not in audit_ids),
                     key=lambda pair_id:hashlib.sha256(pair_id.encode()).hexdigest())
    audit_ids.append(remaining[0])
    audit_rows=[]
    for pair_id in audit_ids:
        clean,fault=pairs[pair_id]['clean'],pairs[pair_id]['fault']
        audit_rows.append({'pair_id':pair_id,'category':fault['category'],'domain':fault['domain'],
                           'query_id':fault['query_id'],'question':fault['question'],
                           'current_value':fault['oracle']['current_value'],
                           'prohibited_or_stale_value':fault['oracle']['old_value'],
                           'abstention_required':fault['oracle']['abstention_required'],
                           'clean_context':prepared[(pair_id,'clean')]['context_text'],
                           'fault_context':prepared[(pair_id,'fault')]['context_text'],
                           'clean_answer':clean['answer'],'fault_answer':fault['answer'],
                           'automated_clean_defect':clean['answer_defect'],
                           'automated_fault_defect':fault['answer_defect'],
                           'automated_disclosure':fault['unsupported_disclosure'],
                           'automated_stale_claim':fault['stale_claim'],
                           'human_clean_defect':'','human_fault_defect':'',
                           'human_disclosure':'','human_stale_claim':'',
                           'review_status':'PENDING_HUMAN'})
    cwrite(audit_path,audit_rows)
    gates={**pre['gates'],'G-P6':repro['identical'] and repro['arms']>=14}
    gates['G-P5']=bool(gates['G-P5'] and unparseable==0)
    jwrite(ROOT/'gate_summary.json',{'gates':gates,'all_pass':all(gates.values()),
                                     'sample_pairs':len(pair_list),'generated_arms':len(rows),
                                     'context_failure_propagated':sum(r['classification']=='context_failure_propagated' for r in answer_metrics),
                                     'context_failure_resisted':sum(r['classification']=='context_failure_resisted' for r in answer_metrics),
                                     'clean_context_answer_errors':sum(c['answer_defect'] for c,f in pair_list),
                                     'evaluator_tier_counts':dict(tier_counts),
                                     'unparseable_answers':unparseable,
                                     'human_audit_status':'PENDING_HUMAN',
                                     'human_audit_pairs':len(audit_rows)})
    figures(primary,propagation,domains,revoke,cache,partial,supersede_mixed_rate)
    tables=ROOT/'tables';tables.mkdir(exist_ok=True)
    composition=[{'category':category,'pairs':sum(f['category']==category for c,f in pair_list),
                  'domains':len({f['domain'] for c,f in pair_list if f['category']==category})}
                 for category in CATEGORIES]
    cwrite(tables/'01_sample_composition.csv',composition)
    cwrite(tables/'02_model_runtime.csv',[json.loads((ROOT/'generation_policy.json').read_text())])
    cwrite(tables/'03_primary_outcomes.csv',[dict(category='OVERALL',**primary)])
    for index,name,source in [(4,'transition_propagation','transition_analysis.csv'),
                              (5,'revoke_confidentiality','revoke_disclosure.csv'),
                              (6,'cache_propagation','cache_answer_propagation.csv'),
                              (7,'partial_exploratory','partial_exploratory.csv')]:
        (tables/f'{index:02d}_{name}.csv').write_bytes((ROOT/source).read_bytes())
    if audit_path.exists():
        cwrite(tables/'08_evaluator_audit_agreement.csv',list(csv.DictReader(audit_path.open())))
    summary=[f'# Generation propagation v1', '',
             f"- Paired conditions: {len(pair_list)}; generated arms: {len(rows)}; domains: {len(domains)}.",
             f"- Clean answer-defect risk: {primary['clean_risk']:.3f}; fault risk: {primary['fault_risk']:.3f}; paired difference {primary['risk_difference']:+.3f} [{primary['ci95_low']:+.3f}, {primary['ci95_high']:+.3f}].",
             f"- Fault-context defects propagated: {sum(r['classification']=='context_failure_propagated' for r in answer_metrics)}; resisted: {sum(r['classification']=='context_failure_resisted' for r in answer_metrics)}; clean-context answer errors: {sum(c['answer_defect'] for c,f in pair_list)}.",
             f"- REVOKE A0 disclosure: {revoke['a0_disclosure']:.3f}; A4: {revoke['a4_disclosure']:.3f}; A5: {revoke['a5_disclosure']:.3f}.",
             f"- Cache A0 stale/prohibited-answer rate: {cache['a0_answer_harm']:.3f}; A3: {cache['a3_answer_harm']:.3f}; A5: {cache['a5_answer_harm']:.3f}.",
             f"- Fresh-process deterministic reproduction: {'PASS' if repro['identical'] else 'FAIL'}.",
             '- Deterministic evaluator self-tests pass. A 15-pair human review packet is prepared; independent human agreement is pending.',
             f"- Gates: {'PASS' if all(gates.values()) else 'PENDING'}.", '',
             'Generation is secondary to the validated retrieval/state campaign. The descriptive chain is not a formal mediation estimate.']
    (ROOT/'summary.md').write_text('\n'.join(summary)+'\n')
    return {'primary':primary,'revoke':revoke,'cache':cache,'partial':partial,
            'domains':domains,'propagation':propagation,'gates':gates}


if __name__=='__main__':
    print(json.dumps(analyze(),sort_keys=True))
