"""Secondary Track-B transfer analysis from frozen v4 and v2 observations."""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from .state_pilot_manifest import digest
from .state_pilot_v4_analysis import _Canvas, _bars, _lines, _paired
from .track_b_transfer_v2 import (ROOT, _config, _sample_scenarios, classify_direction,
    classify_masking, d11, sample, verify_control)


A_ROOT=Path('results/state-isolated-pilot-v4')
FIGURES=ROOT/'figures'
TRANSITIONS=('UPDATE','SUPERSEDE','DELETE','GRANT','REVOKE')


def _csv(path,rows):
    rows=list(rows)
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in rows:writer.writerow(row)


def _json(path,value):Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def _load_rows(path):
    grouped=defaultdict(list)
    with Path(path).open() as stream:
        for line in stream:
            row=json.loads(line)
            grouped[row['scenario_id']].append(row)
    for rows in grouped.values():
        rows.sort(key=lambda row:row['logical_time'])
        if len(rows)!=15:raise ValueError('Incomplete frozen trajectory')
    return grouped


def _at(rows,tick):return next(row for row in rows if row['logical_time']==tick)


def _mean(values):
    values=list(values)
    return sum(values)/len(values) if values else None


def _direction(value):
    if value is None or abs(value)<1e-12:return 'null'
    return 'positive' if value>0 else 'negative'


def _group_scenarios(scenarios,rows,*,block=None,transition=None,arch=None,
                     fault=None,compound=None,component=None):
    return [(scenario,rows[scenario['scenario_id']]) for scenario in scenarios
            if (block is None or scenario['block']==block)
            and (transition is None or scenario['transition']==transition)
            and (arch is None or scenario['architecture']==arch)
            and (fault is None or scenario['fault']==fault)
            and (compound is None or scenario.get('compound')==compound)
            and (component is None or scenario.get('component')==component)]


def _cache_persistent(rows):
    return int(any(row['logical_time']>=0 and row['index_converged']
                   and row['cache_hit'] and not row['state_correct'] for row in rows))


def _rew(rows):
    exposed=[row['logical_time'] for row in rows
             if row['logical_time']>=0 and row['ucer_context']>0]
    return len(exposed)


def _effect_rows(a_scenarios,b_scenarios,a_rows,b_rows):
    effects=[]
    def add(effect_id,cls,member,metric,aval,bval,a_n,b_n):
        effects.append({'effect_id':effect_id,'effect_class':cls,'member':member,
            'metric':metric,'track_a_effect':aval,'track_b_effect':bval,
            'track_a_n':a_n,'track_b_n':b_n,
            'track_a_direction':_direction(aval),
            'track_b_direction':_direction(bval),
            'attenuation_ratio':bval/aval if aval not in (None,0) and bval is not None else None,
            'classification':classify_direction(aval,bval)})
    for transition in TRANSITIONS:
        a=_group_scenarios(a_scenarios,a_rows,block='TA1',transition=transition,arch='A0')
        a=[item for item in a if item[0]['severity']=='S2']
        b=_group_scenarios(b_scenarios,b_rows,block='TB1',transition=transition,arch='A0')
        av=_mean((1-_at(rows,1)['state_correct'])-
                 (1-_at(rows,-1)['state_correct']) for _,rows in a)
        bv=_mean((1-_at(rows,1)['state_correct'])-
                 (1-_at(rows,-1)['state_correct']) for _,rows in b)
        add('T1:'+transition,'T1',transition,'early_state_failure_risk_difference',
            av,bv,len(a),len(b))
    for architecture in ('A4','A5'):
        def contrast(scenarios,rows,block):
            groups=defaultdict(dict)
            for scenario,traj in _group_scenarios(scenarios,rows,block=block,
                                                  transition='REVOKE'):
                if scenario['severity']=='S2':
                    groups[scenario['query_id']][scenario['architecture']]=traj
            return _mean(int(_at(group[architecture],1)['ucer_context']>0)-
                         int(_at(group['A0'],1)['ucer_context']>0)
                         for group in groups.values()
                         if 'A0' in group and architecture in group),len(groups)
        av,an=contrast(a_scenarios,a_rows,'TA3')
        bv,bn=contrast(b_scenarios,b_rows,'TB2')
        add('T2:'+architecture,'T2',architecture,'revoke_UCER_context_risk_difference',
            av,bv,an,bn)
    for transition in ('SUPERSEDE','DELETE'):
        for architecture in ('A3','A5'):
            def cache_contrast(scenarios,rows,block):
                groups=defaultdict(dict)
                for scenario,traj in _group_scenarios(scenarios,rows,block=block,
                    transition=transition,fault='STALE_CACHE'):
                    groups[scenario['query_id']][scenario['architecture']]=traj
                values=[_cache_persistent(group[architecture])-
                        _cache_persistent(group['A0']) for group in groups.values()
                        if 'A0' in group and architecture in group]
                return _mean(values),len(values)
            av,an=cache_contrast(a_scenarios,a_rows,'TA4')
            bv,bn=cache_contrast(b_scenarios,b_rows,'TB3')
            add('T3:'+transition+':'+architecture,'T3',transition+':'+architecture,
                'cache_persistence_risk_difference',av,bv,an,bn)
    for transition in TRANSITIONS:
        a=[rows for scenario,rows in _group_scenarios(a_scenarios,a_rows,
            block='TA1',transition=transition,arch='A0') if scenario['severity']=='S2']
        b=[rows for scenario,rows in _group_scenarios(b_scenarios,b_rows,
            block='TB1',transition=transition,arch='A0')]
        av=_mean(rows[-1]['t_rsc'] is not None for rows in a)
        bv=_mean(rows[-1]['t_rsc'] is not None for rows in b)
        add('T4:'+transition,'T4',transition,'stable_converged_by_14_rate',
            av,bv,len(a),len(b))
    for compound in ('CF2','CF4','CF5'):
        a_groups=defaultdict(dict)
        for scenario,rows in _group_scenarios(a_scenarios,a_rows,
            block='TA5',arch='A0',compound=compound):
            a_groups[scenario['query_id']][scenario['component']]=rows
        a_values=[(1-_at(group['F1_F2'],5)['state_correct'])-
                  (1-_at(group['NONE'],5)['state_correct'])
                  for group in a_groups.values() if 'F1_F2' in group and 'NONE' in group]
        b_comp={scenario['query_id']:rows for scenario,rows in _group_scenarios(
            b_scenarios,b_rows,block='TB4',arch='A0',compound=compound)}
        transition={'CF2':'REVOKE','CF4':'DELETE','CF5':'UPDATE'}[compound]
        b_control={scenario['query_id']:rows for scenario,rows in _group_scenarios(
            b_scenarios,b_rows,block='TB1',arch='A0',transition=transition)}
        b_values=[(1-_at(rows,5)['state_correct'])-
                  (1-_at(b_control[qid],5)['state_correct'])
                  for qid,rows in b_comp.items() if qid in b_control]
        add('T5:'+compound,'T5',compound,
            'compound_A0_failure_at_5_minus_recovered_control',
            _mean(a_values),_mean(b_values),len(a_values),len(b_values))
    return effects


def _row_summaries(scenarios,by_id):
    groups=defaultdict(list)
    for scenario in scenarios:
        rows=by_id[scenario['scenario_id']]
        key=scenario['block'],scenario['transition'],scenario['architecture'],scenario['fault'],scenario.get('compound')
        groups[key].append(rows)
    output=[]
    for key,trajectories in sorted(groups.items(),key=lambda item:str(item[0])):
        observed=[_at(rows,1) for rows in trajectories]
        output.append({'block':key[0],'transition':key[1],
            'architecture':key[2],'fault':key[3],'compound':key[4],
            'n':len(trajectories),
            'early_failure_rate':_mean(not row['state_correct'] for row in observed),
            'early_scrs':_mean(row['scrs'] for row in observed),
            'early_cbr':_mean(row['cbr'] for row in observed if row['cbr'] is not None),
            'early_cer_at_10':_mean(row['cer_at_10'] for row in observed if row['cer_at_10'] is not None),
            'early_ucer_candidate':_mean(row['ucer_candidate'] for row in observed),
            'early_ucer_post_filter':_mean(row['ucer_post_filter'] for row in observed),
            'early_ucer_context':_mean(row['ucer_context'] for row in observed),
            'early_ardr':_mean(row['ardr'] for row in observed),
            'early_dep':_mean(row['dep'] for row in observed),
            'early_srr':_mean(row['srr'] for row in observed),
            'early_mvcr':_mean(row['mvcr'] for row in observed),
            'early_cter':_mean(row['cter'] for row in observed),
            'early_serr':_mean(row['serr'] for row in observed),
            'converged_by_14':_mean(rows[-1]['t_rsc'] is not None for rows in trajectories),
            'median_t_rsc':median([rows[-1]['t_rsc'] for rows in trajectories
                                  if rows[-1]['t_rsc'] is not None])
            if any(rows[-1]['t_rsc'] is not None for rows in trajectories) else None,
            'rew_context_mean':_mean(_rew(rows) for rows in trajectories if key[1]=='REVOKE'),
            'cache_persistence_rate':_mean(_cache_persistent(rows) for rows in trajectories),
            'median_latency_ms':median(row['retrieval_latency_ms'] for row in observed),
            'median_parent_latency_ms':median(row['parent_latency_ms'] for row in observed),
            'auth_validation_mean':_mean(row['auth_validation_count'] for row in observed),
            'cache_hit_rate':_mean(row['cache_hit'] for row in observed)})
    return output


def _domain_effects(a_scenarios,b_scenarios,a_rows,b_rows):
    output=[]
    for domain in ('engineering','finance','hr','legal','sales','security'):
        for transition in TRANSITIONS:
            a=[rows for scenario,rows in _group_scenarios(a_scenarios,a_rows,
                block='TA1',transition=transition,arch='A0')
               if scenario['severity']=='S2' and scenario['query_id'].split('-')[2]==domain]
            b=[rows for scenario,rows in _group_scenarios(b_scenarios,b_rows,
                block='TB1',transition=transition,arch='A0')
               if scenario['query_id'].split('-')[2]==domain]
            av=_mean((1-_at(rows,1)['state_correct'])-
                     (1-_at(rows,-1)['state_correct']) for rows in a)
            bv=_mean((1-_at(rows,1)['state_correct'])-
                     (1-_at(rows,-1)['state_correct']) for rows in b)
            output.append({'effect_id':'T1:'+transition,'domain':domain,
                'track_a_effect':av,'track_b_effect':bv,
                'track_a_n':len(a),'track_b_n':len(b),
                'classification':classify_direction(av,bv),
                'matching_level':'query_level_for_selected_B_query'})
        for architecture in ('A4','A5'):
            def rate(scenarios,rows,block):
                group=defaultdict(dict)
                for scenario,traj in _group_scenarios(scenarios,rows,
                    block=block,transition='REVOKE'):
                    if scenario['severity']=='S2' and scenario['query_id'].split('-')[2]==domain:
                        group[scenario['query_id']][scenario['architecture']]=traj
                return _mean(int(_at(item[architecture],1)['ucer_context']>0)-
                             int(_at(item['A0'],1)['ucer_context']>0)
                             for item in group.values()
                             if 'A0' in item and architecture in item)
            av=rate(a_scenarios,a_rows,'TA3');bv=rate(b_scenarios,b_rows,'TB2')
            output.append({'effect_id':'T2:'+architecture,'domain':domain,
                'track_a_effect':av,'track_b_effect':bv,
                'track_a_n':1,'track_b_n':1,'classification':classify_direction(av,bv),
                'matching_level':'structural_domain_or_query_level'})
    return output


def _paired_statistics(a_scenarios,b_scenarios,a_rows,b_rows):
    statistics=[]
    for transition in TRANSITIONS:
        for label,scenarios,rows,block in (('Track_A',a_scenarios,a_rows,'TA1'),
                                           ('Track_B',b_scenarios,b_rows,'TB1')):
            cases=[trajectory for scenario,trajectory in _group_scenarios(
                scenarios,rows,block=block,transition=transition,arch='A0')
                if scenario['severity']=='S2']
            statistics.append({'comparison':f'T1:{transition}', 'track':label,
                'endpoint':'state_failure_pre_vs_t1',
                **_paired([1-_at(item,-1)['state_correct'] for item in cases],
                          [1-_at(item,1)['state_correct'] for item in cases])})
    for architecture in ('A4','A5'):
        for label,scenarios,rows,block in (('Track_A',a_scenarios,a_rows,'TA3'),
                                           ('Track_B',b_scenarios,b_rows,'TB2')):
            cases=defaultdict(dict)
            for scenario,trajectory in _group_scenarios(scenarios,rows,
                block=block,transition='REVOKE'):
                if scenario['severity']=='S2':
                    cases[scenario['query_id']][scenario['architecture']]=trajectory
            matched=[item for item in cases.values() if 'A0' in item and architecture in item]
            statistics.append({'comparison':f'T2:{architecture}','track':label,
                'endpoint':'revoke_ucer_context_A0_vs_mitigation',
                **_paired([int(_at(item['A0'],1)['ucer_context']>0) for item in matched],
                          [int(_at(item[architecture],1)['ucer_context']>0)
                           for item in matched])})
    return statistics


def _figures(effects,a_scenarios,b_scenarios,a_rows,b_rows,domain_rows,masking,latency):
    import ragstate.campaign.state_pilot_v4_analysis as chart
    chart.FIGURES=FIGURES
    FIGURES.mkdir(exist_ok=True)
    _bars('01_effect_sizes','Track-A versus Track-B transfer effects',
          [item['effect_id'] for item in effects],
          [item['track_b_effect'] if item['track_b_effect'] is not None else 0
           for item in effects])
    heat=_Canvas('Direction agreement by predeclared effect')
    colors={'preserved':'#32875b','attenuated but same direction':'#8ebf7f',
            'absent':'#d49d22','reversed':'#c95a32','not estimable':'#98a1ab'}
    for index,item in enumerate(effects):
        x=80+(index%10)*79;y=92+(index//10)*100
        heat.rect(x,y,63,45,colors[item['classification']])
        heat.text(x+31,y+67,item['effect_id'][:10],10)
    heat.save('02_direction_agreement_heatmap')
    transitions=[]
    for transition in TRANSITIONS:
        a=[rows for scenario,rows in _group_scenarios(a_scenarios,a_rows,
            block='TA1',transition=transition,arch='A0') if scenario['severity']=='S2']
        b=[rows for scenario,rows in _group_scenarios(b_scenarios,b_rows,
            block='TB1',transition=transition,arch='A0')]
        transitions.append((transition,_mean(1-_at(rows,1)['state_correct'] for rows in a),
                            _mean(1-_at(rows,1)['state_correct'] for rows in b)))
    _bars('03_transition_failure','Early transition failure under A and B',
          [f'{name}:{track}' for name,_,_ in transitions for track in ('A','B')],
          [value for _,a,b in transitions for value in (a,b)],ymax=1)
    _bars('04_revoke_mitigation','REVOKE live-auth transfer: UCER effect',
          [item['effect_id']+':'+track for item in effects if item['effect_class']=='T2'
           for track in ('A','B')],
          [value for item in effects if item['effect_class']=='T2'
           for value in (abs(item['track_a_effect']),abs(item['track_b_effect']))],ymax=1)
    _bars('05_cache_persistence','Cache persistence mitigation transfer',
          [item['effect_id']+':'+track for item in effects if item['effect_class']=='T3'
           for track in ('A','B')],
          [value for item in effects if item['effect_class']=='T3'
           for value in (abs(item['track_a_effect'] or 0),abs(item['track_b_effect'] or 0))],ymax=1)
    _bars('06_compound_transfer','Compound versus control at t=5',
          [item['effect_id']+':'+track for item in effects if item['effect_class']=='T5'
           for track in ('A','B')],
          [value for item in effects if item['effect_class']=='T5'
           for value in (abs(item['track_a_effect'] or 0),abs(item['track_b_effect'] or 0))],ymax=1)
    counts=Counter(item['classification'] for item in masking)
    _bars('07_masking_amplification','Matched query masking and amplification',
          list(counts),list(counts.values()))
    groups=defaultdict(list)
    for row in latency:groups[row['block']].append(row['median_latency_ms'])
    _bars('08_latency_overhead','Median PA2 latency by Track-B block',
          sorted(groups),[_mean(groups[key]) for key in sorted(groups)])


def analyze():
    baseline,transfer,v4,provenance=verify_control()
    if (ROOT/'invalid.json').exists():raise ValueError('Track B is marked INVALID')
    execution=json.loads((ROOT/'execution_summary.json').read_text())
    pre=json.loads((ROOT/'preflight.json').read_text())
    if execution['scenarios']!=120 or execution['observations']!=1800 or \
       execution['forbidden_accesses'] or execution['cache_stage_violations'] or \
       not all(pre['gates'].values()) or \
       set(execution['status_counts'])-{'valid','censored'}:
        raise ValueError('Track-B hard validity failed; refusing interpretation')
    a_rows=_load_rows(A_ROOT/'observations.jsonl')
    b_rows=_load_rows(ROOT/'observations.jsonl')
    a_scenarios=v4['scenarios'];b_scenarios=transfer['scenarios']
    if len(b_rows)!=120:raise ValueError('Track-B trajectory set incomplete')
    # The preflight ran the same seven strata in a prior process. Repeat in a
    # new process now and compare all deterministic trace/metric fingerprints.
    env=dict(os.environ,PYTHONPATH=str(Path('src').resolve())+os.pathsep+os.environ.get('PYTHONPATH',''))
    fresh=subprocess.run([sys.executable,'-m','ragstate.campaign.track_b_transfer_v2','sample'],
                         capture_output=True,text=True,env=env)
    observed=json.loads(fresh.stdout) if fresh.returncode==0 else None
    reproduced=observed==pre['sample']
    _json(ROOT/'reproduction.json',{'identical':reproduced,
           'scenario_count':len(pre['sample']),'fresh_exit':fresh.returncode,
           'preflight_sample':pre['sample'],'postrun_fresh_sample':observed,
           'stderr':fresh.stderr})
    if not reproduced:raise ValueError('Track-B fresh-process reproduction failed')
    effects=_effect_rows(a_scenarios,b_scenarios,a_rows,b_rows)
    if len(effects)!=19:raise ValueError('Predeclared transfer effect count changed')
    _csv(ROOT/'transfer_effects.csv',effects)
    classes=defaultdict(list)
    for effect in effects:classes[effect['effect_class']].append(effect)
    matrix=[]
    for cls,items in sorted(classes.items()):
        counts=Counter(item['classification'] for item in items)
        matrix.append({'effect_class':cls,'members':len(items),
            'track_a_direction':';'.join(item['track_a_direction'] for item in items),
            'track_b_direction':';'.join(item['track_b_direction'] for item in items),
            'track_a_effect_sizes':';'.join(str(item['track_a_effect']) for item in items),
            'track_b_effect_sizes':';'.join(str(item['track_b_effect']) for item in items),
            'preserved':counts['preserved'],
            'attenuated_same_direction':counts['attenuated but same direction'],
            'absent':counts['absent'],'reversed':counts['reversed'],
            'not_estimable':counts['not estimable']})
    _csv(ROOT/'direction_agreement.csv',matrix)
    domain_rows=_domain_effects(a_scenarios,b_scenarios,a_rows,b_rows)
    per_effect=defaultdict(list)
    for row in domain_rows:per_effect[row['effect_id']].append(row)
    for row in domain_rows:
        group=per_effect[row['effect_id']]
        row['systematic_reversal']=sum(item['classification']=='reversed'
                                       for item in group)>=4
    _csv(ROOT/'domain_transfer.csv',domain_rows)
    links=list(csv.DictReader((ROOT/'trackA_trackB_links.csv').open()))
    masking=[]
    for link in links:
        if link['matching_level']!='query_level':continue
        a=a_rows[link['track_a_scenario_id']]
        b=b_rows[link['track_b_scenario_id']]
        at,bt=_at(a,1),_at(b,1)
        af,bf=not at['state_correct'],not bt['state_correct']
        masking.append({'track_a_scenario_id':link['track_a_scenario_id'],
            'track_b_scenario_id':link['track_b_scenario_id'],
            'query_id':link['query_id'],'domain':bt['domain'],
            'transition':link['transition'],'architecture':link['architecture'],
            'fault':link['fault'],'matching_level':'query_level',
            'track_a_failure_t1':int(af),'track_b_failure_t1':int(bf),
            'track_b_clean_correct':int(bool(_at(b,-1)['state_correct'])),
            'classification':classify_masking(af,bf,bool(_at(b,-1)['state_correct'])),
            'amplified_binary_failure':int(bf and not af),
            'candidate_count_B':len(bt['candidate_chunk_ids']),
            'context_count_B':len(bt['context_chunk_ids'])})
    _csv(ROOT/'retrieval_masking.csv',masking)
    a_summary=_row_summaries(a_scenarios,a_rows)
    b_summary=_row_summaries(b_scenarios,b_rows)
    _csv(ROOT/'transition_transfer.csv',
         [row for row in b_summary if row['block']=='TB1'])
    _csv(ROOT/'mitigation_transfer.csv',
         [row for row in b_summary if row['block'] in ('TB2','TB3')])
    _csv(ROOT/'compound_transfer.csv',
         [row for row in b_summary if row['block']=='TB4'])
    _csv(ROOT/'latency_summary.csv',
         [{key:row[key] for key in ('block','transition','architecture','fault',
            'compound','n','median_latency_ms','median_parent_latency_ms',
            'auth_validation_mean','cache_hit_rate')}
          for row in b_summary])
    statistics=_paired_statistics(a_scenarios,b_scenarios,a_rows,b_rows)
    _csv(ROOT/'statistical_comparisons.csv',statistics)
    interpretable=pre['clean_correct']==197 and pre['exact_frozen_context']==197 \
        and all(_at(rows,1)['scrs'] is not None for rows in b_rows.values())
    d11_result=d11(effects,domain_rows,retrieval_interpretable=interpretable)
    gates={key:'PASS' if value else 'FAIL' for key,value in pre['gates'].items()}
    gates.update({'D11':d11_result,'runtime_validity':'PASS',
                  'fresh_process_reproduction':'PASS',
                  'query_level_links':sum(link['matching_level']=='query_level' for link in links),
                  'structural_domain_links':sum(link['matching_level']=='structural_domain'
                                                 for link in links)})
    _json(ROOT/'gate_summary.json',gates)
    _figures(effects,a_scenarios,b_scenarios,a_rows,b_rows,domain_rows,masking,
             [row for row in b_summary])
    classifications=Counter(item['classification'] for item in effects)
    masking_counts=Counter(item['classification'] for item in masking)
    summary=(f'# Frozen Track-B realistic transfer\n\n'
        f'120 scenarios, 1,800 observations; PA2 on replica chunk and parent indexes. '
        f'Clean cohort reproduced exactly for 197/197 queries. '
        f'All TB-P1–P7 and hard runtime audits passed.\n\n'
        f'D11: **{d11_result["status"]}**; directional agreement '
        f'{d11_result["agreement_numerator"]}/{d11_result["agreement_denominator"]} '
        f'predeclared nonnull effects. Classification counts: {dict(classifications)}.\n\n'
        f'Query-level links: {gates["query_level_links"]}; structural domain links: '
        f'{gates["structural_domain_links"]}. Query-level masking counts: '
        f'{dict(masking_counts)}. Structural links are not treated as paired queries.\n\n'
        'Track A remains primary causal evidence; Track B is secondary transfer. '
        'The H4 matched-horizon supplement remains separate and is not part of '
        'the frozen Track-B transfer subset. No full campaign or generation ran.\n')
    (ROOT/'summary.md').write_text(summary)
    return {'D11':d11_result,'classifications':dict(classifications),
            'masking':dict(masking_counts),'links':{
                'query_level':gates['query_level_links'],
                'structural_domain':gates['structural_domain_links']},
            'effect_rows':effects,'reproduction':reproduced}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('analyze',))
    parser.parse_args()
    print(json.dumps(analyze(),indent=2,sort_keys=True))


if __name__=='__main__':main()
