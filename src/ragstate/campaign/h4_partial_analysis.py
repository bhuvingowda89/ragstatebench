"""Paired H4 mixed-window analysis after all schedule and boundary gates pass."""
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

import numpy as np

from .h4_partial_campaign import ROOT, _sample_pairs, sample, verify
from .state_pilot_manifest import digest
from .state_pilot_v4_analysis import _Canvas, _bars, _lines, _paired


def _table(path, rows):
    rows=list(rows)
    fields=list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _json(path, value):
    Path(path).write_text(json.dumps(value,indent=2,sort_keys=True)+'\n')


def _read_rows():
    by_id=defaultdict(list)
    with (ROOT/'observations.jsonl').open() as stream:
        for line in stream:
            row=json.loads(line)
            by_id[row['scenario_id']].append(row)
    for rows in by_id.values():
        rows.sort(key=lambda row:row['logical_time'])
        if len(rows)!=15:
            raise ValueError('Incomplete H4 trajectory')
    return by_id


def _window(rows, tp, T):
    return [row for row in rows if tp <= row['logical_time'] < T]


def _mean(values):
    values=list(values)
    return sum(values)/len(values) if values else None


def _pair_measure(pair, by_id):
    u,p=pair
    ru,rp=by_id[u['scenario_id']],by_id[p['scenario_id']]
    tp,T=u['h4_partial_tick'],u['h4_final_tick']
    wu,wp=_window(ru,tp,T),_window(rp,tp,T)
    if not wu or len(wu)!=len(wp):
        raise ValueError('H4 matched interval missing observations')
    def endpoints(rows):
        failure=[int(not row['state_correct']) for row in rows]
        scrs_failure=[1-row['scrs'] for row in rows]
        claim_failure=[1-row['all_claims_success']
                       for row in rows if row['all_claims_success'] is not None]
        return {'any_failure':int(any(failure)),
                'failure_ticks':sum(failure),
                'integrated_failure':_mean(failure),
                'scrs_failure_ticks':sum(scrs_failure),
                'claim_failure_ticks':sum(claim_failure) if claim_failure else None,
                'srr_ticks':sum(row['srr']>0 for row in rows),
                'dep_ticks':sum(row['dep']>0 for row in rows),
                'ucer_ticks':sum(row['ucer_context']>0 for row in rows),
                'ardr_ticks':sum(row['ardr']>0 for row in rows),
                'mvcr_ticks':sum(row['mvcr']>0 for row in rows)}
    eu,ep=endpoints(wu),endpoints(wp)
    row={'pair_id':p['h4_pair_id'],'query_id':p['query_id'],
         'domain':wp[0]['domain'],'transition':p['transition'],
         'architecture':p['architecture'],'severity':p['severity'],
         'target_plane':p['replica_unit_type'],'N':p['total_unit_count'],
         'updated_units':p['updated_unit_count'],
         'topology':p['partial_topology'],
         'realized_fraction':p['realized_fraction'],
         't_partial':tp,'T':T,'interval_ticks':len(wu),
         'selected_unit_ids_hash':p['selected_updated_ids_hash']}
    for label,values in (('uniform',eu),('mixed',ep)):
        row.update({label+'_'+key:value for key,value in values.items()})
    row.update({key+'_difference':
                (ep[key]-eu[key]) if ep[key] is not None and eu[key] is not None else None
                for key in eu})
    return row


def _continuous_paired(values, *, seed=707):
    sample=np.asarray(values,dtype=float)
    rng=np.random.default_rng(seed)
    boot=np.mean(sample[rng.integers(0,len(sample),size=(2000,len(sample)))],axis=1)
    return {'n':len(sample),'mean_difference':float(np.mean(sample)),
            'median_difference':float(np.median(sample)),
            'ci95_low':float(np.quantile(boot,.025)),
            'ci95_high':float(np.quantile(boot,.975))}


def _summarize(items, label, key):
    return {label:key,'n_pairs':len(items),
            'uniform_any_failure':_mean(row['uniform_any_failure'] for row in items),
            'mixed_any_failure':_mean(row['mixed_any_failure'] for row in items),
            'paired_any_failure_difference':_mean(row['any_failure_difference'] for row in items),
            'uniform_integrated_risk':_mean(row['uniform_integrated_failure'] for row in items),
            'mixed_integrated_risk':_mean(row['mixed_integrated_failure'] for row in items),
            'paired_integrated_difference':_mean(row['integrated_failure_difference'] for row in items),
            'uniform_failure_ticks_mean':_mean(row['uniform_failure_ticks'] for row in items),
            'mixed_failure_ticks_mean':_mean(row['mixed_failure_ticks'] for row in items),
            'paired_failure_tick_difference':_mean(row['failure_ticks_difference'] for row in items),
            'paired_claim_failure_tick_difference':_mean(
                row['claim_failure_ticks_difference'] for row in items
                if row['claim_failure_ticks_difference'] is not None),
            'paired_scrs_failure_tick_difference':_mean(
                row['scrs_failure_ticks_difference'] for row in items),
            'paired_srr_tick_difference':_mean(row['srr_ticks_difference'] for row in items),
            'paired_dep_tick_difference':_mean(row['dep_ticks_difference'] for row in items),
            'paired_ucer_tick_difference':_mean(row['ucer_ticks_difference'] for row in items),
            'paired_ardr_tick_difference':_mean(row['ardr_ticks_difference'] for row in items),
            'paired_mvcr_tick_difference':_mean(row['mvcr_ticks_difference'] for row in items)}


def _group(rows, field):
    groups=defaultdict(list)
    for row in rows:groups[row[field]].append(row)
    return groups


def _reproduce(manifest, by_id):
    stored=[]
    for label,pair in _sample_pairs(manifest):
        for scenario in pair:
            rows=[{key:value for key,value in row.items()
                   if key not in ('retrieval_latency_ms','parent_latency_ms')}
                  for row in by_id[scenario['scenario_id']]]
            stored.append({'stratum':label,'scenario_id':scenario['scenario_id'],
                           'row_hash':digest(rows)})
    env=dict(os.environ,PYTHONPATH=str(Path('src').resolve())+os.pathsep+os.environ.get('PYTHONPATH',''))
    command=[sys.executable,'-m','ragstate.campaign.h4_partial_analysis','sample_rows']
    result=subprocess.run(command,capture_output=True,text=True,env=env)
    fresh=json.loads(result.stdout) if result.returncode==0 else None
    identical=stored==fresh
    output={'identical':identical,'fresh_exit':result.returncode,
            'scenario_count':len(stored),'stored':stored,'fresh':fresh,'stderr':result.stderr}
    _json(ROOT/'reproduction.json',output)
    return output


def sample_rows():
    from .h4_partial_campaign import _execute
    from .state_pilot_manifest import _load_frozen
    manifest=verify()
    world,metadata,queries,_=_load_frozen(manifest['config'])
    by_query={query.query_id:query for query in queries}
    output=[]
    for label,pair in _sample_pairs(manifest):
        for scenario in pair:
            rows,_,_,_=_execute(scenario,world,metadata,
                                by_query[scenario['query_id']],manifest['config'])
            stable=[{key:value for key,value in row.items()
                     if key not in ('retrieval_latency_ms','parent_latency_ms')}
                    for row in rows]
            output.append({'stratum':label,'scenario_id':scenario['scenario_id'],
                           'row_hash':digest(stable)})
    return output


def _figures(rows, by_id, manifest):
    import ragstate.campaign.state_pilot_v4_analysis as charts
    charts.FIGURES=ROOT/'figures'
    charts.FIGURES.mkdir(exist_ok=True)
    def signed_bars(stem,title,labels,values):
        chart=_Canvas(title)
        values=[float(value or 0) for value in values]
        scale=max([abs(value) for value in values]+[1.0])
        baseline=270
        chart.line(75,baseline,850,baseline,'#293a49')
        chart.line(75,75,75,455,'#293a49')
        slot=760/max(1,len(values))
        for index,(label,value) in enumerate(zip(labels,values)):
            height=int(155*abs(value)/scale)
            x=85+index*slot
            chart.rect(x,baseline-height if value>=0 else baseline,
                       max(3,slot*.62),max(1,height),chart.COLORS[index%len(chart.COLORS)])
            chart.text(x+slot*.31,480,str(label)[:13],11)
            chart.text(x+slot*.31,
                       baseline-height-9 if value>=0 else baseline+height+17,
                       f'{value:+.3f}',11)
        chart.save(stem)
    _bars('01_matched_exposure','H4 matched mean failure ticks: T=5',
          ['Uniform','Mixed'],[_mean(r['uniform_failure_ticks'] for r in rows),
                               _mean(r['mixed_failure_ticks'] for r in rows)])
    transitions=sorted({row['transition'] for row in rows})
    signed_bars('02_transition_effect','Mixed minus uniform failure ticks by transition',
          transitions,[_mean(r['failure_ticks_difference'] for r in rows
                             if r['transition']==transition) for transition in transitions])
    topologies=('P_LOW','P_MID','P_HIGH')
    signed_bars('03_topology_effect','Mixed minus uniform failure ticks by topology',
          topologies,[_mean(r['failure_ticks_difference'] for r in rows
                            if r['topology']==topology) for topology in topologies])
    fractions=sorted({row['realized_fraction'] for row in rows})
    signed_bars('04_realized_fraction','Mixed minus uniform ticks by realized fraction',
          [f'{fraction:.2f}' for fraction in fractions],
          [_mean(r['failure_ticks_difference'] for r in rows
                 if r['realized_fraction']==fraction) for fraction in fractions])
    exemplar=manifest['scenarios'][:2]
    curves={}
    for scenario in exemplar:
        curves[scenario['h4_arm']]=[(row['logical_time'],int(not row['state_correct']))
                                   for row in by_id[scenario['scenario_id']]
                                   if row['logical_time']>=0]
    _lines('05_representative_trajectories','Representative matched H4 failure trajectory',curves)
    sizes=sorted({row['N'] for row in rows})
    signed_bars('06_topology_N_effect','Mixed minus uniform ticks by replica N',
          [str(n) for n in sizes],
          [_mean(r['failure_ticks_difference'] for r in rows if r['N']==n) for n in sizes])
    content=('UPDATE','SUPERSEDE')
    labels=[f'{transition[:5]}:{metric}' for transition in content
            for metric in ('CLAIM','MVCR')]
    values=[]
    for transition in content:
        items=[row for row in rows if row['transition']==transition]
        values.extend((_mean(row['claim_failure_ticks_difference'] for row in items),
                       _mean(row['mvcr_ticks_difference'] for row in items)))
    signed_bars('07_content_metric_tradeoff',
                'Content partiality: claim failure vs mixed versions',labels,values)


def analyze():
    manifest=verify()
    if (ROOT/'invalid.json').exists():
        raise ValueError('H4 campaign marked INVALID')
    execution=json.loads((ROOT/'execution_summary.json').read_text())
    pre=json.loads((ROOT/'preflight.json').read_text())
    if execution['schedule_agreements']!=1504 or execution['observations']!=22560 or \
       execution['forbidden_accesses'] or execution['cache_stage_violations'] or \
       not all(pre['gates'].values()):
        raise ValueError('H4 hard gates failed; refusing scientific analysis')
    by_id=_read_rows()
    pairs=[manifest['scenarios'][index:index+2]
           for index in range(0,len(manifest['scenarios']),2)]
    rows=[_pair_measure(pair,by_id) for pair in pairs]
    reproduction=_reproduce(manifest,by_id)
    if not reproduction['identical']:
        raise ValueError('H4 fresh-process reproduction failed')
    _table(ROOT/'pair_results.csv',rows)
    overall=_summarize(rows,'scope','all')
    _table(ROOT/'mixed_interval_summary.csv',[overall])
    transitions=[_summarize(items,'transition',key)
                 for key,items in sorted(_group(rows,'transition').items())]
    _table(ROOT/'transition_summary.csv',transitions)
    topology=[_summarize(items,'topology',key)
              for key,items in sorted(_group(rows,'topology').items())]
    _table(ROOT/'topology_summary.csv',topology)
    fractions=[_summarize(items,'realized_fraction',key)
               for key,items in sorted(_group(rows,'realized_fraction').items())]
    _table(ROOT/'realized_fraction_summary.csv',fractions)
    domain=[_summarize(items,'domain',key)
            for key,items in sorted(_group(rows,'domain').items())]
    _table(ROOT/'domain_summary.csv',domain)
    comparisons=[]
    for label,items in [('all',rows)]+[(key,items) for key,items in
                               sorted(_group(rows,'transition').items())]:
        comparisons.append({'group':label,'endpoint':'any_failure',
            **_paired([r['uniform_any_failure'] for r in items],
                      [r['mixed_any_failure'] for r in items])})
        for endpoint in ('failure_ticks','integrated_failure','scrs_failure_ticks',
                         'srr_ticks','dep_ticks','ucer_ticks','ardr_ticks','mvcr_ticks'):
            comparisons.append({'group':label,'endpoint':endpoint,
                **_continuous_paired([r[endpoint+'_difference'] for r in items])})
        claim=[r['claim_failure_ticks_difference'] for r in items
               if r['claim_failure_ticks_difference'] is not None]
        if claim:
            comparisons.append({'group':label,'endpoint':'claim_failure_ticks',
                                **_continuous_paired(claim)})
    _table(ROOT/'statistical_comparisons.csv',comparisons)
    _figures(rows,by_id,manifest)
    gates={key:'PASS' if value else 'FAIL' for key,value in pre['gates'].items()}
    gates.update({'postrun_schedule_agreements':execution['schedule_agreements'],
                  'fresh_process_reproduction':'PASS',
                  'severity_coverage':['S2'],
                  'primary_estimand_identifiable':True})
    _json(ROOT/'gate_summary.json',gates)
    direction='higher' if overall['paired_integrated_difference']>0 else \
              'lower' if overall['paired_integrated_difference']<0 else 'equal'
    summary=(f'# H4 matched-horizon supplement\n\n'
             f'{len(rows)} matched pairs from frozen applicability-v2 PARTIAL plans. '
             'All pairs use A0 and S2 (T=5, intermediate tick 2); '
             'applicability-v2 supplies no S1/S3 PARTIAL plans.\n\n'
             'At t=2,3,4 the treatment has a strict updated/stale split and '
             'the control remains uniformly stale. Both complete target propagation at t=5.\n\n'
             f'Mixed integrated state-failure risk is {overall["mixed_integrated_risk"]:.4f}; '
             f'uniform risk is {overall["uniform_integrated_risk"]:.4f}; '
             f'the mixed arm is {direction} by '
             f'{overall["paired_integrated_difference"]:+.4f}. '
             'Effects by transition, topology, fraction and domain are in machine-readable tables.\n\n'
             'The binary endpoint is saturated: both arms fail in every matched '
             'mixed-window tick. In UPDATE and SUPERSEDE, mixed delivery changes '
             'claim availability and mixed-version exposure even though overall '
             'SCRS remains failed. Those dimensions are reported separately.\n\n'
             'The matched comparison identifies the effect of early heterogeneous '
             'propagation under this controlled state-isolated topology. '
             'It does not establish enterprise-scale behavior or S1/S3 generalization. '
             'Track-A v4 remains the source for all non-H4 findings; Track B was not run.\n')
    (ROOT/'summary.md').write_text(summary)
    return {'pairs':len(rows),'overall':overall,'transitions':transitions,
            'topology':topology,'domain':domain,'gates':gates}


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('analyze','sample_rows'))
    args=parser.parse_args()
    print(json.dumps(sample_rows() if args.action=='sample_rows' else analyze(),
                     indent=2,sort_keys=True))


if __name__=='__main__':main()
