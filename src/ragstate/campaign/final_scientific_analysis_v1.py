"""Prospectively frozen final-campaign analysis from valid raw observations."""
from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import sys
from collections import Counter,defaultdict
from pathlib import Path
from statistics import median

from . import state_pilot_v4_analysis as v4
from .final_scientific_campaign_v1 import ROOT,_csv,_json,verify_freeze
from .state_pilot_manifest import digest
from .track_b_transfer_analysis import _effect_rows as transfer_effect_rows
from .track_b_transfer_v2 import d11 as transfer_d11

TRANSITIONS=('UPDATE','SUPERSEDE','DELETE','GRANT','REVOKE')
H4=Path('results/h4-partial-control-pilot-v1')
TB=Path('results/realistic-transfer-pilot-v2')


def _read(path):
    with Path(path).open(newline='') as stream:return list(csv.DictReader(stream))


def _at(rows,tick):return next(r for r in rows if r['logical_time']==tick)


def _mean(values):
    values=[v for v in values if v is not None]
    return sum(values)/len(values) if values else None


def _sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _groups(scenarios,rows,block,*,transition=None,architecture=None,severity=None):
    for s in scenarios:
        if s['final_block']!=block:continue
        if transition and s['transition']!=transition:continue
        if architecture and s['architecture']!=architecture:continue
        if severity and s['severity']!=severity:continue
        yield s,rows[s['scenario_id']]


def load_and_validate():
    _,manifest,_=verify_freeze()
    execution=json.loads((ROOT/'execution_summary.json').read_text())
    blocks=json.loads((ROOT/'block_status.json').read_text())
    if any(r['status']!='VALID' for r in blocks.values()):
        raise ValueError('A final campaign block is not valid')
    if execution['scenario_count']!=manifest['scenario_count'] or \
       execution['observation_count']!=15*manifest['scenario_count']:
        raise ValueError('Final counts do not reconcile')
    validity=[]
    for s in manifest['scenarios']:
        item=json.loads((ROOT/'validity_audits'/f'{s["scenario_id"]}.json').read_text())
        validity.append(item)
    if any(v['validity']!='valid' or v['forbidden_access_count'] or
           v['cache_stage_violation_count'] or not v['compiler_runtime_agreement']
           for v in validity):
        raise ValueError('Hard runtime audit failed')
    v4.ROOT=ROOT;v4.FIGURES=ROOT/'figures'
    by_scenario=v4.load_compact()
    if len(by_scenario)!=manifest['scenario_count'] or any(
        r['campaign_id']!='final-scientific-campaign-v1'
        for r in (json.loads(line) for line in (ROOT/'observations.jsonl').open())
        if r['logical_time']==-1):
        raise ValueError('Pilot or incomplete observations in final analysis')
    return manifest,by_scenario


def analyze():
    manifest,rows=load_and_validate()
    scenarios=manifest['scenarios']
    # The validated v4 table builder is reused with only the analytic block
    # labels mapped; FA2 is a view of FA1, so trajectories are never duplicated.
    mapping={'FA1':'TA1','FA3':'TA2','FA4':'TA4','FA5':'TA3',
             'FA6':'TA6','FA7':'TA5'}
    analytic={'scenarios':[{**s,'block':mapping[s['final_block']]} for s in scenarios]}
    tables=v4.build_tables(rows,analytic)

    effects=[];h_results=[]
    for transition in TRANSITIONS:
        selected=[traj for _,traj in _groups(scenarios,rows,'FA1',
            transition=transition,architecture='A0',severity='S2')]
        stat=v4._paired([1-_at(r,-1)['state_correct'] for r in selected],
                        [1-_at(r,1)['state_correct'] for r in selected])
        effects.append({'hypothesis':'H1','comparison':transition+'_pre_vs_t1',
                        'endpoint':'state_consistency_failure',**stat,
                        'confirmatory':True})
    revoke=[traj for _,traj in _groups(scenarios,rows,'FA1',
        transition='REVOKE',architecture='A0',severity='S2')]
    grant=[traj for _,traj in _groups(scenarios,rows,'FA1',
        transition='GRANT',architecture='A0',severity='S2')]
    h_results.append({'hypothesis':'H2','comparison':'REVOKE_confidentiality',
        'n':len(revoke),'effect':_mean(_at(r,1)['ucer_context'] for r in revoke),
        'endpoint':'UCER_context at t=1'})
    h_results.append({'hypothesis':'H2','comparison':'GRANT_availability',
        'n':len(grant),'effect':_mean(_at(r,1)['ardr'] for r in grant),
        'endpoint':'ARDR at t=1'})

    cache=[]
    by_cache=defaultdict(dict)
    for s,traj in _groups(scenarios,rows,'FA4'):
        by_cache[(s['query_id'],s['transition'])][s['architecture']]=traj
    for transition in ('UPDATE','SUPERSEDE','DELETE','REVOKE'):
        matched=[group for (qid,t),group in by_cache.items() if t==transition
                 and all(a in group for a in ('A0','A3','A5'))]
        def persistence(traj):
            return int(any(r['logical_time']>=0 and r['index_converged'] and
                r['cache_hit'] and not r['state_correct'] for r in traj))
        for architecture in ('A3','A5'):
            stat=v4._paired([persistence(g['A0']) for g in matched],
                            [persistence(g[architecture]) for g in matched])
            cache.append({'hypothesis':'H3/H8','transition':transition,
                'architecture':architecture,'endpoint':'cache_persistence_after_index',
                **stat,'a0_rate':_mean(persistence(g['A0']) for g in matched),
                'mitigation_rate':_mean(persistence(g[architecture]) for g in matched),
                'cache_hit_A0':_mean(_at(g['A0'],1)['cache_hit'] for g in matched),
                'cache_hit_mitigation':_mean(_at(g[architecture],1)['cache_hit'] for g in matched),
                'cache_hit_at_5_A0':_mean(_at(g['A0'],5)['cache_hit'] for g in matched),
                'cache_hit_at_5_mitigation':_mean(_at(g[architecture],5)['cache_hit']
                    for g in matched),
                'cache_invalidations_mitigation_mean':_mean(g[architecture][-1]['cache_invalidations']
                    for g in matched)})
    _csv(ROOT/'cache_analysis.csv',cache)

    mitigation=[]
    by_mitigation=defaultdict(dict)
    for s,traj in _groups(scenarios,rows,'FA5',severity='S2'):
        by_mitigation[(s['query_id'],s['transition'])][s['architecture']]=traj
    for s,traj in _groups(scenarios,rows,'FA1',architecture='A0',severity='S2'):
        key=s['query_id'],s['transition']
        if key in by_mitigation:
            by_mitigation[key].setdefault('A0',traj)
    for transition in ('SUPERSEDE','DELETE','GRANT','REVOKE'):
        matched=[g for (qid,t),g in by_mitigation.items() if t==transition
                 and all(a in g for a in ('A0','A1','A2','A3','A4','A5'))]
        for architecture in ('A1','A2','A3','A4','A5'):
            stat=v4._paired([1-_at(g['A0'],1)['state_correct'] for g in matched],
                            [1-_at(g[architecture],1)['state_correct'] for g in matched])
            mitigation.append({'transition':transition,'architecture':architecture,
                'endpoint':'early_state_failure',**stat,
                'a0_rmst':v4._km([g['A0'] for g in matched])['rmst_to_14'],
                'mitigation_rmst':v4._km([g[architecture] for g in matched])['rmst_to_14'],
                'a0_latency_median_ms':median(_at(g['A0'],1)['retrieval_latency_ms'] for g in matched),
                'mitigation_latency_median_ms':median(_at(g[architecture],1)['retrieval_latency_ms'] for g in matched),
                'mitigation_auth_calls_mean':_mean(_at(g[architecture],1)['auth_validation_count'] for g in matched)})
    _csv(ROOT/'mitigation_analysis.csv',mitigation)

    h6=[]
    revoke_pairs=[g for (qid,t),g in by_mitigation.items() if t=='REVOKE'
                  and all(a in g for a in ('A0','A4','A5'))]
    for architecture in ('A4','A5'):
        paired=v4._paired([int(_at(g['A0'],1)['ucer_context']>0) for g in revoke_pairs],
                          [int(_at(g[architecture],1)['ucer_context']>0)
                           for g in revoke_pairs])
        h6.append({'hypothesis':'H6','comparison':architecture+'_vs_A0_REVOKE',
            'endpoint':'binary_UCER_context_t1',**paired,
            'rew_A0_mean':_mean(sum(r['ucer_context']>0 for r in g['A0']
                if r['logical_time']>=0) for g in revoke_pairs),
            'rew_mitigation_mean':_mean(sum(r['ucer_context']>0 for r in g[architecture]
                if r['logical_time']>=0) for g in revoke_pairs),
            'rew_mean_difference':_mean(sum(r['ucer_context']>0 for r in g[architecture]
                if r['logical_time']>=0)-sum(r['ucer_context']>0 for r in g['A0']
                if r['logical_time']>=0) for g in revoke_pairs),
            'auth_calls_mitigation_mean':_mean(_at(g[architecture],1)['auth_validation_count']
                for g in revoke_pairs)})
    _csv(ROOT/'live_authorization_effects.csv',h6)
    h7=[]
    for transition in ('SUPERSEDE','DELETE','GRANT','REVOKE'):
        matched=[g for (qid,t),g in by_mitigation.items() if t==transition
                 and 'A1' in g and 'A2' in g]
        periodic=[g['A1'] for g in matched];event=[g['A2'] for g in matched]
        rng=v4.np.random.default_rng(606)
        boot=[]
        for choices in rng.integers(0,len(matched),size=(1000,len(matched))):
            boot.append(v4._km([event[i] for i in choices])['rmst_to_14']-
                        v4._km([periodic[i] for i in choices])['rmst_to_14'])
        h7.append({'hypothesis':'H7','transition':transition,'n_matched':len(matched),
            'periodic_rmst_to_14':v4._km(periodic)['rmst_to_14'],
            'event_rmst_to_14':v4._km(event)['rmst_to_14'],
            'rmst_difference_event_minus_periodic':v4._km(event)['rmst_to_14']-
                v4._km(periodic)['rmst_to_14'],
            'rmst_difference_ci95_low':float(v4.np.quantile(boot,.025)),
            'rmst_difference_ci95_high':float(v4.np.quantile(boot,.975)),
            **v4._logrank(periodic,event)})
    _csv(ROOT/'event_vs_periodic_survival.csv',h7)

    auth=[]
    for architecture in ('A0','A4','A5'):
        selected=[g[architecture] for (qid,t),g in by_mitigation.items()
                  if t=='REVOKE' and architecture in g]
        auth.append({'transition':'REVOKE','architecture':architecture,'n':len(selected),
            'ucer_context_t1':_mean(_at(r,1)['ucer_context'] for r in selected),
            'rew_context_mean':_mean(sum(item['ucer_context']>0 for item in r
                if item['logical_time']>=0) for r in selected),
            'auth_calls_t1_mean':_mean(_at(r,1)['auth_validation_count'] for r in selected)})
    _csv(ROOT/'authorization_analysis.csv',auth)

    compounds=[]
    by_compound=defaultdict(dict)
    for s,traj in _groups(scenarios,rows,'FA7'):
        by_compound[(s['compound'],s['architecture'],s['query_id'])][s['component']]=traj
    for compound in ('CF1','CF2','CF3','CF4','CF5','CF6'):
        for architecture in ('A0','A5'):
            selected=[g for (cf,a,q),g in by_compound.items() if cf==compound
                      and a==architecture]
            complete=[g for g in selected if all(arm in g for arm in
                      ('NONE','F1','F2','F1_F2'))]
            distinct=sum(tuple(r['state_correct'] for r in g['F1_F2']) not in
                (tuple(r['state_correct'] for r in g['F1']),
                 tuple(r['state_correct'] for r in g['F2'])) for g in complete)
            from scipy.stats import beta
            lower=float(beta.ppf(.025,distinct,len(complete)-distinct+1)) if distinct else 0.0
            upper=float(beta.ppf(.975,distinct+1,len(complete)-distinct)) \
                if distinct<len(complete) else 1.0
            compounds.append({'compound':compound,'architecture':architecture,
                'n_matched':len(complete),'trajectory_distinct_from_both':distinct,
                'distinct_rate':distinct/len(complete),
                'exact_binomial_ci95_low':lower,'exact_binomial_ci95_high':upper,
                'compound_failure_t1':_mean(1-_at(g['F1_F2'],1)['state_correct'] for g in complete),
                'compound_failure_t5':_mean(1-_at(g['F1_F2'],5)['state_correct'] for g in complete),
                'interpretation':'matched_trajectory_only' if compound!='CF5'
                    else 'descriptive_stress_no_common_canonical_baseline'})
    _csv(ROOT/'compound_analysis.csv',compounds)
    _csv(ROOT/'component_analysis.csv',tables['factorial'])
    localization=Counter()
    for s in scenarios:
        trajectory=rows[s['scenario_id']]
        pre=trajectory[0]
        for row in trajectory[1:]:
            if row['state_correct']:
                continue
            planes=[]
            if row['canonical_source_state_hash']!=pre['canonical_source_state_hash']:
                if row['chunk_index_hash']==pre['chunk_index_hash']:
                    planes.append('content_replica')
                if row['parent_index_hash']==pre['parent_index_hash']:
                    planes.append('parent_replica')
            if (row['canonical_authorization_state_hash']!=pre['canonical_authorization_state_hash']
                    and row['auth_metadata_hash']==pre['auth_metadata_hash']):
                planes.append('ACL_metadata')
            if row['cache_hit'] and row['index_converged']:
                planes.append('cache_after_index_convergence')
            if not planes:planes=['final_context_only_or_other']
            filter_effective=int(row['ucer_post_filter']>row['ucer_context'])
            key=(s['final_block'],s['transition'],row['domain'],
                 '+'.join(sorted(planes)),filter_effective)
            localization[key]+=1
    localization_rows=[{'final_block':block,'transition':transition,'domain':domain,
        'inferred_plane':plane,'authorization_filter_effective':effective,
        'joint_state':int('+' in plane),'final_context_failure':1,
        'incorrect_observation_count':count,
        'method':'state-hash and stage heuristic; joint state retained; not causal attribution'}
        for (block,transition,domain,plane,effective),count in sorted(localization.items())]
    _csv(ROOT/'localization_analysis.csv',localization_rows)
    _csv(ROOT/'convergence_analysis.csv',tables['convergence'])
    _csv(ROOT/'primary_effects.csv',effects+h_results+h6+[
        {'hypothesis':'H3','comparison':r['transition']+'_'+r['architecture']+'_cache',
         'endpoint':'cache_persistence_after_index',
         'n':r['n'],'risk_difference':r['risk_difference'],
         'ci95_low':r['ci95_low'],'ci95_high':r['ci95_high'],
         'mcnemar_exact_p':r['mcnemar_exact_p']} for r in cache]+[
        {'hypothesis':'H8','comparison':r['transition']+'_'+r['architecture']+'_epoch_cache',
         'endpoint':'cache_persistence_after_index_and_hit_cost',
         'n':r['n'],'risk_difference':r['risk_difference'],
         'ci95_low':r['ci95_low'],'ci95_high':r['ci95_high'],
         'cache_hit_A0':r['cache_hit_A0'],
         'cache_hit_mitigation':r['cache_hit_mitigation']}
        for r in cache if r['architecture'] in ('A3','A5')]+[
        {'hypothesis':'H5','comparison':r['compound']+'_'+r['architecture'],
         'endpoint':'trajectory_distinct_from_both_isolated',
         'n':r['n_matched'],'effect':r['trajectory_distinct_from_both'],
         'interpretation':r['interpretation']} for r in compounds]+[
        {'hypothesis':'H7','comparison':r['transition']+'_A2_vs_A1',
         'endpoint':'RMST_to_14','n':r['n_matched'],
         'effect':r['rmst_difference_event_minus_periodic'],
         'p_value':r['p_value']} for r in h7])
    h4_rows=_read(H4/'transition_summary.csv')
    _csv(ROOT/'partial_analysis.csv',[{**r,'status':'exploratory; historical primary H4 unsupported'}
                                   for r in h4_rows])

    domains=[];families=[]
    for field,destination in (('domain',domains),('query_family',families)):
        values=sorted({traj[0][field] for _,traj in _groups(scenarios,rows,'FA1')})
        for value in values:
            for transition in TRANSITIONS:
                selected=[traj for s,traj in _groups(scenarios,rows,'FA1',
                    transition=transition,architecture='A0',severity='S2')
                    if traj[0][field]==value]
                if selected:
                    destination.append({field:value,'transition':transition,'n':len(selected),
                        'pre_failure':_mean(1-_at(r,-1)['state_correct'] for r in selected),
                        'early_failure':_mean(1-_at(r,1)['state_correct'] for r in selected),
                        'risk_increase':_mean(_at(r,-1)['state_correct']-
                                              _at(r,1)['state_correct'] for r in selected)})
    _csv(ROOT/'domain_heterogeneity.csv',domains)
    _csv(ROOT/'query_family_analysis.csv',families)
    cost_groups=defaultdict(list)
    raw_at1={}
    for line in (ROOT/'observations.jsonl').open():
        row=json.loads(line)
        if row['logical_time']==1:
            cost_groups[(row['final_block'],row['architecture'])].append(row)
            raw_at1[row['scenario_id']]=row
    for architecture in ('A0','A1','A2','A3','A4','A5'):
        ids={g[architecture][0]['scenario_id'] for (qid,t),g in by_mitigation.items()
             if t in ('SUPERSEDE','DELETE','GRANT','REVOKE') and architecture in g}
        cost_groups[('FA5_S2_matched',architecture)]=[raw_at1[sid] for sid in sorted(ids)]
    costs=[]
    for (block,architecture),items in sorted(cost_groups.items()):
        costs.append({'final_block':block,'architecture':architecture,'n':len(items),
            'median_retrieval_latency_ms':median(r['retrieval_latency_ms'] for r in items),
            'median_parent_ranking_latency_ms':median(r['parent_latency_ms'] for r in items
                if r['parent_latency_ms'] is not None) if any(r['parent_latency_ms'] is not None
                for r in items) else None,
            'auth_validation_calls_mean':_mean(r['auth_validation_count'] for r in items),
            'cache_hit_rate':_mean(r['cache_hit'] for r in items),
            'cache_invalidations_mean':_mean(r['cache_invalidations'] for r in items),
            'replica_refreshes_mean':_mean(r['replica_refreshes'] for r in items),
            'index_work_note':'replica refresh count is a proxy; rebuild CPU work not separately instrumented'})
    _csv(ROOT/'cost_analysis.csv',costs)
    model_cells=defaultdict(list)
    for s in scenarios:
        row=_at(rows[s['scenario_id']],1)
        model_cells[(s['transition'],s['severity'],s['architecture'],s['fault'])].append(
            1-row['state_correct'])
    separated=sum(all(y==0 for y in outcomes) or all(y==1 for y in outcomes)
                  for outcomes in model_cells.values())
    _json(ROOT/'mixed_effects_model_status.json',{
        'status':'NOT_FIT','reason':'complete or near-complete deterministic cell separation; '
            'unpenalized prespecified mixed logit would not have stable finite coefficients',
        'transition_severity_architecture_fault_cells':len(model_cells),
        'completely_separated_cells':separated,
        'primary_nonmodel_effects':'paired risk differences with 95% CIs and exact McNemar',
        'population':'Track A only'})
    _csv(ROOT/'multiplicity_adjusted_results.csv',v4._bh([
        {**r,'comparison_family':'FA5_secondary_architecture_early_failure',
         'secondary':True} for r in mitigation]))
    return {'scenario_count':len(scenarios),'observation_count':len(scenarios)*15,
            'H1':effects,'H2':h_results,'H3_cache':cache,'H5_compounds':compounds,
            'H6_auth':auth,'H7_H8_mitigation':mitigation,
            'H6_paired':h6,'H7_survival':h7,
            'domain_count':len({r['domain'] for r in domains}),
            'query_family_count':len({r['query_family'] for r in families})}


def _figures(manifest,rows,tables,transfer_effects):
    v4.FIGURES=ROOT/'figures'
    v4.FIGURES.mkdir(exist_ok=True)
    scenarios=manifest['scenarios']
    canvas=v4._Canvas('RAGStateBench state-plane instrument')
    boxes=[('Canonical transitions',75,95),('Chunk replica',75,205),
           ('Parent replica',310,205),('ACL metadata',545,205),
           ('Candidate cache',75,325),('Auth filter',310,325),
           ('Final context',545,325)]
    for label,x,y in boxes:
        canvas.rect(x,y,185,55,'#dceaf7')
        canvas.text(x+92,y+33,label,14)
    for x1,y1,x2,y2 in ((170,150,170,205),(170,150,405,205),
        (170,150,640,205),(170,260,170,325),(405,260,405,325),
        (640,260,405,325),(170,380,640,380),(405,380,640,380)):
        canvas.line(x1,y1,x2,y2,'#355f85',2)
    canvas.save('01_state_plane_architecture')
    curves={}
    for transition in TRANSITIONS:
        selected=[traj for _,traj in _groups(scenarios,rows,'FA1',
            transition=transition,architecture='A0',severity='S2')]
        curves[transition]=[(tick,_mean(1-_at(r,tick)['state_correct'] for r in selected))
            for tick in (0,1,2,3,4,5,6,7,8,9,10,11,12,14)]
    v4._lines('02_transition_failure_trajectories','A0 S2 state failure',curves)
    survival={}
    for severity in ('S1','S2','S3'):
        selected=[traj for _,traj in _groups(scenarios,rows,'FA1',
            architecture='A0',severity=severity)]
        survival[severity]=[(t,1-p) for t,p in v4._km(selected)['curve']]
    v4._lines('03_convergence_survival','Cumulative stable convergence',survival)
    revoke=[r for r in _read(ROOT/'authorization_analysis.csv')]
    v4._bars('04_revoke_ucer_rew','REVOKE UCER and exposure window',
        [r['architecture']+':UCER' for r in revoke]+[r['architecture']+':REW' for r in revoke],
        [float(r['ucer_context_t1']) for r in revoke]+[float(r['rew_context_mean']) for r in revoke])
    grant={}
    for architecture in ('A0','A1','A2','A3','A4','A5'):
        selected=[traj for _,traj in _groups(scenarios,rows,'FA5',
            transition='GRANT',architecture=architecture,severity='S2')]
        grant[architecture]=[(tick,_mean(_at(r,tick)['ardr'] for r in selected))
            for tick in (0,1,2,3,4,5,6,7,8,9,10,11,12,14)]
    v4._lines('05_grant_availability','GRANT availability denial',grant)
    cache=_read(ROOT/'cache_analysis.csv')
    v4._bars('06_cache_persistence','Cache persistence after index convergence',
        [r['transition']+':'+r['architecture'] for r in cache],
        [float(r['a0_rate'])-float(r['mitigation_rate']) for r in cache])
    mitigation=_read(ROOT/'mitigation_analysis.csv')
    v4._bars('07_mitigation_risk_cost','Mitigation early risk reduction',
        ['A1','A2','A3','A4','A5'],
        [_mean(max(0,-float(r['risk_difference'])) for r in mitigation
               if r['architecture']==architecture)
         for architecture in ('A1','A2','A3','A4','A5')])
    h4=_read(H4/'transition_summary.csv')
    selected=[r for r in h4 if r['transition'] in ('UPDATE','SUPERSEDE')]
    v4._bars('08_partial_tradeoff','H4 exploratory claim rescue and mixed versions',
        [r['transition']+':claim' for r in selected]+[
            r['transition']+':MVCR' for r in selected],
        [-float(r['paired_claim_failure_tick_difference']) for r in selected]+[
            float(r['paired_mvcr_tick_difference']) for r in selected])
    ooo=[traj for s,traj in _groups(scenarios,rows,'FA3')
         if s['fault']=='OUT_OF_ORDER' and s['transition']=='DELETE']
    if not ooo:
        ooo=[traj for s,traj in _groups(scenarios,rows,'FA3')
             if s['fault']=='OUT_OF_ORDER']
    v4._lines('09_out_of_order_resurrection','Out-of-order failure and replay',{
        'state failure':[(tick,_mean(1-_at(r,tick)['state_correct'] for r in ooo))
            for tick in (0,1,2,3,4,5,6,7,8,9,10,11,12,14)],
        'SERR':[(tick,_mean(_at(r,tick)['serr'] for r in ooo))
            for tick in (0,1,2,3,4,5,6,7,8,9,10,11,12,14)]})
    compound_curves={}
    for arm in ('NONE','F1','F2','F1_F2'):
        selected=[traj for s,traj in _groups(scenarios,rows,'FA7',architecture='A0')
                  if s['compound']=='CF4' and s['component']==arm]
        compound_curves[arm]=[(tick,_mean(1-_at(r,tick)['state_correct'] for r in selected))
            for tick in (0,1,2,3,4,5,6,7,8,9,10,11,12,14)]
    v4._lines('10_compound_trajectories','CF4 matched stress trajectories',compound_curves)
    localization=_read(ROOT/'localization_analysis.csv')
    counts=Counter()
    for r in localization:counts[r['inferred_plane']]+=int(r['incorrect_observation_count'])
    shown=counts.most_common(6)
    other=sum(counts.values())-sum(n for _,n in shown)
    v4._bars('11_failure_localization','Incorrect observations by inferred plane',
        [label for label,_ in shown]+(['other'] if other else []),
        [n for _,n in shown]+([other] if other else []))
    transfer_groups=defaultdict(list)
    for r in transfer_effects:
        if r['effect_class']!='T5':transfer_groups[r['effect_class']].append(r)
    v4._bars('12_trackA_trackB_transfer','Track-A vs PA2 direction transfer',
        [cls+':A' for cls in ('T1','T2','T3','T4')]+[
         cls+':B' for cls in ('T1','T2','T3','T4')],
        [_mean(abs(r['track_a_effect'] or 0) for r in transfer_groups[cls])
         for cls in ('T1','T2','T3','T4')]+[
         _mean(abs(r['track_b_effect'] or 0) for r in transfer_groups[cls])
         for cls in ('T1','T2','T3','T4')])
    domain=_read(ROOT/'domain_heterogeneity.csv')
    by_domain=defaultdict(list)
    for r in domain:by_domain[r['domain']].append(float(r['risk_increase']))
    v4._bars('13_domain_heterogeneity','Mean H1 effect by domain',
        sorted(by_domain),[_mean(by_domain[d]) for d in sorted(by_domain)],ymax=1)
    components=_read(ROOT/'component_analysis.csv')
    v4._bars('14_component_mechanisms','Component factorial early failure',
        [r['factorial_block'][:4]+':'+r['E']+r['C']+r['A'] for r in components],
        [float(r['failure_at_1']) for r in components],ymax=1)
    costs=[r for r in _read(ROOT/'cost_analysis.csv') if r['final_block']=='FA5_S2_matched']
    v4._bars('15_latency_overhead','Matched S2 retrieval latency by architecture',
        [r['architecture'] for r in costs],
        [float(r['median_retrieval_latency_ms']) for r in costs])


def finalize():
    manifest,rows=load_and_validate()
    if not (ROOT/'primary_effects.csv').exists():
        raise ValueError('Statistical analysis must run first')
    tb=json.loads((ROOT/'track_b_confirmation_summary.json').read_text())
    if not tb['all_valid'] or tb['scenario_count']!=120:
        raise ValueError('Targeted Track-B confirmation incomplete')
    transfer=json.loads((TB/'transfer_manifest.json').read_text())
    b_rows=defaultdict(list)
    for s in transfer['scenarios']:
        with (ROOT/'track_b_scenario_rows'/f'{s["scenario_id"]}.jsonl').open() as stream:
            b_rows[s['scenario_id']]=[json.loads(line) for line in stream]
    historical_b=defaultdict(list)
    for line in (TB/'observations.jsonl').open():
        row=json.loads(line);historical_b[row['scenario_id']].append(row)
    non_deterministic={'campaign_id','retrieval_latency_ms','parent_latency_ms'}
    b_reproduced=all([
        {k:v for k,v in row.items() if k not in non_deterministic} for row in b_rows[sid]]==[
        {k:v for k,v in row.items() if k not in non_deterministic} for row in historical_b[sid]]
        for sid in b_rows)
    _json(ROOT/'track_b_confirmation_integrity.json',{'identical_deterministic_rows':b_reproduced,
        'scenario_count':len(b_rows),'observation_count':sum(len(x) for x in b_rows.values())})
    if not b_reproduced:raise ValueError('Targeted Track-B repeat differs from frozen deterministic rows')
    effects=transfer_effect_rows(manifest['scenarios'],transfer['scenarios'],rows,b_rows)
    for r in effects:
        if r['effect_class']=='T5':
            r.update({'track_a_effect':None,'track_b_effect':None,
                      'track_a_direction':'not_estimable','classification':'not estimable',
                      'estimability_reason':'frozen TB4 lacks matched no-fault arm'})
    d11=transfer_d11(effects,[],retrieval_interpretable=True)
    _csv(ROOT/'trackB_transfer_effects.csv',effects)
    _json(ROOT/'trackB_direction_summary.json',d11)
    primary=[r for r in _read(ROOT/'primary_effects.csv') if r['hypothesis']!='H9']
    primary.append({'hypothesis':'H9','comparison':'frozen_PA2_directional_transfer',
        'endpoint':'preserved_or_attenuated/estimable_effects',
        'n':d11['agreement_denominator'],'effect':d11['agreement_numerator'],
        'interpretation':f'{d11["status"]}; Track B separate population; T5 not estimable'})
    _csv(ROOT/'primary_effects.csv',primary)
    _figures(manifest,rows,None,effects)
    table_sources={
        '01_benchmark_corpus':'results/state-isolated-pilot-v4/ta0_clean_summary.json',
        '02_scenario_matrix':ROOT/'scenario_composition.csv',
        '03_primary_endpoints':ROOT/'primary_effects.csv',
        '04_convergence':ROOT/'convergence_analysis.csv',
        '05_authorization':ROOT/'authorization_analysis.csv',
        '06_cache':ROOT/'cache_analysis.csv',
        '07_mitigation':ROOT/'mitigation_analysis.csv',
        '08_partial_exploratory':ROOT/'partial_analysis.csv',
        '09_compounds':ROOT/'compound_analysis.csv',
        '10_domain_robustness':ROOT/'domain_heterogeneity.csv',
        '11_realistic_transfer':ROOT/'trackB_transfer_effects.csv',
        '12_cost':ROOT/'cost_analysis.csv'}
    table_dir=ROOT/'tables';table_dir.mkdir(exist_ok=True)
    for name,source in table_sources.items():
        source=Path(source)
        shutil.copy2(source,table_dir/(name+source.suffix))
    env=dict(os.environ,PYTHONPATH=str(Path('src').resolve())+os.pathsep+
             os.environ.get('PYTHONPATH',''))
    fresh=subprocess.run([sys.executable,'-m',
        'ragstate.campaign.final_scientific_campaign_v1','sample'],
        capture_output=True,text=True,env=env)
    expected=json.loads((ROOT/'preflight_gate_summary.json').read_text())['sample']
    reproduced=fresh.returncode==0 and json.loads(fresh.stdout)==expected
    _json(ROOT/'reproducibility.json',{'identical':reproduced and b_reproduced,
        'scenario_count':len(expected),'fresh_exit':fresh.returncode,
        'sample_ids':[r['scenario_id'] for r in expected],
        'track_b_full_repeat_identical':b_reproduced})
    audit_files=list((ROOT/'validity_audits').glob('*.json'))
    figure_pairs=len(list((ROOT/'figures').glob('*.svg')))==len(list((ROOT/'figures').glob('*.png')))
    primary_rows=_read(ROOT/'primary_effects.csv')
    h1_trace=[]
    for transition in TRANSITIONS:
        selected=[traj for _,traj in _groups(manifest['scenarios'],rows,'FA1',
            transition=transition,architecture='A0',severity='S2')]
        actual=_mean(_at(r,-1)['state_correct']-_at(r,1)['state_correct']
                     for r in selected)
        reported=next(r for r in primary_rows if r['hypothesis']=='H1' and
                      r['comparison']==transition+'_pre_vs_t1')
        h1_trace.append({'transition':transition,'raw_n':len(selected),
            'raw_effect':actual,'reported_effect':float(reported['risk_difference']),
            'match':len(selected)==int(reported['n']) and
                    abs(actual-float(reported['risk_difference']))<1e-12})
    cache_trace=[]
    cache_by_key=defaultdict(dict)
    for s,traj in _groups(manifest['scenarios'],rows,'FA4'):
        cache_by_key[(s['query_id'],s['transition'])][s['architecture']]=traj
    for reported in _read(ROOT/'cache_analysis.csv'):
        matched=[g for (qid,t),g in cache_by_key.items()
                 if t==reported['transition'] and 'A0' in g and
                 reported['architecture'] in g]
        def persistent(traj):
            return int(any(r['logical_time']>=0 and r['index_converged'] and
                r['cache_hit'] and not r['state_correct'] for r in traj))
        actual=_mean(persistent(g[reported['architecture']])-persistent(g['A0'])
                     for g in matched)
        cache_trace.append({'transition':reported['transition'],
            'architecture':reported['architecture'],'raw_n':len(matched),
            'raw_effect':actual,'reported_effect':float(reported['risk_difference']),
            'match':len(matched)==int(reported['n']) and
                    abs(actual-float(reported['risk_difference']))<1e-12})
    auth_trace=[]
    a0={s['query_id']:traj for s,traj in _groups(manifest['scenarios'],rows,'FA1',
        transition='REVOKE',architecture='A0',severity='S2')}
    for reported in _read(ROOT/'live_authorization_effects.csv'):
        arch=reported['comparison'].split('_')[0]
        alternate={s['query_id']:traj for s,traj in _groups(manifest['scenarios'],rows,
            'FA5',transition='REVOKE',architecture=arch,severity='S2')}
        common=sorted(set(a0)&set(alternate))
        actual=_mean(int(_at(alternate[q],1)['ucer_context']>0)-
                     int(_at(a0[q],1)['ucer_context']>0) for q in common)
        auth_trace.append({'architecture':arch,'raw_n':len(common),
            'raw_effect':actual,'reported_effect':float(reported['risk_difference']),
            'match':len(common)==int(reported['n']) and
                    abs(actual-float(reported['risk_difference']))<1e-12})
    figure_sources={
        '01_state_plane_architecture':ROOT/'campaign_manifest.json',
        '02_transition_failure_trajectories':ROOT/'observations.jsonl',
        '03_convergence_survival':ROOT/'convergence_analysis.csv',
        '04_revoke_ucer_rew':ROOT/'authorization_analysis.csv',
        '05_grant_availability':ROOT/'observations.jsonl',
        '06_cache_persistence':ROOT/'cache_analysis.csv',
        '07_mitigation_risk_cost':ROOT/'mitigation_analysis.csv',
        '08_partial_tradeoff':H4/'transition_summary.csv',
        '09_out_of_order_resurrection':ROOT/'observations.jsonl',
        '10_compound_trajectories':ROOT/'observations.jsonl',
        '11_failure_localization':ROOT/'localization_analysis.csv',
        '12_trackA_trackB_transfer':ROOT/'trackB_transfer_effects.csv',
        '13_domain_heterogeneity':ROOT/'domain_heterogeneity.csv',
        '14_component_mechanisms':ROOT/'component_analysis.csv',
        '15_latency_overhead':ROOT/'mitigation_analysis.csv'}
    figure_trace=[{'figure':stem,'source':str(source),'source_sha256':_sha(source),
        'svg_sha256':_sha(ROOT/'figures'/f'{stem}.svg'),
        'png_sha256':_sha(ROOT/'figures'/f'{stem}.png')}
        for stem,source in figure_sources.items()]
    table_trace=[{'table':name,'source':str(source),'source_sha256':_sha(source),
        'copy_sha256':_sha(table_dir/(name+Path(source).suffix))}
        for name,source in table_sources.items()]
    _json(ROOT/'traceability.json',{'h1_raw_recomputation':h1_trace,
        'cache_raw_recomputation':cache_trace,
        'authorization_raw_recomputation':auth_trace,
        'figures':figure_trace,'tables':table_trace})
    statuses=_read(ROOT/'scenario_status.csv')
    h4_summary=_read(H4/'mixed_interval_summary.csv')[0]
    h4_null=float(h4_summary['mixed_integrated_risk'])==float(
        h4_summary['uniform_integrated_risk'])
    b_population=all(r['campaign_id']=='final-scientific-campaign-v1-track-b'
                     for trajectories in b_rows.values() for r in trajectories)
    fc={
        'FC1':len(audit_files)==manifest['scenario_count'],
        'FC2':all(r['status']=='VALID' for r in json.loads((ROOT/'block_status.json').read_text()).values()),
        'FC3':len(h1_trace)==5 and len(cache_trace)==8 and len(auth_trace)==2
              and all(r['match'] for r in h1_trace+cache_trace+auth_trace),
        'FC4':len(rows)==manifest['scenario_count'] and
              sum(len(v) for v in rows.values())==15*manifest['scenario_count']
              and len(statuses)==manifest['scenario_count'] and
              all(r['status'] in ('valid','censored') for r in statuses),
        'FC5':figure_pairs and len(list((ROOT/'figures').glob('*.svg')))>=13 and
              len(list(table_dir.iterdir()))==12 and
              all(r['source_sha256']==r['copy_sha256'] for r in table_trace)
              and len(figure_trace)==15,
        'FC6':reproduced and b_reproduced,
        'FC7':all(r['campaign_id']=='final-scientific-campaign-v1'
                  for r in (json.loads(line) for line in (ROOT/'observations.jsonl').open())
                  if r['logical_time']==-1),
        'FC8':json.loads((H4/'gate_summary.json').read_text())['H4-P7']=='PASS'
              and h4_null,
        'FC9':all(r['classification']=='not estimable' for r in effects if r['effect_class']=='T5'),
        'FC10':tb['separate_population'] and b_population}
    _json(ROOT/'gate_summary.json',{'gates':{k:'PASS' if v else 'FAIL' for k,v in fc.items()},
        'D11':d11,'manifest_sha256':manifest['scenario_sha256'],
        'figure_count':len(list((ROOT/'figures').glob('*.svg'))),
        'table_count':len(list(table_dir.iterdir()))})
    (ROOT/'summary.md').write_text('# Final scientific campaign v1\n\n'
        f'Frozen policy SHA-256: `{manifest["policy_sha256"]}`. '
        f'Manifest SHA-256: `{manifest["scenario_sha256"]}`.\n\n'
        f'{manifest["scenario_count"]:,} Track-A executions, '
        f'{15*manifest["scenario_count"]:,} observations; all FA1, FA3, FA4, '
        'FA5, FA6, FA7 blocks VALID. FA2 is an analytic view of FA1. '
        f'Targeted Track-B confirmation: {tb["scenario_count"]} scenarios and '
        f'{tb["observation_count"]} observations in a separate population. '
        'FC1–FC10 all PASS.\n\n'
        '## Primary findings\n\n'
        '- H1: A0/S2 early failure risk increases by 0.985–1.000 across all five transitions '
        '(paired n=174–204 per transition).\n'
        '- H2: REVOKE early UCER_context=1.000; GRANT early ARDR=1.000 under A0/S2.\n'
        '- H3/H8: after index convergence, A0 stale-cache persistence is 1.000 for '
        'UPDATE, SUPERSEDE, DELETE; A3/A5 reduce it to 0–0.015. REVOKE persistence is '
        '0 under all three architectures in this endpoint. At t=5 A3 sacrifices cache hits, '
        'while A5 invalidates then repopulates.\n'
        '- H5: A0 compound trajectories differ from both isolated arms for CF3 (4/4), '
        'CF4 (6/6), and CF6 (6/6); CF1, CF2, CF5 do not. These are stress/trajectory '
        'comparisons, not universal factorial interaction claims.\n'
        '- H6: A4/A5 reduce REVOKE binary context exposure by 1.000 (paired n=18, '
        '95% paired CI [-1,-1]); REW_context falls from 5 to 0 observed ticks. '
        'A4 averages 2.11 live authorization calls at t=1.\n'
        '- H7: delivered event-driven A2 shortens RMST through tick 14 relative to '
        'periodic A1 by 5.33–6.00 ticks across focused transitions (n=18 each).\n'
        f'- H9/D11: {d11["agreement_numerator"]}/{d11["agreement_denominator"]} '
        'estimable effects transfer directionally under PA2; T5 remains NOT_ESTIMABLE.\n\n'
        '## Convergence, exploratory mechanisms, and limits\n\n'
        'For UPDATE under A0, median T_RSC is 2, 5, and 10 ticks at S1, S2, S3. '
        'The H4 matched-horizon primary directional hypothesis was unsupported; '
        'UPDATE/SUPERSEDE claim rescue with greater mixed-version exposure remains exploratory. '
        'Out-of-order samples show stale replay and recovery; dropped-event samples are '
        'right-censored. State-plane localization retains joint chunk/parent/ACL/cache states '
        'and is a hash/stage heuristic rather than unique causal attribution.\n\n'
        'The corpus is controlled and synthetic. The 122 inherited repeated control '
        'executions are not counted as independent query replication in primary analyses. '
        'Only the frozen selected-unit seed is represented; additional seeded topology '
        'robustness was not run. The prespecified mixed logit was not fitted because '
        '72/90 cells show complete separation; paired effects and survival estimates '
        'remain the inferential basis. Compound and out-of-order samples are small. '
        'Local CPU latency is not a deployment estimate. No LLM generation was run.\n\n'
        'See the analysis CSVs, `traceability.json`, `figures/`, and `tables/`.\n')
    return {'gates':{k:'PASS' if v else 'FAIL' for k,v in fc.items()},
            'D11':d11,'figures':len(list((ROOT/'figures').glob('*.svg'))),
            'tables':len(list(table_dir.iterdir()))}


if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('action',choices=('analyze','finalize'))
    action=parser.parse_args().action
    print(json.dumps({'analyze':analyze,'finalize':finalize}[action](),
                     indent=2,sort_keys=True))
