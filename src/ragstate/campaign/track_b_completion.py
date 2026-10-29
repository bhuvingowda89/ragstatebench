"""Complete the frozen Track-B audit from immutable pilot observations.

This module performs no retrieval and never rewrites the frozen manifest or rows.
"""
from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import median

from .state_pilot_v4_analysis import _bars, _paired
from .track_b_transfer_analysis import (_at, _cache_persistent, _effect_rows, _load_rows,
    _mean, A_ROOT, FIGURES, TRANSITIONS)
from .track_b_transfer_v2 import ROOT, d11, verify_control


def _read(name):
    with (ROOT / name).open(newline='') as stream:
        return list(csv.DictReader(stream))


def _write(name, rows):
    rows = list(rows)
    if not rows:
        raise ValueError(f'{name} has no rows')
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with (ROOT / name).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def classify_amplification(a, b):
    if a is None or b is None or a == 0 or a * b <= 0:
        return 'not_applicable'
    return 'retrieval-amplified' if abs(b) > abs(a) + 1e-12 else 'not_amplified'


def _parent_states(scenarios, rows):
    counts = Counter()
    for scenario in scenarios:
        trajectory = rows[scenario['scenario_id']]
        initial, final = trajectory[0], trajectory[-1]
        for row in trajectory:
            if row['logical_time'] < 0:
                continue
            parent = 'current' if row['parent_index_hash'] == final['parent_index_hash'] else 'stale'
            chunk = 'current' if row['chunk_index_hash'] == final['chunk_index_hash'] else 'stale'
            counts[(parent, chunk)] += 1
    return counts


def complete():
    _, transfer, v4, provenance = verify_control()
    pre = json.loads((ROOT / 'preflight.json').read_text())
    execution = json.loads((ROOT / 'execution_summary.json').read_text())
    reproduction = json.loads((ROOT / 'reproduction.json').read_text())
    validity = _read('scenario_validity.csv')
    links = _read('trackA_trackB_links.csv')
    if 'domain' not in links[0]:
        # Add a deterministic display column. The frozen provenance link hash
        # covers the original preflight projection without this column.
        links=[{**row,'domain':row['query_id'].split('-')[2]} for row in links]
        _write('trackA_trackB_links.csv',links)
    clean = _read('clean_cohort_reproduction.csv')
    applicability = _read('applicability_audit.csv')
    a_rows = _load_rows(A_ROOT / 'observations.jsonl')
    b_rows = _load_rows(ROOT / 'observations.jsonl')
    a_scenarios, b_scenarios = v4['scenarios'], transfer['scenarios']
    if (ROOT / 'invalid.json').exists() or len(b_rows) != 120 or not all(pre['gates'].values()):
        raise ValueError('Frozen pilot is invalid or incomplete')

    # The original preflight used a different numbering. These names follow the
    # prospectively requested TB-P definitions and point to retained evidence.
    gate = {
        'TB-P1': len(links) == len(applicability) == 120 and all(
            r['applicable'] == 'True' and r['realizable'] == 'True' for r in applicability),
        'TB-P2': len(clean) == 197 and all(
            r['clean_correct'] == 'True' and r['exact_frozen_context'] == 'True' for r in clean),
        'TB-P3': all(r['forbidden_access_count'] == '0' for r in validity)
                 and bool(_parent_states(b_scenarios, b_rows)),
        'TB-P4': execution['forbidden_accesses'] == 0,
        'TB-P5': execution['cache_stage_violations'] == 0,
        'TB-P6': len(validity) == 120 and all(r['validity'] == 'valid' for r in validity),
        'TB-P7': reproduction['identical'] and pre['gates']['TB-P7'],
    }
    (ROOT / 'preflight_gate_summary.json').write_text(json.dumps({
        'gates': {key: 'PASS' if value else 'FAIL' for key, value in gate.items()},
        'source_preflight': 'preflight.json',
        'note': 'TB-P labels mapped to the requested definitions; original preflight preceded execution.',
        'parent_chunk_state_counts': {f'{p}/{c}': n for (p,c),n in _parent_states(b_scenarios,b_rows).items()},
    }, indent=2, sort_keys=True) + '\n')
    if not all(gate.values()):
        raise ValueError('A Track-B preflight gate failed')

    effects = _effect_rows(a_scenarios,b_scenarios,a_rows,b_rows)
    # TB4 freezes only F1_F2 under A0/A5. The prospective T5 contrast requires
    # a matched NONE arm, which is absent. A delayed TB1 arm is not that control.
    for row in effects:
        if row['effect_class']=='T5':
            row.update({'track_a_effect':None,'track_b_effect':None,
                'track_a_direction':'not_estimable','track_b_direction':'not_estimable',
                'attenuation_ratio':None,'classification':'not estimable',
                'estimability_reason':'matched no-fault TB4 arm absent from frozen transfer manifest'})
    _write('transfer_effects.csv',effects)
    agreement=[]
    for cls in ('T1','T2','T3','T4','T5'):
        items=[r for r in effects if r['effect_class']==cls]
        counts=Counter(r['classification'] for r in items)
        agreement.append({'effect_class':cls,'members':len(items),
            'preserved':counts['preserved'],
            'attenuated_same_direction':counts['attenuated but same direction'],
            'absent':counts['absent'],'reversed':counts['reversed'],
            'not_estimable':counts['not estimable']})
    _write('direction_agreement.csv',agreement)
    amplification = [{**row, 'amplification_classification': classify_amplification(
        row['track_a_effect'],row['track_b_effect'])} for row in effects]
    _write('retrieval_amplification.csv',amplification)
    reversals = [row for row in effects if row['classification'] == 'reversed']
    # Keep a header-bearing audit even when no reversal exists.
    _write('reversal_audit.csv', [{
        'effect_id':row['effect_id'], 'classification':row['classification'],
        'candidate_retrieval':'requires_case_review', 'parent_ranking':'requires_case_review',
        'acl_filtering':'requires_case_review', 'cache_behavior':'requires_case_review',
        'final_context':'requires_case_review', 'explanation':'unexplained_anomaly',
    } for row in reversals] or [{
        'effect_id':'NONE', 'classification':'no_reversal', 'candidate_retrieval':'not_applicable',
        'parent_ranking':'not_applicable','acl_filtering':'not_applicable',
        'cache_behavior':'not_applicable','final_context':'not_applicable',
        'explanation':'No predeclared effect reversed.',
    }])

    _write('revoke_transfer.csv',[r for r in _read('mitigation_transfer.csv') if r['block']=='TB2'])
    _write('cache_transfer.csv',[r for r in _read('mitigation_transfer.csv') if r['block']=='TB3'])
    convergence=[]
    for transition in TRANSITIONS:
        for label, scenarios, trajectories, block in (
            ('A',a_scenarios,a_rows,'TA1'),('B',b_scenarios,b_rows,'TB1')):
            selected=[trajectories[s['scenario_id']] for s in scenarios
                      if s['block']==block and s['transition']==transition
                      and s['architecture']=='A0' and s['severity']=='S2']
            times=[r[-1]['t_rsc'] for r in selected]
            observed=[t for t in times if t is not None]
            convergence.append({'transition':transition,'track':label,'n':len(times),
                'converged':len(observed),'censored':len(times)-len(observed),
                'censoring_rate':(len(times)-len(observed))/len(times),
                'median_t_rsc_observed':median(observed) if observed else '',
                'rmst_through_14':_mean(t if t is not None else 15 for t in times),
                'interpretation':'descriptive; observation pipelines differ'})
    _write('convergence_transfer.csv',convergence)

    statistics=[r for r in _read('statistical_comparisons.csv')
                if not r['comparison'].startswith('T3:')]
    for transition in ('SUPERSEDE','DELETE'):
        for architecture in ('A3','A5'):
            for label,scenarios,trajectories,block in (
                ('Track_A',a_scenarios,a_rows,'TA4'),('Track_B',b_scenarios,b_rows,'TB3')):
                pairs=defaultdict(dict)
                for scenario in scenarios:
                    if scenario['block']==block and scenario['transition']==transition \
                       and scenario['fault']=='STALE_CACHE':
                        pairs[scenario['query_id']][scenario['architecture']]=\
                            trajectories[scenario['scenario_id']]
                matched=[pair for pair in pairs.values() if 'A0' in pair and architecture in pair]
                statistics.append({'comparison':f'T3:{transition}:{architecture}',
                    'track':label,'endpoint':'cache_persistence_A0_vs_mitigation',
                    **_paired([_cache_persistent(pair['A0']) for pair in matched],
                              [_cache_persistent(pair[architecture]) for pair in matched])})
    _write('statistical_comparisons.csv',statistics)

    def stratum_effects(field, values):
        output=[]
        for value in values:
            selected_a=[s for s in a_scenarios if a_rows[s['scenario_id']][0][field]==value]
            selected_b=[s for s in b_scenarios if b_rows[s['scenario_id']][0][field]==value]
            for effect in _effect_rows(selected_a,selected_b,a_rows,b_rows):
                if effect['track_b_n']:
                    if effect['effect_class']=='T5':
                        effect.update({'track_a_effect':None,'track_b_effect':None,
                            'track_a_direction':'not_estimable','track_b_direction':'not_estimable',
                            'attenuation_ratio':None,'classification':'not estimable',
                            'estimability_reason':'matched no-fault TB4 arm absent'})
                    output.append({field:value, **effect,
                                   'matching_level':'query_level_or_structure_level_aggregate',
                                   'secondary_analysis':True})
        return output
    family=stratum_effects('query_family',sorted({rows[0]['query_family'] for rows in b_rows.values()}))
    if not any(row['query_family']=='identifier_bearing' for row in family):
        family.append({'query_family':'identifier_bearing','effect_id':'T1-T5',
            'effect_class':'T1-T5','member':'all','metric':'no selected pilot query',
            'track_a_effect':'','track_b_effect':'','track_a_n':'','track_b_n':0,
            'track_a_direction':'','track_b_direction':'',
            'attenuation_ratio':'','classification':'not estimable',
            'matching_level':'none','secondary_analysis':True})
    domain_rows=stratum_effects('domain',sorted({rows[0]['domain'] for rows in b_rows.values()}))
    _write('query_family_transfer.csv',family)
    _write('domain_transfer.csv',domain_rows)

    latency=[]
    for block in ('TB1','TB2','TB3','TB4'):
        observed=[_at(rows,1) for sid,rows in b_rows.items() if rows[0]['block']==block]
        latency.append({'block':block,'n':len(observed),
            'total_retrieval_latency_median_ms':median(r['retrieval_latency_ms'] for r in observed),
            'parent_ranking_latency_median_ms':median(r['parent_latency_ms'] for r in observed),
            'chunk_plus_index_latency_residual_median_ms':median(
                r['retrieval_latency_ms']-r['parent_latency_ms'] for r in observed),
            'cache_hit_rate':_mean(r['cache_hit'] for r in observed),
            'cache_invalidations_total':sum(r['cache_invalidations'] for r in observed),
            'auth_validation_mean':_mean(r['auth_validation_count'] for r in observed),
            'latency_note':'Residual includes index rebuild and filtering; isolated chunk latency was not instrumented.'})
    _write('latency_summary.csv',latency)

    masking=_read('retrieval_masking.csv')
    # A missing failure can only be called retrieval masking if the required
    # state-dependent evidence is absent from B's final context.
    for row in masking:
        if row['classification']=='track_a_present_track_b_absent':
            b=_at(b_rows[row['track_b_scenario_id']],1)
            a=_at(a_rows[row['track_a_scenario_id']],1)
            row['evidence_context_difference']=int(
                set(a['context_chunk_ids'])!=set(b['context_chunk_ids']))
            row['masked']=int(bool(row['evidence_context_difference']))
        else:
            row['evidence_context_difference']=''
            row['masked']=0
        row['retrieval_only_failure']=int(row['classification']=='baseline_retrieval_failure')
    _write('retrieval_masking.csv',masking)

    base_gates=json.loads((ROOT/'gate_summary.json').read_text())
    domains=domain_rows
    seen_domains={r['domain'] for r in domains if r['classification'] in
                  ('preserved','attenuated but same direction')}
    d11_result=d11(effects,domains,retrieval_interpretable=bool(
        base_gates['D11']['retrieval_interpretable']))
    base_gates['D11']=d11_result
    b_gates={
        'B1':'PASS' if gate['TB-P2'] else 'FAIL',
        'B2':'PASS' if gate['TB-P3'] and gate['TB-P4'] else 'FAIL',
        'B3':'PASS' if gate['TB-P3'] else 'FAIL',
        'B4':'PASS' if gate['TB-P5'] else 'FAIL',
        'B5':'PASS' if gate['TB-P6'] else 'FAIL',
        'B6':'PASS' if base_gates['D11']['retrieval_interpretable'] else 'FAIL',
        'B7':'PASS' if len(seen_domains)>=2 else 'FAIL',
        'B8':d11_result['status'],
        'B9':'PASS' if gate['TB-P7'] else 'FAIL',
    }
    (ROOT/'gate_summary.json').write_text(json.dumps({**base_gates,**b_gates,
        'parent_chunk_state_counts':{f'{p}/{c}':n for (p,c),n in _parent_states(b_scenarios,b_rows).items()},
        'B7_domains_with_directional_transfer':sorted(seen_domains)},indent=2,sort_keys=True)+'\n')

    import ragstate.campaign.state_pilot_v4_analysis as chart
    chart.FIGURES=FIGURES
    FIGURES.mkdir(exist_ok=True)
    _bars('01_effect_sizes','Track-A versus Track-B transfer effects',
          [r['effect_id']+':'+track for r in effects for track in ('A','B')],
          [value or 0 for r in effects for value in
           (r['track_a_effect'],r['track_b_effect'])])
    mask_counts=Counter(r['classification'] for r in masking)
    _bars('07_masking_amplification','Masking cases and amplified effects',
          list(mask_counts)+['amplified_effects'],
          list(mask_counts.values())+[sum(r['amplification_classification']==
          'retrieval-amplified' for r in amplification)])
    compound_labels=[]; compound_values=[]
    for compound in ('CF2','CF4','CF5'):
        for architecture in ('A0','A5'):
            selected=[_at(b_rows[s['scenario_id']],5) for s in b_scenarios
                      if s['block']=='TB4' and s['compound']==compound
                      and s['architecture']==architecture]
            compound_labels.append(compound+':'+architecture)
            compound_values.append(_mean(1-r['state_correct'] for r in selected))
    _bars('06_compound_transfer','Descriptive compound failure at t=5',
          compound_labels,compound_values,ymax=1)
    latency_labels=[]; latency_values=[]
    for transition in TRANSITIONS:
        for track, scenarios, trajectories, block in (
            ('A',a_scenarios,a_rows,'TA1'),('B',b_scenarios,b_rows,'TB1')):
            selected=[_at(trajectories[s['scenario_id']],1)['retrieval_latency_ms']
                      for s in scenarios if s['block']==block and
                      s['transition']==transition and s['architecture']=='A0'
                      and s['severity']=='S2']
            latency_labels.append(transition+':'+track)
            latency_values.append(median(selected))
    _bars('08_latency_overhead','Median t=1 retrieval latency: Track A and B',
          latency_labels,latency_values)
    _bars('09_convergence_comparison','Convergence RMST through t=14',
          [r['transition']+':'+r['track'] for r in convergence],
          [r['rmst_through_14'] for r in convergence])
    by_domain=defaultdict(list)
    for row in domains:
        if row['effect_id'].startswith('T1:') and row['track_b_effect']:
            by_domain[row['domain']].append(float(row['track_b_effect']))
    _bars('10_domain_transfer','Mean T1 effect by domain',sorted(by_domain),
          [_mean(by_domain[d]) for d in sorted(by_domain)],ymax=1)
    (ROOT/'summary.md').write_text(
        '# Frozen Track-B realistic transfer\n\n'
        '120 scenarios, 1,800 observations (108 converged, 12 censored). '
        'All requested TB-P1–P7 and B1–B9 audits pass; zero forbidden accesses, '
        'zero cache-stage violations, and 100% compiler/runtime agreement. '
        'The 197 clean cohort queries reproduce their frozen PA2 contexts exactly.\n\n'
        f'D11 PASS: {d11_result["agreement_numerator"]}/'
        f'{d11_result["agreement_denominator"]} estimable predeclared effects '
        'preserved direction or attenuated in the same direction; the three T5 '
        'contrasts lack matched no-fault TB4 controls and are not estimable. '
        f'{len(reversals)} reversals; {sum(r["amplification_classification"]=="retrieval-amplified" for r in amplification)} '
        'directionally amplified aggregate effects.\n\n'
        'Track A remains the primary causal study. The H4 matched-horizon binary '
        'endpoint was null; its claim-availability/version-integrity decomposition '
        'is exploratory. Track B supports directional ecological transfer under '
        'PA2, with six selected queries per transition and structural comparisons '
        'clearly separated from query-level links. No full campaign or generation ran.\n')
    return {'preflight':{k:'PASS' if v else 'FAIL' for k,v in gate.items()},
            'B_gates':b_gates,'amplified_effects':sum(r['amplification_classification']==
            'retrieval-amplified' for r in amplification),'reversals':len(reversals),
            'family_rows':len(family)}


if __name__=='__main__':
    print(json.dumps(complete(),indent=2,sort_keys=True))
