"""Repeated-measures analysis of the frozen PARTIAL seed supplement."""
from __future__ import annotations

import csv
import json
import os
import statistics
import subprocess
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from .h4_partial_analysis import _pair_measure
from .h4_partial_runtime import prepare_prestate, transition_event
from .evidence_spans import claim_span, normalize
from .partial_seed_robustness import ROOT, SEEDS, source_pairs, verify, write_csv, write_json, sample
from .state_pilot_manifest import _load_frozen, digest
from ragstate.retrieval.chunking import chunk_document
from ragstate.replicas.scientific import _document


def mean(values):
    values = list(values)
    return float(sum(values)/len(values)) if values else None


def clustered_ci(rows, key, *, seed=711):
    grouped = defaultdict(list)
    for row in rows:
        grouped[row['query_id']].append(row[key])
    keys = sorted(grouped)
    if not keys:
        return (None, None)
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(2000):
        selected = rng.choice(keys, size=len(keys), replace=True)
        values = [value for query in selected for value in grouped[query]]
        draws.append(mean(values))
    return tuple(float(x) for x in np.quantile(draws, [.025, .975]))


def summary(items, label, value):
    row = {label: value, 'n_pairs': len(items), 'n_queries': len({r['query_id'] for r in items})}
    for endpoint, prefix in [('claim_failure_ticks_difference', 'claim_failure'),
                             ('mvcr_ticks_difference', 'mixed_version')]:
        values = [r[endpoint] for r in items]
        row[prefix+'_mean_difference'] = mean(values)
        row[prefix+'_median_difference'] = statistics.median(values) if values else None
        lo, hi = clustered_ci(items, endpoint)
        row[prefix+'_ci95_low'], row[prefix+'_ci95_high'] = lo, hi
    return row


def classify_query(rows):
    claim = [r['claim_failure_ticks_difference'] for r in rows]
    mvcr = [r['mvcr_ticks_difference'] for r in rows]
    # A stable version cost alone does not establish the joint rescue/cost
    # tradeoff for a query; "no effect" refers to claim rescue here.
    if all(c == 0 for c in claim):
        return 'no effect'
    if any(c > 0 for c in claim) or any(m < 0 for m in mvcr):
        return 'sign-sensitive'
    if all(c < 0 and m > 0 for c, m in zip(claim, mvcr)) and len(set(zip(claim, mvcr))) == 1:
        return 'seed-stable benefit/tradeoff'
    return 'magnitude-sensitive'


def unique_rows(rows):
    by = {}
    for row in rows:
        by[(row['source_pair_id'], row['subset_hash'])] = row
    return list(by.values())


def _signed_bars(stem, title, labels, values):
    from .state_pilot_v4_analysis import _Canvas
    chart = _Canvas(title)
    values = [float(value or 0) for value in values]
    scale = max([abs(value) for value in values]+[1.0])
    baseline = 270
    chart.line(75, baseline, 850, baseline, '#293a49')
    chart.line(75, 75, 75, 455, '#293a49')
    slot = 760/max(1, len(values))
    for i, (label, value) in enumerate(zip(labels, values)):
        height = int(160*abs(value)/scale)
        x = 85+i*slot
        chart.rect(x, baseline-height if value >= 0 else baseline,
                   max(3, slot*.60), max(1, height), chart.COLORS[i%len(chart.COLORS)])
        chart.text(x+slot*.30, 480, str(label)[:12], 11)
        chart.text(x+slot*.30, baseline-height-9 if value >= 0 else baseline+height+17,
                   f'{value:+.2f}', 11)
    chart.save(stem)


def figures(per_seed, unique, queries, topologies, domains):
    import ragstate.campaign.state_pilot_v4_analysis as charts
    charts.FIGURES = ROOT/'figures'
    charts.FIGURES.mkdir(exist_ok=True)
    labels = [f"{r['seed']}:{r['transition'][:3]}" for r in per_seed if r['transition'] != 'ALL']
    selected = [r for r in per_seed if r['transition'] != 'ALL']
    _signed_bars('01_claim_failure_by_seed', 'Claim failure: mixed minus uniform', labels,
                 [r['claim_failure_mean_difference'] for r in selected])
    _signed_bars('02_mixed_version_by_seed', 'Mixed versions: mixed minus uniform', labels,
                 [r['mixed_version_mean_difference'] for r in selected])
    from .state_pilot_v4_analysis import _Canvas
    chart = _Canvas('Paired claim availability and version integrity')
    chart.line(75, 450, 850, 450, '#293a49')
    chart.line(450, 75, 450, 450, '#293a49')
    for row in unique:
        x = 450+95*row['claim_failure_ticks_difference']
        y = 450-110*row['mvcr_ticks_difference']
        chart.rect(x-3, y-3, 6, 6, chart.COLORS[0 if row['transition']=='UPDATE' else 1])
    chart.text(450, 500, 'Claim failure effect (ticks)', 12)
    chart.text(185, 70, 'Mixed-version effect (ticks)', 12)
    chart.save('03_two_dimensional_tradeoff')
    categories = sorted({r['classification'] for r in queries})
    _signed_bars('04_query_sensitivity', 'Query sensitivity across unique subsets', categories,
                 [sum(r['classification']==c for r in queries) for c in categories])
    _signed_bars('05_topology', 'Claim failure effect by topology',
                 [r['topology'] for r in topologies], [r['claim_failure_mean_difference'] for r in topologies])
    n_fraction = sorted({(r['N'],r['realized_fraction']) for r in unique})
    _signed_bars('06_N_fraction', 'Claim failure effect by N and fraction',
                 [f'{n}:{fraction:.2f}' for n,fraction in n_fraction],
                 [mean(r['claim_failure_ticks_difference'] for r in unique
                       if r['N']==n and r['realized_fraction']==fraction)
                  for n,fraction in n_fraction])
    _signed_bars('07_domain', 'Claim failure effect by domain',
                 [r['domain'] for r in domains], [r['claim_failure_mean_difference'] for r in domains])


def analyze():
    manifest = verify()
    if (ROOT/'invalid.json').exists():
        raise ValueError('Supplement INVALID')
    pre = json.loads((ROOT/'preflight_gate_summary.json').read_text())
    execution = json.loads((ROOT/'execution_summary.json').read_text())
    if not all(pre['gates'].values()) or execution != {'pairs':1608,'scenarios':3216,
            'observations':48240,'valid_arms':3216}:
        raise ValueError('Execution gates or counts failed')
    by_id = defaultdict(list)
    with (ROOT/'observations.jsonl').open() as stream:
        for line in stream:
            row = json.loads(line)
            by_id[row['scenario_id']].append(row)
    pairs = [manifest['scenarios'][i:i+2] for i in range(0,len(manifest['scenarios']),2)]
    rows = []
    for pair in pairs:
        result = _pair_measure(pair, by_id)
        result.update(seed=pair[1]['ps_seed'], source_pair_id=pair[1]['ps_source_pair_id'],
                      subset_hash=digest(pair[1]['selected_unit_ids']),
                      selected_unit_ids=pair[1]['selected_unit_ids'])
        rows.append(result)
    # Independent fresh-process execution, including all state and audit fields.
    env = dict(os.environ, PYTHONPATH=str(Path('src').resolve()) + os.pathsep + os.environ.get('PYTHONPATH',''))
    fresh = subprocess.run([sys.executable, '-m', 'ragstate.campaign.partial_seed_robustness', 'sample'],
                           capture_output=True, text=True, env=env)
    reproduced = json.loads(fresh.stdout) if fresh.returncode == 0 else []
    with (ROOT/'runtime_audit.csv').open() as stream:
        runtime_audits={row['scenario_id']:row for row in csv.DictReader(stream)}
    stored = []
    from .partial_seed_robustness import stratified_pairs
    for pair in stratified_pairs(manifest):
        for scenario in pair:
            stable = [{key:value for key,value in row.items()
                       if key not in ('retrieval_latency_ms','parent_latency_ms')}
                      for row in by_id[scenario['scenario_id']]]
            stored.append({'scenario_id':scenario['scenario_id'], 'hash':digest(stable)})
    identical = fresh.returncode == 0 and all(a['scenario_id']==b['scenario_id'] and a['hash']==b['hash']
                    and a['audit']['valid'] and runtime_audits[a['scenario_id']]['valid']=='True'
                    for a,b in zip(reproduced,stored)) and len(reproduced)==len(stored)
    write_json(ROOT/'reproduction.json', {'identical':identical,'sample_count':len(stored),
                                         'preflight_identical':pre['gates']['PS-P8'],
                                         'fresh_exit':fresh.returncode,'fresh_stderr':fresh.stderr})
    if not identical:
        write_json(ROOT/'invalid.json', {'status':'INVALID','reason':'fresh_process_reproduction'})
        raise ValueError('Fresh-process reproduction failed')
    historical = {pair[1]['h4_pair_id']:pair[1] for pair in source_pairs()[1]}
    historical_rows=[]
    with Path('results/h4-partial-control-pilot-v1/pair_results.csv').open() as stream:
        for old in csv.DictReader(stream):
            source_id=old['pair_id']
            if source_id not in historical:
                continue
            scenario=historical[source_id]
            historical_rows.append({'source_pair_id':source_id,'seed':'seed_0',
                                    'query_id':old['query_id'],'transition':old['transition'],
                                    'domain':old['domain'],'topology':old['topology'],
                                    'N':int(old['N']),'realized_fraction':float(old['realized_fraction']),
                                    'subset_hash':digest(scenario['selected_unit_ids']),
                                    'selected_unit_ids':scenario['selected_unit_ids'],
                                    'claim_failure_ticks_difference':float(old['claim_failure_ticks_difference']),
                                    'mvcr_ticks_difference':float(old['mvcr_ticks_difference'])})
    if len(historical_rows)!=402:
        raise ValueError('Historical seed_0 observations unavailable')
    all_rows=historical_rows+rows
    write_csv(ROOT/'statistical_comparisons.csv', all_rows)
    uniqueness = []
    for source_id in sorted(historical):
        relevant = [r for r in rows if r['source_pair_id']==source_id]
        hashes = {digest(historical[source_id]['selected_unit_ids'])} | {r['subset_hash'] for r in relevant}
        uniqueness.append({'source_pair_id':source_id,'query_id':historical[source_id]['query_id'],
                           'transition':historical[source_id]['transition'],
                           'topology':historical[source_id]['partial_topology'],
                           'N':historical[source_id]['total_unit_count'],
                           'requested_seeds':5,'unique_subset_count':len(hashes),
                           'duplicate_subset_count':5-len(hashes),
                           'historical_seed_0_subset_hash':digest(historical[source_id]['selected_unit_ids'])})
    write_csv(ROOT/'subset_uniqueness.csv', uniqueness)
    unique = unique_rows(all_rows)
    per_seed = [dict(seed=seed,transition=transition,**summary(
                    [r for r in all_rows if r['seed']==seed and
                     (transition=='ALL' or r['transition']==transition)],'scope','all'))
                for seed in ('seed_0',*SEEDS) for transition in (*('UPDATE','SUPERSEDE'),'ALL')]
    write_csv(ROOT/'per_seed_effects.csv',per_seed)
    grouped = defaultdict(list)
    for row in unique:
        grouped[(row['query_id'],row['transition'])].append(row)
    queries = []
    for (query,transition), items in sorted(grouped.items()):
        claim=[r['claim_failure_ticks_difference'] for r in items]
        mvcr=[r['mvcr_ticks_difference'] for r in items]
        queries.append({'query_id':query,'transition':transition,'unique_subsets':len(items),
                        'min_claim_effect':min(claim),'max_claim_effect':max(claim),
                        'mean_claim_effect':mean(claim),'min_mixed_version_effect':min(mvcr),
                        'max_mixed_version_effect':max(mvcr),'mean_mixed_version_effect':mean(mvcr),
                        'claim_sign_changes':min(claim)<0<max(claim),
                        'classification':classify_query(items)})
    write_csv(ROOT/'per_query_sensitivity.csv',queries)
    topologies = [dict(topology=t,N=None,realized_fraction=None,
                       **summary([r for r in unique if r['topology']==t],'scope','all'))
                  for t in ('P_LOW','P_MID','P_HIGH') if any(r['topology']==t for r in unique)]
    nf_rows=[dict(topology='N_FRACTION',N=n,realized_fraction=f,
                  **summary([r for r in unique if r['N']==n and r['realized_fraction']==f],'scope','all'))
             for n,f in sorted({(r['N'],r['realized_fraction']) for r in unique})]
    write_csv(ROOT/'topology_sensitivity.csv',topologies+nf_rows)
    domains = [dict(domain=d,**summary([r for r in unique if r['domain']==d],'scope','all'))
               for d in sorted({r['domain'] for r in unique})]
    write_csv(ROOT/'domain_sensitivity.csv',domains)
    world, metadata, query_list, _ = _load_frozen(manifest['config'])
    by_query={query.query_id:query for query in query_list}
    evidence=[]
    for source_id, original in historical.items():
        query=by_query[original['query_id']]
        pre=prepare_prestate(world,query,original['transition'])
        event, updates=transition_event(pre,metadata,query,original['transition'])
        doc=_document(event.payload)
        chunks=chunk_document(doc,max_words=160)
        claim=next(c for c in query.required_atomic_claims if c['logical_document_id']==query.logical_document_ids[0])
        revised=updates[query.logical_document_ids[0]]
        span=claim_span(doc,metadata[query.logical_document_ids[0]],revised)['span_text']
        claim_units={c.chunk_id for c in chunks if normalize(span) in normalize(c.text)}
        old_claim_units={c.chunk_id for c in chunks if c.chunk_id.rsplit(':c',1)[1] ==
                         next((u.rsplit(':c',1)[1] for u in claim_units),'-1')}
        for row in [r for r in rows if r['source_pair_id']==source_id]:
            selected=set(row['selected_unit_ids'])
            evidence.append({'source_pair_id':source_id,'seed':row['seed'],
                             'subset_hash':row['subset_hash'],
                             'selected_contains_current_claim':bool(selected & claim_units),
                             'selected_contains_superseded_position':bool(selected & old_claim_units),
                             'claim_failure_difference':row['claim_failure_ticks_difference'],
                             'mixed_version_difference':row['mvcr_ticks_difference']})
    write_csv(ROOT/'evidence_position_analysis.csv',evidence)
    pooled=summary(unique,'scope','unique_new_subsets')
    seed_means=[r['claim_failure_mean_difference'] for r in per_seed if r['transition']=='ALL']
    version_means=[r['mixed_version_mean_difference'] for r in per_seed if r['transition']=='ALL']
    both=sum(r['claim_failure_ticks_difference']<0 and r['mvcr_ticks_difference']>0 for r in unique)
    nonzero=sum(r['claim_failure_ticks_difference']!=0 or r['mvcr_ticks_difference']!=0 for r in unique)
    no_reversal=all(r['claim_failure_mean_difference']<=0 and r['mixed_version_mean_difference']>=0
                    for r in per_seed if r['transition']=='ALL')
    domains_with_both=sum(r['claim_failure_mean_difference']<0 and r['mixed_version_mean_difference']>0
                          for r in domains)
    by_source=defaultdict(list)
    for row in unique:
        by_source[row['source_pair_id']].append(row)
    changed_pair_count=sum(len({r['claim_failure_ticks_difference']<0 for r in items})>1 or
                           len({r['mvcr_ticks_difference']>0 for r in items})>1
                           for items in by_source.values())
    selected_claim=[r for r in evidence if r['selected_contains_current_claim']]
    unselected_claim=[r for r in evidence if not r['selected_contains_current_claim']]
    classification=('NOT_ESTIMABLE' if sum(r['unique_subset_count']>1 for r in uniqueness)==0 else
                    'SEED_SENSITIVE' if not no_reversal or pooled['claim_failure_mean_difference']>=0 or
                    pooled['mixed_version_mean_difference']<=0 else
                    'ROBUST' if both > len(unique)/2 and domains_with_both>=2 and
                    pooled['claim_failure_ci95_high']<0 and pooled['mixed_version_ci95_low']>0 and
                    sum(r['classification']=='sign-sensitive' for r in queries)==0 else
                    'PARTIALLY_ROBUST')
    pooled.update(classification=classification,unique_queries=len({r['query_id'] for r in all_rows}),
                  seed_count=5,additional_seed_count=4,unique_new_subsets=len(unique_rows(rows)),
                  unique_subsets_including_seed_0=sum(r['unique_subset_count'] for r in uniqueness),
                  duplicate_subsets_including_seed_0=sum(r['duplicate_subset_count'] for r in uniqueness),
                  both_direction_count=both,nonzero_effect_count=nonzero,
                  seed_claim_effect_range=[min(seed_means),max(seed_means)],
                  seed_claim_effect_sd=statistics.stdev(seed_means),
                  seed_version_effect_range=[min(version_means),max(version_means)],
                  seed_version_effect_sd=statistics.stdev(version_means),
                  domains_with_both_directions=domains_with_both,
                  qualitative_change_pair_count=changed_pair_count,
                  qualitative_change_pair_proportion=changed_pair_count/len(by_source),
                  selected_claim_count=len(selected_claim),
                  selected_claim_mean_effect=mean(r['claim_failure_difference'] for r in selected_claim),
                  unselected_claim_count=len(unselected_claim),
                  unselected_claim_mean_effect=mean(r['claim_failure_difference'] for r in unselected_claim),
                  query_sign_sensitive_count=sum(r['classification']=='sign-sensitive' for r in queries))
    write_csv(ROOT/'pooled_robustness.csv',[pooled])
    figures(per_seed,unique,queries,topologies,domains)
    summary_md = [f'# PARTIAL selected-unit seed robustness: {classification}', '',
                  'Historical H4 remains unsupported at its primary binary endpoint. This is an exploratory mechanism sensitivity study.', '',
                  f"- Queries: {pooled['unique_queries']}; source pairs: {len(uniqueness)}; additional seed pairs: {len(rows)}.",
                  f"- Unique subsets across seed_0–seed_4: {pooled['unique_subsets_including_seed_0']}; duplicates: {pooled['duplicate_subsets_including_seed_0']}.",
                  f"- Claim-failure effect: {pooled['claim_failure_mean_difference']:.3f} ticks, query-cluster bootstrap CI [{pooled['claim_failure_ci95_low']:.3f}, {pooled['claim_failure_ci95_high']:.3f}].",
                  f"- Mixed-version effect: {pooled['mixed_version_mean_difference']:.3f} ticks, query-cluster bootstrap CI [{pooled['mixed_version_ci95_low']:.3f}, {pooled['mixed_version_ci95_high']:.3f}].",
                  f"- Domains with both directions: {domains_with_both}/{len(domains)}.",
                  f"- Source pairs whose qualitative benefit/cost status changes across subsets: {changed_pair_count}/{len(by_source)}.",
                  f"- Early selection included the current claim in {len(selected_claim)} additional-seed pairs (mean claim effect {mean(r['claim_failure_difference'] for r in selected_claim):.1f}); it did not in {len(unselected_claim)} pairs (mean {mean(r['claim_failure_difference'] for r in unselected_claim):.1f}).",
                  '- Query classification treats no claim rescue as "no effect" on the joint tradeoff even where mixed-version exposure remains.',
                  f"- Fresh-process reproduction: {'PASS' if identical else 'FAIL'}.", '']
    (ROOT/'summary.md').write_text('\n'.join(summary_md))
    return pooled


if __name__=='__main__':
    print(json.dumps(analyze(),sort_keys=True))
