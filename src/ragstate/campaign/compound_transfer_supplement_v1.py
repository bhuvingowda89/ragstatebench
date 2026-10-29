"""Prospective four-arm compound supplement preflight.

Never executes science when frozen CF5 component semantics violate the
same-canonical-transition matched-set requirement.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from .scenario_applicability import _transition_facts, compile_candidate, dry_run
from .state_pilot_manifest import _load_frozen, digest
from .track_b_transfer_v2 import _config, verify_control

ROOT=Path('results/compound-transfer-supplement-v1')
ARMS={'C0':'NONE','C1':'F1','C2':'F2','C12':'F1_F2'}
COMPOUNDS={'CF2':('DELAY_AUTH','STALE_CACHE'),
           'CF4':('DELAY_CONTENT','STALE_INGESTION_REPLAY'),
           'CF5':('CONTENT_UPDATE','AUTH_REVOKE')}


def _write_csv(path,rows):
    rows=list(rows)
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(dict.fromkeys(
            key for row in rows for key in row)))
        writer.writeheader()
        writer.writerows(rows)


def _events(compound,component,transition):
    """Canonical events actually selected by the frozen v4 runtime."""
    if compound!='CF5':return ('PERMISSION_REVOKE' if transition=='REVOKE'
                               else 'DOCUMENT_DELETE',)
    if component=='NONE':return ()
    if component=='F1':return ('DOCUMENT_UPDATE',)
    if component=='F2':return ('PERMISSION_REVOKE',)
    return ('DOCUMENT_UPDATE','PERMISSION_REVOKE')


def preflight():
    ROOT.mkdir(parents=True,exist_ok=True)
    if (ROOT/'observations.jsonl').exists():
        raise ValueError('Supplement observations already exist')
    baseline,transfer,_,provenance=verify_control()
    config=_config(baseline)
    world,metadata,queries,_=_load_frozen(config)
    by_query={q.query_id:q for q in queries}
    source=[s for s in transfer['scenarios'] if s['block']=='TB4']
    assert len(source)==36 and {s['compound'] for s in source}==set(COMPOUNDS)
    scenarios=[];sets=[];audit=[]
    for original in source:
        set_id='CTS1__'+original['scenario_id'].split('__',1)[1]
        ids={}
        for arm,component in ARMS.items():
            scenario={**original,'scenario_id':set_id+'__'+arm,
                      'block':'CTS1','component':component,
                      'campaign_version':'compound-transfer-supplement-v1',
                      'source_tb4_scenario_id':original['scenario_id'],
                      'matched_set_id':set_id,'arm':arm}
            ids[arm]=scenario['scenario_id']
            scenarios.append(scenario)
            query=by_query[scenario['query_id']]
            facts=_transition_facts(world,metadata,query,scenario['transition'])
            compiled=compile_candidate(scenario,query,facts,config)
            dry=dry_run(scenario,query,facts,compiled)
            audit.append({'matched_set_id':set_id,'scenario_id':scenario['scenario_id'],
                'arm':arm,'compound':scenario['compound'],'query_id':query.query_id,
                'architecture':scenario['architecture'],
                'applicable':compiled.applicable,'realizable':dry['realizable'],
                'reasons':';'.join(compiled.reasons),
                'component_operators':','.join(COMPOUNDS[scenario['compound']][i]
                    for i,c in enumerate(('F1','F2'))
                    if component in (c,'F1_F2')),
                'canonical_event_types':','.join(_events(scenario['compound'],component,
                                                         scenario['transition']))})
        sets.append({'matched_set_id':set_id,'compound':original['compound'],
            'query_id':original['query_id'],'architecture':original['architecture'],
            'source_tb4_scenario_id':original['scenario_id'],
            'transition':original['transition'],'severity':original['severity'],
            'principal_id':by_query[original['query_id']].principal_id,
            'tenant_id':by_query[original['query_id']].tenant_id,
            **{arm+'_scenario_id':sid for arm,sid in ids.items()}})
    manifest={'campaign_id':'compound-transfer-supplement-v1',
        'source_transfer_manifest_sha256':provenance['transfer_manifest_sha256'],
        'cohort_v2_sha256':provenance['cohort_v2_sha256'],
        'source_tb4_count':len(source),'matched_set_count':len(sets),
        'scenario_count':len(scenarios),'arm_map':ARMS,
        'compound_components':COMPOUNDS,'scenarios':scenarios,
        'scenario_sha256':digest(scenarios)}
    (ROOT/'manifest.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    _write_csv(ROOT/'matched_sets.csv',sets)
    _write_csv(ROOT/'applicability_audit.csv',audit)
    by_set=defaultdict(list)
    for row in audit:by_set[row['matched_set_id']].append(row)
    complete=len(sets)==36 and len(scenarios)==144 and all(
        {r['arm'] for r in rows}==set(ARMS) for rows in by_set.values())
    baseline_equal=all(len({r['canonical_event_types'] for r in rows})==1
                       for rows in by_set.values())
    valid_applicability=all(r['applicable'] and r['realizable'] for r in audit)
    # Frozen CF5 F1/F2 are semantic events, not replica fault mechanisms.
    fault_isolated=all(s['compound']!='CF5' for s in sets)
    controls_clean=all(next(r for r in rows if r['arm']=='C0')['canonical_event_types']
        ==','.join(_events(rows[0]['compound'],'F1_F2',
                          next(s['transition'] for s in sets if s['matched_set_id']==set_id)))
        for set_id,rows in by_set.items())
    gates={'CT-P1':'PASS' if complete else 'FAIL',
           'CT-P2':'PASS' if baseline_equal else 'FAIL',
           'CT-P3':'PASS' if fault_isolated else 'FAIL',
           'CT-P4':'PASS' if controls_clean else 'FAIL',
           'CT-P5':'NOT_RUN','CT-P6':'NOT_RUN','CT-P7':'NOT_RUN'}
    result={'status':'INVALID','gates':gates,
        'compiler_applicable_and_realizable':sum(r['applicable'] and r['realizable'] for r in audit),
        'scenario_count':len(scenarios),'matched_set_count':len(sets),
        'blocking_reason':'Frozen CF5 NONE/F1/F2/F1_F2 arms have different canonical event sets. '
            'A valid CF5 no-fault control requires update plus revoke in every arm, '
            'which would redefine the frozen CF5 component mechanisms.',
        'science_executed':False}
    (ROOT/'preflight_gate_summary.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    (ROOT/'invalid.json').write_text(json.dumps({'status':'INVALID','stage':'preflight',
        'failed_gates':[k for k,v in gates.items() if v=='FAIL'],
        'reason':result['blocking_reason']},indent=2,sort_keys=True)+'\n')
    (ROOT/'summary.md').write_text('# Compound transfer supplement v1\n\n'
        'Preflight stopped before scientific execution. CF5 cannot satisfy '
        'the requested four-arm baseline equivalence under frozen component '
        'semantics. See `preflight_gate_summary.json` and `applicability_audit.csv`.\n')
    return result


if __name__=='__main__':print(json.dumps(preflight(),indent=2,sort_keys=True))
