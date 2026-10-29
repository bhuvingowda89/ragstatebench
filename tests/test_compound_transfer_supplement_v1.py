"""The frozen CF5 semantics must not be silently treated as matched faults."""
import csv
import json
from collections import defaultdict
from pathlib import Path

from ragstate.campaign.compound_transfer_supplement_v1 import _events


ROOT=Path('results/compound-transfer-supplement-v1')


def test_four_arm_manifest_reuses_exact_tb4_population():
    manifest=json.loads((ROOT/'manifest.json').read_text())
    sets=list(csv.DictReader((ROOT/'matched_sets.csv').open()))
    source=json.loads(Path('results/realistic-transfer-pilot-v2/transfer_manifest.json').read_text())
    tb4={s['scenario_id'] for s in source['scenarios'] if s['block']=='TB4'}
    assert len(sets)==len(tb4)==36
    assert len(manifest['scenarios'])==144
    assert {s['source_tb4_scenario_id'] for s in sets}==tb4
    assert {s['compound'] for s in sets}=={'CF2','CF4','CF5'}
    assert {s['architecture'] for s in sets}=={'A0','A5'}
    assert all(all(s[arm+'_scenario_id'] for arm in ('C0','C1','C2','C12')) for s in sets)


def test_cf5_control_cannot_preserve_canonical_events_under_frozen_runtime():
    assert _events('CF5','NONE','UPDATE')==()
    assert _events('CF5','F1','UPDATE')==('DOCUMENT_UPDATE',)
    assert _events('CF5','F2','UPDATE')==('PERMISSION_REVOKE',)
    assert _events('CF5','F1_F2','UPDATE')==('DOCUMENT_UPDATE','PERMISSION_REVOKE')
    assert _events('CF2','NONE','REVOKE')==_events('CF2','F1_F2','REVOKE')
    assert _events('CF4','NONE','DELETE')==_events('CF4','F1_F2','DELETE')


def test_preflight_stops_before_observations():
    audit=list(csv.DictReader((ROOT/'applicability_audit.csv').open()))
    by_set=defaultdict(list)
    for row in audit:by_set[row['matched_set_id']].append(row)
    assert len(by_set)==36 and all(len(rows)==4 for rows in by_set.values())
    assert sum(len({r['canonical_event_types'] for r in rows})>1
               for rows in by_set.values())==12
    gate=json.loads((ROOT/'preflight_gate_summary.json').read_text())
    assert gate['gates']['CT-P1']=='PASS'
    assert gate['gates']['CT-P2']=='FAIL'
    assert gate['science_executed'] is False
    assert not (ROOT/'observations.jsonl').exists()
