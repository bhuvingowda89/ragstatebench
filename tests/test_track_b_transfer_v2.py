"""Frozen transfer design, linkage, and directional classification checks."""

from ragstate.campaign.track_b_transfer_v2 import (
    COHORT, TRANSFER_SHA, classify_direction, classify_masking, d11,
    frozen_inputs, track_a_link,
)
from ragstate.campaign.track_b_completion import classify_amplification
from ragstate.campaign.state_pilot_manifest import digest
import csv
import json
from pathlib import Path


def test_every_frozen_track_b_case_links_to_frozen_track_a_structure():
    baseline, transfer, v4 = frozen_inputs()
    assert baseline['track_b_cohort_sha256'] == COHORT
    assert transfer['sha256'] == TRANSFER_SHA
    a_ids = {scenario['scenario_id'] for scenario in v4['scenarios']}
    assert len(transfer['scenarios']) == 120
    links=[track_a_link(scenario,v4['scenarios']) for scenario in transfer['scenarios']]
    assert all(identifier in a_ids for identifier,_ in links)
    assert {level for _,level in links} == {'query_level','structural_domain'}


def test_direction_classification_preserves_null_and_reverse():
    assert classify_direction(-1, -.5) == 'attenuated but same direction'
    assert classify_direction(-1, -2) == 'preserved'
    assert classify_direction(-1, 0) == 'absent'
    assert classify_direction(-1, 1) == 'reversed'
    assert classify_direction(0, 1) == 'not estimable'


def test_masking_and_amplification_categories_remain_distinct():
    assert classify_masking(True, False, True) == 'track_a_present_track_b_absent'
    assert classify_masking(False, True, True) == 'track_b_only_failure'
    assert classify_masking(False, True, False) == 'baseline_retrieval_failure'
    assert classify_masking(True, True, True) == 'both_present'


def test_d11_requires_majority_and_no_reversal():
    rows = [
        {'effect_id': 'e1', 'track_a_direction': 'positive', 'classification': 'preserved'},
        {'effect_id': 'e2', 'track_a_direction': 'negative', 'classification': 'attenuated but same direction'},
        {'effect_id': 'e3', 'track_a_direction': 'positive', 'classification': 'absent'},
    ]
    assert d11(rows, [], retrieval_interpretable=True)['status'] == 'PASS'
    rows[2]['classification'] = 'reversed'
    assert d11(rows, [], retrieval_interpretable=True)['status'] == 'FAIL'
    rows[2]['classification'] = 'absent'
    assert d11(rows, [], retrieval_interpretable=False)['status'] == 'FAIL'


def test_amplification_only_when_direction_agrees_and_magnitude_increases():
    assert classify_amplification(0.2, 0.4) == 'retrieval-amplified'
    assert classify_amplification(-0.2, -0.4) == 'retrieval-amplified'
    assert classify_amplification(0.2, -0.4) == 'not_applicable'
    assert classify_amplification(0.2, 0.1) == 'not_amplified'


def test_frozen_pilot_provenance_and_fresh_process_record():
    root=Path('results/realistic-transfer-pilot-v2')
    provenance=json.loads((root/'provenance.json').read_text())
    reproduction=json.loads((root/'reproduction.json').read_text())
    links=list(csv.DictReader((root/'trackA_trackB_links.csv').open()))
    assert provenance['transfer_manifest_sha256'] == TRANSFER_SHA
    assert provenance['cohort_v2_sha256'] == COHORT
    assert reproduction['identical'] and reproduction['scenario_count'] == 7
    assert len(links) == 120
    assert sum(r['matching_level']=='query_level' for r in links) == 90
    assert sum(r['matching_level']=='structural_domain' for r in links) == 30
    assert {r['domain'] for r in links} == {
        'engineering','finance','hr','legal','sales','security'}
    projection=[{key: (None if key=='compound' and row[key]=='' else row[key])
                 for key in row if key!='domain'} for row in links]
    assert digest(projection)==provenance['link_sha256']


def test_t5_requires_missing_matched_no_fault_arm():
    root=Path('results/realistic-transfer-pilot-v2')
    transfer=json.loads((root/'transfer_manifest.json').read_text())
    tb4=[s for s in transfer['scenarios'] if s['block']=='TB4']
    assert {s['component'] for s in tb4} == {'F1_F2'}
    effects=list(csv.DictReader((root/'transfer_effects.csv').open()))
    assert all(r['classification']=='not estimable' for r in effects
               if r['effect_class']=='T5')
    assert json.loads((root/'gate_summary.json').read_text())['D11']['agreement_denominator']==16
