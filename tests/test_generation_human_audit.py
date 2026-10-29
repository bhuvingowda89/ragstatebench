"""Blinding and agreement safeguards for independent human validation."""
import csv
import json
import shutil

import pytest

from ragstate.campaign.generation_human_audit import (
    BLIND, BLINDING, COMPLETED, KEY, ROOT, SOURCE, _kappa, _source_pairs,
    finalize, prepare,
)


def test_frozen_selection_and_blinded_answer_count():
    assert len(_source_pairs()) == 15
    assert prepare()['answers'] == 36
    blinded = list(csv.DictReader(BLIND.open()))
    assert len(blinded) == 36
    assert {row['pair_id'] for row in blinded} == {row['pair_id'] for row in _source_pairs()}


def test_reviewer_packet_hides_evaluator_and_arm_labels():
    rows = list(csv.DictReader(BLIND.open()))
    forbidden = {'automated_clean_defect','automated_fault_defect',
                 'automated_disclosure','automated_stale_claim','answer_defect',
                 'stale_claim','unsupported_disclosure','source_arm','architecture'}
    assert not forbidden.intersection(rows[0])
    assert all(row['reviewer_status'] == 'PENDING_HUMAN' for row in rows)
    assert all(row['human_correctness'] == '' for row in rows)


def test_revoke_exposed_and_protected_answers_remain_in_same_frozen_pairs():
    mapping = json.loads(KEY.read_text())['mapping']
    revoke = {row['pair_id'] for row in _source_pairs() if row['category'] == 'REVOKE'}
    assert len(revoke) == 2
    for pair_id in revoke:
        assert {row['source_arm'] for row in mapping if row['pair_id'] == pair_id} == {
            'clean','fault','protected_A4','protected_A5'}


def test_pending_human_labels_cannot_be_finalized(tmp_path, monkeypatch):
    import ragstate.campaign.generation_human_audit as audit
    rows=list(csv.DictReader(COMPLETED.open()))
    rows[0]['reviewer_status']='PENDING_HUMAN'
    pending=tmp_path/'pending.csv'
    with pending.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    monkeypatch.setattr(audit,'COMPLETED',pending)
    with pytest.raises(ValueError, match='pending'):
        finalize()


def test_kappa_balanced_and_single_class_cases():
    assert _kappa(['YES','NO'],['YES','NO']) == 1.0
    assert _kappa(['YES','YES'],['YES','YES']) is None


def test_blinding_record_and_gate_track_human_review():
    blind = json.loads(BLINDING.read_text())
    gate = json.loads((ROOT/'gate_summary.json').read_text())
    assert blind['status'] in ('PENDING_HUMAN','COMPLETED')
    if blind['status']=='PENDING_HUMAN':
        assert blind['reviewer_had_aggregate_results'] == 'UNKNOWN_PENDING_REVIEWER_ATTESTATION'
    assert blind['automated_labels_shown_before_judgment'] is False
    assert gate['gates']['G-P8'] in ('PENDING','PASS','FAIL')
    assert all(gate['gates'][f'G-P{i}'] is True for i in range(1,8))


def test_agreement_pipeline_on_isolated_synthetic_labels(tmp_path, monkeypatch):
    import ragstate.campaign.generation_human_audit as audit
    files=['human_audit.csv','human_review_blinded.csv','human_review_key.json',
           'human_audit_completed.csv','human_audit_blinding.json','observations.jsonl',
           'gate_summary.json','summary.md','manifest.json']
    for name in files:
        shutil.copy2(ROOT/name,tmp_path/name)
    monkeypatch.setattr(audit,'ROOT',tmp_path)
    for key,name in [('SOURCE','human_audit.csv'),('BLIND','human_review_blinded.csv'),
                     ('KEY','human_review_key.json'),('COMPLETED','human_audit_completed.csv'),
                     ('BLINDING','human_audit_blinding.json'),
                     ('AGREEMENT','human_audit_agreement.csv'),
                     ('DISAGREEMENTS','human_audit_disagreements.csv'),
                     ('SUMMARY','human_audit_summary.json')]:
        monkeypatch.setattr(audit,key,tmp_path/name)
    mapping={row['answer_id']:row for row in json.loads((tmp_path/'human_review_key.json').read_text())['mapping']}
    observations={(row['pair_id'],row['arm']):row for row in
                  (json.loads(line) for line in (tmp_path/'observations.jsonl').open())}
    rows=list(csv.DictReader((tmp_path/'human_audit_completed.csv').open()))
    for row in rows:
        key=mapping[row['answer_id']]
        labels=audit._automated(observations[(key['pair_id'],key['source_arm'])])
        row.update(labels)
        row['reviewer_status']='COMPLETED'
    with (tmp_path/'human_audit_completed.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    blind=json.loads((tmp_path/'human_audit_blinding.json').read_text())
    blind['reviewer_id']='SYNTHETIC_TEST_ONLY'
    blind['reviewer_had_aggregate_results']='NO'
    (tmp_path/'human_audit_blinding.json').write_text(json.dumps(blind))
    result=audit.finalize()
    assert result['G-P8']=='PASS'
    assert result['generation_status']=='GENERATION_VALIDATED'
    assert result['overall_exact']['agree']==result['overall_exact']['total']
    assert result['revoke_disclosure']['agree']==8
