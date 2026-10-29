"""Frozen generation provenance, pairing, and structured evaluator tests."""
import json

from ragstate.campaign.generation_propagation import ROOT, evaluate, evaluator_self_test, verify
from ragstate.campaign.generation_analysis import paired_effect, wilson


def test_frozen_sample_is_unique_and_balanced():
    manifest=verify()
    rows=manifest['conditions']
    assert len(rows)==120
    assert len({r['query_id'] for r in rows})==120
    assert all(r['clean']['query_id']==r['fault']['query_id'] for r in rows)
    assert all(r['clean']['context_hash']!=r['fault']['context_hash'] for r in rows)
    assert all(len(r['query_id'])>0 for r in rows)


def test_all_domains_and_predeclared_categories():
    rows=verify()['conditions']
    assert {r['category'] for r in rows}=={
        'UPDATE','SUPERSEDE','DELETE','GRANT','REVOKE','CACHE','PARTIAL'}
    assert len({r['domain'] for r in rows})==6
    assert all(len({r['domain'] for r in rows if r['category']==category})==6
               for category in {r['category'] for r in rows})


def test_context_hashes_match_frozen_sources():
    provenance=json.loads((ROOT/'preflight.json').read_text())
    assert provenance['gates']['G-P3']


def test_one_value_oracle_is_valid_for_every_selected_query():
    from ragstate.campaign.state_pilot_manifest import _load_frozen
    config=json.load(open('results/h4-partial-control-pilot-v1/manifest.json'))['config']
    _,_,queries,_=_load_frozen(config)
    by_id={query.query_id:query for query in queries}
    assert all(len(by_id[row['query_id']].required_atomic_claims)==1
               for row in verify()['conditions'])


def test_structured_evaluator_primary_cases():
    assert evaluator_self_test()
    current={'old_value':42,'current_value':49,'abstention_required':False}
    denied={'old_value':42,'current_value':42,'abstention_required':True}
    assert evaluate('VALUE: 49',current,'UPDATE')['answer_defect']==0
    assert evaluate('VALUE: 42',current,'UPDATE')['stale_claim']==1
    assert evaluate('VALUE: 42',denied,'REVOKE')['unsupported_disclosure']==1
    assert evaluate('VALUE: 10',denied,'REVOKE')['unsupported_disclosure']==0
    assert evaluate('VALUE: 10',denied,'REVOKE')['answer_defect']==1
    assert evaluate('INSUFFICIENT_EVIDENCE',denied,'REVOKE')['answer_defect']==0
    assert evaluate('VALUE: 42 and 49',current,'SUPERSEDE')['mixed_version_synthesis']==1


def test_paired_effect_and_exact_mcnemar():
    pairs=[({'answer_defect':0},{'answer_defect':1}) for _ in range(4)]
    result=paired_effect(pairs)
    assert result['risk_difference']==1
    assert result['clean_risk']==0 and result['fault_risk']==1
    assert result['mcnemar_p']==0.125


def test_wilson_interval_contains_observed_rate():
    lo,hi=wilson(3,10)
    assert lo<.3<hi
