"""Deterministic clean-only inclusion criteria for the transition cohort."""

import json
from pathlib import Path

from ragstate.campaign.evidence_spans import mapping_for_queries
from ragstate.campaign.transition_cohort import (
    assess_query, build, cohort_hash, generate, verify,
)
from ragstate.retrieval.chunking import chunk_document
from ragstate.world.ledger import WorldLedger
from ragstate.world.models import DocumentVersion, Principal
from ragstate.world.scientific_v3 import V3Query


CONFIG = Path('configs/transition-cohort-v1.json')


def _fixture():
    world = WorldLedger()
    world.add_principal(Principal('p', 'tenant_a'))
    sentence_a = 'For domestic employees, the approved cap for travel meals is 42 dollars.'
    sentence_b = 'For domestic employees, the approved exception for travel meals is 47 dollars.'
    for logical_id, sentence in (('a', sentence_a), ('b', sentence_b)):
        world.add_document(DocumentVersion(f'{logical_id}-v1', logical_id, 'tenant_a',
                                           1, 0, content=sentence + ' Background notes.'))
    world.add_document(DocumentVersion('blocked-v1', 'blocked', 'tenant_b', 1, 0,
                                       content='Restricted target answer.'))
    chunks = {chunk.chunk_id: chunk for doc in world.documents.values()
              for chunk in chunk_document(doc, max_words=160)}
    return world, chunks


def _query(query_id='q', docs=('a',), answerable=True):
    claims = tuple({'document_id': f'{doc}-v1', 'value': 42 if doc == 'a' else 47,
                    'unit': 'dollars', 'claim_type': 'default_rule'} for doc in docs)
    return V3Query(query_id, 'What is the approved travel amount?', 'hr',
                   'multi_document' if len(docs) > 1 else 'natural_descriptive',
                   'p', 'tenant_a', 0, docs,
                   tuple(f'{doc}-v1' for doc in docs) if answerable else (),
                   claims if answerable else (), answerable, False, False, False,
                   'UPDATE', f'{docs[0]}-v1', 0, None, 'independent_sources')


def _row(chunk_ids, scrs):
    return {'context_chunk_ids': json.dumps(chunk_ids),
            'candidate_chunk_ids': json.dumps(chunk_ids), 'scrs': str(scrs)}


def _claims(world, query, chunks):
    metadata = {doc: {'axis_one': 'domestic', 'axis_two': 'employees'}
                for doc in query.logical_document_ids}
    return mapping_for_queries(world, metadata, [query], list(chunks.values()))


def test_answerable_requires_scrs_documents_and_complete_claim_spans():
    world, chunks = _fixture()
    query = _query()
    claim_rows = _claims(world, query, chunks)
    complete = assess_query(query, _row(['a-v1:c0'], 1), world, chunks, claim_rows)
    assert complete['eligible'] == 1
    assert complete['scrs'] == complete['document_complete'] == complete['claim_complete'] == 1
    incomplete_claim = assess_query(query, _row(['a-v1:c0'], 1), world, chunks,
                                    [{**claim_rows[0], 'containing_chunk_ids': []}])
    assert incomplete_claim['eligible'] == 0
    assert incomplete_claim['document_complete'] == 1
    assert incomplete_claim['ineligibility_reasons'] == 'missing_claim_bearing_chunk'


def test_multi_document_rule_never_relaxes_missing_evidence():
    world, chunks = _fixture()
    query = _query(docs=('a', 'b'))
    claims = _claims(world, query, chunks)
    both = assess_query(query, _row(['a-v1:c0', 'b-v1:c0'], 1), world, chunks, claims)
    one = assess_query(query, _row(['a-v1:c0'], 0), world, chunks, claims)
    assert both['eligible'] == 1
    assert one['eligible'] == 0
    assert 'missing_document' in one['ineligibility_reasons']
    assert 'missing_claim_bearing_chunk' in one['ineligibility_reasons']


def test_negative_query_has_separate_absence_and_authorization_rule():
    world, chunks = _fixture()
    query = _query('negative', docs=('blocked',), answerable=False)
    empty = assess_query(query, _row([], 1), world, chunks, [])
    exposed = assess_query(query, _row(['blocked-v1:c0'], 0), world, chunks, [])
    assert empty['eligible'] == 1
    assert empty['negative_absence_success'] == 1
    assert empty['claim_complete'] is None
    assert exposed['eligible'] == 0
    assert 'target_evidence_in_context' in exposed['ineligibility_reasons']
    assert 'relevant_cross_tenant_context' in exposed['ineligibility_reasons']


def test_cohort_hash_sorts_ids_and_binds_rules_corpus_and_architecture():
    config = json.loads(CONFIG.read_text())
    first, payload = cohort_hash(['q2', 'q1'], ['n2', 'n1'], config, 'corpus')
    assert first == cohort_hash(['q1', 'q2'], ['n1', 'n2'], config, 'corpus')[0]
    assert payload['eligible_answerable_query_ids'] == ['q1', 'q2']
    reordered = {**config, 'answerable_rules': list(reversed(config['answerable_rules']))}
    assert first == cohort_hash(['q1', 'q2'], ['n1', 'n2'], reordered, 'corpus')[0]
    assert first != cohort_hash(['q1', 'q2'], ['n1', 'n2'], config, 'new-corpus')[0]
    changed = {**config, 'answerable_rules': config['answerable_rules'] + ['new-rule']}
    assert first != cohort_hash(['q1', 'q2'], ['n1', 'n2'], changed, 'corpus')[0]


def test_frozen_corpus_eligibility_has_no_manual_exclusions_and_rebuilds_identically():
    config = json.loads(CONFIG.read_text())
    first = build(config)
    second = build(config)
    records = first[0]
    assert first[4:7] == second[4:7]
    assert len(records) == 216
    assert all(row['eligible'] == int(not row['ineligibility_reasons']) for row in records)
    assert {row['query_id'] for row in records if row['eligible'] and row['answerable']} == set(first[4])
    assert {row['query_id'] for row in records if row['eligible'] and not row['answerable']} == set(first[5])


def test_generated_analysis_policy_and_separate_process_verification(tmp_path):
    output = tmp_path / 'cohort'
    manifest = generate(CONFIG, output)
    policy = json.loads((output / 'analysis_policy.json').read_text())
    assert policy['primary_transition_analysis']['retrieval_architecture_id'] == \
           'RA1-hybrid-BM25-E5-no-reranker-k10'
    assert policy['primary_transition_analysis']['fault_outcomes_used_for_inclusion'] is False
    assert policy['secondary_generalization_analysis']['included_in_primary_causal_factorial'] is False
    assert policy['historical_phase7d_gate_A4']['status'] == 'FAIL'
    assert verify(CONFIG, output)['cohort_sha256'] == manifest['cohort_sha256']
