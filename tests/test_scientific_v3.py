from collections import Counter
from dataclasses import replace

from ragstate.retrieval.bm25 import BM25Retriever
from ragstate.retrieval.indexing import build_chunks_from_ledger
from ragstate.world.scientific_v3 import (ARCHETYPES, VERSION_CHANGES, corpus_hash,
    generate_v3_queries, generate_v3_world)
from ragstate.world.v3_validation import cramers_v, leakage_rows, validate_gold


def fixture():
    world, metadata = generate_v3_world(101)
    queries = generate_v3_queries(world, 202)
    return world, metadata, queries


def test_v3_deterministic_generation_and_archetype_diversity():
    world, metadata, queries = fixture()
    again_world, again_meta, again_queries = fixture()
    assert corpus_hash(world, metadata, queries) == corpus_hash(again_world, again_meta, again_queries)
    assert len(metadata) == 390 and len(world.documents) == 450 and len(queries) == 216
    counts = Counter(row['archetype'] for row in metadata.values())
    assert set(counts) == set(ARCHETYPES)
    assert max(counts.values()) / len(metadata) < 0.40


def test_v3_archetypes_have_distinct_body_structures():
    world, metadata, _ = fixture()
    markers = {
        'policy_memo': 'Purpose\n', 'wiki_page': '## Overview',
        'engineering_runbook': 'Preconditions / step',
        'customer_support': 'Customer tier:', 'legal_notice': 'Clause 1',
        'internal_email': 'Hello team,', 'incident_ticket': 'Summary entry',
        'change_note': '- Change summary:',
    }
    for archetype, marker in markers.items():
        examples = [world.document_at(lid, 9).content for lid, row in metadata.items()
                    if row['archetype'] == archetype]
        assert examples and all(marker in text for text in examples)


def test_v3_length_buckets_and_multichunk_cover_partial_propagation():
    world, metadata, _ = fixture()
    chunks = build_chunks_from_ledger(world, 9)
    by_logical = Counter(chunk.logical_document_id for chunk in chunks)
    words = sorted(len(world.document_at(lid, 9).content.split()) for lid in metadata)
    assert words[len(words) // 2] >= 150
    assert words[int(.9 * len(words))] >= 300
    assert sum(n > 1 for n in by_logical.values()) / len(metadata) >= .40
    assert sum(row['version_sensitive'] and by_logical[lid] > 1 for lid, row in metadata.items()) >= 30
    assert sum(row['kind'] == 'EXC' and by_logical[lid] > 1 for lid, row in metadata.items()) >= 100


def test_v3_tenant_scope_is_not_deterministically_linked():
    _, metadata, _ = fixture()
    for axis in ('axis_one', 'axis_two'):
        result = cramers_v(list(metadata.values()), axis)
        assert result['cramers_v'] < .5
        assert all(sum(scopes.values()) > 0 for scopes in result['contingency'].values())


def test_v3_authority_and_version_changes_vary_independently():
    world, metadata, _ = fixture()
    assert {m['version_change_type'] for m in metadata.values() if m['version_sensitive']} == set(VERSION_CHANGES)
    assert {m['authority_level'] for m in metadata.values()} >= {1, 2, 3, 4}
    versioned = [lid for lid, m in metadata.items() if m['version_sensitive']]
    assert any(metadata[lid]['authority_level'] == 4 for lid in versioned)
    assert any(metadata[lid]['authority_level'] != 4 for lid in versioned)
    for lid in versioned:
        old, current = world.document_at(lid, 4), world.document_at(lid, 9)
        assert old.version == 1 and current.version == 2
        assert current.supersedes == old.document_id
        assert old.content != current.content


def test_v3_multi_document_gold_has_independent_claims_and_sources():
    world, metadata, queries = fixture()
    assert validate_gold(world, metadata, queries) == []
    multi = [q for q in queries if q.family == 'multi_document']
    assert len(multi) == 30
    assert len({q.evidence_relation for q in multi}) == 6
    for q in multi:
        assert len(q.required_document_ids) == 2 and q.precedence_matters
        assert {claim['precedence'] for claim in q.required_atomic_claims} == {1, 2}
        assert len({claim['logical_document_id'] for claim in q.required_atomic_claims}) == 2
        assert {metadata[lid]['kind'] for lid in q.logical_document_ids} == {'BASE', 'EXC'}
    broken = replace(multi[0], required_document_ids=multi[0].required_document_ids[:1])
    assert validate_gold(world, metadata, [broken])


def test_v3_lexical_leakage_and_negative_gold_controls():
    world, metadata, queries = fixture()
    rows = leakage_rows(world, metadata, queries)
    answerable = [r for r in rows if r['answerable']]
    assert sum(r['gold_identifier_overlap'] for r in answerable) <= .15 * len(answerable)
    assert all(not r['unique_gold_token_leak'] for r in answerable if r['family'] != 'identifier_bearing')
    assert all(not r['construction_warning'] for r in rows)
    assert all(not q.required_document_ids for q in queries if q.family == 'negative')
    natural = next(q for q in queries if q.family == 'natural_descriptive')
    contaminated = replace(natural, text=natural.text + ' ' + natural.logical_document_ids[0])
    assert leakage_rows(world, metadata, [contaminated])[0]['construction_warning']


def test_v3_clean_candidate_ranking_is_deterministic_without_model_download():
    world, _, queries = fixture()
    chunks = build_chunks_from_ledger(world, 9)
    q = queries[0].text
    first = BM25Retriever(chunks).retrieve(q, 10)
    second = BM25Retriever(chunks).retrieve(q, 10)
    assert [h.chunk.chunk_id for h in first] == [h.chunk.chunk_id for h in second]
