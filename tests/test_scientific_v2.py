from collections import Counter
from dataclasses import replace

from ragstate.world.scientific_v2 import (DOMAINS, FAMILY_COUNTS, LOGICAL_TIME,
    corpus_hash, generate_v2_queries, generate_v2_world)
from ragstate.world.v2_validation import leakage_rows, repair_lexical_leakage, validate_gold


def fixture():
    world = generate_v2_world(101)
    queries = generate_v2_queries(world, 202)
    return world, queries


def test_v2_repeated_generation_has_identical_hash_and_allocations():
    first_world, first_queries = fixture()
    second_world, second_queries = fixture()
    assert corpus_hash(first_world, first_queries) == corpus_hash(second_world, second_queries)
    assert len(first_queries) == 216
    assert len({d.logical_document_id for d in first_world.documents.values()}) == 360
    assert Counter(q.domain for q in first_queries) == {d.name: 36 for d in DOMAINS}
    assert Counter(q.family for q in first_queries) == {k: v * 6 for k, v in FAMILY_COUNTS.items()}


def test_v2_gold_is_current_authorized_and_multi_claims_are_complete():
    world, queries = fixture()
    assert validate_gold(world, queries) == []
    multi = [q for q in queries if q.family == 'multi_document']
    assert len(multi) == 30
    for q in multi:
        assert len(q.required_document_ids) == 2
        assert q.precedence_matters
        assert {claim['precedence'] for claim in q.required_atomic_claims} == {1, 2}
        for claim in q.required_atomic_claims:
            assert f"{claim['value']} {claim['unit']}" in world.documents[claim['document_id']].content
    assert any(q.version_sensitive for q in queries)
    for q in queries:
        assert all(world.document_at(lid, LOGICAL_TIME) for lid in q.logical_document_ids)
        assert all(world.allowed(q.principal_id, lid, LOGICAL_TIME)
                   for lid in q.logical_document_ids if world.document_at(lid, LOGICAL_TIME).document_id in q.required_document_ids)


def test_v2_negative_queries_are_denied_without_exposing_access_status_in_text():
    world, queries = fixture()
    negatives = [q for q in queries if q.family == 'negative']
    assert len(negatives) == 12
    assert all(not q.answerable and not q.required_document_ids for q in negatives)
    assert all(not world.allowed(q.principal_id, q.logical_document_ids[0], LOGICAL_TIME) for q in negatives)
    assert all('authorized' not in q.text.lower() and 'confidential' not in q.text.lower() for q in negatives)


def test_v2_exception_access_and_current_version_resolution():
    world, queries = fixture()
    multi = next(q for q in queries if q.family == 'multi_document')
    exception = next(lid for lid in multi.logical_document_ids if lid.endswith('EXC'))
    assert world.allowed(multi.principal_id, exception, LOGICAL_TIME)
    generalist = multi.principal_id.replace('specialist', 'generalist')
    assert not world.allowed(generalist, exception, LOGICAL_TIME)
    versioned = next(q for q in queries if q.version_sensitive and q.answerable)
    lid = versioned.logical_document_ids[0]
    old = world.document_at(lid, 4)
    current = world.document_at(lid, LOGICAL_TIME)
    assert old.version == 1 and current.version == 2
    assert current.supersedes == old.document_id
    assert current.document_id in versioned.required_document_ids


def test_v2_lexical_detector_flags_accidental_identifier_and_repairs_it():
    world, queries = fixture()
    natural = next(q for q in queries if q.family == 'natural_descriptive')
    contaminated = replace(natural, text=natural.text + ' Record ' + natural.logical_document_ids[0])
    flagged = leakage_rows(world, [contaminated])[0]
    assert flagged['explicit_identifier'] == 1
    assert flagged['construction_warning'] == 1
    repaired, rows = repair_lexical_leakage(world, [contaminated])
    assert rows[0]['construction_warning'] == 0
    assert repaired[0].text != contaminated.text


def test_v2_lexical_detector_flags_nonidentifier_gold_only_token():
    world, queries = fixture()
    natural = next(q for q in queries if q.family == 'natural_descriptive')
    document_id = natural.required_document_ids[0]
    world.documents[document_id] = replace(world.documents[document_id],
                                           content=world.documents[document_id].content + ' goldonlyword')
    contaminated = replace(natural, text=natural.text + ' goldonlyword')
    row = leakage_rows(world, [contaminated])[0]
    assert row['unique_gold_token_leak'] == 1
    assert 'goldonlyword' in row['unique_gold_tokens']
    assert row['construction_warning'] == 1


def test_v2_validator_rejects_ambiguous_or_incorrect_gold():
    world, queries = fixture()
    multi = next(q for q in queries if q.family == 'multi_document')
    broken = replace(multi, required_document_ids=multi.required_document_ids[:1])
    assert any('gold' in error or 'multi-document' in error for error in validate_gold(world, [broken]))
