"""Offline checks for the four frozen parent-aware policies."""

from ragstate.campaign.phase7g import _outcome, _parent_index, _policy_hits
from ragstate.retrieval.models import Chunk, RetrievalHit
from ragstate.retrieval.parent_aware import diversity_cap, parent_chunk_rrf
from ragstate.world.ledger import WorldLedger
from ragstate.world.models import DocumentVersion, Principal


def _hits():
    parents = ['a', 'a', 'a', 'b', 'c', 'd']
    return [RetrievalHit(Chunk(f'{parent}:{i}', f'{parent}-v1', parent, 1,
                               'tenant', 0, f'passage {parent} {i}'),
                         1 / (i + 1), (('bm25', i + 1, 1.0),))
            for i, parent in enumerate(parents)]


def test_diversity_cap_preserves_order_and_parent_provenance():
    hits = _hits()
    selected = diversity_cap(hits, top_k=5, max_per_logical_document=2)
    assert [hit.chunk.chunk_id for hit in selected] == ['a:0', 'a:1', 'b:3', 'c:4', 'd:5']
    assert selected[0] is hits[0] and selected[2] is hits[3]
    assert max(sum(hit.chunk.logical_document_id == parent for hit in selected)
               for parent in ('a', 'b', 'c', 'd')) == 2
    assert diversity_cap(hits, top_k=0) == []


def test_parent_chunk_rrf_promotes_independent_parent_rank_without_changing_provenance():
    hits = _hits()
    parent_ranks = {'a': 20, 'b': 1, 'c': 2, 'd': 3}
    fused = parent_chunk_rrf(hits, parent_ranks, rrf_k=60)
    assert fused[0].chunk.logical_document_id == 'b'
    assert {hit.chunk.chunk_id for hit in fused} == {hit.chunk.chunk_id for hit in hits}
    assert all(hit.components == next(original.components for original in hits
                                      if original.chunk == hit.chunk) for hit in fused)
    assert [hit.chunk.chunk_id for hit in fused] == [
        hit.chunk.chunk_id for hit in parent_chunk_rrf(hits, parent_ranks, 60)]


def test_rrf_tie_breaks_by_original_chunk_rank():
    hits = _hits()[:2]
    # Both chunks have the same parent rank; their chunk ranks decide the order.
    ranked = parent_chunk_rrf(hits, {'a': 1}, rrf_k=60)
    assert [hit.chunk.chunk_id for hit in ranked] == ['a:0', 'a:1']


def test_policy_matrix_applies_same_rule_regardless_of_query_type():
    hits = _hits()
    config = {'top_k': 5, 'hybrid': {'rrf_k': 60}}
    parents = {'a': 20, 'b': 1, 'c': 2, 'd': 3}
    selected, _ = _policy_hits(hits, parents, config)
    assert [hit.chunk.chunk_id for hit in selected['PA0']] == \
           [hit.chunk.chunk_id for hit in hits[:5]]
    assert [hit.chunk.chunk_id for hit in selected['PA1']] == \
           [hit.chunk.chunk_id for hit in diversity_cap(hits, 5, 2)]
    assert selected['PA2'] == parent_chunk_rrf(hits, parents, 60)[:5]
    assert selected['PA3'] == diversity_cap(parent_chunk_rrf(hits, parents, 60), 5, 2)
    assert _outcome(1, 0) == 'regression'
    assert _outcome(0, 1) == 'rescue'


class ToyEncoder:
    def encode(self, texts, batch_size):
        return self.encode_documents(texts, batch_size)

    def encode_documents(self, texts, batch_size):
        return [self._vector(text) for text in texts]

    def encode_query(self, texts, batch_size):
        return [self._vector(text) for text in texts]

    @staticmethod
    def _vector(text):
        return [float(text.lower().count('alpha') + 1),
                float(text.lower().count('beta') + 1)]


def test_parent_index_uses_full_document_text_and_no_gold_fields():
    world = WorldLedger()
    world.add_principal(Principal('p', 'tenant'))
    world.add_document(DocumentVersion('a-v1', 'a', 'tenant', 1, 0,
                                       content='alpha ' * 170))
    world.add_document(DocumentVersion('b-v1', 'b', 'tenant', 1, 0,
                                       content='beta ' * 170))
    config = {'embedding_model': 'toy', 'embedding_revision': 'fixed',
              'hybrid': {'sparse_weight': 1.0, 'dense_weight': 1.0,
                         'rrf_k': 60, 'candidate_k': 100}}
    docs, chunks, index, _, vectors, _ = _parent_index(world, config, ToyEncoder())
    assert len(docs) == 2 and len(chunks) == 4 and vectors == 2
    assert len(index.sparse.chunks[0].text.split()) == 170
    ranked = index.retrieve('alpha', 2)
    assert ranked[0].chunk.logical_document_id == 'a'
    assert [hit.chunk.chunk_id for hit in ranked] == \
           [hit.chunk.chunk_id for hit in index.retrieve('alpha', 2)]


def test_selected_policy_cohort_v2_regenerates_from_unchanged_rules():
    import csv
    import json
    from collections import defaultdict
    from pathlib import Path

    from ragstate.campaign.evidence_spans import mapping_for_queries
    from ragstate.campaign.phase7c import _load
    from ragstate.campaign.transition_cohort import assess_query, cohort_hash
    from ragstate.retrieval.architecture import chunks_for_architecture

    root = Path('data/transition-cohort-v2')
    manifest = json.loads((root / 'cohort_manifest.json').read_text())
    config = json.loads((root / 'eligibility_config.json').read_text())
    prior = json.loads(Path('configs/transition-cohort-v1.json').read_text())
    assert config['answerable_rules'] == prior['answerable_rules']
    assert config['negative_rules'] == prior['negative_rules']
    assert manifest['selected_policy'] == 'PA2'
    data = Path('data/generated-v3')
    world, metadata, queries = _load(data, json.loads((data / 'manifest.json').read_text()))
    _, chunks = chunks_for_architecture(world, 9, 'RA1')
    by_chunk = {chunk.chunk_id: chunk for chunk in chunks}
    claims = defaultdict(list)
    for claim in mapping_for_queries(world, metadata, queries, chunks):
        claims[claim['query_id']].append(claim)
    with Path('results/phase7g-parent-aware-v1/per_query_results.csv').open(newline='') as stream:
        rows = {row['query_id']: row for row in csv.DictReader(stream) if row['policy'] == 'PA2'}
    records = [assess_query(query, {'context_chunk_ids': rows[query.query_id]['context_chunk_ids'],
                                    'candidate_chunk_ids': rows[query.query_id]['selected_chunk_ids'],
                                    'scrs': rows[query.query_id]['scrs']}, world, by_chunk,
                            claims[query.query_id]) for query in queries]
    answerable = sorted(row['query_id'] for row in records if row['answerable'] and row['eligible'])
    negative = sorted(row['query_id'] for row in records if not row['answerable'] and row['eligible'])
    digest, _ = cohort_hash(answerable, negative, config, manifest['corpus_sha256'])
    assert digest == manifest['cohort_sha256']
    assert answerable == json.loads((root / 'eligible_answerable_query_ids.json').read_text())
    assert negative == json.loads((root / 'eligible_negative_query_ids.json').read_text())
