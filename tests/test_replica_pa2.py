"""PA2 parent and chunk indices must observe replica state independently."""
from dataclasses import asdict

from ragstate.replicas.scientific import ReplicaContentPlane
from ragstate.retrieval.replica_pa2 import PA2ReplicaRetriever
from ragstate.transitions.engine import TransitionEngine
from ragstate.transitions.events import Transition
from ragstate.world.ledger import WorldLedger
from ragstate.world.models import DocumentVersion, Principal


class TinyEncoder:
    cache_identity = 'tiny-replica-test'

    def encode_documents(self, texts, batch_size):
        return [[float('new' in text), float('old' in text), 1.0] for text in texts]

    def encode_query(self, texts, batch_size):
        return [[1.0, 0.0, 1.0] for _ in texts]


CONFIG = {'embedding_model': 'mock', 'embedding_revision': 'local',
          'hybrid': {'sparse_weight': 1.0, 'dense_weight': 1.0,
                     'rrf_k': 60, 'candidate_k': 100},
          'candidate_k': 30, 'top_k': 10, 'parent_chunk_rrf_k': 60}


def test_pa2_ranks_materialized_replicas_not_new_canonical_content():
    world = WorldLedger()
    world.add_principal(Principal('p', 'tenant'))
    old = DocumentVersion('d-v1', 'd', 'tenant', 1, 0,
                          content='old policy ' * 200)
    world.add_document(old)
    chunk = ReplicaContentPlane(world, 9, chunked=True)
    parent = ReplicaContentPlane(world, 9, chunked=False)
    new = DocumentVersion('d-v2', 'd', 'tenant', 2, 10,
                          content='new policy ' * 200)
    event = Transition('update', 10, 'DOCUMENT_UPDATE', 'd', asdict(new))
    TransitionEngine(world).apply(event)
    retriever = PA2ReplicaRetriever(CONFIG, encoder=TinyEncoder())
    assert {hit.chunk.version for hit in retriever.retrieve('new policy', chunk, parent)} == {1}
    chunk.apply(event)
    assert {hit.chunk.version for hit in retriever.retrieve('new policy', chunk, parent)} == {2}
    assert parent.documents['d'].version == 1
    parent.apply(event)
    assert {hit.chunk.version for hit in retriever.retrieve('new policy', chunk, parent)} == {2}


def test_pa2_handles_chunk_parent_delete_disagreement_without_oracle_fallback():
    world = WorldLedger()
    world.add_principal(Principal('p', 'tenant'))
    old = DocumentVersion('d-v1', 'd', 'tenant', 1, 0,
                          content='old policy ' * 200)
    world.add_document(old)
    chunk = ReplicaContentPlane(world, 9, chunked=True)
    parent = ReplicaContentPlane(world, 9, chunked=False)
    delete = Transition('delete', 10, 'DOCUMENT_DELETE', old.document_id,
                        {'logical_document_id': 'd'})
    parent.apply(delete)
    retriever = PA2ReplicaRetriever(CONFIG, encoder=TinyEncoder())
    assert retriever.retrieve('old policy', chunk, parent) == []
    assert chunk.ordered_chunks()  # stale chunks still exist independently


def test_pa2_all_four_parent_chunk_version_combinations():
    world = WorldLedger()
    world.add_principal(Principal('p', 'tenant'))
    old = DocumentVersion('d-v1', 'd', 'tenant', 1, 0, content='old policy ' * 200)
    world.add_document(old)
    new = DocumentVersion('d-v2', 'd', 'tenant', 2, 10, content='new policy ' * 200)
    event = Transition('update', 10, 'DOCUMENT_UPDATE', 'd', asdict(new))
    chunks = ReplicaContentPlane(world, 9, chunked=True)
    parents = ReplicaContentPlane(world, 9, chunked=False)
    TransitionEngine(world).apply(event)
    retriever = PA2ReplicaRetriever(CONFIG, encoder=TinyEncoder())

    def check(parent_version, chunk_version):
        assert parents.ordered_parent_chunks()[0].version == parent_version
        assert {hit.chunk.version for hit in retriever.retrieve('new policy', chunks, parents)} == {chunk_version}

    check(1, 1)  # parent stale / chunk stale
    chunks.apply(event)
    check(1, 2)  # parent stale / chunk current
    # Rebuild a stale chunk plane from the original version, independently.
    stale_world = WorldLedger()
    stale_world.add_principal(Principal('p', 'tenant'))
    stale_world.add_document(old)
    stale_chunks = ReplicaContentPlane(stale_world, 9, chunked=True)
    parents.apply(event)
    assert parents.ordered_parent_chunks()[0].version == 2
    assert {hit.chunk.version for hit in retriever.retrieve('new policy', stale_chunks, parents)} == {1}
    check(2, 2)  # parent current / chunk current
