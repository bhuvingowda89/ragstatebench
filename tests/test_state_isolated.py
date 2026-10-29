"""Replica-only evidence resolution and independent content-plane behavior."""

from dataclasses import asdict

from ragstate.replicas.scientific import ReplicaContentPlane
from ragstate.retrieval.state_isolated import StateIsolatedEvidenceResolver
from ragstate.transitions.events import Transition
from ragstate.world.ledger import WorldLedger
from ragstate.world.models import DocumentVersion, Principal


def _fixture():
    world = WorldLedger()
    world.add_principal(Principal('p', 'tenant'))
    old = DocumentVersion('doc-v1', 'doc', 'tenant', 1, 0,
                          content='old rule ' * 340)
    world.add_document(old)
    return world, old


def _event(old, kind='DOCUMENT_UPDATE'):
    new = DocumentVersion('doc-v2', 'doc', 'tenant', 2, 10,
                          supersedes=old.document_id,
                          content='new rule ' * 340)
    return Transition('e2', 10, kind, 'doc', asdict(new))


def test_resolver_never_falls_back_to_new_canonical_content():
    world, old = _fixture()
    chunks = ReplicaContentPlane(world, 9, chunked=True)
    parents = ReplicaContentPlane(world, 9, chunked=False)
    from ragstate.transitions.engine import TransitionEngine
    TransitionEngine(world).apply(_event(old))  # canonical changes; replicas do not
    resolved = StateIsolatedEvidenceResolver().resolve('q', 'p', 10, ('doc',), chunks, parents)
    assert resolved.resolved_chunks
    assert {chunk.version for chunk in resolved.resolved_chunks} == {1}
    assert all('old rule' in chunk.text and 'new rule' not in chunk.text
               for chunk in resolved.resolved_chunks)


def test_deleted_content_persists_until_each_replica_receives_event():
    world, old = _fixture()
    chunks = ReplicaContentPlane(world, 9, chunked=True)
    parents = ReplicaContentPlane(world, 9, chunked=False)
    delete = Transition('del', 10, 'DOCUMENT_DELETE', old.document_id,
                        {'logical_document_id': 'doc'})
    from ragstate.transitions.engine import TransitionEngine
    TransitionEngine(world).apply(delete)
    resolver = StateIsolatedEvidenceResolver()
    assert resolver.resolve('q', 'p', 10, ('doc',), chunks, parents).resolved_chunks
    parents.apply(delete)
    assert not resolver.resolve('q', 'p', 10, ('doc',), chunks, parents).resolved_chunks
    assert resolver.resolve('q', 'p', 10, ('doc',), chunks, parents).missing_parent_ids == ('doc',)
    chunks.apply(delete)
    assert not chunks.ordered_chunks()


def test_partial_chunks_and_out_of_order_replay_are_visible_as_replica_state():
    world, old = _fixture()
    chunks = ReplicaContentPlane(world, 9, chunked=True)
    parents = ReplicaContentPlane(world, 9, chunked=False)
    update = _event(old)
    effect = chunks.apply(update, fraction=0.5)
    assert effect == {'applied_units': 2, 'total_units': 5}
    versions = [chunk.version for chunk in chunks.ordered_chunks()]
    assert versions == [2, 2, 1, 1, 1]
    resolved = StateIsolatedEvidenceResolver().resolve('q', 'p', 10, ('doc',), chunks, parents)
    assert {chunk.version for chunk in resolved.resolved_chunks} == {1, 2}
    chunks.apply(update)
    parents.apply(update)
    assert {chunk.version for chunk in chunks.ordered_chunks()} == {2}
    replay = Transition('old-replay', 11, 'DOCUMENT_CREATE', 'doc', asdict(old))
    chunks.apply(replay)
    assert {chunk.version for chunk in chunks.ordered_chunks()} == {1}
    assert parents.documents['doc'].version == 2


def test_parent_and_chunk_replicas_can_disagree_in_both_directions():
    world, old = _fixture()
    chunks = ReplicaContentPlane(world, 9, chunked=True)
    parents = ReplicaContentPlane(world, 9, chunked=False)
    update = _event(old)
    chunks.apply(update)
    assert ({chunk.version for chunk in chunks.ordered_chunks()},
            parents.documents['doc'].version) == ({2}, 1)
    parents.apply(update)
    assert ({chunk.version for chunk in chunks.ordered_chunks()},
            parents.documents['doc'].version) == ({2}, 2)
    chunks = ReplicaContentPlane(world, 9, chunked=True)
    assert ({chunk.version for chunk in chunks.ordered_chunks()},
            parents.documents['doc'].version) == ({1}, 2)
