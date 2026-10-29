"""Resolve oracle-specified logical IDs exclusively through replica-visible state."""
from __future__ import annotations

from dataclasses import dataclass

from .models import Chunk, RetrievalHit


@dataclass(frozen=True)
class StateIsolatedResolution:
    query_id: str
    principal_id: str
    logical_time: int
    oracle_logical_ids: tuple[str, ...]
    resolved_chunks: tuple[Chunk, ...]
    missing_parent_ids: tuple[str, ...]
    missing_chunk_ids: tuple[str, ...]

    def hits(self, top_k: int = 10) -> list[RetrievalHit]:
        return [RetrievalHit(chunk, 0.0, (('replica_source_order', rank, 0.0),))
                for rank, chunk in enumerate(self.resolved_chunks[:top_k], 1)]


class StateIsolatedEvidenceResolver:
    """Oracle supplies IDs only; content, versions and ordering come from replicas."""

    ordering_policy = 'replica_chunk_index_order'

    def resolve(self, query_id: str, principal_id: str, logical_time: int,
                logical_evidence_ids, chunk_replica, parent_replica) -> StateIsolatedResolution:
        ids = tuple(dict.fromkeys(logical_evidence_ids))
        requested = set(ids)
        parent_visible = requested.intersection(parent_replica.documents)
        chunks = tuple(chunk for chunk in chunk_replica.ordered_chunks()
                       if chunk.logical_document_id in parent_visible)
        chunk_visible = {chunk.logical_document_id for chunk in chunks}
        return StateIsolatedResolution(
            query_id=query_id, principal_id=principal_id, logical_time=logical_time,
            oracle_logical_ids=ids, resolved_chunks=chunks,
            missing_parent_ids=tuple(lid for lid in ids if lid not in parent_visible),
            missing_chunk_ids=tuple(lid for lid in ids if lid in parent_visible
                                    and lid not in chunk_visible),
        )
