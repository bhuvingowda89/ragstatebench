"""Deterministic parent-aware policies over an unchanged chunk candidate pool."""
from __future__ import annotations

from .models import RetrievalHit


def diversity_cap(hits: list[RetrievalHit], top_k: int = 10,
                  max_per_logical_document: int = 2) -> list[RetrievalHit]:
    """Keep input ranking order and cap repeated logical-document chunks."""
    if top_k < 0 or max_per_logical_document < 1:
        raise ValueError('Invalid diversity-cap settings')
    if top_k == 0:
        return []
    counts = {}
    selected = []
    for hit in hits:
        parent = hit.chunk.logical_document_id
        if counts.get(parent, 0) >= max_per_logical_document:
            continue
        selected.append(hit)
        counts[parent] = counts.get(parent, 0) + 1
        if len(selected) == top_k:
            break
    return selected


def parent_chunk_rrf(hits: list[RetrievalHit], parent_ranks: dict[str, int],
                     rrf_k: int = 60) -> list[RetrievalHit]:
    """Equal rank fusion using only chunk rank and independent parent rank."""
    if rrf_k < 0:
        raise ValueError('RRF constant must be nonnegative')
    ranked = []
    for chunk_rank, hit in enumerate(hits, 1):
        parent = hit.chunk.logical_document_id
        parent_rank = parent_ranks.get(parent)
        if parent_rank is None or parent_rank < 1:
            raise ValueError('Candidate parent is absent from full-document ranking')
        score = 1 / (rrf_k + chunk_rank) + 1 / (rrf_k + parent_rank)
        ranked.append((chunk_rank, RetrievalHit(hit.chunk, score, hit.components)))
    ranked.sort(key=lambda item: (-item[1].score, item[0], item[1].chunk.chunk_id))
    return [hit for _, hit in ranked]
