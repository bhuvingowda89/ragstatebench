"""Deterministic weighted reciprocal rank fusion over chunk IDs."""
from __future__ import annotations

from .models import RetrievalHit


class HybridRetriever:
    def __init__(self, sparse, dense, *, sparse_weight: float = 1.0,
                 dense_weight: float = 1.0, rrf_k: int = 60, candidate_k: int = 100):
        if sparse_weight < 0 or dense_weight < 0 or sparse_weight + dense_weight == 0:
            raise ValueError("Fusion weights must be nonnegative and at least one positive")
        if rrf_k < 0 or candidate_k < 1:
            raise ValueError("Invalid fusion depth")
        self.sparse, self.dense = sparse, dense
        self.weights = (sparse_weight, dense_weight)
        self.rrf_k, self.candidate_k = rrf_k, candidate_k

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalHit]:
        if top_k <= 0:
            return []
        merged = {}
        depth = max(top_k, self.candidate_k)
        for name, retriever, weight in zip(("bm25", "dense"),
                                            (self.sparse, self.dense), self.weights):
            if weight == 0:
                continue
            for rank, hit in enumerate(retriever.retrieve(query, depth), start=1):
                key = hit.chunk.chunk_id
                if key not in merged:
                    merged[key] = [hit.chunk, 0.0, []]
                elif merged[key][0] != hit.chunk:
                    raise ValueError("Conflicting provenance for chunk ID")
                merged[key][1] += weight / (self.rrf_k + rank)
                merged[key][2].append((name, rank, hit.score))
        hits = [RetrievalHit(c, s, tuple(parts)) for c, s, parts in merged.values()]
        return sorted(hits, key=lambda h: (-h.score, h.chunk.chunk_id))[:top_k]
