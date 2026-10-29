from __future__ import annotations
from dataclasses import dataclass

@dataclass
class RetrievalResult:
    raw_hits: list
    context_hits: list

def run_retrieval(retriever, query: str, principal_id: str, authorizer, top_k: int = 10) -> RetrievalResult:
    raw = retriever.retrieve(query, top_k=top_k)
    context = [h for h in raw if authorizer.allowed_chunk(principal_id, h.chunk)]
    return RetrievalResult(raw_hits=raw, context_hits=context)
