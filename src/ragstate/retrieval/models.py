from __future__ import annotations
from dataclasses import dataclass

@dataclass(frozen=True)
class Chunk:
    chunk_id: str
    document_id: str
    logical_document_id: str
    version: int
    tenant_id: str
    acl_version: int
    text: str

@dataclass(frozen=True)
class RetrievalHit:
    chunk: Chunk
    score: float
    components: tuple[tuple[str, int, float], ...] = ()
