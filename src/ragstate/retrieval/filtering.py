from __future__ import annotations

class IndexedAuthorizer:
    def __init__(self, allowed_logical_ids_by_principal: dict[str, set[str]]):
        self.allowed = allowed_logical_ids_by_principal

    def allowed_chunk(self, principal_id: str, chunk) -> bool:
        return chunk.logical_document_id in self.allowed.get(principal_id, set())

class LiveAuthorizer:
    def __init__(self, ledger, logical_time: int):
        self.ledger = ledger
        self.logical_time = logical_time

    def allowed_chunk(self, principal_id: str, chunk) -> bool:
        return self.ledger.allowed(
            principal_id,
            chunk.logical_document_id,
            self.logical_time,
        )

def filter_hits(hits, principal_id: str, authorizer):
    return [h for h in hits if authorizer.allowed_chunk(principal_id, h.chunk)]
