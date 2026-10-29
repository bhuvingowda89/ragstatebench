from __future__ import annotations
from .chunking import chunk_document

def build_chunks_from_ledger(ledger, logical_time: int, include_all_versions: bool = False):
    docs = []
    if include_all_versions:
        docs = list(ledger.documents.values())
    else:
        seen = set()
        for d in ledger.documents.values():
            if d.logical_document_id in seen:
                continue
            current = ledger.document_at(d.logical_document_id, logical_time)
            if current is not None:
                docs.append(current)
                seen.add(d.logical_document_id)
    chunks = []
    for d in docs:
        chunks.extend(chunk_document(d, acl_version=0))
    return chunks
