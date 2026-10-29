from __future__ import annotations
from .models import Chunk

def chunk_document(doc, acl_version: int = 0, max_words: int = 80) -> list[Chunk]:
    words = doc.content.split()
    if not words:
        return []
    chunks = []
    for i in range(0, len(words), max_words):
        part = words[i:i+max_words]
        chunks.append(Chunk(
            chunk_id=f"{doc.document_id}:c{i//max_words}",
            document_id=doc.document_id,
            logical_document_id=doc.logical_document_id,
            version=doc.version,
            tenant_id=doc.tenant_id,
            acl_version=acl_version,
            text=" ".join(part),
        ))
    return chunks
