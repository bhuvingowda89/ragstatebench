"""Predeclared Phase 7D chunking and two-stage retrieval adapters."""
from __future__ import annotations

import re
import time
from collections import defaultdict

from .bm25 import BM25Retriever
from .chunking import chunk_document
from .dense import DenseRetriever
from .hybrid import HybridRetriever
from .models import Chunk


SECTION_PREFIXES = (
    'Purpose', 'Scope', 'Rule', 'Exceptions', 'Effective date',
    '## Overview', '## Procedure', '## Notes', '## Related systems',
    'Preconditions', 'Steps', 'Rollback', 'Escalation',
    'Customer tier:', 'SLA:', 'Escalation policy:',
    'Clause ', 'Summary entry', 'Investigation entry', 'Timeline entry',
    'Resolution entry', 'Follow-up entry', '- Change summary:',
    '- Compatibility:', '- Deprecation:', '- Migration guidance:',
)


def _section_key(paragraph):
    first=paragraph.split('\n',1)[0].strip()
    for prefix in SECTION_PREFIXES:
        if first.startswith(prefix):return prefix
    return 'body'


def section_aware_chunks(doc, maximum_words=200):
    if maximum_words<1:raise ValueError('maximum_words must be positive')
    # Double-newline blocks are source sections. Change notes use line bullets.
    raw=[]
    for paragraph in re.split(r'\n\s*\n',doc.content):
        if not paragraph.strip():continue
        if '\n- ' in paragraph:
            raw.extend(x for x in re.split(r'(?=\n- )',paragraph) if x.strip())
        else:raw.append(paragraph)
    groups=[];current=[];key=None
    for paragraph in raw:
        words=paragraph.split();new_key=_section_key(paragraph)
        if current and ((new_key!=key and key!='body') or len(current)+len(words)>maximum_words):
            groups.append(current);current=[]
        key=new_key
        for i in range(0,len(words),maximum_words):
            part=words[i:i+maximum_words]
            if current and len(current)+len(part)>maximum_words:
                groups.append(current);current=[]
            current.extend(part)
            if len(current)==maximum_words:
                groups.append(current);current=[]
    if current:groups.append(current)
    return [Chunk(f'{doc.document_id}:s{i}',doc.document_id,doc.logical_document_id,
                  doc.version,doc.tenant_id,0,' '.join(words))
            for i,words in enumerate(groups)]


def chunks_for_architecture(world,logical_time,architecture):
    docs=[];seen=set()
    for d in world.documents.values():
        if d.logical_document_id in seen:continue
        current=world.document_at(d.logical_document_id,logical_time)
        if current is not None:docs.append(current);seen.add(d.logical_document_id)
    if architecture=='RA0' or architecture=='RA3':
        return docs,[c for d in docs for c in chunk_document(d,max_words=80)]
    if architecture=='RA1':
        return docs,[c for d in docs for c in chunk_document(d,max_words=160)]
    if architecture=='RA2':
        return docs,[c for d in docs for c in section_aware_chunks(d,200)]
    raise ValueError('Unknown frozen architecture: '+architecture)


class PrecomputedEncoder:
    """Use globally encoded vectors for a document-restricted dense index."""
    def __init__(self,encoder,by_text):
        self.encoder=encoder;self.by_text=by_text

    def encode_documents(self,texts,batch_size):
        return [self.by_text[text] for text in texts]

    def encode_query(self,texts,batch_size):
        return self.encoder.encode_query(texts,batch_size)

    def encode(self,texts,batch_size):
        return self.encoder.encode_query(texts,batch_size)


def retrieval_bundle(chunks,encoder,model_id,revision,fusion):
    bm25=BM25Retriever(chunks)
    dense=DenseRetriever(chunks,model_name=model_id,model_version=revision,encoder=encoder)
    hybrid=HybridRetriever(bm25,dense,**fusion)
    return {'bm25':bm25,'e5':dense,'hybrid':hybrid}


class HierarchicalRetriever:
    def __init__(self,documents,chunks,encoder,model_id,revision,fusion,document_top_n=8):
        self.documents=documents;self.chunks=chunks;self.encoder=encoder
        self.model_id=model_id;self.revision=revision;self.fusion=fusion
        self.document_top_n=document_top_n
        doc_chunks=[Chunk(d.document_id+':document',d.document_id,d.logical_document_id,
                          d.version,d.tenant_id,0,' '.join(d.content.split())) for d in documents]
        self.document_indices=retrieval_bundle(doc_chunks,encoder,model_id,revision,fusion)
        self.chunk_indices=retrieval_bundle(chunks,encoder,model_id,revision,fusion)
        self.chunk_vector_by_text={c.text:v for c,v in zip(chunks,self.chunk_indices['e5'].vectors)}
        self.by_document=defaultdict(list)
        for chunk in chunks:self.by_document[chunk.document_id].append(chunk)
        self.last_document_ms=0.0;self.last_chunk_ms=0.0
        self.last_document_ids=()

    def retrieve(self,query,method,top_k=10):
        start=time.perf_counter()
        doc_hits=self.document_indices[method].retrieve(query,self.document_top_n)
        self.last_document_ms=(time.perf_counter()-start)*1000
        self.last_document_ids=tuple(h.chunk.document_id for h in doc_hits)
        start=time.perf_counter()
        selected=[c for docid in self.last_document_ids for c in self.by_document[docid]]
        precomputed=PrecomputedEncoder(self.encoder,self.chunk_vector_by_text)
        indices=retrieval_bundle(selected,precomputed,self.model_id,self.revision,self.fusion)
        hits=indices[method].retrieve(query,top_k)
        self.last_chunk_ms=(time.perf_counter()-start)*1000
        return hits

    @property
    def vector_count(self):
        return len(self.chunk_indices['e5'].vectors)+len(self.document_indices['e5'].vectors)
