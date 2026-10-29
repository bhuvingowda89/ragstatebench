"""Deterministic character-span oracle for current v3 atomic claims."""
from __future__ import annotations

import hashlib
import re
from collections import defaultdict

from ragstate.world.scientific_v2 import DOMAINS, LOGICAL_TIME


def normalize(text):
    return ' '.join(text.split())


def claim_span(document,metadata,claim):
    scope=re.escape(metadata['axis_one']+' '+metadata['axis_two'])
    value=re.escape(str(claim['value']))
    unit=re.escape(claim['unit'])
    pattern=rf'For\s+{scope},\s+the\s+[^.\n]{{0,300}}?\b{value}\s+{unit}\.'
    match=re.search(pattern,document.content)
    if match is None:
        raise ValueError('Unresolvable claim sentence in '+document.document_id)
    span=document.content[match.start():match.end()]
    return {'document_id':document.document_id,'version':document.version,
            'start_char':match.start(),'end_char':match.end(),
            'span_text':span,'span_sha256':hashlib.sha256(normalize(span).encode()).hexdigest()}


def mapping_for_queries(world,metadata,queries,chunks):
    by_document=defaultdict(list)
    for chunk in chunks:by_document[chunk.document_id].append(chunk)
    rows=[]
    for query in queries:
        for claim in query.required_atomic_claims:
            document=world.documents[claim['document_id']]
            info=claim_span(document,metadata[document.logical_document_id],claim)
            normalized=normalize(info['span_text'])
            containing=[c.chunk_id for c in by_document[document.document_id]
                        if normalized in normalize(c.text)]
            rows.append({'query_id':query.query_id,'domain':query.domain,
                         'query_family':query.family,'claim_type':claim['claim_type'],
                         'document_id':document.document_id,'version':document.version,
                         'logical_document_id':document.logical_document_id,
                         'claim_value':claim['value'],'claim_unit':claim['unit'],
                         **info,'containing_chunk_ids':containing,
                         'complete_span_available':bool(containing)})
    return rows


def numeric_pair_boundary(document,chunks,value,unit):
    words=document.content.split()
    targets=[i for i in range(len(words)-1)
             if words[i]==str(value) and words[i+1].rstrip('.,;:')==unit]
    if not targets:raise ValueError('No numeric pair in '+document.document_id)
    target=targets[0]
    # Chunk text is contiguous, nonoverlapping words for RA0/RA1. RA2 is
    # section based, so inspect actual chunk text instead of assuming offsets.
    pair=f'{value} {unit}'
    return not any(pair in c.text for c in chunks if c.document_id==document.document_id)
