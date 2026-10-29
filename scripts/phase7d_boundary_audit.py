"""Audit all current v3 atomic claims under the frozen Phase 7D chunk schemes."""
from __future__ import annotations

import csv
import json
import re
from collections import defaultdict
from pathlib import Path

from ragstate.campaign.evidence_spans import claim_span, normalize, numeric_pair_boundary
from ragstate.campaign.phase7c import _load
from ragstate.retrieval.architecture import chunks_for_architecture
from ragstate.world.scientific_v2 import DOMAINS


def run(data=Path('data/generated-v3'), output=Path('results/phase7d-retrieval-architecture-v1')):
    manifest=json.loads((output/'manifest.json').read_text())
    frozen=json.loads((data/'manifest.json').read_text())
    if manifest['v3_corpus_sha256']!=frozen['corpus_sha256']:
        raise ValueError('Frozen v3 corpus mismatch')
    world,metadata,_=_load(data,frozen)
    by_domain={d.name:d for d in DOMAINS}
    rows=[]
    for arch in ('RA0','RA1','RA2','RA3'):
        _,chunks=chunks_for_architecture(world,9,arch)
        by_doc=defaultdict(list)
        for chunk in chunks:by_doc[chunk.document_id].append(chunk)
        for lid,meta in sorted(metadata.items()):
            if meta['kind'] not in ('BASE','EXC'):continue
            doc=world.document_at(lid,9)
            unit=by_domain[meta['domain']].topics[meta['topic_index']].unit
            values=re.findall(r'\b(\d+) '+re.escape(unit)+r'\b',doc.content)
            if not values:raise ValueError('Unresolvable numeric claim: '+doc.document_id)
            claim={'value':int(values[0]),'unit':unit}
            span=claim_span(doc,meta,claim)
            containing=[chunk.chunk_id for chunk in by_doc[doc.document_id]
                        if normalize(span['span_text']) in normalize(chunk.text)]
            rows.append({'architecture':arch,'logical_document_id':lid,
                'document_id':doc.document_id,'version':doc.version,
                'numeric_pair_crosses':int(numeric_pair_boundary(doc,by_doc[doc.document_id],claim['value'],unit)),
                'complete_span_available':int(bool(containing)),
                'containing_chunk_ids':json.dumps(containing)})
    path=output/'corpus_boundary_audit.csv'
    if path.exists():raise ValueError('Boundary audit exists; refusing overwrite')
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    return rows


if __name__=='__main__':
    rows=run()
    for arch in ('RA0','RA1','RA2','RA3'):
        subset=[r for r in rows if r['architecture']==arch]
        print(arch,'numeric_pairs_split',sum(r['numeric_pair_crosses'] for r in subset),
              'complete_spans_unavailable',sum(1-r['complete_span_available'] for r in subset))
