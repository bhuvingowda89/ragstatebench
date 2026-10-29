"""Derive observed answer-bearing chunk positions from the frozen v3 corpus."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from ragstate.retrieval.chunking import chunk_document
from ragstate.world.models import DocumentVersion


def run(data=Path('data/generated-v3'), output=Path('results/phase7c-enterprise-realism-v3')):
    manifest=json.loads((data/'manifest.json').read_text())
    result_manifest=json.loads((output/'manifest.json').read_text())
    if manifest['corpus_sha256']!=result_manifest['corpus_sha256']:
        raise ValueError('v3 data and result corpus hashes differ')
    metadata=[json.loads(line) for line in (data/'metadata.jsonl').read_text().splitlines()]
    documents={}
    for line in (data/'documents.jsonl').read_text().splitlines():
        raw=json.loads(line)
        if raw['valid_until'] is None:
            documents[raw['logical_document_id']]=raw
    actual=Counter();declared=Counter();cross=defaultdict(Counter);by_domain=defaultdict(Counter)
    split_claims=0
    for meta in metadata:
        if meta['kind'] not in ('BASE','EXC'):continue
        raw=documents[meta['logical_document_id']]
        doc=DocumentVersion(**{**raw,'required_groups':frozenset(raw['required_groups'])})
        chunks=chunk_document(doc)
        # Numeric claim values are unique to the operative rule sentence in v3.
        words=doc.content.split()
        units={'dollars','days','months','minutes','hours','percent','reviewers','approvers'}
        locations=[i for i in range(len(words)-1) if words[i].isdigit() and
                   words[i+1].rstrip('.,;:').lower() in units]
        if not locations:raise ValueError('No atomic value for '+meta['logical_document_id'])
        i=locations[0]//80
        split_claims+=int(locations[0]//80 != (locations[0]+1)//80)
        position='beginning' if i<len(chunks)/3 else 'middle' if i<2*len(chunks)/3 else 'end'
        actual[position]+=1;declared[meta['evidence_position']]+=1
        cross[meta['evidence_position']][position]+=1
        by_domain[meta['domain']][position]+=1
    result={'corpus_sha256':manifest['corpus_sha256'],'n_answer_bearing_documents':sum(actual.values()),
            'declared_positions':dict(declared),'observed_chunk_positions':dict(actual),
            'claims_split_across_chunk_boundary':split_claims,
            'declared_by_observed':{k:dict(v) for k,v in cross.items()},
            'observed_by_domain':{k:dict(v) for k,v in by_domain.items()},
            'method':'Chunk containing the first numeric token of the current claim; relative chunk thirds. Adjacent unit may fall across a chunk boundary.'}
    path=output/'evidence_position_statistics_actual.json'
    if path.exists():raise ValueError('Audit artifact already exists')
    path.write_text(json.dumps(result,indent=2,sort_keys=True))
    return result


if __name__=='__main__':
    print(json.dumps(run(),indent=2,sort_keys=True))
