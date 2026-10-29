"""Track-A clean check: logical relevance resolved only through replica planes."""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from ragstate.replicas.scientific import ReplicaAuthorizationPlane, ReplicaContentPlane
from ragstate.retrieval.metrics import (authorization_violation, cross_tenant_exposure,
    deleted_evidence_persistence, invalid_evidence_absence_success,
    mixed_version_context_rate, state_consistent_retrieval_success,
    superseded_retrieval_rate)
from ragstate.retrieval.state_isolated import StateIsolatedEvidenceResolver
from .evidence_spans import mapping_for_queries, normalize
from .phase7b import _write_csv, _write_jsonl, _json
from .state_pilot_manifest import _load_frozen


def clean_rows(config):
    world, metadata, queries, _ = _load_frozen(config)
    chunks = ReplicaContentPlane(world, 9, chunked=True)
    parents = ReplicaContentPlane(world, 9, chunked=False)
    acl = ReplicaAuthorizationPlane(world)
    claims = defaultdict(list)
    for claim in mapping_for_queries(world, metadata, queries, chunks.ordered_chunks()):
        claims[claim['query_id']].append(claim)
    resolver = StateIsolatedEvidenceResolver()
    rows = []
    for q in queries:
        resolved = resolver.resolve(q.query_id, q.principal_id, 9,
                                    q.logical_document_ids, chunks, parents)
        candidates = resolved.hits(config['top_k'])
        context = [hit for hit in candidates if acl.allowed(q.principal_id, hit.chunk, 9)]
        gold = set(q.required_document_ids) if q.answerable else set()
        scrs = int(state_consistent_retrieval_success(context, gold, q.principal_id, world, 9))
        claim_complete = int(all(any(normalize(claim['span_text']) in normalize(hit.chunk.text)
                                         for hit in context) for claim in claims[q.query_id]))
        ucer = int(any(authorization_violation(hit.chunk, q.principal_id, world, 9)
                       for hit in context))
        dep = int(deleted_evidence_persistence(context, world, 9))
        srr = superseded_retrieval_rate(context, gold, world, 9)
        cter = int(cross_tenant_exposure(context, q.principal_id, world))
        mvcr = mixed_version_context_rate(context)
        absence = invalid_evidence_absence_success(context, gold, set(q.logical_document_ids))
        valid = (scrs and claim_complete and not (ucer or dep or srr or cter or mvcr)
                 if q.answerable else absence == 1 and not (ucer or cter))
        rows.append({'query_id': q.query_id, 'domain': q.domain,
                     'query_family': q.family, 'answerable': q.answerable,
                     'valid': int(valid), 'scrs': scrs,
                     'claim_complete': claim_complete if q.answerable else None,
                     'negative_absence': absence,
                     'ucer_context': ucer, 'dep': dep, 'srr': srr,
                     'cter': cter, 'mvcr': mvcr,
                     'oracle_logical_evidence_ids': list(q.logical_document_ids),
                     'replica_resolved_evidence_ids': [hit.chunk.document_id for hit in candidates],
                     'replica_resolved_versions': [hit.chunk.version for hit in candidates],
                     'context_chunk_ids': [hit.chunk.chunk_id for hit in context],
                     'canonical_source_state_hash': None,
                     'replica_chunk_index_hash': chunks.state_hash(),
                     'replica_parent_index_hash': parents.state_hash(),
                     'replica_acl_hash': acl.state_hash(9)})
    return rows


def run(config_path, output):
    config = json.loads(Path(config_path).read_text())
    output = Path(output)
    if (output / 'ta0_clean_summary.json').exists():
        raise ValueError('TA0 clean output is immutable')
    rows = clean_rows(config)
    answerable = [r for r in rows if r['answerable']]
    negative = [r for r in rows if not r['answerable']]
    summary = {'answerable_valid': sum(r['valid'] for r in answerable),
               'answerable_total': len(answerable),
               'answerable_rate': sum(r['valid'] for r in answerable) / len(answerable),
               'negative_valid': sum(r['valid'] for r in negative),
               'negative_total': len(negative),
               'D1': 'PASS' if sum(r['valid'] for r in answerable) / len(answerable)
                       >= config['pilot_gates']['D1']['state_isolated_clean_answerable_min']
                       else 'FAIL',
               'failures': [r['query_id'] for r in rows if not r['valid']]}
    output.mkdir(parents=True, exist_ok=True)
    _write_jsonl(output / 'ta0_clean_observations.jsonl', rows)
    flat = [{k: json.dumps(v) if isinstance(v, list) else v for k, v in r.items()}
            for r in rows]
    _write_csv(output / 'ta0_clean_observations.csv', flat)
    (output / 'ta0_clean_summary.json').write_text(_json(summary) + '\n')
    return summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/state-isolated-pilot-v1.json')
    parser.add_argument('--output', default='results/state-isolated-pilot-v1')
    args = parser.parse_args()
    print(json.dumps(run(args.config, args.output), indent=2))


if __name__ == '__main__':
    main()
