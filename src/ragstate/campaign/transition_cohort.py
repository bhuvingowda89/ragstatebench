"""Freeze clean-state query eligibility before any transition observations.

This module reads frozen v3 truth and frozen RA1 hybrid results. It never
constructs a retriever, injects a fault, or consults a faulted-state result.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

from ragstate.retrieval.architecture import chunks_for_architecture
from ragstate.retrieval.metrics import (
    authorization_violation,
    cross_tenant_exposure,
    deleted_evidence_persistence,
    invalid_evidence_absence_success,
    mixed_version_context_rate,
    state_consistent_retrieval_success,
    superseded_retrieval_rate,
)
from ragstate.retrieval.models import RetrievalHit

from .evidence_spans import mapping_for_queries
from .phase7b import _json, _sha, _write_csv
from .phase7c import _load


def canonical_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=False).encode()


def cohort_hash(answerable_ids, negative_ids, config, corpus_hash):
    """Hash exactly the declared inclusion result, rules, corpus and architecture."""
    payload = {
        'eligible_answerable_query_ids': sorted(answerable_ids),
        'eligible_negative_query_ids': sorted(negative_ids),
        'answerable_rules': sorted(config['answerable_rules']),
        'negative_rules': sorted(config['negative_rules']),
        'corpus_sha256': corpus_hash,
        'retrieval_architecture_id': config['retrieval_architecture_id'],
    }
    return hashlib.sha256(canonical_bytes(payload)).hexdigest(), payload


def _csv_rows(path):
    with Path(path).open(newline='') as stream:
        return list(csv.DictReader(stream))


def _load_inputs(config):
    corpus_path = Path(config['corpus_manifest'])
    corpus_manifest = json.loads(corpus_path.read_text())
    world, metadata, queries = _load(corpus_path.parent, corpus_manifest)
    result_manifest = json.loads(Path(config['clean_result_manifest']).read_text())
    frozen = result_manifest['config']
    if result_manifest['v3_corpus_sha256'] != corpus_manifest['corpus_sha256']:
        raise ValueError('Clean result uses a different v3 corpus')
    if (frozen['architecture'], frozen['chunk_words'], frozen['overlap_words'],
            frozen['final_top_k'], frozen['first_stage_candidate_k']) != ('RA1', 160, 0, 10, 30):
        raise ValueError('Clean result does not use frozen RA1 hybrid')
    if (config['chunk_words'], config['chunk_overlap_words'], config['top_k']) != (160, 0, 10):
        raise ValueError('Cohort config changes frozen retrieval architecture')
    if frozen['hybrid'] != corpus_manifest['config']['hybrid']:
        raise ValueError('Clean result changes frozen hybrid fusion')
    clean = _csv_rows(config['clean_result'])
    independent = {row['query_id']: row for row in _csv_rows(config['independent_ra1_result'])
                   if row['architecture'] == 'RA1' and row['retriever'] == 'hybrid'}
    if len(clean) != len(queries) or len(independent) != len(queries):
        raise ValueError('Incomplete frozen clean results')
    clean_by_id = {}
    for row in clean:
        if row['configuration'] != 'RR0' or row['query_id'] in clean_by_id:
            raise ValueError('Clean input must have one RR0 row per query')
        peer = independent.get(row['query_id'])
        if peer is None or any(row[key] != peer[key] for key in
                               ('context_chunk_ids', 'scrs', 'required_document_ids')):
            raise ValueError('Phase 7D and Phase 7E RA1 clean results disagree')
        clean_by_id[row['query_id']] = row
    if set(clean_by_id) != {q.query_id for q in queries}:
        raise ValueError('Clean input query IDs differ from frozen v3 queries')
    _, chunks = chunks_for_architecture(world, 9, 'RA1')
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    claim_by_query = defaultdict(list)
    for claim in mapping_for_queries(world, metadata, queries, chunks):
        claim_by_query[claim['query_id']].append(claim)
    return corpus_manifest, world, metadata, queries, clean_by_id, chunk_by_id, claim_by_query


def assess_query(query, clean_row, world, chunk_by_id, claim_rows):
    """Evaluate post-retrieval clean correctness; this never alters rankings."""
    context_ids = json.loads(clean_row['context_chunk_ids'])
    candidate_ids = json.loads(clean_row['candidate_chunk_ids'])
    if len(context_ids) != len(set(context_ids)) or not set(context_ids) <= set(candidate_ids[:10]):
        raise ValueError('Frozen context is not a unique subset of final top-10 candidates')
    if any(chunk_id not in chunk_by_id for chunk_id in context_ids):
        raise ValueError('Frozen result references unknown RA1 chunk')
    hits = [RetrievalHit(chunk_by_id[chunk_id], 0.0) for chunk_id in context_ids]
    gold = set(query.required_document_ids)
    present = {hit.chunk.document_id for hit in hits}
    doc_complete = gold.issubset(present) if query.answerable else None
    claim_complete = (len(claim_rows) == len(query.required_atomic_claims)
                      and all(set(context_ids).intersection(claim['containing_chunk_ids'])
                              for claim in claim_rows)) if query.answerable else None
    scrs = int(state_consistent_retrieval_success(hits, gold, query.principal_id,
                                                   world, query.logical_time))
    if scrs != int(clean_row['scrs']):
        raise ValueError('Clean SCRS disagrees with canonical recomputation')
    unauthorized = any(authorization_violation(hit.chunk, query.principal_id,
                                                world, query.logical_time) for hit in hits)
    deleted = deleted_evidence_persistence(hits, world, query.logical_time)
    superseded = superseded_retrieval_rate(hits, set(), world, query.logical_time) > 0
    cross_tenant = cross_tenant_exposure(hits, query.principal_id, world)
    mixed_version = mixed_version_context_rate(hits) > 0
    target_ids = set(query.logical_document_ids)
    target_hits = [hit for hit in hits if hit.chunk.logical_document_id in target_ids]
    negative_absence = invalid_evidence_absence_success(hits, gold, target_ids)
    canonical_negative = (not query.answerable and not gold and
                          not query.required_atomic_claims and
                          all(world.document_at(logical_id, query.logical_time) is None or
                              not world.allowed(query.principal_id, logical_id, query.logical_time)
                              for logical_id in target_ids))
    current_gold = all((doc := world.document_at(logical_id, query.logical_time)) is not None
                       and doc.document_id in gold and
                       world.allowed(query.principal_id, logical_id, query.logical_time)
                       for logical_id in query.logical_document_ids) if query.answerable else None
    reasons = []
    if query.answerable:
        if not gold or not current_gold:
            reasons.append('canonical_gold_invalid')
        if not scrs:
            reasons.append('scrs_failure')
        if not doc_complete:
            reasons.append('missing_document')
        if not claim_complete:
            reasons.append('missing_claim_bearing_chunk')
        if unauthorized:
            reasons.append('unauthorized_context')
        if deleted:
            reasons.append('deleted_context')
        if superseded:
            reasons.append('superseded_context')
        if cross_tenant:
            reasons.append('cross_tenant_context')
        if mixed_version:
            reasons.append('mixed_version_context')
    else:
        if not canonical_negative:
            reasons.append('canonical_negative_invalid')
        if target_hits:
            reasons.append('target_evidence_in_context')
        if negative_absence != 1:
            reasons.append('negative_absence_failure')
        if any(authorization_violation(hit.chunk, query.principal_id, world,
                                       query.logical_time) for hit in target_hits):
            reasons.append('relevant_unauthorized_context')
        if any(hit.chunk.tenant_id != query.tenant_id for hit in target_hits):
            reasons.append('relevant_cross_tenant_context')
    return {
        'query_id': query.query_id, 'domain': query.domain, 'query_family': query.family,
        'answerable': int(query.answerable), 'required_document_count': len(gold),
        'scrs': scrs, 'claim_complete': None if claim_complete is None else int(claim_complete),
        'document_complete': None if doc_complete is None else int(doc_complete),
        'negative_absence_success': negative_absence,
        'unauthorized_context': int(unauthorized), 'deleted_context': int(deleted),
        'superseded_context': int(superseded), 'cross_tenant_context': int(cross_tenant),
        'mixed_version_context': int(mixed_version),
        'eligible': int(not reasons), 'ineligibility_reasons': ';'.join(reasons),
    }


def _diversity(queries, records, metadata):
    by_id = {record['query_id']: record for record in records}
    categories = defaultdict(lambda: [0, 0])
    for query in queries:
        record = by_id[query.query_id]
        dimensions = {
            'domain': {query.domain}, 'query_family': {query.family},
            'domain_answerable' if query.answerable else 'domain_negative': {query.domain},
            'document_structure': {'multi_document' if len(query.required_document_ids) > 1
                                   else 'negative' if not query.answerable else 'single_document'},
            'answerability': {'answerable' if query.answerable else 'negative'},
            'required_document_count': {str(len(query.required_document_ids))},
            'source_archetype': {metadata[logical_id]['archetype'] for logical_id in query.logical_document_ids},
            'evidence_position': {metadata[logical_id]['evidence_position'] for logical_id in query.logical_document_ids},
        }
        for dimension, values in dimensions.items():
            for value in values:
                tally = categories[(dimension, value)]
                tally[0] += 1
                tally[1] += record['eligible']
    return [{'dimension': dimension, 'category': category, 'raw_count': counts[0],
             'eligible_count': counts[1], 'eligibility_rate': counts[1] / counts[0],
             'unit': 'query-category incidence' if dimension in ('source_archetype', 'evidence_position')
                     else 'query'}
            for (dimension, category), counts in sorted(categories.items())]


def _gates(records, diversity, config, deterministic):
    thresholds = config['gates_frozen_before_cohort_generation']
    answerable = [row for row in records if row['answerable']]
    negative = [row for row in records if not row['answerable']]
    rate = lambda rows: sum(row['eligible'] for row in rows) / len(rows)
    by_domain = defaultdict(list)
    for row in answerable:
        by_domain[row['domain']].append(row)
    domain_rates = {name: rate(group) for name, group in sorted(by_domain.items())}
    families = [row for row in diversity if row['dimension'] == 'query_family'
                and row['category'] != 'negative']
    multi = [row for row in records if row['query_family'] == 'multi_document']
    violations = sum(bool(row['unauthorized_context'] or row['deleted_context'] or
                          row['superseded_context'] or row['cross_tenant_context'] or
                          row['mixed_version_context'] or
                          (row['answerable'] and (not row['scrs'] or not row['claim_complete'] or
                                                  not row['document_complete'])) or
                          (not row['answerable'] and row['negative_absence_success'] != 1))
                     for row in records if row['eligible'])
    values = {
        'F1': {'value': rate(answerable), 'threshold': thresholds['F1']['answerable_retention_min']},
        'F2': {'value': min(domain_rates.values()),
               'threshold': thresholds['F2']['each_domain_answerable_retention_min'],
               'by_domain': domain_rates},
        'F3': {'value': rate(multi), 'threshold': thresholds['F3']['multi_document_retention_min']},
        'F4': {'value': min(row['eligibility_rate'] for row in families),
               'threshold': thresholds['F4']['each_nonnegative_family_retention_min'],
               'by_family': {row['category']: row['eligibility_rate'] for row in families}},
        'F5': {'value': rate(negative), 'threshold': thresholds['F5']['negative_retention_min']},
        'F6': {'value': violations, 'threshold': thresholds['F6']['eligible_integrity_violations_max']},
        'F7': {'value': int(deterministic), 'threshold': 1},
    }
    for key, value in values.items():
        value['status'] = 'PASS' if ((value['value'] <= value['threshold']) if key == 'F6'
                                     else (value['value'] >= value['threshold'])) else 'FAIL'
    return values


def build(config):
    corpus_manifest, world, metadata, queries, clean, chunks, claims = _load_inputs(config)
    records = [assess_query(query, clean[query.query_id], world, chunks,
                            claims[query.query_id]) for query in queries]
    answerable_ids = sorted(row['query_id'] for row in records if row['answerable'] and row['eligible'])
    negative_ids = sorted(row['query_id'] for row in records if not row['answerable'] and row['eligible'])
    hash_value, hash_payload = cohort_hash(answerable_ids, negative_ids, config,
                                            corpus_manifest['corpus_sha256'])
    return records, queries, metadata, corpus_manifest, answerable_ids, negative_ids, hash_value, hash_payload


def generate(config_path, output):
    config_path, output = Path(config_path), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Frozen transition cohort already exists; refusing overwrite')
    config = json.loads(config_path.read_text())
    first = build(config)
    second = build(config)  # independent fresh loads of corpus and frozen clean CSVs
    deterministic = first[4:7] == second[4:7] and first[0] == second[0]
    records, queries, metadata, corpus_manifest, answerable, negative, digest, payload = first
    if not deterministic:
        raise ValueError('Cohort regeneration is nondeterministic')
    diversity = _diversity(queries, records, metadata)
    gates = _gates(records, diversity, config, deterministic)
    excluded = []
    for row in records:
        if not row['answerable'] or row['eligible']:
            continue
        reasons = row['ineligibility_reasons'].split(';')
        causes = []
        if 'missing_document' in reasons:
            causes.append('missing document')
        if 'missing_claim_bearing_chunk' in reasons:
            causes.append('missing claim-bearing chunk')
        if any(reason in reasons for reason in ('unauthorized_context', 'deleted_context',
               'superseded_context', 'cross_tenant_context', 'mixed_version_context')) or \
               ('scrs_failure' in reasons and 'missing_document' not in reasons):
            causes.append('integrity violation')
        if not causes:
            causes.append('other')
        excluded.append({key: row[key] for key in ('query_id', 'domain', 'query_family',
                        'required_document_count', 'scrs', 'document_complete', 'claim_complete')}
                        | {'causes': ';'.join(causes),
                           'ineligibility_reasons': row['ineligibility_reasons']})
    policy = {
        'policy_version': 'transition-analysis-policy-v1',
        'cohort_version': config['cohort_version'],
        'primary_transition_analysis': {
            'retrieval_architecture_id': config['retrieval_architecture_id'],
            'answerable_query_ids_file': 'eligible_answerable_query_ids.json',
            'negative_query_ids_file': 'eligible_negative_query_ids.json',
            'answerable_eligibility_rules': config['answerable_rules'],
            'negative_eligibility_rules': config['negative_rules'],
            'purpose': 'Estimate causal state-transition effects without baseline retrieval failures',
            'fault_outcomes_used_for_inclusion': False,
        },
        'full_corpus_clean_analysis': {
            'query_cohort': 'all frozen v3 questions',
            'purpose': 'Report ordinary retrieval difficulty and retain clean failures',
        },
        'secondary_generalization_analysis': {
            'retrievers': ['bm25', 'e5'],
            'cohort_rule': 'retriever-specific clean eligibility must be frozen separately',
            'included_in_primary_causal_factorial': False,
        },
        'historical_phase7d_gate_A4': {'status': 'FAIL', 'observed': '25/30',
                                      'threshold': 0.85},
    }
    source_paths = [Path(__file__), Path(__file__).parent / 'evidence_spans.py',
                    Path(__file__).parent.parent / 'retrieval' / 'metrics.py']
    manifest = {
        'cohort_version': config['cohort_version'], 'corpus_version': 'v3',
        'retrieval_architecture_id': config['retrieval_architecture_id'],
        'corpus_sha256': corpus_manifest['corpus_sha256'],
        'cohort_sha256': digest, 'cohort_hash_payload': payload,
        'config_sha256': _sha(config_path), 'clean_result_sha256': _sha(config['clean_result']),
        'clean_result_manifest_sha256': _sha(config['clean_result_manifest']),
        'independent_ra1_result_sha256': _sha(config['independent_ra1_result']),
        'source_sha256': {path.name: _sha(path) for path in source_paths},
        'total_queries': len(records), 'answerable_total': sum(row['answerable'] for row in records),
        'answerable_eligible': len(answerable),
        'negative_total': sum(not row['answerable'] for row in records),
        'negative_eligible': len(negative),
        'generated_before_fault_observations': True,
        'generation_repeated_identically': deterministic,
        'gates': gates,
    }
    output.mkdir(parents=True)
    for name, ids in (('eligible_answerable_query_ids.json', answerable),
                      ('eligible_negative_query_ids.json', negative),
                      ('ineligible_query_ids.json', sorted(row['query_id'] for row in records
                                                        if not row['eligible']))):
        (output / name).write_text(_json(ids))
    _write_csv(output / 'eligibility_reasons.csv', records)
    _write_csv(output / 'excluded_query_analysis.csv', excluded)
    _write_csv(output / 'diversity_audit.csv', diversity)
    (output / 'analysis_policy.json').write_text(_json(policy))
    (output / 'gate_summary.json').write_text(_json(gates))
    (output / 'cohort_manifest.json').write_text(_json(manifest))
    return manifest


def verify(config_path, output):
    config_path, output = Path(config_path), Path(output)
    manifest = json.loads((output / 'cohort_manifest.json').read_text())
    if _sha(config_path) != manifest['config_sha256']:
        raise ValueError('Frozen cohort rules changed')
    config = json.loads(config_path.read_text())
    rebuilt = build(config)
    answerable, negative, digest = rebuilt[4:7]
    if digest != manifest['cohort_sha256'] or \
       answerable != json.loads((output / 'eligible_answerable_query_ids.json').read_text()) or \
       negative != json.loads((output / 'eligible_negative_query_ids.json').read_text()):
        raise ValueError('Frozen cohort did not regenerate identically')
    return {'cohort_sha256': digest, 'answerable_eligible': len(answerable),
            'negative_eligible': len(negative), 'independent_process_verification': True}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('generate', 'verify'))
    parser.add_argument('--config', default='configs/transition-cohort-v1.json')
    parser.add_argument('--output', default='data/transition-cohort-v1')
    args = parser.parse_args()
    result = generate(args.config, args.output) if args.stage == 'generate' else verify(args.config, args.output)
    print(_json(result))


if __name__ == '__main__':
    main()
