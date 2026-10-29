"""Frozen parent-aware clean retrieval comparison; no transition faults."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import platform
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

from ragstate.retrieval.architecture import chunks_for_architecture, retrieval_bundle
from ragstate.retrieval.dense import SentenceTransformerEncoder
from ragstate.retrieval.models import Chunk, RetrievalHit
from ragstate.retrieval.parent_aware import diversity_cap, parent_chunk_rrf
from ragstate.world.scientific_v2 import DOMAINS

from .evidence_spans import mapping_for_queries
from .phase7a import PromptedEncoder
from .phase7b import _json, _mean, _quantile, _sha, _write_csv
from .phase7c import _load
from .phase7d import _paired
from .phase7e import _evaluate
from .runner import _git_commit
from .transition_cohort import assess_query, cohort_hash, _diversity, _gates


POLICIES = ('PA0', 'PA1', 'PA2', 'PA3')


def _source_paths():
    root = Path(__file__).parent.parent
    return [Path(__file__), root / 'retrieval' / 'parent_aware.py',
            root / 'retrieval' / 'architecture.py', root / 'retrieval' / 'metrics.py',
            root / 'retrieval' / 'hybrid.py', root / 'retrieval' / 'dense.py',
            Path(__file__).parent / 'evidence_spans.py',
            Path(__file__).parent / 'transition_cohort.py']


def freeze(config_path, output):
    config_path, output = Path(config_path), Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError('Phase 7G output exists; refusing overwrite')
    config = json.loads(config_path.read_text())
    corpus_path = Path(config['corpus_manifest'])
    corpus = json.loads(corpus_path.read_text())
    phase7e = json.loads(Path(config['phase7e_manifest']).read_text())
    candidate_summary = json.loads(Path(config['phase7e_candidate_summary']).read_text())
    v1 = json.loads(Path(config['cohort_v1_manifest']).read_text())
    if (config['corpus_version'], config['architecture'], config['chunk_words'],
        config['overlap_words'], config['candidate_k'], config['top_k']) != \
       ('v3', 'RA1', 160, 0, 30, 10):
        raise ValueError('Frozen RA1 or retrieval depth changed')
    if tuple(config['policies']) != POLICIES or \
       config['policies']['PA1']['max_chunks_per_logical_document'] != 2 or \
       config['policies']['PA2']['rrf_k'] != 60 or \
       config['policies']['PA3']['rrf_k'] != 60 or \
       config['policies']['PA3']['max_chunks_per_logical_document'] != 2:
        raise ValueError('Predeclared parent-aware policy changed')
    if config['hybrid'] != corpus['config']['hybrid'] or \
       config['embedding_revision'] != corpus['config']['embedding_revision'] or \
       phase7e['v3_corpus_sha256'] != corpus['corpus_sha256'] or \
       phase7e['config']['architecture'] != 'RA1' or \
       v1['cohort_sha256'] != '2b920c186513a7a13b98d2d5ba4c9b1c61530a61d19f37b65d8c7305eb16a804':
        raise ValueError('Frozen source differs from v3, RA1, or cohort v1')
    if _sha(config['phase7e_candidate_pool']) != candidate_summary['candidate_sha256']:
        raise ValueError('Frozen top-30 candidate pool changed')
    if not Path(config['embedding_snapshot']).is_dir():
        raise ValueError('Pinned local E5 snapshot unavailable')
    inputs = ['corpus_manifest', 'phase7e_manifest', 'phase7e_candidate_pool',
              'phase7e_candidate_summary', 'phase7e_rr0', 'cohort_v1_manifest', 'cohort_v1_config']
    manifest = {
        'phase': config['phase'], 'config': config, 'config_sha256': _sha(config_path),
        'corpus_sha256': corpus['corpus_sha256'],
        'input_sha256': {key: _sha(config[key]) for key in inputs},
        'source_sha256': {path.name: _sha(path) for path in _source_paths()},
        'frozen_before_policy_evaluation': True, 'git_commit': _git_commit(),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / 'manifest.json').write_text(_json(manifest))
    return manifest


def _load_frozen(config_path, output):
    config_path, output = Path(config_path), Path(output)
    manifest = json.loads((output / 'manifest.json').read_text())
    if _sha(config_path) != manifest['config_sha256']:
        raise ValueError('Frozen Phase 7G config changed')
    config = manifest['config']
    for key, expected in manifest['input_sha256'].items():
        if _sha(config[key]) != expected:
            raise ValueError('Frozen input changed: ' + key)
    for path in _source_paths():
        if _sha(path) != manifest['source_sha256'][path.name]:
            raise ValueError('Frozen retrieval source changed: ' + path.name)
    corpus_path = Path(config['corpus_manifest'])
    corpus_manifest = json.loads(corpus_path.read_text())
    world, metadata, queries = _load(corpus_path.parent, corpus_manifest)
    return manifest, world, metadata, queries


def _candidate_rows(config):
    return {row['query_id']: row for line in Path(config['phase7e_candidate_pool']).read_text().splitlines()
            for row in [json.loads(line)]}


def _parent_index(world, config, encoder):
    documents, chunks = chunks_for_architecture(world, 9, 'RA1')
    parent_chunks = [Chunk(doc.document_id + ':document', doc.document_id,
                           doc.logical_document_id, doc.version, doc.tenant_id, 0,
                           ' '.join(doc.content.split())) for doc in documents]
    start = time.perf_counter()
    index = retrieval_bundle(parent_chunks, encoder, config['embedding_model'],
                             config['embedding_revision'], config['hybrid'])['hybrid']
    build_ms = (time.perf_counter() - start) * 1000
    vector_count = len(index.dense.vectors)
    vector_bytes = sum(len(vector) for vector in index.dense.vectors) * 4
    return documents, chunks, index, build_ms, vector_count, vector_bytes


def _policy_hits(candidates, parent_ranks, config):
    top_k, rrf_k = config['top_k'], config['hybrid']['rrf_k']
    out, overhead = {}, {}
    out['PA0'] = candidates[:top_k]
    overhead['PA0'] = (0.0, 0.0)
    start = time.perf_counter()
    out['PA1'] = diversity_cap(candidates, top_k, 2)
    overhead['PA1'] = (0.0, (time.perf_counter() - start) * 1000)
    start = time.perf_counter()
    fused = parent_chunk_rrf(candidates, parent_ranks, rrf_k)
    fusion_ms = (time.perf_counter() - start) * 1000
    out['PA2'] = fused[:top_k]
    overhead['PA2'] = (fusion_ms, 0.0)
    start = time.perf_counter()
    out['PA3'] = diversity_cap(fused, top_k, 2)
    overhead['PA3'] = (fusion_ms, (time.perf_counter() - start) * 1000)
    return out, overhead


def _groups(rows, field):
    grouped = defaultdict(list)
    for row in rows:
        if row['answerable']:
            grouped[(row['policy'], row[field])].append(row)
    return [{'policy': policy, 'scope': scope, 'n_queries': len(group),
             'cer_at_10': _mean([row['cer_at_10'] for row in group]),
             'scrs': _mean([row['scrs'] for row in group]),
             'cbr_at_10': _mean([row['cbr_at_10'] for row in group]),
             'all_required_claim_success': _mean([row['all_required_claim_success'] for row in group]),
             'all_required_document_success': _mean([row['all_required_document_success'] for row in group])}
            for (policy, scope), group in sorted(grouped.items())]


def _outcome(before, after):
    if before and after:
        return 'preserved success'
    if before and not after:
        return 'regression'
    if not before and after:
        return 'rescue'
    return 'unchanged failure'


def _cohort_v2(config, selected, rows, world, metadata, queries, chunks,
               corpus_hash, output, phase_manifest):
    """Apply exactly the frozen v1 inclusion function after retrieval selection."""
    destination = Path('data/transition-cohort-v2')
    if destination.exists() and any(destination.iterdir()):
        raise ValueError('transition-cohort-v2 already exists; refusing overwrite')
    v1_config = json.loads(Path(config['cohort_v1_config']).read_text())
    v2_config = {**v1_config, 'cohort_version': 'transition-cohort-v2',
                 'retrieval_architecture_id': f'RA1-hybrid-E5-{selected}-k10'}
    by_query = {row['query_id']: row for row in rows if row['policy'] == selected}
    chunk_by_id = {chunk.chunk_id: chunk for chunk in chunks}
    claims = defaultdict(list)
    for claim in mapping_for_queries(world, metadata, queries, chunks):
        claims[claim['query_id']].append(claim)
    def evaluate():
        return [assess_query(query, {
            'context_chunk_ids': by_query[query.query_id]['context_chunk_ids'],
            'candidate_chunk_ids': by_query[query.query_id]['selected_chunk_ids'],
            'scrs': by_query[query.query_id]['scrs'],
        }, world, chunk_by_id, claims[query.query_id]) for query in queries]
    eligibility = evaluate()
    answerable_ids = sorted(row['query_id'] for row in eligibility if row['answerable'] and row['eligible'])
    negative_ids = sorted(row['query_id'] for row in eligibility if not row['answerable'] and row['eligible'])
    digest, payload = cohort_hash(answerable_ids, negative_ids, v2_config, corpus_hash)
    repeated = evaluate() == eligibility
    diversity = _diversity(queries, eligibility, metadata)
    gates = _gates(eligibility, diversity, v2_config, repeated)
    policy = json.loads((Path('data/transition-cohort-v1') / 'analysis_policy.json').read_text())
    policy['cohort_version'] = 'transition-cohort-v2'
    policy['policy_version'] = 'transition-analysis-policy-v2'
    policy['primary_transition_analysis']['retrieval_architecture_id'] = v2_config['retrieval_architecture_id']
    manifest = {'cohort_version': 'transition-cohort-v2', 'source_cohort_v1_sha256':
                json.loads(Path(config['cohort_v1_manifest']).read_text())['cohort_sha256'],
                'phase7g_manifest_sha256': _sha(output / 'manifest.json'),
                'phase7g_per_query_sha256': _sha(output / 'per_query_results.csv'),
                'selected_policy': selected, 'corpus_sha256': corpus_hash,
                'retrieval_architecture_id': v2_config['retrieval_architecture_id'],
                'cohort_sha256': digest, 'cohort_hash_payload': payload,
                'answerable_total': sum(row['answerable'] for row in eligibility),
                'answerable_eligible': len(answerable_ids),
                'negative_total': sum(not row['answerable'] for row in eligibility),
                'negative_eligible': len(negative_ids), 'gates': gates,
                'same_frozen_eligibility_rules_as_v1': True,
                'generated_before_fault_observations': True,
                'generation_repeated_identically': repeated}
    destination.mkdir(parents=True)
    for filename, value in [('eligible_answerable_query_ids.json', answerable_ids),
                            ('eligible_negative_query_ids.json', negative_ids),
                            ('ineligible_query_ids.json', sorted(row['query_id'] for row in eligibility
                                                               if not row['eligible'])),
                            ('analysis_policy.json', policy), ('cohort_manifest.json', manifest),
                            ('gate_summary.json', gates), ('eligibility_config.json', v2_config)]:
        (destination / filename).write_text(_json(value))
    _write_csv(destination / 'eligibility_reasons.csv', eligibility)
    _write_csv(destination / 'diversity_audit.csv', diversity)
    return manifest


def run(config_path, output):
    output = Path(output)
    if (output / 'gate_summary.json').exists():
        raise ValueError('Completed Phase 7G result is immutable')
    manifest, world, metadata, queries = _load_frozen(config_path, output)
    config = manifest['config']
    os.environ.update(HF_HUB_OFFLINE='1', TRANSFORMERS_OFFLINE='1',
                      HF_DATASETS_OFFLINE='1', TOKENIZERS_PARALLELISM='false')
    import torch, sentence_transformers
    torch.manual_seed(0)
    torch.set_num_threads(1)
    start = time.perf_counter()
    encoder = PromptedEncoder(SentenceTransformerEncoder(config['embedding_snapshot'],
                             config['embedding_revision']), config['embedding_query_prefix'],
                             config['embedding_passage_prefix'])
    model_load_ms = (time.perf_counter() - start) * 1000
    documents, chunks, parent_index, parent_build_ms, vectors, vector_bytes = \
        _parent_index(world, config, encoder)
    parent_index.retrieve(queries[0].text, len(documents))  # timing warm-up
    by_chunk = {chunk.chunk_id: chunk for chunk in chunks}
    candidates = _candidate_rows(config)
    if set(candidates) != {query.query_id for query in queries}:
        raise ValueError('Frozen top-30 candidate query IDs mismatch')
    claim_by_query = defaultdict(list)
    for claim in mapping_for_queries(world, metadata, queries, chunks):
        claim_by_query[claim['query_id']].append(claim)
    baseline = {row['query_id']: row for row in csv.DictReader(Path(config['phase7e_rr0']).open())}
    rows = []
    rankings = {policy: {} for policy in POLICIES}
    repeated_identical = True
    for query in queries:
        frozen = candidates[query.query_id]
        chunk_hits = [RetrievalHit(by_chunk[chunk_id], float(score),
                                  tuple(tuple(item) for item in components))
                      for chunk_id, score, components in zip(frozen['chunk_ids'], frozen['scores'],
                                                             frozen['components'])]
        if len(chunk_hits) != config['candidate_k']:
            raise ValueError('Frozen query does not have exactly 30 candidates')
        start = time.perf_counter()
        parent_hits = parent_index.retrieve(query.text, len(documents))
        parent_ms = (time.perf_counter() - start) * 1000
        parent_ranks = {hit.chunk.logical_document_id: rank
                        for rank, hit in enumerate(parent_hits, 1)}
        selected, overhead = _policy_hits(chunk_hits, parent_ranks, config)
        # Recompute parent ranks and all policies independently, without timing.
        repeat_hits = parent_index.retrieve(query.text, len(documents))
        repeat_ranks = {hit.chunk.logical_document_id: rank
                        for rank, hit in enumerate(repeat_hits, 1)}
        repeated, _ = _policy_hits(chunk_hits, repeat_ranks, config)
        for policy in POLICIES:
            hits = selected[policy]
            chunk_ids = [hit.chunk.chunk_id for hit in hits]
            rankings[policy][query.query_id] = chunk_ids
            repeated_identical &= chunk_ids == [hit.chunk.chunk_id for hit in repeated[policy]]
            fusion_ms, cap_ms = overhead[policy]
            required_parent_ms = parent_ms if policy in ('PA2', 'PA3') else 0.0
            result = _evaluate(query, policy, hits, world, claim_by_query[query.query_id],
                               frozen['first_stage_ms'], fusion_ms + cap_ms + required_parent_ms,
                               chunk_hits)
            result.pop('pairs_scored')
            result['policy'] = result.pop('configuration')
            result['selected_chunk_ids'] = json.dumps(chunk_ids)
            result['selected_count'] = len(hits)
            counts = defaultdict(int)
            for hit in hits:
                counts[hit.chunk.logical_document_id] += 1
            result['distinct_logical_documents_at_10'] = len(counts)
            result['largest_parent_fraction_at_10'] = max(counts.values()) / len(hits) if hits else 0.0
            result['parent_document_ranks'] = json.dumps({logical_id: parent_ranks[logical_id]
                for logical_id in sorted({hit.chunk.logical_document_id for hit in chunk_hits})}, sort_keys=True)
            result['candidate_retrieval_ms'] = frozen['first_stage_ms']
            result['parent_rank_ms'] = required_parent_ms
            result['fusion_ms'] = fusion_ms
            result['diversity_cap_ms'] = cap_ms
            result['total_ms'] = frozen['first_stage_ms'] + required_parent_ms + fusion_ms + cap_ms
            rows.append(result)
            if policy == 'PA0':
                reference = baseline[query.query_id]
                reference_cbr = None if reference['cbr_at_10'] == '' else float(reference['cbr_at_10'])
                if result['context_chunk_ids'] != reference['context_chunk_ids'] or \
                   result['scrs'] != int(reference['scrs']) or \
                   result['cbr_at_10'] != reference_cbr:
                    raise ValueError('PA0 does not reproduce frozen RR0')
    domain_rows, family_rows = _groups(rows, 'domain'), _groups(rows, 'query_family')
    summary, latency = [], []
    by_policy = {policy: [row for row in rows if row['policy'] == policy] for policy in POLICIES}
    for policy in POLICIES:
        group = by_policy[policy]
        answerable = [row for row in group if row['answerable']]
        negative = [row for row in group if not row['answerable']]
        multi = [row for row in answerable if row['query_family'] == 'multi_document']
        single = [row for row in answerable if row['query_family'] != 'multi_document']
        record = {'policy': policy, 'n_queries': len(group), 'n_answerable': len(answerable),
            'cer_at_10': _mean([row['cer_at_10'] for row in answerable]),
            'scrs': _mean([row['scrs'] for row in answerable]),
            'cbr_at_10': _mean([row['cbr_at_10'] for row in answerable]),
            'all_required_claim_success': _mean([row['all_required_claim_success'] for row in answerable]),
            'multi_document_doc_success': _mean([row['all_required_document_success'] for row in multi]),
            'multi_claim_success': _mean([row['all_required_claim_success'] for row in multi]),
            'single_scrs': _mean([row['scrs'] for row in single]),
            'single_claim_success': _mean([row['all_required_claim_success'] for row in single]),
            'negative_absence_success': _mean([row['negative_absence_success'] for row in negative]),
            'mean_distinct_logical_documents': _mean([row['distinct_logical_documents_at_10'] for row in group]),
            'median_distinct_logical_documents': statistics.median(row['distinct_logical_documents_at_10'] for row in group),
            'mean_largest_parent_fraction': _mean([row['largest_parent_fraction_at_10'] for row in group]),
            'median_candidate_ms': statistics.median(row['candidate_retrieval_ms'] for row in group),
            'p95_candidate_ms': _quantile([row['candidate_retrieval_ms'] for row in group], .95),
            'median_parent_ms': statistics.median(row['parent_rank_ms'] for row in group),
            'p95_parent_ms': _quantile([row['parent_rank_ms'] for row in group], .95),
            'median_fusion_ms': statistics.median(row['fusion_ms'] for row in group),
            'median_diversity_cap_ms': statistics.median(row['diversity_cap_ms'] for row in group),
            'median_total_ms': statistics.median(row['total_ms'] for row in group),
            'p95_total_ms': _quantile([row['total_ms'] for row in group], .95)}
        summary.append(record)
        latency.append({key: record[key] for key in record if key in ('policy', 'median_candidate_ms',
                'p95_candidate_ms', 'median_parent_ms', 'p95_parent_ms', 'median_fusion_ms',
                'median_diversity_cap_ms', 'median_total_ms', 'p95_total_ms')})
    by_id = {policy: {row['query_id']: row for row in by_policy[policy]} for policy in POLICIES}
    multi_analysis, single_regressions = [], []
    for query in queries:
        if not query.answerable:
            continue
        baseline_row = by_id['PA0'][query.query_id]
        record = {'query_id': query.query_id, 'domain': query.domain,
                  'query_family': query.family, 'required_document_count': len(query.required_document_ids),
                  'required_document_ids': baseline_row['required_document_ids'],
                  'required_claim_chunk_ids': baseline_row['required_claim_chunk_ids']}
        for policy in POLICIES:
            row = by_id[policy][query.query_id]
            record[f'{policy}_distinct_logical_documents'] = row['distinct_logical_documents_at_10']
            record[f'{policy}_document_ranks'] = row['final_document_ranks']
            record[f'{policy}_claim_ranks'] = row['final_claim_ranks']
            record[f'{policy}_scrs'] = row['scrs']
            record[f'{policy}_document_success'] = row['all_required_document_success']
            record[f'{policy}_claim_success'] = row['all_required_claim_success']
            if policy != 'PA0':
                record[f'{policy}_doc_outcome'] = _outcome(baseline_row['all_required_document_success'],
                                                         row['all_required_document_success'])
                record[f'{policy}_claim_outcome'] = _outcome(baseline_row['all_required_claim_success'],
                                                           row['all_required_claim_success'])
                record[f'{policy}_scrs_outcome'] = _outcome(baseline_row['scrs'], row['scrs'])
        if query.family == 'multi_document':
            multi_analysis.append(record)
        else:
            single_regressions.append(record)
    paired = []
    settings = config['paired_analysis']
    for policy in POLICIES[1:]:
        for scope, condition, endpoints in (
            ('all_answerable', lambda row: bool(row['answerable']), ('scrs', 'all_required_claim_success')),
            ('multi_document', lambda row: row['query_family'] == 'multi_document',
             ('all_required_document_success', 'all_required_claim_success')),
            ('single_document', lambda row: bool(row['answerable']) and row['query_family'] != 'multi_document',
             ('scrs', 'all_required_claim_success'))):
            left = sorted((row for row in by_policy['PA0'] if condition(row)), key=lambda row: row['query_id'])
            right = sorted((row for row in by_policy[policy] if condition(row)), key=lambda row: row['query_id'])
            if [row['query_id'] for row in left] != [row['query_id'] for row in right]:
                raise ValueError('Unmatched paired queries')
            for endpoint in endpoints:
                paired.append({'policy': policy, 'scope': scope, 'endpoint': endpoint,
                               **_paired(left, right, endpoint, settings['bootstrap_seed'],
                                         settings['bootstrap_replicates'])})
    diversity = []
    for policy in POLICIES:
        for scope, group in (('all_answerable', [row for row in by_policy[policy] if row['answerable']]),
                             ('multi_document', [row for row in by_policy[policy] if row['query_family'] == 'multi_document'])):
            by_count = defaultdict(list)
            for row in group:
                by_count[row['distinct_logical_documents_at_10']].append(row)
            for count, subgroup in sorted(by_count.items()):
                diversity.append({'policy': policy, 'scope': scope,
                    'distinct_logical_documents_at_10': count, 'n_queries': len(subgroup),
                    'scrs': _mean([row['scrs'] for row in subgroup]),
                    'all_required_document_success': _mean([row['all_required_document_success'] for row in subgroup]),
                    'all_required_claim_success': _mean([row['all_required_claim_success'] for row in subgroup]),
                    'mean_largest_parent_fraction': _mean([row['largest_parent_fraction_at_10'] for row in subgroup])})
    scores = {}
    for policy in POLICIES:
        domains = [row for row in domain_rows if row['policy'] == policy]
        record = next(row for row in summary if row['policy'] == policy)
        scores[policy] = {'macro_domain_scrs': _mean([row['scrs'] for row in domains]),
            'multi_claim_success': record['multi_claim_success'],
            'minimum_domain_scrs': min(row['scrs'] for row in domains),
            'overall_claim_success': record['all_required_claim_success'],
            'median_total_ms': record['median_total_ms']}
    selected = max(POLICIES, key=lambda policy: (
        scores[policy]['macro_domain_scrs'], scores[policy]['multi_claim_success'],
        scores[policy]['minimum_domain_scrs'], scores[policy]['overall_claim_success'],
        -scores[policy]['median_total_ms']))
    thresholds = config['gates_frozen_before_execution']
    baseline_single = next(row['single_scrs'] for row in summary if row['policy'] == 'PA0')
    gates = {}
    for policy in POLICIES:
        record = next(row for row in summary if row['policy'] == policy)
        values = {
            'G1': record['scrs'] >= thresholds['G1']['overall_scrs_min'],
            'G2': scores[policy]['minimum_domain_scrs'] >= thresholds['G2']['minimum_domain_scrs_min'],
            'G3': record['cbr_at_10'] >= thresholds['G3']['cbr_at_10_min'],
            'G4': record['multi_document_doc_success'] >= thresholds['G4']['multi_document_document_success_min'],
            'G5': record['multi_claim_success'] >= thresholds['G5']['multi_document_claim_success_min'],
            'G6': record['negative_absence_success'] >= thresholds['G6']['negative_absence_min'],
            'G7': repeated_identical,
            'G8': record['single_scrs'] >= baseline_single - thresholds['G8']['single_document_scrs_max_drop'],
        }
        gates[policy] = {key: 'PASS' if value else 'FAIL' for key, value in values.items()}
    environment = {'python': sys.version.split()[0], 'platform': platform.platform(),
        'torch': torch.__version__, 'sentence_transformers': sentence_transformers.__version__,
        'device': 'cpu', 'git_commit': _git_commit(), 'embedding_model': config['embedding_model'],
        'embedding_revision': config['embedding_revision']}
    gate_summary = {'selected_policy': selected, 'selection_scores': scores,
        'gates_by_policy': gates, 'selected_passes_all_retrieval_gates':
        all(value == 'PASS' for value in gates[selected].values()),
        'any_policy_passes_all_retrieval_gates': any(all(value == 'PASS' for value in gate.values())
                                                  for gate in gates.values()),
        'repeated_rankings_identical': repeated_identical,
        'parent_documents': len(documents), 'ra1_chunks': len(chunks),
        'parent_index_build_ms': parent_build_ms, 'parent_model_load_ms': model_load_ms,
        'parent_dense_vectors': vectors, 'parent_dense_vector_bytes': vector_bytes,
        'environment': environment, 'fault_campaign_executed': False}
    _write_csv(output / 'architecture_summary.csv', summary)
    _write_csv(output / 'per_query_results.csv', rows)
    _write_csv(output / 'diversity_statistics.csv', diversity)
    _write_csv(output / 'domain_summary.csv', domain_rows)
    _write_csv(output / 'query_family_summary.csv', family_rows)
    _write_csv(output / 'multi_document_analysis.csv', multi_analysis)
    _write_csv(output / 'single_document_regressions.csv', single_regressions)
    _write_csv(output / 'latency_summary.csv', latency)
    _write_csv(output / 'statistical_comparisons.csv', paired)
    (output / 'gate_summary.json').write_text(_json(gate_summary))
    cohort_v2 = None
    if gate_summary['selected_passes_all_retrieval_gates']:
        cohort_v2 = _cohort_v2(config, selected, rows, world, metadata, queries, chunks,
                              manifest['corpus_sha256'], output, manifest)
        gate_summary['cohort_v2'] = {'sha256': cohort_v2['cohort_sha256'],
                                    'gates': cohort_v2['gates'],
                                    'answerable_eligible': cohort_v2['answerable_eligible'],
                                    'negative_eligible': cohort_v2['negative_eligible']}
        (output / 'gate_summary.json').write_text(_json(gate_summary))
    lines = ['# Phase 7G parent-aware clean retrieval', '',
        f"Frozen v3 corpus: `{manifest['corpus_sha256']}`. PA0 reproduces Phase 7E RR0.", '',
        '## Clean metrics', '',
        '| Policy | CER | SCRS | CBR | All claims | Multi docs | Multi claims | Single SCRS | Negative absence | Median ms |',
        '| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |']
    for row in summary:
        lines.append(f"| {row['policy']} | {row['cer_at_10']:.3f} | {row['scrs']:.3f} | "
                     f"{row['cbr_at_10']:.3f} | {row['all_required_claim_success']:.3f} | "
                     f"{row['multi_document_doc_success']:.3f} | {row['multi_claim_success']:.3f} | "
                     f"{row['single_scrs']:.3f} | {row['negative_absence_success']:.3f} | "
                     f"{row['median_total_ms']:.1f} |")
    lines += ['', f"Selected by the frozen rule: **{selected}**.", '', '## Retrieval gates', '']
    for policy in POLICIES:
        lines.append('- ' + policy + ': ' + ', '.join(f'{name} {status}' for name, status in gates[policy].items()))
    if cohort_v2:
        lines += ['', '## Transition cohort v2', '',
                  f"Hash: `{cohort_v2['cohort_sha256']}`; answerable eligible "
                  f"{cohort_v2['answerable_eligible']}/{cohort_v2['answerable_total']}; "
                  f"negative eligible {cohort_v2['negative_eligible']}/{cohort_v2['negative_total']}.",
                  ', '.join(f'{name} {gate["status"]}' for name, gate in cohort_v2['gates'].items())]
    else:
        lines += ['', 'No cohort v2 was generated because the selected policy did not pass all retrieval gates.']
    lines += ['', 'Diversity-success tables are observational associations, not causal effects.',
              'No transition fault or LLM generation was executed.', '']
    (output / 'summary.md').write_text('\n'.join(lines))
    return gate_summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('stage', choices=('freeze', 'run'))
    parser.add_argument('--config', default='configs/phase7g-parent-aware-v1.json')
    parser.add_argument('--output', default='results/phase7g-parent-aware-v1')
    args = parser.parse_args()
    result = freeze(args.config, args.output) if args.stage == 'freeze' else run(args.config, args.output)
    print(_json({'stage': args.stage, 'result': result if args.stage == 'run'
                 else {'manifest_created': True}}))


if __name__ == '__main__':
    main()
