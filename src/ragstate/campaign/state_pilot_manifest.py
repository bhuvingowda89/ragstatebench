"""Freeze two-track pilot applicability and scenario selection before outcomes.

The manifest is selected from frozen v3 metadata and cohort-v2 IDs only. It
does not instantiate retrieval, deliver an event, or read a transition result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path

from ragstate.world.scientific_v2 import LOGICAL_TIME
from .phase7c import _load


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     default=lambda x: sorted(x) if isinstance(x, (set, frozenset))
                                     else x).encode()).hexdigest()


def _load_frozen(config):
    path = Path(config['corpus_manifest'])
    manifest = json.loads(path.read_text())
    if manifest['corpus_sha256'] != config['corpus_sha256']:
        raise ValueError('Frozen v3 corpus hash differs from pilot config')
    cohort = json.loads(Path(config['track_b_cohort_manifest']).read_text())
    if cohort['cohort_sha256'] != config['track_b_cohort_sha256']:
        raise ValueError('Frozen Track-B cohort hash differs from pilot config')
    world, metadata, queries = _load(path.parent, manifest)
    b_ids = set(json.loads(Path(config['track_b_cohort_ids']).read_text()))
    b_ids.update(json.loads(Path(config['track_b_negative_ids']).read_text()))
    return world, metadata, queries, b_ids


def applicability(world, queries):
    """Classify from clean canonical metadata; no ranking or fault outcome."""
    rows = []
    for q in sorted(queries, key=lambda item: item.query_id):
        gold_valid = (q.answerable and bool(q.required_atomic_claims)
                      and len(q.required_atomic_claims) == len(q.logical_document_ids)
                      and all((doc := world.document_at(lid, LOGICAL_TIME)) is not None
                              and world.allowed(q.principal_id, lid, LOGICAL_TIME)
                              and doc.document_id in q.required_document_ids
                              for lid in q.logical_document_ids))
        for transition in ('UPDATE', 'SUPERSEDE', 'DELETE', 'GRANT', 'REVOKE'):
            reason = None
            if not gold_valid:
                reason = ('negative_query_not_semantically_affected_by_primary_transition'
                          if not q.answerable else 'ambiguous_or_invalid_canonical_gold')
            elif transition in ('DELETE', 'GRANT', 'REVOKE') and len(q.logical_document_ids) > 1:
                # The frozen multi-source question has no declared answer rule
                # for a single required source becoming inaccessible.
                reason = 'post_transition_multi_source_answerability_undefined'
            rows.append({'query_id': q.query_id, 'domain': q.domain,
                         'query_family': q.family, 'answerable': q.answerable,
                         'transition': transition, 'applicable': reason is None,
                         'exclusion_reason': reason,
                         'logical_evidence_ids': list(q.logical_document_ids),
                         'principal_id': q.principal_id})
    return rows


def _first(rows, transition, *, cohort=None, per_domain=1):
    by_domain = defaultdict(list)
    for row in rows:
        if row['transition'] == transition and row['applicable'] and (
                cohort is None or row['query_id'] in cohort):
            by_domain[row['domain']].append(row['query_id'])
    return [qid for domain in sorted(by_domain)
            for qid in sorted(by_domain[domain])[:per_domain]]


def _scenario(block, query_id, transition, severity, architecture, fault='DELAY',
              *, component=None, compound=None, component_tuple=None, fraction=None):
    parts = [block, query_id, transition, severity, architecture, fault]
    if component: parts.append(component)
    if compound: parts.append(compound)
    if component_tuple is not None: parts.append(''.join(map(str, component_tuple)))
    if fraction is not None: parts.append(str(fraction))
    return {'scenario_id': '__'.join(parts), 'block': block, 'query_id': query_id,
            'transition': transition, 'severity': severity, 'architecture': architecture,
            'fault': fault, 'component': component, 'compound': compound,
            'component_tuple': component_tuple, 'partial_fraction': fraction}


def build(config):
    world, metadata, queries, cohort = _load_frozen(config)
    rows = applicability(world, queries)
    q_by_id = {q.query_id: q for q in queries}
    a = []
    b = []
    # TA1: all semantically applicable queries, as promised by the design.
    for row in rows:
        if row['applicable']:
            for severity in ('S1', 'S2', 'S3'):
                a.append(_scenario('TA1', row['query_id'], row['transition'], severity, 'A0'))
    # TA2: predeclared compact fault-mode subset. DELAY is matched to TA1.
    for transition in config['transition_types']:
        for qid in _first(rows, transition, per_domain=config['ta2_delay_queries_per_domain']):
            for severity in ('S1', 'S2', 'S3'):
                a.append(_scenario('TA2', qid, transition, severity, 'A0'))
    for transition in ('UPDATE', 'SUPERSEDE', 'DELETE', 'REVOKE'):
        selected = _first(rows, transition, per_domain=2)[:config['ta2_partial_max_queries']]
        for qid in selected:
            for fraction in config['ta2_partial_fractions']:
                a.append(_scenario('TA2', qid, transition, 'S2', 'A0', 'PARTIAL', fraction=fraction))
    for transition in ('UPDATE', 'SUPERSEDE', 'DELETE', 'REVOKE'):
        for qid in _first(rows, transition, per_domain=1)[:config['ta2_out_of_order_per_sequence']]:
            a.append(_scenario('TA2', qid, transition, 'S2', 'A0', 'OUT_OF_ORDER'))
    for transition in config['transition_types']:
        for qid in _first(rows, transition, per_domain=1)[:config['ta2_dropped_per_transition']]:
            a.append(_scenario('TA2', qid, transition, 'S2', 'A0', 'DROPPED'))
    # TA3: paired architecture comparison, independent of outcomes.
    for transition in ('SUPERSEDE', 'DELETE', 'GRANT', 'REVOKE'):
        selected = _first(rows, transition, per_domain=config['ta3_queries_per_domain_transition'])
        for qid in selected:
            for architecture in config['architecture_components']:
                a.append(_scenario('TA3', qid, transition, 'S2', architecture))
                if q_by_id[qid].domain in ('engineering', 'finance'):
                    a.append(_scenario('TA3', qid, transition, 'S3', architecture))
    for transition in ('UPDATE', 'SUPERSEDE', 'DELETE', 'REVOKE'):
        for qid in _first(rows, transition, per_domain=config['ta4_queries_per_domain_transition']):
            for architecture in ('A0', 'A3', 'A5'):
                a.append(_scenario('TA4', qid, transition, 'S2', architecture, 'STALE_CACHE'))
    compound_transition = {'CF1': 'SUPERSEDE', 'CF2': 'REVOKE',
                           'CF3': 'REVOKE', 'CF4': 'DELETE',
                           'CF5': 'UPDATE', 'CF6': 'REVOKE'}
    for compound, transition in compound_transition.items():
        for qid in _first(rows, transition, per_domain=config['ta5_queries_per_domain_compound']):
            for architecture in ('A0', 'A5'):
                for part in ('NONE', 'F1', 'F2', 'F1_F2'):
                    a.append(_scenario('TA5', qid, transition, 'S2', architecture,
                                       'COMPOUND', component=part, compound=compound))
    factorial = []
    for block, transition in (('authorization_revoke_S2', 'REVOKE'),
                              ('content_cache_S2', 'SUPERSEDE')):
        for qid in _first(rows, transition, per_domain=config['ta6_queries_per_domain_block']):
            for e, c, auth in config['component_factorial_blocks'][block]:
                entry = _scenario('TA6', qid, transition, 'S2', 'ECA', 'STALE_CACHE',
                                  component=block, component_tuple=[e, c, auth])
                a.append(entry)
                factorial.append(entry)
    # Transfer subset is fixed in this same pre-outcome manifest build.
    for transition in config['transition_types']:
        for qid in _first(rows, transition, cohort=cohort,
                          per_domain=config['track_b_queries_per_domain_transition']):
            b.append(_scenario('TB1', qid, transition, 'S2', 'A0'))
    for qid in _first(rows, 'REVOKE', cohort=cohort):
        for architecture in ('A0', 'A4', 'A5'):
            b.append(_scenario('TB2', qid, 'REVOKE', 'S2', architecture))
    for transition in ('SUPERSEDE', 'DELETE'):
        for qid in _first(rows, transition, cohort=cohort):
            for architecture in ('A0', 'A3', 'A5'):
                b.append(_scenario('TB3', qid, transition, 'S2', architecture, 'STALE_CACHE'))
    for compound in ('CF2', 'CF4', 'CF5'):
        transition = compound_transition[compound]
        for qid in _first(rows, transition, cohort=cohort,
                          per_domain=config['track_b_queries_per_domain_compound']):
            for architecture in ('A0', 'A5'):
                b.append(_scenario('TB4', qid, transition, 'S2', architecture,
                                   'COMPOUND', component='F1_F2', compound=compound))
    if config['campaign_version'].endswith('-v2'):
        # The scenario structure is intentionally equivalent to the invalid
        # v1 design, but the executable/cache/capability model is not. Include
        # those identities in every v2 manifest hash without changing cases.
        for scenario in a:
            scenario.update(campaign_version=config['campaign_version'],
                            runtime_model_version=config['runtime_model_version'],
                            cache_semantic_version=config['cache_semantic_version'],
                            capability_policy_version=config['capability_policy_version'])
        for scenario in b:
            scenario.update(campaign_version=config['transfer_version'],
                            runtime_model_version=config['runtime_model_version'],
                            cache_semantic_version=config['cache_semantic_version'],
                            capability_policy_version=config['capability_policy_version'])
    if len({s['scenario_id'] for s in a}) != len(a) or len({s['scenario_id'] for s in b}) != len(b):
        raise ValueError('Scenario IDs are not unique')
    return {'applicability': rows, 'track_a': a, 'track_b': b,
            'component_factorial': factorial, 'queries': q_by_id,
            'world': world, 'metadata': metadata}


def freeze(config_path, root=Path('.')):
    config_path = Path(config_path)
    config = json.loads(config_path.read_text())
    built = build(config)
    root = Path(root)
    version = config['campaign_version'].rsplit('-', 1)[-1]
    if version not in ('v1', 'v2') or config['transfer_version'] != f'realistic-transfer-pilot-{version}':
        raise ValueError('Campaign and transfer versions must match')
    data = root / f'data/state-isolated-track-{version}'
    out_a = root / f"results/{config['campaign_version']}"
    out_b = root / f"results/{config['transfer_version']}"
    for path in (data, out_a, out_b):
        if path.exists() and any(path.iterdir()):
            raise ValueError(f'Frozen output already exists: {path}')
    payload = {name: built[name] for name in ('applicability', 'track_a', 'track_b',
                                              'component_factorial')}
    hashes = {name: digest(value) for name, value in payload.items()}
    policy = {
        'version': 'analysis-policy-v2', 'primary': 'Track A',
        'Track A': {'purpose': 'Isolate causal state convergence without ordinary search-ranking error',
                    'population': 'all semantically applicable v3 queries',
                    'retrieval': 'oracle logical IDs -> replica-visible content/parent/ACL/cache -> context'},
        'Track B': {'purpose': 'Ecological transfer under realistic retrieval noise',
                    'population': 'frozen transition-cohort-v2', 'retrieval': 'frozen PA2'},
        'full_clean_benchmark': {'population': 'all v3 queries',
                                 'purpose': 'Report ordinary retrieval difficulty separately'},
        'transition_subset_selected_before_outcomes': True,
        'hypotheses': [f'H{i}' for i in range(1, 10)],
        'research_questions': [f'RQ{i}' for i in range(1, 7)],
        'no_canonical_content_in_retrieval': True,
        'negative_query_exclusions': 'cross-tenant frozen negatives lack a semantically applicable primary transition',
        'multi_source_delete_grant_revoke_exclusions': 'frozen post-transition answerability for loss of one of multiple required sources is undefined',
    }
    for path in (data, out_a, out_b): path.mkdir(parents=True)
    def write(path, value):
        path.write_text(json.dumps(value, indent=2, sort_keys=True) + '\n')
    write(data / 'analysis_policy_v2.json', policy)
    write(data / 'applicability_manifest.json', {'rows': payload['applicability'],
          'sha256': hashes['applicability'], 'corpus_sha256': config['corpus_sha256']})
    write(data / 'track_a_query_ids.json', sorted({row['query_id'] for row in payload['applicability']
                                                   if row['applicable']}))
    write(data / 'exclusions.json', [row for row in payload['applicability'] if not row['applicable']])
    write(out_a / 'scenario_manifest.json', {'scenarios': payload['track_a'],
                                              'sha256': hashes['track_a']})
    write(out_a / 'component_factorial_manifest.json', {'scenarios': payload['component_factorial'],
                                                         'sha256': hashes['component_factorial']})
    write(out_b / 'transfer_manifest.json', {'scenarios': payload['track_b'],
                                               'sha256': hashes['track_b']})
    common = {'config': config, 'config_sha256': hashlib.sha256(config_path.read_bytes()).hexdigest(),
              'corpus_sha256': config['corpus_sha256'],
              'track_b_cohort_sha256': config['track_b_cohort_sha256'],
              'manifest_hashes': hashes, 'analysis_policy_sha256': digest(policy),
              'runtime_model_version': config.get('runtime_model_version', 'v1_implicit'),
              'cache_semantic_version': config.get('cache_semantic_version', 'v1_untyped'),
              'capability_policy_version': config.get('capability_policy_version', 'v1_implicit'),
              'frozen_before_transition_outcomes': True}
    write(data / 'manifest.json', common)
    write(out_a / 'manifest.json', {**common, 'track': 'A',
                                   'scenario_count': len(payload['track_a'])})
    write(out_b / 'manifest.json', {**common, 'track': 'B',
                                   'scenario_count': len(payload['track_b'])})
    return {'hashes': hashes, 'track_a_scenarios': len(payload['track_a']),
            'track_b_scenarios': len(payload['track_b']),
            'applicable_by_transition': dict(Counter(row['transition'] for row in payload['applicability']
                                                      if row['applicable']))}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='configs/state-isolated-pilot-v1.json')
    parser.add_argument('--root', default='.')
    args = parser.parse_args()
    print(json.dumps(freeze(args.config, args.root), indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
