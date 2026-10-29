"""Compile frozen Track-A scenario candidates into structurally realizable cases.

This module reads the frozen corpus and scenario definitions, but never reads
observation rows, retrieval metrics, or faulted scientific outcomes. A dry run
applies transitions to disposable replica copies without evaluating queries.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

from ragstate.replicas.scientific import (ReplicaAuthorizationPlane,
    ReplicaContentPlane, ReplicaEvidenceCache)
from ragstate.retrieval.models import RetrievalHit
from ragstate.retrieval.state_isolated import StateIsolatedEvidenceResolver
from ragstate.transitions.engine import TransitionEngine
from ragstate.transitions.events import Transition
from .state_pilot_manifest import _load_frozen, digest
from .state_pilot_runtime import prepare_prestate, transition_event


POLICY_VERSION = 'scenario-applicability-v1'
OUTPUT = Path('results/scenario-applicability-v1')
CONFIG = Path('configs/state-isolated-pilot-v2.json')
SOURCE_MANIFEST = Path('results/state-isolated-pilot-v3/scenario_manifest.json')
TICK = 10
CONTENT = ('UPDATE', 'SUPERSEDE', 'DELETE')
AUTH = ('GRANT', 'REVOKE')


@dataclass(frozen=True)
class ReplicaUnit:
    plane: str
    unit_type: str
    identity: str


@dataclass(frozen=True)
class FaultOperatorSpec:
    operator_id: str
    target_plane: str
    replica_unit_type: str
    required_transition_types: tuple[str, ...]
    structural_preconditions: tuple[str, ...]
    state_preconditions: tuple[str, ...]
    minimum_replica_units: int
    event_preconditions: tuple[str, ...]
    cache_preconditions: tuple[str, ...]
    authorization_preconditions: tuple[str, ...]
    applicability_predicate: str
    effectiveness_predicate: str
    convergence_expectation: str
    valid_severities: tuple[str, ...] = ('S1', 'S2', 'S3')


def _spec(operator_id, plane, unit, transitions, *, structural=(), state=(),
          minimum=1, events=(), cache=(), authorization=(), predicate=None,
          effect=None, convergence='eventual'):
    return FaultOperatorSpec(operator_id, plane, unit, transitions, structural,
        state, minimum, events, cache, authorization, predicate or operator_id.lower(),
        effect or operator_id.lower(), convergence)


OPERATORS = {
    'DELAY_CONTENT': _spec('DELAY_CONTENT', 'content_chunk_and_parent', 'chunk', CONTENT,
        state=('canonical content changes',), events=('deliverable document event',),
        effect='content_delay'),
    'DELAY_AUTH': _spec('DELAY_AUTH', 'replica_acl', 'acl_entry', AUTH,
        state=('effective authorization changes',), events=('deliverable ACL event',),
        effect='auth_delay'),
    'PARTIAL_CONTENT': _spec('PARTIAL_CONTENT', 'replica_chunk_units', 'chunk', CONTENT,
        structural=('updated and stale chunks coexist',), minimum=2,
        events=('chunkwise document event',), predicate='partial_content',
        effect='partial_content_split'),
    'PARTIAL_AUTH': _spec('PARTIAL_AUTH', 'replica_acl_units', 'acl_entry', AUTH,
        structural=('updated and stale ACL chunk entries coexist',), minimum=2,
        events=('chunkwise ACL event',), predicate='partial_auth',
        effect='partial_acl_split'),
    'OUT_OF_ORDER': _spec('OUT_OF_ORDER', 'replica_content_or_acl', 'ordered_event',
        ('UPDATE', 'SUPERSEDE', 'DELETE', 'REVOKE'),
        structural=('two causally ordered events on same target',), minimum=2,
        events=('new event then prior-state replay',), effect='state_reversal'),
    'DROPPED_CONTENT': _spec('DROPPED_CONTENT', 'content_chunk_and_parent', 'document_event', CONTENT,
        events=('omittable content event',), effect='event_absent', convergence='right_censored'),
    'DROPPED_AUTH': _spec('DROPPED_AUTH', 'replica_acl', 'authorization_event', AUTH,
        events=('omittable ACL event',), effect='event_absent', convergence='right_censored'),
    'STALE_CACHE': _spec('STALE_CACHE', 'candidate_cache', 'candidate_cache_entry',
        ('UPDATE', 'SUPERSEDE', 'DELETE', 'REVOKE'),
        cache=('pretransition candidate exists', 'posttransition lookup opportunity'),
        effect='candidate_cache_reuse_or_invalidation'),
    'STALE_ACL_METADATA': _spec('STALE_ACL_METADATA', 'replica_acl_metadata', 'acl_metadata_entry',
        ('REVOKE',), structural=('cross-tenant same-topic chunk exists',),
        authorization=('canonical deny and replica allow differ',),
        effect='stale_acl_cross_tenant'),
    'STALE_INGESTION_REPLAY': _spec('STALE_INGESTION_REPLAY', 'content_chunk_and_parent',
        'ordered_event', ('DELETE',),
        structural=('predecessor ingestion payload exists',), minimum=2,
        events=('delete then stale create replay',), effect='resurrected_replica_content'),
    'CROSS_TENANT_CACHE': _spec('CROSS_TENANT_CACHE', 'candidate_cache', 'candidate_cache_entry',
        ('REVOKE',), structural=('cross-tenant same-topic chunk exists',),
        cache=('injected candidate cache entry can be populated',),
        effect='cross_tenant_cache_injection'),
    'CONTENT_UPDATE': _spec('CONTENT_UPDATE', 'replica_chunk_units', 'chunk', ('UPDATE',),
        state=('changed successor content exists',), effect='content_update'),
    'AUTH_REVOKE': _spec('AUTH_REVOKE', 'replica_acl', 'acl_entry', ('UPDATE',),
        state=('principal allowed before revoke',),
        authorization=('extra revoke changes effective authorization',),
        effect='auth_revoke'),
}


COMPOUNDS = {
    'CF1': ('DELAY_CONTENT', 'STALE_CACHE'),
    'CF2': ('DELAY_AUTH', 'STALE_CACHE'),
    'CF3': ('DELAY_AUTH', 'PARTIAL_AUTH'),
    'CF4': ('DELAY_CONTENT', 'STALE_INGESTION_REPLAY'),
    'CF5': ('CONTENT_UPDATE', 'AUTH_REVOKE'),
    'CF6': ('CROSS_TENANT_CACHE', 'STALE_ACL_METADATA'),
}


@dataclass(frozen=True)
class ApplicabilityResult:
    scenario_id: str
    applicable: bool
    reasons: tuple[str, ...]
    structural_facts: dict
    unit_counts: dict
    transition_preconditions: dict
    fault_preconditions: dict
    architecture_preconditions: dict
    expected_target_planes: tuple[str, ...]
    effectiveness_predicates: tuple[str, ...]
    component_applicability: dict
    updated_unit_count: int | None


def partial_updated_units(n: int, fraction: float) -> int | None:
    """Frozen floor rule. No clamping: impossible splits are excluded."""
    if n < 2 or not 0 < fraction < 1:
        return None
    updated = math.floor(n * fraction)
    return updated if 1 <= updated < n else None


def _cross_tenant_lid(query, metadata, pre):
    origin = metadata[query.logical_document_ids[0]]
    matches = sorted(lid for lid, item in metadata.items()
                     if item['domain'] == origin['domain']
                     and item['topic_index'] == origin['topic_index']
                     and item['tenant_id'] != origin['tenant_id']
                     and pre.document_at(lid, TICK - 1) is not None)
    return matches[0] if matches else None


def _operator_names(scenario):
    if scenario['fault'] == 'COMPOUND':
        pair = COMPOUNDS[scenario['compound']]
        arm = scenario['component']
        return tuple((component, pair[index]) for index, component in enumerate(('F1', 'F2'))
                     if arm in (component, 'F1_F2'))
    fault, transition = scenario['fault'], scenario['transition']
    if fault in ('DELAY', 'PARTIAL', 'DROPPED'):
        prefix = {'DELAY': 'DELAY', 'PARTIAL': 'PARTIAL', 'DROPPED': 'DROPPED'}[fault]
        return ((fault, prefix + ('_AUTH' if transition in AUTH else '_CONTENT')),)
    if fault == 'OUT_OF_ORDER':
        return ((fault, 'OUT_OF_ORDER'),)
    if fault == 'STALE_CACHE':
        return ((fault, 'STALE_CACHE'),)
    raise ValueError(f'Unsupported frozen fault: {fault}')


def _transition_facts(world, metadata, query, transition):
    pre = prepare_prestate(world, query, transition, TICK)
    lid = query.logical_document_ids[0]
    old = pre.document_at(lid, TICK - 1)
    before_allowed = pre.allowed(query.principal_id, lid, TICK - 1)
    event, updates = transition_event(pre, metadata, query, transition, TICK)
    post = prepare_prestate(world, query, transition, TICK)
    TransitionEngine(post).apply(event)
    after_allowed = post.allowed(query.principal_id, lid, TICK)
    changed_content = (old.content != post.document_at(lid, TICK).content
                       if old and post.document_at(lid, TICK) else old is not None)
    chunk = ReplicaContentPlane(pre, TICK - 1, chunked=True)
    parent = ReplicaContentPlane(pre, TICK - 1, chunked=False)
    acl = ReplicaAuthorizationPlane(pre)
    old_units = tuple(ReplicaUnit('content_chunk', 'chunk', c.chunk_id)
                      for c in chunk.chunks.get(lid, ()))
    new_units = tuple(ReplicaUnit('content_chunk', 'chunk', c.chunk_id)
                      for c in ReplicaContentPlane(post, TICK, chunked=True).chunks.get(lid, ()))
    cross_lid = _cross_tenant_lid(query, metadata, pre)
    candidates = StateIsolatedEvidenceResolver().resolve(
        query.query_id, query.principal_id, TICK - 1,
        query.logical_document_ids, chunk, parent).hits(10)
    return {'pre': pre, 'post': post, 'event': event, 'updates': updates,
            'lid': lid, 'old': old, 'old_units': old_units, 'new_units': new_units,
            'chunk': chunk, 'parent': parent, 'acl': acl,
            'before_allowed': before_allowed, 'after_allowed': after_allowed,
            'changed_content': changed_content, 'cross_lid': cross_lid,
            'cacheable_candidates': bool(candidates), 'candidates': candidates}


def _transition_check(scenario, query, facts):
    transition = scenario['transition']
    old, event = facts['old'], facts['event']
    claim = any(c['logical_document_id'] == facts['lid']
                for c in query.required_atomic_claims)
    if transition == 'UPDATE':
        ok = old is not None and claim and facts['changed_content'] and event.event_type == 'DOCUMENT_UPDATE'
    elif transition == 'SUPERSEDE':
        ok = old is not None and event.event_type == 'DOCUMENT_SUPERSEDE' and \
             event.payload['supersedes'] == old.document_id and facts['changed_content']
    elif transition == 'DELETE':
        ok = old is not None and event.event_type == 'DOCUMENT_DELETE'
    elif transition == 'GRANT':
        ok = not facts['before_allowed'] and facts['after_allowed']
    elif transition == 'REVOKE':
        ok = facts['before_allowed'] and not facts['after_allowed']
    else:
        ok = False
    return {'transition': transition, 'effective': bool(ok),
            'pre_content_exists': old is not None,
            'pre_authorized': facts['before_allowed'],
            'post_authorized': facts['after_allowed'],
            'content_changed': facts['changed_content'],
            'event_type': event.event_type}


def _operator_check(name, scenario, query, facts):
    spec = OPERATORS[name]
    transition = scenario['transition']
    reasons = []
    if transition not in spec.required_transition_types:
        reasons.append('operator_transition_mismatch')
    if scenario['severity'] not in spec.valid_severities:
        reasons.append('invalid_severity')
    old_n, new_n = len(facts['old_units']), len(facts['new_units'])
    acl_n = len(facts['chunk'].chunks.get(facts['lid'], ()))
    unit_count = (new_n if transition in ('UPDATE', 'SUPERSEDE') else old_n) \
        if spec.replica_unit_type == 'chunk' else acl_n if spec.replica_unit_type == 'acl_entry' else 1
    updated = None
    if name in ('PARTIAL_CONTENT', 'PARTIAL_AUTH'):
        fraction = .5 if scenario.get('compound') == 'CF3' else scenario.get('partial_fraction')
        unit_count = acl_n if name == 'PARTIAL_AUTH' else (new_n if transition in ('UPDATE', 'SUPERSEDE') else old_n)
        updated = partial_updated_units(unit_count, fraction)
        if updated is None:
            reasons.append('partial_split_not_representable')
        if name == 'PARTIAL_CONTENT' and transition != 'DELETE' and old_n < 2:
            reasons.append('no_stale_chunk_can_remain')
    elif name == 'OUT_OF_ORDER':
        unit_count = 2 if facts['event'] and (facts['old'] or transition == 'REVOKE') else 0
        if unit_count < 2:
            reasons.append('fewer_than_two_ordered_events')
    elif name == 'STALE_INGESTION_REPLAY':
        unit_count = 2 if facts['old'] is not None else 0
        if unit_count < 2:
            reasons.append('prior_ingestion_event_absent')
    elif name in ('STALE_CACHE', 'CROSS_TENANT_CACHE'):
        unit_count = 1 if facts['cacheable_candidates'] else 0
        if not facts['cacheable_candidates']:
            reasons.append('cache_population_impossible')
        if name == 'CROSS_TENANT_CACHE' and facts['cross_lid'] is None:
            reasons.append('cross_tenant_candidate_absent')
    elif name == 'STALE_ACL_METADATA':
        unit_count = 1 if facts['cross_lid'] is not None else 0
        if facts['cross_lid'] is None:
            reasons.append('cross_tenant_acl_target_absent')
        if facts['cross_lid'] is not None and facts['pre'].allowed(
                query.principal_id, facts['cross_lid'], TICK - 1):
            reasons.append('canonical_cross_tenant_deny_absent')
    elif name == 'AUTH_REVOKE':
        unit_count = 1
        if not facts['before_allowed']:
            reasons.append('extra_revoke_would_be_noop')
    elif name in ('DELAY_CONTENT', 'DELAY_AUTH', 'DROPPED_CONTENT', 'DROPPED_AUTH', 'CONTENT_UPDATE'):
        if facts['event'] is None:
            reasons.append('target_event_absent')
    if name == 'DELAY_AUTH' and not (facts['before_allowed'] != facts['after_allowed']):
        reasons.append('authorization_change_absent')
    if name == 'CONTENT_UPDATE' and not facts['changed_content']:
        reasons.append('content_change_absent')
    if unit_count < spec.minimum_replica_units:
        reasons.append('insufficient_' + spec.replica_unit_type + '_units')
    return {'applicable': not reasons, 'reasons': reasons,
            'unit_count': unit_count, 'updated_unit_count': updated,
            'target_plane': spec.target_plane, 'unit_type': spec.replica_unit_type,
            'effectiveness_predicate': spec.effectiveness_predicate}


def _joint_check(scenario, components, facts, query):
    if not scenario.get('compound') or scenario['component'] != 'F1_F2':
        return {'applicable': True, 'reason': None}
    compound = scenario['compound']
    if not all(part['applicable'] for part in components.values()):
        return {'applicable': False, 'reason': 'component_precondition_failed'}
    # These checks encode co-existence, not just logical conjunction.
    checks = {
        'CF1': facts['old'] is not None and facts['cacheable_candidates'],
        'CF2': facts['before_allowed'] and facts['cacheable_candidates'],
        'CF3': components['F2']['updated_unit_count'] is not None and
               facts['before_allowed'] != facts['after_allowed'],
        'CF4': facts['old'] is not None and facts['event'].event_type == 'DOCUMENT_DELETE',
        'CF5': facts['changed_content'] and facts['before_allowed'],
        'CF6': facts['cross_lid'] is not None and facts['cacheable_candidates'] and
               not facts['pre'].allowed(query.principal_id, facts['cross_lid'], TICK - 1),
    }
    return {'applicable': bool(checks[compound]),
            'reason': None if checks[compound] else 'compound_components_cannot_coexist'}


def compile_candidate(scenario, query, facts, config):
    transition = _transition_check(scenario, query, facts)
    names = _operator_names(scenario)
    components = {component: _operator_check(operator, scenario, query, facts)
                  for component, operator in names}
    joint = _joint_check(scenario, components, facts, query)
    architecture = config['architecture_components'].get(scenario['architecture'])
    if scenario['architecture'] == 'ECA':
        e, c, a = scenario['component_tuple']
        architecture = {'event_driven_invalidation': e,
                        'epoch_cache_validation': c, 'live_authorization': a,
                        'periodic_sync': 0}
    runnable = architecture is not None
    cache_scenario = scenario['fault'] == 'STALE_CACHE' or any(
        name in ('STALE_CACHE', 'CROSS_TENANT_CACHE') for _, name in names)
    informative = {
        'event_driven_invalidation': bool(architecture and architecture['event_driven_invalidation'] and facts['event']),
        'epoch_cache_validation': bool(architecture and architecture['epoch_cache_validation'] and cache_scenario),
        'live_authorization': bool(architecture and architecture['live_authorization'] and
                                    (scenario['transition'] in AUTH or scenario['compound'] == 'CF5')),
    }
    reasons = []
    if not transition['effective'] and not (scenario.get('compound') == 'CF5' and
                                            scenario.get('component') == 'F2'):
        reasons.append('transition_precondition_failed')
    if not runnable:
        reasons.append('unknown_architecture')
    for component, result in components.items():
        reasons.extend(f'{component}:{reason}' for reason in result['reasons'])
    if not joint['applicable']:
        reasons.append('joint:' + joint['reason'])
    unit_counts = {component: result['unit_count'] for component, result in components.items()}
    updated = next((result['updated_unit_count'] for result in components.values()
                    if result['updated_unit_count'] is not None), None)
    return ApplicabilityResult(scenario['scenario_id'], not reasons, tuple(reasons),
        {'old_chunk_units': len(facts['old_units']), 'new_chunk_units': len(facts['new_units']),
         'cross_tenant_candidate_exists': facts['cross_lid'] is not None,
         'pretransition_candidate_cacheable': facts['cacheable_candidates'],
         'joint_applicability': joint},
        unit_counts, transition, components,
        {'runnable': runnable, 'informative': informative},
        tuple(result['target_plane'] for result in components.values()),
        tuple(result['effectiveness_predicate'] for result in components.values()),
        components, updated)


def dry_run(scenario, query, facts, result):
    """Apply only state transformations on disposable replicas; never evaluate metrics."""
    if not result.applicable:
        return {'scenario_id': scenario['scenario_id'], 'realizable': False,
                'reason': 'excluded_by_applicability', 'dry_run_performed': False}
    event = facts['event']
    chunk = ReplicaContentPlane(facts['pre'], TICK - 1, chunked=True)
    parent = ReplicaContentPlane(facts['pre'], TICK - 1, chunked=False)
    acl = ReplicaAuthorizationPlane(facts['pre'])
    cache = ReplicaEvidenceCache()
    lid = facts['lid']
    names = dict(_operator_names(scenario))
    active = set(names.values())
    checks = []
    if 'STALE_CACHE' in active or 'CROSS_TENANT_CACHE' in active:
        candidates = facts['candidates']
        if 'CROSS_TENANT_CACHE' in active:
            cross = facts['cross_lid']
            cross_chunk = chunk.chunks[cross][0]
            candidates = [RetrievalHit(cross_chunk, 0.0)] + candidates[:9]
        cache.put('candidate', candidates, query.principal_id, chunk.epoch,
                  parent.epoch, acl.epoch, tenant_id=query.tenant_id,
                  creation_time=TICK - 1, producer_architecture=scenario['architecture'],
                  source_state_hashes={'chunk': chunk.state_hash(),
                                       'parent': parent.state_hash(), 'auth': acl.state_hash(TICK-1)})
        checks.append(bool(cache.entries))
    if 'STALE_ACL_METADATA' in active:
        cross = facts['cross_lid']
        cross_chunk = chunk.chunks[cross][0]
        acl.chunk_overrides[(query.principal_id, cross_chunk.chunk_id)] = True
        checks.append(acl.allowed(query.principal_id, cross_chunk, TICK) and
                      not facts['pre'].allowed(query.principal_id, cross, TICK - 1))
    if 'PARTIAL_CONTENT' in active:
        fraction = scenario['partial_fraction']
        old_ids = {c.chunk_id for c in chunk.chunks[lid]}
        applied = chunk.apply(event, fraction=fraction)
        remaining = chunk.chunks.get(lid, [])
        stale = [c for c in remaining if c.chunk_id in old_ids]
        updated = [c for c in remaining if c.chunk_id not in old_ids]
        if scenario['transition'] == 'DELETE':
            checks.append(applied['applied_units'] == result.updated_unit_count and
                          0 < len(stale) < len(old_ids))
        else:
            checks.append(applied['applied_units'] == result.updated_unit_count and
                          bool(stale) and bool(updated))
    elif 'PARTIAL_AUTH' in active:
        units = [c for c in chunk.ordered_chunks() if c.logical_document_id == lid]
        fraction = .5 if scenario.get('compound') == 'CF3' else scenario['partial_fraction']
        applied = acl.apply(event, fraction=fraction, units=units)
        stale_count = sum((query.principal_id, c.chunk_id) in acl.chunk_overrides for c in units)
        checks.append(applied['applied_units'] == result.updated_unit_count and
                      stale_count == len(units) - result.updated_unit_count and stale_count > 0)
    elif 'DROPPED_CONTENT' in active or 'DROPPED_AUTH' in active:
        # A dropped target event leaves the initialized replica unchanged.
        before = (chunk.state_hash(), parent.state_hash(), acl.state_hash(TICK-1))
        after = (chunk.state_hash(), parent.state_hash(), acl.state_hash(TICK))
        checks.append(before == after)
    elif event is not None and not (scenario.get('compound') == 'CF5' and
                                    scenario.get('component') in ('F2', 'NONE')):
        before_chunk = tuple(c.chunk_id for c in chunk.chunks.get(lid, ()))
        if event.event_type.startswith('DOCUMENT_'):
            chunk.apply(event); parent.apply(event); acl.apply(event)
            checks.append(tuple(c.chunk_id for c in chunk.chunks.get(lid, ())) != before_chunk)
        else:
            units = [c for c in chunk.ordered_chunks() if c.logical_document_id == lid]
            before = tuple(acl.allowed(query.principal_id, c, TICK-1) for c in units)
            acl.apply(event, units=units)
            after = tuple(acl.allowed(query.principal_id, c, TICK) for c in units)
            checks.append(before != after)
    if 'OUT_OF_ORDER' in active or 'STALE_INGESTION_REPLAY' in active:
        if event.event_type.startswith('DOCUMENT_'):
            old = facts['old']
            replay = Transition(event.event_id + '-replay', TICK + 2,
                                'DOCUMENT_CREATE', lid, asdict(old))
            chunk.apply(replay); parent.apply(replay); acl.apply(replay)
            checks.append(any(c.document_id == old.document_id
                              for c in chunk.chunks.get(lid, ())))
        else:
            replay = Transition(event.event_id + '-allow-replay', TICK + 2,
                                'PERMISSION_GRANT', lid, event.payload)
            units = [c for c in chunk.ordered_chunks() if c.logical_document_id == lid]
            acl.apply(replay, units=units)
            checks.append(all(acl.allowed(query.principal_id, c, TICK + 2) for c in units))
    if 'AUTH_REVOKE' in active:
        revoke = Transition('dry-run-extra-revoke', TICK, 'PERMISSION_REVOKE', lid,
                            {'principal_id': query.principal_id, 'logical_document_id': lid})
        units = [c for c in chunk.ordered_chunks() if c.logical_document_id == lid]
        before = tuple(acl.allowed(query.principal_id, c, TICK-1) for c in units)
        acl.apply(revoke, units=units)
        after = tuple(acl.allowed(query.principal_id, c, TICK) for c in units)
        checks.append(before != after)
    return {'scenario_id': scenario['scenario_id'], 'realizable': all(checks),
            'reason': None if all(checks) else 'state_transformation_not_realizable',
            'dry_run_performed': True, 'checks': checks}


def _csv(path, rows, fields):
    with Path(path).open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(value, sort_keys=True)
                             if isinstance(value, (dict, list, tuple)) else value
                             for key, value in row.items()})


def _candidate_row(scenario, query, result):
    names = _operator_names(scenario)
    units = result.unit_counts
    return {'scenario_id': scenario['scenario_id'], 'query_id': scenario['query_id'],
            'domain': query.domain, 'query_family': query.family,
            'block': scenario['block'], 'transition': scenario['transition'],
            'fault': scenario['fault'], 'severity': scenario['severity'],
            'architecture': scenario['architecture'],
            'compound': scenario.get('compound'), 'component': scenario.get('component'),
            'component_operators': [name for _, name in names],
            'applicable': result.applicable,
            'exclusion_reason': ';'.join(result.reasons),
            'target_plane': list(result.expected_target_planes),
            'replica_unit_type': [OPERATORS[name].replica_unit_type for _, name in names],
            'replica_unit_count': units,
            'requested_partial_fraction': scenario.get('partial_fraction')
            if scenario['fault'] == 'PARTIAL' else .5 if scenario.get('compound') == 'CF3'
            and scenario.get('component') in ('F2', 'F1_F2') else None,
            'updated_unit_count': result.updated_unit_count,
            'transition_preconditions': result.transition_preconditions,
            'fault_preconditions': result.fault_preconditions,
            'architecture_preconditions': result.architecture_preconditions,
            'structural_facts': result.structural_facts,
            'effectiveness_predicates': result.effectiveness_predicates}


def _matrix(rows, key_fields):
    groups = defaultdict(list)
    for row in rows:
        groups[tuple(row[key] for key in key_fields)].append(row)
    return [{**dict(zip(key_fields, key)), 'considered': len(items),
             'applicable': sum(item['applicable'] for item in items),
             'excluded': sum(not item['applicable'] for item in items)}
            for key, items in sorted(groups.items(), key=lambda item: str(item[0]))]


def compile_all(output=OUTPUT, *, write=True):
    """Exhaustively inspect frozen candidates; no scientific outcome file is read."""
    config = json.loads(CONFIG.read_text())
    source = json.loads(SOURCE_MANIFEST.read_text())
    scenarios = source['scenarios']
    if digest(scenarios) != source['sha256']:
        raise ValueError('Frozen candidate source manifest hash mismatch')
    world, metadata, queries, _ = _load_frozen(config)
    by_id = {query.query_id: query for query in queries}
    fact_cache = {}
    rows, audits, emitted = [], [], []
    for scenario in scenarios:
        query = by_id[scenario['query_id']]
        key = (query.query_id, scenario['transition'])
        if key not in fact_cache:
            try:
                fact_cache[key] = _transition_facts(world, metadata, query,
                                                    scenario['transition'])
            except Exception as exc:
                fact_cache[key] = exc
        facts = fact_cache[key]
        if isinstance(facts, Exception):
            result = ApplicabilityResult(scenario['scenario_id'], False,
                ('transition_construction_failed:' + type(facts).__name__,),
                {}, {}, {'effective': False}, {}, {'runnable': False}, (), (), {}, None)
            audit = {'scenario_id': scenario['scenario_id'], 'realizable': False,
                     'reason': 'excluded_by_applicability', 'dry_run_performed': False}
        else:
            result = compile_candidate(scenario, query, facts, config)
            audit = dry_run(scenario, query, facts, result)
        row = _candidate_row(scenario, query, result)
        rows.append(row)
        audits.append(audit)
        if result.applicable and audit['realizable']:
            emitted.append(scenario)
    candidate_hash = digest(rows)
    excluded = [row for row in rows if not row['applicable']]
    failed_realizability = [audit for audit in audits if
                           next(row for row in rows if row['scenario_id'] == audit['scenario_id'])['applicable']
                           and not audit['realizable']]
    output = Path(output)
    if write:
        if output.exists() and any(output.iterdir()):
            raise ValueError('Applicability output already exists; refusing to overwrite')
        output.mkdir(parents=True, exist_ok=True)
        fields = list(rows[0])
        _csv(output / 'candidate_scenarios.csv', rows, fields)
        _csv(output / 'executable_scenarios.csv',
             [row for row, audit in zip(rows, audits) if row['applicable'] and audit['realizable']],
             fields)
        _csv(output / 'excluded_scenarios.csv', excluded, fields)
        matrix_fields = ['transition', 'fault', 'severity', 'domain', 'architecture']
        _csv(output / 'transition_fault_applicability.csv',
             _matrix(rows, matrix_fields), matrix_fields + ['considered', 'applicable', 'excluded'])
        compound_rows = [row for row in rows if row['compound']]
        compound_fields = ['compound', 'component', 'domain', 'architecture']
        _csv(output / 'compound_applicability.csv',
             _matrix(compound_rows, compound_fields),
             compound_fields + ['considered', 'applicable', 'excluded'])
        _csv(output / 'realizability_audit.csv', audits,
             ['scenario_id', 'realizable', 'reason', 'dry_run_performed', 'checks'])
    return {'rows': rows, 'audits': audits, 'emitted': emitted,
            'candidate_hash': candidate_hash, 'source_hash': source['sha256'],
            'failed_realizability': failed_realizability,
            'excluded': excluded, 'config': config, 'output': output}


def summarize(compiled):
    rows, emitted, audits = compiled['rows'], compiled['emitted'], compiled['audits']
    by_id = {row['scenario_id']: row for row in rows}
    executable = [by_id[s['scenario_id']] for s in emitted]
    cells = _matrix(executable, ['transition', 'domain', 'fault'])
    sizes = sorted(row['considered'] for row in cells)
    domain_transition = _matrix(executable, ['transition', 'domain'])
    partial = [row for row in executable if row['fault'] == 'PARTIAL']
    compounds = [row for row in executable if row['compound']]
    partial_cells = _matrix(partial, ['transition', 'requested_partial_fraction', 'domain'])
    partial_fraction_domains = {
        str(fraction): sorted({row['domain'] for row in partial
                               if row['requested_partial_fraction'] == fraction})
        for fraction in (.25, .5, .75)}
    exclusions = dict(Counter(reason for row in compiled['excluded']
                              for reason in row['exclusion_reason'].split(';') if reason))
    return {'considered': len(rows), 'applicable': len(executable),
            'excluded': len(compiled['excluded']),
            'realizability_failed': len(compiled['failed_realizability']),
            'exclusion_reasons': exclusions,
            'cells': {'count': len(sizes), 'minimum': sizes[0] if sizes else 0,
                      'median': sizes[len(sizes)//2] if sizes else 0,
                      'maximum': sizes[-1] if sizes else 0},
            'transition_domain': domain_transition,
            'partial': {'count': len(partial),
                        'by_transition': dict(Counter(row['transition'] for row in partial)),
                        'by_fraction': dict(Counter(str(row['requested_partial_fraction']) for row in partial)),
                        'fraction_domains': partial_fraction_domains,
                        'cells': partial_cells},
            'compound': {'count': len(compounds),
                         'by_compound': dict(Counter(row['compound'] for row in compounds))},
            'by_block': dict(Counter(row['block'] for row in executable)),
            'by_domain': dict(Counter(row['domain'] for row in executable)),
            'by_transition': dict(Counter(row['transition'] for row in executable))}


def operator_smoke(compiled):
    """Stratified state-operator execution without query evaluation or metrics."""
    config = compiled['config']
    world, metadata, queries, _ = _load_frozen(config)
    by_id = {q.query_id: q for q in queries}
    rows = {row['scenario_id']: row for row in compiled['rows']}
    strata = {}
    for scenario in compiled['emitted']:
        for _, operator in _operator_names(scenario):
            strata.setdefault(operator, scenario)
    records = []
    for operator, scenario in sorted(strata.items()):
        query = by_id[scenario['query_id']]
        facts = _transition_facts(world, metadata, query, scenario['transition'])
        result = compile_candidate(scenario, query, facts, config)
        actual = dry_run(scenario, query, facts, result)
        records.append({'operator': operator, 'scenario_id': scenario['scenario_id'],
                        'compiler_applicable': result.applicable,
                        'runtime_replica_realizable': actual['realizable'],
                        'updated_unit_count': result.updated_unit_count,
                        'state_transform_checks': actual['checks']})
    # The v3 stop case must remain excluded; a one-unit split is impossible.
    stopped = rows['TA2__v3-q-finance-01__DELETE__S2__A0__PARTIAL__0.25']
    agreement = (len(records) == len(OPERATORS) and
                 all(record['compiler_applicable'] and record['runtime_replica_realizable']
                     for record in records) and not stopped['applicable'])
    return {'agreement': agreement, 'operator_count': len(records),
            'records': records, 'one_unit_partial_excluded': not stopped['applicable'],
            'scope': 'actual replica/cache operators; scientific evaluator never called'}


def fingerprint(compiled):
    return {'candidate_table_sha256': compiled['candidate_hash'],
            'executable_manifest_sha256': digest(compiled['emitted']),
            'exclusions_sha256': digest(compiled['excluded']),
            'source_candidate_manifest_sha256': compiled['source_hash']}


def finalize(compiled, *, smoke=None, repeat=None):
    output = compiled['output']
    summary = summarize(compiled)
    manifest = {'version': POLICY_VERSION, 'status': 'APPLICABILITY_VALIDATION_ONLY',
        'source_candidate_manifest_sha256': compiled['source_hash'],
        'candidate_table_sha256': compiled['candidate_hash'],
        'executable_manifest_sha256': digest(compiled['emitted']),
        'executable_scenarios': compiled['emitted'],
        'operator_specs': {name: asdict(spec) for name, spec in OPERATORS.items()},
        'compound_components': COMPOUNDS,
        'partial_rounding': 'floor(N * requested_fraction); require 1 <= updated < N; no clamping',
        'corpus_sha256': compiled['config']['corpus_sha256'],
        'scientific_outcomes_consulted': False,
        'scientific_campaign_frozen': False}
    (output / 'applicability_manifest.json').write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + '\n')
    (output / 'coverage_summary.json').write_text(
        json.dumps(summary, indent=2, sort_keys=True) + '\n')
    emitted_ids = {s['scenario_id'] for s in compiled['emitted']}
    ap1 = all(row['applicable'] for row in compiled['rows'] if row['scenario_id'] in emitted_ids)
    ap2 = not compiled['failed_realizability'] and all(audit['realizable']
        for audit in compiled['audits'] if audit['scenario_id'] in emitted_ids)
    # Pilot H4 needs each partial severity represented by more than a single
    # domain and each transition/fraction cell to have at least two examples.
    # This is a minimal coverage rule, not a statistical power claim.
    partial_counts = Counter((row['transition'], row['requested_partial_fraction'])
                             for row in compiled['rows'] if row['scenario_id'] in emitted_ids
                             and row['fault'] == 'PARTIAL')
    partial_coverage = (all(len(summary['partial']['fraction_domains'][str(fraction)]) >= 2
                            for fraction in (.25, .5, .75))
                        and all(partial_counts[(transition, fraction)] >= 2
                                for transition in CONTENT + ('REVOKE',)
                                for fraction in (.25, .5, .75)))
    ap4 = (len(summary['by_transition']) == 5 and len(summary['by_domain']) == 6
           and len(summary['compound']['by_compound']) == 6 and partial_coverage)
    gates = {'AP1': 'PASS' if ap1 else 'FAIL',
             'AP2': 'PASS' if ap2 else 'FAIL',
             'AP3': 'PASS' if smoke and smoke['agreement'] else 'FAIL',
             'AP4': 'PASS' if ap4 else 'FAIL',
             'AP5': 'PASS' if repeat and repeat['identical'] else 'FAIL',
             'notes': {'AP4': 'Requires at least two domains per partial fraction and two cases per transition/fraction; 25% partial is undercovered.',
                       'AP3': 'Stratified state-operator smoke, no scientific metrics.'}}
    (output / 'gate_summary.json').write_text(json.dumps(gates, indent=2, sort_keys=True) + '\n')
    (output / 'summary.md').write_text(
        '# Scenario applicability v1\n\n' +
        f"Candidates: {summary['considered']}; applicable: {summary['applicable']}; "
        f"excluded: {summary['excluded']}.\n\n" +
        'Partial propagation uses floor(N × p) and excludes unsplittable targets. '
        'No v2/v3 scientific observations were read. No Track-A or Track-B campaign was run.\n\n' +
        '\n'.join(f'- {key}: {value}' for key, value in gates.items() if key.startswith('AP')) + '\n')
    return gates, summary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=str(OUTPUT))
    parser.add_argument('--fingerprint-only', action='store_true')
    parser.add_argument('--reassess-existing', action='store_true')
    args = parser.parse_args()
    if args.fingerprint_only:
        print(json.dumps(fingerprint(compile_all(write=False)), sort_keys=True))
        return
    if args.reassess_existing:
        compiled = compile_all(args.output, write=False)
        existing = json.loads((Path(args.output) / 'determinism_audit.json').read_text())
        if fingerprint(compiled) != existing['local']:
            raise ValueError('Existing applicability artifacts differ from recomputation')
        smoke = json.loads((Path(args.output) / 'runtime_agreement_smoke.json').read_text())
        gates, summary = finalize(compiled, smoke=smoke, repeat=existing)
        print(json.dumps({'gates': gates, 'summary': summary}, indent=2, sort_keys=True))
        return
    compiled = compile_all(args.output)
    smoke = operator_smoke(compiled)
    (compiled['output'] / 'runtime_agreement_smoke.json').write_text(
        json.dumps(smoke, indent=2, sort_keys=True) + '\n')
    env = dict(os.environ)
    env['PYTHONPATH'] = str(Path('src').resolve()) + os.pathsep + env.get('PYTHONPATH', '')
    fresh = subprocess.run([sys.executable, '-m',
        'ragstate.campaign.scenario_applicability', '--fingerprint-only'],
        capture_output=True, text=True, env=env)
    fresh_hashes = json.loads(fresh.stdout) if fresh.returncode == 0 else None
    local_hashes = fingerprint(compiled)
    repeat = {'identical': fresh_hashes == local_hashes,
              'fresh_process_exit': fresh.returncode,
              'local': local_hashes, 'fresh': fresh_hashes,
              'stderr': fresh.stderr}
    (compiled['output'] / 'determinism_audit.json').write_text(
        json.dumps(repeat, indent=2, sort_keys=True) + '\n')
    gates, summary = finalize(compiled, smoke=smoke, repeat=repeat)
    print(json.dumps({'gates': gates, 'summary': summary}, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
