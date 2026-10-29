"""Replica-only transition execution for the state-isolated causal track.

Oracle access is confined to scenario setup and evaluation. The retrieval
path receives logical IDs and materialized replica planes only.
"""
from __future__ import annotations

import copy
import hashlib
import json
import math
import re
import time
from dataclasses import asdict, replace

from ragstate.replicas.scientific import (ReplicaAuthorizationPlane,
    ReplicaContentPlane, ReplicaEvidenceCache)
from ragstate.retrieval.models import RetrievalHit
from ragstate.retrieval.metrics import (authorized_retrieval_denial,
    cross_tenant_exposure, current_evidence_recall, deleted_evidence_persistence,
    invalid_evidence_absence_success, mixed_version_context_rate,
    state_consistent_retrieval_success, superseded_retrieval_rate,
    unauthorized_context_rate)
from ragstate.retrieval.state_isolated import StateIsolatedEvidenceResolver
from ragstate.transitions.engine import TransitionEngine
from ragstate.transitions.events import Transition
from ragstate.world.models import PermissionRule
from .evidence_spans import claim_span, normalize
from .state_capabilities import (AccessAudit, AuthoritativeAuthorizationView,
                                  AuthorizationPolicyView, CanonicalEvaluationView)
from .state_pilot_manifest import digest


def transition_event(base_world, metadata, query, transition, tick=10):
    """Create a deterministic intervention using frozen claim metadata.

    Only the explicitly targeted claim changes for UPDATE/SUPERSEDE. No query
    wording, original gold, or corpus file is rewritten.
    """
    lid = query.logical_document_ids[0]
    old = base_world.document_at(lid, tick - 1)
    if old is None:
        raise ValueError('Transition target absent in prestate')
    if transition in ('UPDATE', 'SUPERSEDE'):
        claim = next((c for c in query.required_atomic_claims
                      if c['logical_document_id'] == lid), None)
        if claim is None:
            raise ValueError('Transition target has no required atomic claim')
        span = claim_span(old, metadata[lid], claim)
        old_sentence = span['span_text']
        value = str(claim['value'])
        new_sentence, count = re.subn(r'\b' + re.escape(value) + r'\b',
                                      str(int(value) + 7), old_sentence, count=1)
        if count != 1 or old.content.count(old_sentence) != 1:
            raise ValueError('Cannot deterministically change targeted claim')
        new_content = old.content.replace(old_sentence, new_sentence, 1)
        new = replace(old, document_id=f'{lid}-v{old.version + 1}',
                      version=old.version + 1, valid_from=tick, valid_until=None,
                      supersedes=old.document_id, deleted_at=None, content=new_content)
        kind = 'DOCUMENT_UPDATE' if transition == 'UPDATE' else 'DOCUMENT_SUPERSEDE'
        return Transition(f'{query.query_id}-{transition}', tick, kind,
                          lid, asdict(new)), {lid: {**claim, 'value': int(value) + 7,
                                                   'document_id': new.document_id}}
    if transition == 'DELETE':
        return Transition(f'{query.query_id}-DELETE', tick, 'DOCUMENT_DELETE',
                          old.document_id, {'logical_document_id': lid}), {}
    if transition in ('GRANT', 'REVOKE'):
        return Transition(f'{query.query_id}-{transition}', tick,
                          'PERMISSION_' + transition, lid,
                          {'principal_id': query.principal_id,
                           'logical_document_id': lid}), {}
    raise ValueError('Unsupported primary transition: ' + transition)


def prepare_prestate(base_world, query, transition, tick=10):
    world = copy.deepcopy(base_world)
    if transition == 'GRANT':
        # An applicable clean answerable query becomes a matched same-tenant
        # denied prestate, then receives a direct grant at the transition.
        world.add_permission(PermissionRule(query.principal_id,
                                            query.logical_document_ids[0],
                                            False, tick - 1))
    return world


def _gold(world, query, tick):
    return {doc.document_id for lid in query.logical_document_ids
            if (doc := world.document_at(lid, tick)) is not None
            and world.allowed(query.principal_id, lid, tick)}


def _claim_success(world, query, metadata, updates, context, tick):
    claims = []
    for claim in query.required_atomic_claims:
        lid = claim['logical_document_id']
        if not world.allowed(query.principal_id, lid, tick):
            continue
        doc = world.document_at(lid, tick)
        if doc is None:
            continue
        revised = updates.get(lid)
        effective = revised if revised and revised['document_id'] == doc.document_id else claim
        span = claim_span(doc, metadata[lid], effective)['span_text']
        claims.append(any(hit.chunk.document_id == doc.document_id
                          and normalize(span) in normalize(hit.chunk.text)
                          for hit in context))
    return (sum(claims) / len(claims) if claims else None,
            int(all(claims)) if claims else None)


def _hash_hits(hits):
    return digest([(h.chunk.chunk_id, h.chunk.text) for h in hits])


def _stability(rows):
    """Stable convergence is the first success sustained through final tick."""
    post = [r for r in rows if r['logical_time'] >= 0]
    valid = [bool(r['state_correct']) for r in post]
    for index, ok in enumerate(valid):
        if ok and all(valid[index:]):
            return post[index]['logical_time'], False
    return None, True


def execute_track_a(scenario, base_world, metadata, query, config, *,
                    pa2_retriever=None, audit_detail=False):
    tick0 = config['transition_absolute_tick']
    access_audit = AccessAudit(query.query_id)
    access_audit.record('scheduler', 'canonical_content', tick0 - 1,
                        'construct scenario prestate and event payload')
    pre = prepare_prestate(base_world, query, scenario['transition'], tick0)
    event, updates = transition_event(pre, metadata, query, scenario['transition'], tick0)
    compound = scenario.get('compound')
    component = scenario.get('component')
    f1 = component in ('F1', 'F1_F2')
    f2 = component in ('F2', 'F1_F2')
    extra_event = None
    if compound == 'CF5':
        revoke = Transition(f'{query.query_id}-CF5-REVOKE', tick0,
                            'PERMISSION_REVOKE', query.logical_document_ids[0],
                            {'principal_id': query.principal_id,
                             'logical_document_id': query.logical_document_ids[0]})
        if not f1:
            event, updates = (revoke, {}) if f2 else (None, {})
        elif f2:
            extra_event = revoke
    canonical = copy.deepcopy(pre)
    access_audit.record('scheduler', 'canonical_content', tick0 - 1,
                        'materialize initial content replicas')
    chunk = ReplicaContentPlane(pre, tick0 - 1, chunked=True)
    parent = ReplicaContentPlane(pre, tick0 - 1, chunked=False)
    access_audit.record('scheduler', 'canonical_authorization', tick0 - 1,
                        'materialize initial indexed authorization metadata')
    indexed_auth = ReplicaAuthorizationPlane(pre)
    access_audit.record('scheduler', 'canonical_content', tick0 - 1,
                        'derive baseline ACL metadata before experiment')
    access_audit.record('scheduler', 'canonical_authorization', tick0 - 1,
                        'initialize authoritative ACL snapshot')
    live_auth = AuthoritativeAuthorizationView(
        AuthorizationPolicyView.from_world(pre, tick0 - 1), access_audit)
    cache = ReplicaEvidenceCache()
    resolver = StateIsolatedEvidenceResolver()
    principal = query.principal_id
    lid = query.logical_document_ids[0]
    architecture = config['architecture_components'].get(scenario['architecture'], {})
    if scenario['architecture'] == 'ECA':
        e, c, a = scenario['component_tuple']
        architecture = {'event_driven_invalidation': e, 'periodic_sync': 0,
                        'epoch_cache_validation': c, 'live_authorization': a}
    delay = config['severity_ticks'][scenario['severity']]
    fault = scenario['fault']
    if compound in ('CF1', 'CF2', 'CF3', 'CF4') and not f1:
        delay = 0
    if compound == 'CF6':
        delay = 0
    event_driven = bool(architecture['event_driven_invalidation'])
    periodic = bool(architecture['periodic_sync'])
    # A successfully delivered invalidation carries event payload; no
    # authoritative content lookup occurs. Explicit DROPPED suppresses it.
    delivery = 0 if event_driven and fault != 'DROPPED' else delay
    if periodic:
        delivery = math.ceil((tick0 + delay) / config['periodic_sync_interval']) * \
                   config['periodic_sync_interval'] - tick0
    if fault == 'DROPPED':
        delivery = None
    fraction = scenario.get('partial_fraction') if fault == 'PARTIAL' else None
    if compound == 'CF3' and f2:
        fraction = .5
    cache_fault = fault == 'STALE_CACHE' or (compound in ('CF1', 'CF2') and f2) \
                  or (compound == 'CF6' and f1)
    if event is None:
        delivery = None
    cache_key = f'{principal}:{query.query_id}'
    def fresh_hits(absolute):
        if pa2_retriever is not None:
            return pa2_retriever.retrieve(query.text, chunk, parent)
        return resolver.resolve(query.query_id, principal, absolute,
                                query.logical_document_ids, chunk, parent).hits(config['top_k'])
    if cache_fault:
        pre_hits = fresh_hits(tick0 - 1)
        cache.put(cache_key, pre_hits, principal,
                  chunk.epoch, parent.epoch, indexed_auth.epoch,
                  tenant_id=query.tenant_id, creation_time=tick0 - 1,
                  producer_architecture=scenario['architecture'],
                  source_state_hashes={'chunk': chunk.state_hash(),
                                       'parent': parent.state_hash(),
                                       'auth': indexed_auth.state_hash(tick0 - 1)})
    cross_chunk = None
    if compound == 'CF6':
        origin = metadata[lid]
        matching = [other for other, facts in metadata.items()
                    if facts['domain'] == origin['domain']
                    and facts['topic_index'] == origin['topic_index']
                    and facts['tenant_id'] != origin['tenant_id']
                    and other in chunk.chunks]
        if not matching:
            raise ValueError('CF6 lacks a cross-tenant same-topic near duplicate')
        cross_chunk = chunk.chunks[sorted(matching)[0]][0]
    source_hash_pre = digest([(key, asdict(doc)) for key, doc in sorted(pre.documents.items())])
    auth_hash_pre = digest([asdict(rule) for rule in pre.permissions])
    source_hash_post = auth_hash_post = None
    rows = []
    applied = False
    completed_partial = False
    replayed = False
    recovered_replay = False
    replica_refreshes = 0
    for relative in config['observation_relative_ticks']:
        absolute = tick0 + relative
        if relative == 0:
            if event is not None:
                access_audit.record('scheduler',
                                    'canonical_authorization' if event.event_type.startswith(('PERMISSION_', 'GROUP_'))
                                    else 'canonical_content', absolute,
                                    'apply canonical transition')
                TransitionEngine(canonical).apply(event)
            if extra_event is not None:
                access_audit.record('scheduler', 'canonical_authorization', absolute,
                                    'apply compound authorization transition')
                TransitionEngine(canonical).apply(extra_event)
            access_audit.record('evaluator', 'canonical_content', absolute,
                                'hash canonical source state for trace')
            source_hash_post = digest([(key, asdict(doc))
                                       for key, doc in sorted(canonical.documents.items())])
            access_audit.record('evaluator', 'canonical_authorization', absolute,
                                'hash canonical authorization state for trace')
            auth_hash_post = digest([asdict(rule) for rule in canonical.permissions])
            # A4/A5 receive only authorization events. DELETE is content.
            if event is not None and event.event_type.startswith(('PERMISSION_', 'GROUP_')):
                live_auth.apply(event)
            if extra_event is not None:
                live_auth.apply(extra_event)
            if compound == 'CF6' and f2:
                indexed_auth.chunk_overrides[(principal, cross_chunk.chunk_id)] = True
                indexed_auth.epoch += 1
                indexed_auth._state_hash = None
            if compound == 'CF6' and f1:
                normal = fresh_hits(absolute)
                cache.put(cache_key, [RetrievalHit(cross_chunk, 0.0)] + normal[:9],
                          principal, chunk.epoch, parent.epoch, indexed_auth.epoch,
                          tenant_id=query.tenant_id, creation_time=absolute,
                          producer_architecture=scenario['architecture'],
                          source_state_hashes={'chunk': chunk.state_hash(),
                                               'parent': parent.state_hash(),
                                               'auth': indexed_auth.state_hash(absolute)})
        if delivery is not None and relative >= delivery and not applied:
            if fraction is not None and event.event_type.startswith('DOCUMENT_'):
                chunk.apply(event, fraction=fraction)
            else:
                chunk.apply(event)
            if not (fraction is not None and event.event_type == 'DOCUMENT_DELETE'):
                parent.apply(event)
            units = [c for c in chunk.ordered_chunks() if c.logical_document_id == lid]
            indexed_auth.apply(event, fraction=fraction or 1.0, units=units)
            if extra_event is not None:
                indexed_auth.apply(extra_event, units=units)
            applied = True
            replica_refreshes += 1
            if event_driven:
                cache.invalidate(cache_key)
        if fraction is not None and applied and not completed_partial \
                and relative >= delivery + 2:
            if event.event_type.startswith('DOCUMENT_'):
                chunk.apply(event)
            if event.event_type == 'DOCUMENT_DELETE':
                parent.apply(event)
            indexed_auth.complete_partial(principal, [c.chunk_id for c in chunk.ordered_chunks()
                                                      if c.logical_document_id == lid])
            completed_partial = True
            replica_refreshes += 1
        if (fault == 'OUT_OF_ORDER' or (compound == 'CF4' and f2)) and applied and not replayed \
                and relative >= delivery + 2 and event.event_type.startswith('DOCUMENT_'):
            old = pre.document_at(lid, tick0 - 1)
            if old is not None:
                old_event = Transition(event.event_id + '-replay', absolute,
                                       'DOCUMENT_CREATE', lid, asdict(old))
                chunk.apply(old_event)
                parent.apply(old_event)
                indexed_auth.apply(old_event)
                replayed = True
        if fault == 'OUT_OF_ORDER' and applied and not replayed \
                and relative >= delivery + 2 and event.event_type == 'PERMISSION_REVOKE':
            stale_allow = Transition(event.event_id + '-allow-replay', absolute,
                                     'PERMISSION_GRANT', lid, event.payload)
            indexed_auth.apply(stale_allow)
            replayed = True
        if replayed and not recovered_replay and relative >= delivery + 4:
            if event.event_type.startswith('DOCUMENT_'):
                chunk.apply(event)
                parent.apply(event)
                indexed_auth.apply(event)
            else:
                indexed_auth.apply(event)
            recovered_replay = True
        audit_start = len(access_audit.events)
        cache_audit_start = len(cache.stage_audit)
        started = time.perf_counter()
        resolution = (resolver.resolve(query.query_id, principal, absolute,
                                       query.logical_document_ids, chunk, parent)
                      if pa2_retriever is None else None)
        live_cached = cache.get(cache_key, principal,
                                tenant_id=query.tenant_id, logical_time=absolute,
                                epoch_aware=bool(architecture['epoch_cache_validation']),
                                chunk_epoch=chunk.epoch, parent_epoch=parent.epoch,
                                auth_epoch=indexed_auth.epoch) if cache_fault else None
        cache_hit = live_cached is not None
        candidates = ((fresh_hits(absolute) if resolution is None else
                       resolution.hits(config['top_k'])) if live_cached is None else
                      live_cached[:config['top_k']])
        if cache_fault and live_cached is None:
            cache.put(cache_key, candidates, principal,
                      chunk.epoch, parent.epoch, indexed_auth.epoch,
                      tenant_id=query.tenant_id, creation_time=absolute,
                      producer_architecture=scenario['architecture'],
                      source_state_hashes={'chunk': chunk.state_hash(),
                                           'parent': parent.state_hash(),
                                           'auth': indexed_auth.state_hash(absolute)})
        # C1 always re-enters B2 indexed authorization. It is never context.
        post = [hit for hit in candidates
                if indexed_auth.allowed(principal, hit.chunk, absolute)]
        auth_validation_count = 0
        if architecture['live_authorization']:
            context = []
            for hit in post:
                auth_validation_count += 1
                if live_auth.allowed(principal, hit.chunk.logical_document_id, absolute):
                    context.append(hit)
        else:
            context = post
        latency = (time.perf_counter() - started) * 1000
        evaluation = CanonicalEvaluationView(canonical, access_audit, absolute)
        gold = _gold(evaluation, query, absolute)
        scrs = int(state_consistent_retrieval_success(context, gold, principal,
                                                      evaluation, absolute))
        cbr, all_claims = _claim_success(evaluation, query, metadata, updates,
                                          context, absolute)
        absence = invalid_evidence_absence_success(context, gold,
                                                   set(query.logical_document_ids))
        ucer_candidate = unauthorized_context_rate(candidates, principal, evaluation, absolute)
        ucer_post = unauthorized_context_rate(post, principal, evaluation, absolute)
        ucer_context = unauthorized_context_rate(context, principal, evaluation, absolute)
        dep = int(deleted_evidence_persistence(context, evaluation, absolute))
        srr = superseded_retrieval_rate(context, gold, evaluation, absolute)
        cter = int(cross_tenant_exposure(context, principal, evaluation))
        mvcr = mixed_version_context_rate(context)
        correct = bool(scrs and (all_claims == 1 if gold else absence == 1)
                       and not (ucer_context or dep or srr or cter or mvcr))
        access_audit.assert_clean()
        if any(not item['valid'] for item in cache.stage_audit[cache_audit_start:]):
            raise ValueError('Cache stage audit failed')
        rows.append({'campaign_id': config['campaign_version'],
                     'scenario_id': scenario['scenario_id'],
                     'track': 'B' if pa2_retriever is not None else 'A',
                     'block': scenario['block'], 'query_id': query.query_id,
                     'domain': query.domain, 'query_family': query.family,
                     'principal_id': principal, 'tenant_id': query.tenant_id,
                     'logical_time': relative, 'transition': scenario['transition'],
                     'fault': fault, 'severity': scenario['severity'],
                     'architecture': scenario['architecture'],
                     'retriever': 'PA2_hybrid' if pa2_retriever is not None else 'state_isolated',
                     'oracle_logical_evidence_ids': list(query.logical_document_ids),
                     'replica_resolved_evidence_ids': [h.chunk.document_id for h in candidates],
                     'replica_resolved_versions': [h.chunk.version for h in candidates],
                     'candidate_chunk_ids': [h.chunk.chunk_id for h in candidates],
                     'candidate_cache_input_chunk_ids': [h.chunk.chunk_id for h in live_cached]
                     if live_cached is not None else None,
                     'candidate_cache_output_chunk_ids': [h.chunk.chunk_id for h in candidates]
                     if cache_hit else None,
                     'indexed_authorization_result': [h.chunk.chunk_id for h in post],
                     'live_authorization_result': [h.chunk.chunk_id for h in context]
                     if architecture['live_authorization'] else None,
                     'post_filter_chunk_ids': [h.chunk.chunk_id for h in post],
                     'context_chunk_ids': [h.chunk.chunk_id for h in context],
                     'canonical_source_state_hash': source_hash_pre if relative < 0 else source_hash_post,
                     'canonical_authorization_state_hash': auth_hash_pre if relative < 0 else auth_hash_post,
                     'chunk_index_hash': chunk.state_hash(),
                     'parent_index_hash': parent.state_hash(),
                     'auth_metadata_hash': indexed_auth.state_hash(absolute),
                     'cache_hash': cache.state_hash(), 'context_hash': _hash_hits(context),
                     'scrs': scrs,
                     'cer_at_10': current_evidence_recall(context, gold)
                     if pa2_retriever is not None else None,
                     'cbr': cbr,
                     'all_claims_success': all_claims,
                     'srr': srr, 'dep': dep, 'mvcr': mvcr,
                     'ucer_candidate': ucer_candidate,
                     'ucer_post_filter': ucer_post,
                     'ucer_context': ucer_context,
                     'ardr': int(authorized_retrieval_denial(gold, context)),
                     'cter': cter,
                     'serr': int(replayed and not recovered_replay),
                     'invalid_evidence_absence_success': absence,
                     'cache_hit': int(cache_hit),
                     'cache_type': 'C1_candidate' if cache_fault else 'C0_none',
                     'cache_stage_violation_count': sum(not item['valid'] for item in cache.stage_audit),
                     'canonical_forbidden_access_count': access_audit.forbidden_count,
                     'authoritative_access_count': len(access_audit.events) - audit_start,
                     'authoritative_access_events': access_audit.events[audit_start:]
                     if audit_detail else None,
                     'cache_stage_events': cache.stage_audit[cache_audit_start:]
                     if audit_detail else None,
                     'cache_invalidations': cache.invalidations,
                     'replica_refreshes': replica_refreshes,
                     'auth_validation_count': auth_validation_count,
                     'retrieval_latency_ms': latency,
                     'parent_latency_ms': pa2_retriever.last_parent_latency_ms
                     if pa2_retriever is not None and not cache_hit else 0.0,
                     'index_converged': int(applied and (fraction is None or completed_partial)
                                            and (not replayed or recovered_replay)),
                     'state_correct': int(correct), 'convergence_status': None})
    convergence, censored = _stability(rows)
    for row in rows:
        row['convergence_status'] = 'censored' if censored else 'converged'
        row['t_rsc'] = convergence
    return rows
