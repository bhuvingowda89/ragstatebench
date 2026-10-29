"""Declarative state-plane validity policy for the unchanged Track-A scenarios.

The checks inspect replica/canonical state and recorded event/cache operations.
They never use final SCRS, UCER, DEP, or other scientific outcome as a proxy
for successful fault injection.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass


POLICY_VERSION = 'v3'


@dataclass(frozen=True)
class EffectRule:
    target_plane: str
    required_effect: str
    predicate: str
    prohibited_false_requirement: str
    test_id: str


COMPOUND_RULES: dict[str, dict[str, EffectRule]] = {
    'CF1': {
        'F1': EffectRule('content_chunk_and_parent', 'supersession propagation lag',
                         'content_delay', 'cache hit', 'cf1_content_delay'),
        'F2': EffectRule('candidate_cache', 'pretransition candidate cached and replayed or invalidated',
                         'candidate_cache', 'stale content metric', 'cf1_candidate_cache'),
    },
    'CF2': {
        'F1': EffectRule('replica_acl', 'revocation propagation lag',
                         'auth_delay', 'cache hit', 'cf2_auth_delay'),
        'F2': EffectRule('candidate_cache', 'pretransition candidate cached and replayed or invalidated',
                         'candidate_cache', 'UCER outcome', 'cf2_candidate_cache'),
    },
    'CF3': {
        'F1': EffectRule('replica_acl', 'revocation propagation lag',
                         'auth_delay', 'cache hit', 'cf3_auth_delay'),
        'F2': EffectRule('replica_acl_units', 'some chunk ACL units stale after partial delivery',
                         'partial_acl', 'cache hit', 'cf3_partial_acl'),
    },
    'CF4': {
        'F1': EffectRule('content_chunk_and_parent', 'delete propagation lag',
                         'content_delay', 'cache hit', 'cf4_delete_delay'),
        'F2': EffectRule('content_chunk_parent_acl', 'old ingestion event replays after delete',
                         'stale_ingestion_replay', 'DEP outcome alone', 'cf4_replay'),
    },
    'CF5': {
        'F1': EffectRule('canonical_source_and_replica_content', 'content update event delivered',
                         'content_update', 'cache hit', 'cf5_content_update'),
        'F2': EffectRule('canonical_auth_and_replica_acl', 'permission revoke event delivered',
                         'auth_revoke', 'cache hit', 'cf5_auth_revoke'),
    },
    'CF6': {
        'F1': EffectRule('candidate_cache', 'cross-tenant same-topic candidate injected into C1',
                         'cross_tenant_cache', 'stale ACL override', 'cf6_cache_injection'),
        'F2': EffectRule('replica_acl_metadata', 'cross-tenant candidate has stale allow override',
                         'stale_acl_cross_tenant', 'cache hit', 'cf6_stale_acl'),
    },
}


ISOLATED_RULES: dict[str, EffectRule] = {
    'DELAY_CONTENT': EffectRule('content_chunk_and_parent',
        'replica content lags canonical event or event-driven mitigation refreshes immediately',
        'content_delay', 'SCRS failure alone', 'isolated_content_delay'),
    'DELAY_AUTH': EffectRule('replica_acl',
        'replica ACL lags canonical authorization or event-driven mitigation refreshes immediately',
        'auth_delay', 'UCER or ARDR alone', 'isolated_auth_delay'),
    'PARTIAL_CONTENT': EffectRule('replica_chunk_units',
        'configured fraction of real chunks updated while remainder stays stale',
        'partial_content', 'outcome-label flip', 'isolated_partial_content'),
    'PARTIAL_AUTH': EffectRule('replica_acl_units',
        'configured fraction of ACL chunk units updated while remainder stays stale',
        'partial_acl', 'cache hit', 'isolated_partial_acl'),
    'OUT_OF_ORDER': EffectRule('replica_content_or_acl',
        'older event applied after newer event with visible state reversal',
        'out_of_order', 'SERR flag alone', 'isolated_out_of_order'),
    'DROPPED_CONTENT': EffectRule('content_chunk_and_parent',
        'content event absent from replica through horizon',
        'dropped_content', 'censoring flag alone', 'isolated_dropped_content'),
    'DROPPED_AUTH': EffectRule('replica_acl',
        'authorization event absent from replica through horizon',
        'dropped_auth', 'censoring flag alone', 'isolated_dropped_auth'),
    'STALE_CACHE': EffectRule('candidate_cache',
        'pretransition candidate cached and replayed or invalidated',
        'candidate_cache', 'final context failure', 'isolated_candidate_cache'),
    'ACL_METADATA': EffectRule('replica_acl_metadata',
        'cross-tenant candidate has stale allow override',
        'stale_acl_cross_tenant', 'cache hit', 'isolated_acl_metadata'),
}


def compound_matrix():
    return [{'compound': compound, 'component': component,
             **asdict(rule)} for compound, parts in COMPOUND_RULES.items()
            for component, rule in parts.items()]


def isolated_matrix():
    return [{'fault': fault, **asdict(rule)} for fault, rule in ISOLATED_RULES.items()]


def _at(rows, tick):
    return next((row for row in rows if row['logical_time'] == tick), None)


def _target_lid(query):
    return query.logical_document_ids[0]


def _cross_tenant_chunk(query, metadata, world):
    origin = metadata[_target_lid(query)]
    matching = sorted(lid for lid, item in metadata.items()
                      if item['domain'] == origin['domain']
                      and item['topic_index'] == origin['topic_index']
                      and item['tenant_id'] != origin['tenant_id'])
    if not matching:
        return None
    doc = world.document_at(matching[0], 9)
    return f'{doc.document_id}:c0' if doc is not None else None


def _put_before(cache_ops, tick):
    return any(op['operation'] == 'put' and op['entry']
               and op['entry']['creation_time'] < tick for op in cache_ops)


def _replayed_or_invalidated(cache_ops, tick):
    return any((op['operation'] == 'get' and op.get('hit')
                and op.get('logical_time', -999) >= tick)
               or (op['operation'] == 'invalidate' and op['affected_keys'])
               for op in cache_ops)


def _content_delay(scenario, rows):
    pre, post = rows[0], _at(rows, 0)
    if pre['canonical_source_state_hash'] == post['canonical_source_state_hash']:
        return False, 'canonical content transition absent'
    if scenario['architecture'] in ('A2', 'A5'):
        ok = bool(post['replica_refreshes']) and (
            post['chunk_index_hash'] != pre['chunk_index_hash'] or
            post['parent_index_hash'] != pre['parent_index_hash'])
        return ok, 'event-driven refresh at transition' if ok else 'event-driven refresh absent'
    same = (post['chunk_index_hash'] == pre['chunk_index_hash'] and
            post['parent_index_hash'] == pre['parent_index_hash'])
    later = any(row['chunk_index_hash'] != pre['chunk_index_hash'] or
                row['parent_index_hash'] != pre['parent_index_hash'] for row in rows[2:])
    return same and later, 'stale at t0 then delivered' if same and later else 'content lag not observed'


def _auth_delay(scenario, rows):
    pre, post = rows[0], _at(rows, 0)
    if pre['canonical_authorization_state_hash'] == post['canonical_authorization_state_hash']:
        return False, 'canonical authorization transition absent'
    if scenario['architecture'] in ('A2', 'A5'):
        ok = bool(post['replica_refreshes']) and post['auth_metadata_hash'] != pre['auth_metadata_hash']
        return ok, 'event-driven ACL refresh' if ok else 'event-driven ACL refresh absent'
    same = post['auth_metadata_hash'] == pre['auth_metadata_hash']
    later = any(row['auth_metadata_hash'] != pre['auth_metadata_hash'] for row in rows[2:])
    return same and later, 'ACL stale at t0 then delivered' if same and later else 'ACL lag not observed'


def evaluate_predicate(predicate, scenario, rows, cache_ops, snapshots,
                       query, metadata, world):
    """Return (effective, state evidence), without reading scientific metrics."""
    tick0 = 10
    pre = rows[0]
    if predicate == 'content_delay':
        return _content_delay(scenario, rows)
    if predicate == 'auth_delay':
        return _auth_delay(scenario, rows)
    if predicate == 'candidate_cache':
        populated = _put_before(cache_ops, tick0)
        observed = _replayed_or_invalidated(cache_ops, tick0)
        return populated and observed, {'pretransition_put': populated,
                                        'posttransition_hit_or_invalidation': observed}
    if predicate == 'cross_tenant_cache':
        cross_id = _cross_tenant_chunk(query, metadata, world)
        injected = [op for op in cache_ops if op['operation'] == 'put'
                    and op['entry'] and op['entry']['creation_time'] == tick0
                    and cross_id in op['entry']['evidence_ids']]
        observed = _replayed_or_invalidated(cache_ops, tick0)
        return bool(injected and observed), {'cross_tenant_chunk_id': cross_id,
            'injected_at_transition': bool(injected),
            'hit_or_mitigation_invalidation': observed}
    if predicate == 'stale_acl_cross_tenant':
        cross_id = _cross_tenant_chunk(query, metadata, world)
        principal = query.principal_id
        target = (principal, cross_id, True)
        pre_acl = snapshots['auth'].get(pre['auth_metadata_hash'], {})
        observed = []
        for row in rows[1:]:
            auth = snapshots['auth'].get(row['auth_metadata_hash'], {})
            if target in auth.get('overrides', ()):
                observed.append(row['logical_time'])
        cross_tenant = cross_id is not None and metadata[
            next(lid for lid, item in metadata.items()
                 if world.document_at(lid, 9) is not None
                 and cross_id.startswith(world.document_at(lid, 9).document_id + ':'))
            ]['tenant_id'] != query.tenant_id
        ok = cross_tenant and target not in pre_acl.get('overrides', ()) and bool(observed)
        return ok, {'cross_tenant_chunk_id': cross_id,
                    'canonical_tenant_denies': cross_tenant,
                    'stale_allow_override_ticks': observed,
                    'cache_hit_not_required': True}
    if predicate == 'partial_acl':
        partial = []
        for row in rows[1:]:
            auth = snapshots['auth'].get(row['auth_metadata_hash'], {})
            target = [entry for entry in auth.get('overrides', ())
                      if entry[0] == query.principal_id]
            if target:
                partial.append((row['logical_time'], len(target)))
        final = snapshots['auth'].get(rows[-1]['auth_metadata_hash'], {})
        ok = bool(partial) and not [entry for entry in final.get('overrides', ())
                                    if entry[0] == query.principal_id]
        return ok, {'partial_acl_ticks_and_stale_units': partial,
                    'fully_propagated_at_final_tick': ok}
    if predicate == 'partial_content':
        lid = _target_lid(query)
        old = snapshots['chunk'].get(pre['chunk_index_hash'], {}).get(lid, ())
        fraction = scenario['partial_fraction']
        if fraction is None:
            return False, 'missing declared fraction'
        count = max(1, math.floor(len(old) * fraction))
        partial = []
        for row in rows[1:]:
            state = snapshots['chunk'].get(row['chunk_index_hash'], {}).get(lid, ())
            if scenario['transition'] == 'DELETE':
                changed = len(old) - len(state)
            else:
                changed = sum(version != old[0][1] for _, version in state)
            if changed == count and len(state) > 0:
                partial.append(row['logical_time'])
        return bool(partial), {'expected_updated_units': count,
                               'partial_state_ticks': partial,
                               'pretransition_units': len(old)}
    if predicate in ('out_of_order', 'stale_ingestion_replay'):
        replay = [row for row in rows if row['serr'] and row['logical_time'] > 0]
        reversed_state = any(row['chunk_index_hash'] != _at(rows, row['logical_time'] - 1)['chunk_index_hash']
                             or row['auth_metadata_hash'] != _at(rows, row['logical_time'] - 1)['auth_metadata_hash']
                             for row in replay if _at(rows, row['logical_time'] - 1))
        return bool(replay) and reversed_state, {
            'replay_ticks': [row['logical_time'] for row in replay],
            'replica_hash_reversal': reversed_state}
    if predicate in ('dropped_content', 'dropped_auth'):
        field = 'chunk_index_hash' if predicate == 'dropped_content' else 'auth_metadata_hash'
        absent = all(row[field] == pre[field] for row in rows)
        no_delivery = all(row['replica_refreshes'] == 0 for row in rows)
        return absent and no_delivery, {'target_replica_unchanged': absent,
                                        'delivery_count_zero': no_delivery}
    if predicate == 'content_update':
        changed = pre['canonical_source_state_hash'] != _at(rows, 0)['canonical_source_state_hash']
        delivered = any(row['chunk_index_hash'] != pre['chunk_index_hash'] for row in rows[1:])
        return changed and delivered, {'canonical_changed': changed,
                                       'replica_chunk_event_delivered': delivered}
    if predicate == 'auth_revoke':
        changed = pre['canonical_authorization_state_hash'] != _at(rows, 0)['canonical_authorization_state_hash']
        delivered = any(row['auth_metadata_hash'] != pre['auth_metadata_hash'] for row in rows[1:])
        return changed and delivered, {'canonical_changed': changed,
                                       'replica_acl_event_delivered': delivered}
    raise ValueError('Unknown validity predicate: ' + predicate)


def scenario_rules(scenario):
    compound = scenario.get('compound')
    component = scenario.get('component')
    if compound:
        return [(f'{compound}/{part}', COMPOUND_RULES[compound][part])
                for part in ('F1', 'F2')
                if component in (part, 'F1_F2')]
    fault = scenario['fault']
    transition = scenario['transition']
    if fault == 'DELAY':
        return [('DELAY_' + ('AUTH' if transition in ('GRANT', 'REVOKE') else 'CONTENT'),
                 ISOLATED_RULES['DELAY_' + ('AUTH' if transition in ('GRANT', 'REVOKE') else 'CONTENT')])]
    if fault == 'PARTIAL':
        key = 'PARTIAL_AUTH' if transition in ('GRANT', 'REVOKE') else 'PARTIAL_CONTENT'
        return [(key, ISOLATED_RULES[key])]
    if fault == 'DROPPED':
        key = 'DROPPED_AUTH' if transition in ('GRANT', 'REVOKE') else 'DROPPED_CONTENT'
        return [(key, ISOLATED_RULES[key])]
    if fault == 'OUT_OF_ORDER':
        return [('OUT_OF_ORDER', ISOLATED_RULES['OUT_OF_ORDER'])]
    if fault == 'STALE_CACHE':
        return [('STALE_CACHE', ISOLATED_RULES['STALE_CACHE'])]
    raise ValueError('Unmapped isolated fault: ' + fault)


def validate_components(scenario, rows, cache_ops, snapshots,
                        query, metadata, world):
    results = []
    for name, rule in scenario_rules(scenario):
        effective, evidence = evaluate_predicate(rule.predicate, scenario, rows,
                                                  cache_ops, snapshots, query,
                                                  metadata, world)
        results.append({'component': name, 'target_plane': rule.target_plane,
                        'predicate': rule.predicate, 'effective': bool(effective),
                        'evidence': evidence})
    return results
