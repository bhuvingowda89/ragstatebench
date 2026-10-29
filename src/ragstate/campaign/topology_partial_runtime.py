"""Apply an already frozen topology-aware PARTIAL plan to replica planes.

The operator consumes exact unit IDs from the campaign manifest. It never
selects units, consults gold claims, or reads canonical content at runtime.
"""
from __future__ import annotations

from ragstate.replicas.scientific import _document
from ragstate.retrieval.chunking import chunk_document
from .state_pilot_manifest import digest


PLAN_VERSION = 'topology_units_seeded_v1'


def verify_plan(scenario, event, principal, lid, chunk):
    if scenario.get('partial_plan_version') != PLAN_VERSION:
        raise ValueError('Unsupported frozen PARTIAL plan version')
    old = tuple(chunk.chunks.get(lid, ()))
    if event.event_type in ('DOCUMENT_UPDATE', 'DOCUMENT_SUPERSEDE'):
        new = tuple(chunk_document(_document(event.payload), max_words=160))
        if len(old) != len(new):
            raise ValueError('Frozen PARTIAL plan requires aligned old/new chunk units')
        unit_ids = tuple(part.chunk_id for part in new)
        expected_stale = tuple(old[index].chunk_id for index, part in enumerate(new)
                               if part.chunk_id not in scenario['selected_unit_ids'])
    else:
        new = ()
        unit_ids = tuple(part.chunk_id for part in old)
        expected_stale = tuple(part.chunk_id for part in old
                               if part.chunk_id not in scenario['selected_unit_ids'])
    selected = tuple(scenario['selected_unit_ids'])
    stale = tuple(scenario['selected_stale_unit_ids'])
    n = scenario['total_unit_count']
    if n != len(unit_ids) or len(set(unit_ids)) != n:
        raise ValueError('Frozen replica unit count does not match runtime topology')
    if not 1 <= len(selected) < n or len(selected) != scenario['updated_unit_count']:
        raise ValueError('Frozen PARTIAL plan is not a strict mixed state')
    if not set(selected) <= set(unit_ids) or stale != expected_stale:
        raise ValueError('Frozen selected/stale unit IDs differ from runtime replica')
    if scenario['transition'] in ('REVOKE', 'GRANT') and scenario['replica_unit_type'] != 'acl_entry':
        raise ValueError('Authorization PARTIAL plan targets wrong plane')
    if scenario['transition'] in ('UPDATE', 'SUPERSEDE', 'DELETE') and scenario['replica_unit_type'] != 'content_chunk':
        raise ValueError('Content PARTIAL plan targets wrong plane')
    if scenario['realized_fraction'] != len(selected) / n:
        raise ValueError('Frozen realized fraction differs from unit plan')
    return old, new, selected, stale


def apply_selected_partial(scenario, event, principal, lid, chunk, parent, indexed_auth):
    old, new, selected, stale = verify_plan(scenario, event, principal, lid, chunk)
    selected_set = set(selected)
    if scenario['replica_unit_type'] == 'content_chunk':
        if event.event_type == 'DOCUMENT_DELETE':
            chunk.chunks[lid] = [part for part in old if part.chunk_id not in selected_set]
            actual_stale = tuple(part.chunk_id for part in chunk.chunks[lid])
            actual_updated = tuple(part.chunk_id for part in old if part.chunk_id in selected_set)
            # The parent record remains stale until the remainder arrives.
        else:
            chunk.chunks[lid] = [new[index] if new[index].chunk_id in selected_set
                                 else old[index] for index in range(len(old))]
            actual_updated = tuple(part.chunk_id for part in chunk.chunks[lid]
                                   if part.version == new[0].version)
            actual_stale = tuple(part.chunk_id for part in chunk.chunks[lid]
                                 if part.version == old[0].version)
            parent.apply(event)
        chunk.epoch += 1
        chunk._state_hash = None
        indexed_auth.apply(event, units=chunk.chunks.get(lid, ()))
    else:
        old_allowed = {(principal, part.chunk_id):
                       indexed_auth.allowed(principal, part, event.logical_time)
                       for part in old}
        indexed_auth.policy.apply_authorization_event(event)
        for part in old:
            key = (principal, part.chunk_id)
            if part.chunk_id in selected_set:
                indexed_auth.chunk_overrides.pop(key, None)
            else:
                indexed_auth.chunk_overrides[key] = old_allowed[key]
        indexed_auth.epoch += 1
        indexed_auth._state_hash = None
        actual_updated = tuple(part.chunk_id for part in old
                               if (principal, part.chunk_id) not in indexed_auth.chunk_overrides)
        actual_stale = tuple(part.chunk_id for part in old
                             if (principal, part.chunk_id) in indexed_auth.chunk_overrides)
    if actual_updated != selected or actual_stale != stale:
        raise ValueError('Compiler/runtime selected-unit mismatch')
    return {'target_plane': scenario['replica_unit_type'],
            'planned_updated_unit_ids': list(selected),
            'planned_stale_unit_ids': list(stale),
            'actual_updated_unit_ids': list(actual_updated),
            'actual_stale_unit_ids': list(actual_stale),
            'updated_unit_count': len(actual_updated),
            'stale_unit_count': len(actual_stale),
            'selected_updated_ids_hash': digest(list(selected)),
            'event_id': event.event_id,
            'chunk_index_hash': chunk.state_hash(),
            'parent_index_hash': parent.state_hash(),
            'auth_metadata_hash': indexed_auth.state_hash(event.logical_time)}
