"""Regression cases for the invalid v1 authorization and cache boundaries."""
import json
from pathlib import Path

import pytest

from ragstate.campaign.state_capabilities import (
    AccessAudit, AuthoritativeAuthorizationView, AuthorizationPolicyView,
    ForbiddenCanonicalAccess,
)
from ragstate.campaign.state_pilot_manifest import build
from ragstate.campaign.state_pilot_runtime import execute_track_a
from ragstate.replicas.scientific import ReplicaEvidenceCache
from ragstate.retrieval.models import Chunk, RetrievalHit


def _fixture():
    config = json.loads(Path('configs/state-isolated-pilot-v1.json').read_text())
    built = build(config)
    return config, built, built['queries']['v3-q-engineering-01']


def _rows(transition, architecture, fault='DELAY', audit_detail=False):
    config, built, query = _fixture()
    scenario = {'scenario_id': f'test-{transition}-{architecture}-{fault}',
                'block': 'boundary_test', 'transition': transition,
                'severity': 'S2', 'architecture': architecture, 'fault': fault,
                'component': None, 'compound': None,
                'component_tuple': None, 'partial_fraction': None}
    return {row['logical_time']: row for row in execute_track_a(
        scenario, built['world'], built['metadata'], query, config,
        audit_detail=audit_detail)}


def test_a4_delete_keeps_stale_authorized_content_for_dep_not_ucer():
    rows = _rows('DELETE', 'A4', audit_detail=True)
    early = rows[1]
    assert early['candidate_chunk_ids'] == early['post_filter_chunk_ids']
    assert early['post_filter_chunk_ids'] == early['context_chunk_ids']
    assert early['dep'] == 1 and early['ucer_context'] == 0
    assert early['scrs'] == 0 and early['canonical_forbidden_access_count'] == 0
    assert all(event['accessed_plane'] != 'canonical_content'
               for event in early['authoritative_access_events']
               if event['component'] == 'live_authorization')


def test_a4_revoke_blocks_at_context_only_and_never_creates_dep():
    early = _rows('REVOKE', 'A4')[1]
    assert early['ucer_candidate'] > 0 and early['ucer_post_filter'] > 0
    assert early['ucer_context'] == 0 and early['dep'] == 0
    assert early['candidate_chunk_ids'] and not early['context_chunk_ids']


@pytest.mark.parametrize('transition', ['UPDATE', 'SUPERSEDE'])
def test_a4_does_not_suppress_or_replace_stale_content(transition):
    early = _rows(transition, 'A4')[1]
    assert early['context_chunk_ids'] == early['post_filter_chunk_ids']
    assert early['context_chunk_ids']
    assert early['scrs'] == 0 and early['srr'] > 0
    assert early['ucer_context'] == 0 and early['dep'] == 0


def test_candidate_cache_stage_guard_and_metadata():
    cache = ReplicaEvidenceCache()
    hit = RetrievalHit(Chunk('d-v1:c0', 'd-v1', 'd', 1, 't', 0, 'text'), .75,
                       (('bm25', 1, 3.0),))
    cache.put('q', [hit], 'p', 1, 2, 3, tenant_id='t', creation_time=9,
              producer_architecture='A0', source_state_hashes={'chunk': 'x'})
    entry = cache.entries['q']
    assert (entry.cache_type, entry.pipeline_stage) == ('C1_candidate', 'candidate_retrieval')
    assert entry.evidence_ids == ('d-v1:c0',) and entry.versions == (1,)
    assert entry.principal_id == 'p' and entry.tenant_id == 't'
    assert entry.creation_time == 9 and entry.producer_architecture == 'A0'
    assert cache.get('q', 'p', tenant_id='t', logical_time=10,
                     epoch_aware=False, chunk_epoch=1, parent_epoch=2,
                     auth_epoch=3) == [hit]
    with pytest.raises(ValueError, match='cannot be consumed'):
        cache.get('q', 'p', tenant_id='t', logical_time=10,
                  consumer_stage='final_context', epoch_aware=False,
                  chunk_epoch=1, parent_epoch=2, auth_epoch=3)


def test_revoke_candidate_cache_rechecks_indexed_acl_after_convergence():
    for architecture in ('A0', 'A3'):
        rows = _rows('REVOKE', architecture, 'STALE_CACHE', audit_detail=True)
        early, late = rows[1], rows[6]
        assert early['cache_hit'] == 1 and early['ucer_context'] > 0
        assert late['index_converged'] == 1
        assert late['cache_hit'] == 1 and late['candidate_chunk_ids']
        assert late['post_filter_chunk_ids'] == []
        assert late['context_chunk_ids'] == [] and late['ucer_context'] == 0
        assert late['cache_stage_violation_count'] == 0


@pytest.mark.parametrize('architecture', ['A4', 'A5'])
def test_live_auth_blocks_replayed_revoked_candidates(architecture):
    rows = _rows('REVOKE', architecture, 'STALE_CACHE')
    early = rows[1]
    assert early['ucer_context'] == 0
    assert early['canonical_forbidden_access_count'] == 0


def test_delete_candidate_cache_tracks_dep_without_authorization_contamination():
    early = _rows('DELETE', 'A0', 'STALE_CACHE')[1]
    late = _rows('DELETE', 'A0', 'STALE_CACHE')[6]
    assert early['cache_hit'] == 1 and early['dep'] == 1
    assert early['ucer_context'] == 0
    assert late['cache_hit'] == 1 and late['dep'] == 1
    assert late['ucer_context'] == 0


def test_grant_candidate_cache_cannot_pull_canonical_content():
    rows = _rows('GRANT', 'A3', 'STALE_CACHE')
    assert rows[1]['cache_hit'] == 1 and rows[1]['ardr'] == 1
    assert rows[1]['context_chunk_ids'] == []
    assert rows[6]['ardr'] == 0 and rows[6]['scrs'] == 1


def test_authoritative_view_exposes_no_content_lifecycle_and_audit_rejects_it():
    _, built, query = _fixture()
    audit = AccessAudit(query.query_id)
    view = AuthoritativeAuthorizationView(
        AuthorizationPolicyView.from_world(built['world'], 9), audit)
    assert view.allowed(query.principal_id, query.logical_document_ids[0], 9)
    assert not hasattr(view, 'document_at')
    assert not hasattr(view, 'documents')
    with pytest.raises(ForbiddenCanonicalAccess):
        audit.record('live_authorization', 'canonical_content', 10,
                     'forbidden content lookup')


def test_a5_has_only_event_epoch_and_auth_capabilities():
    rows = _rows('DELETE', 'A5', 'STALE_CACHE', audit_detail=True)
    assert rows[0]['canonical_forbidden_access_count'] == 0
    assert rows[1]['dep'] == 0  # delivered replica delete, not live content lookup
    for row in rows.values():
        assert row['canonical_forbidden_access_count'] == 0
        assert row['cache_stage_violation_count'] == 0
        assert all(event['accessed_plane'] == 'canonical_authorization'
                   for event in row['authoritative_access_events']
                   if event['component'] == 'live_authorization')


def test_transition_metric_orthogonality_across_all_primary_types():
    delete, revoke = _rows('DELETE', 'A0')[1], _rows('REVOKE', 'A0')[1]
    supersede, grant = _rows('SUPERSEDE', 'A0')[1], _rows('GRANT', 'A0')[1]
    update = _rows('UPDATE', 'A0')[1]
    assert delete['dep'] == 1 and delete['ucer_context'] == 0
    assert revoke['ucer_context'] > 0 and revoke['dep'] == 0
    assert supersede['srr'] > 0 and supersede['dep'] == 0
    assert grant['ardr'] == 1 and grant['ucer_context'] == 0
    assert update['scrs'] == 0 and update['ucer_context'] == 0
