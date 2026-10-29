"""Focused causal-pilot safeguards before any scientific transition run."""
import json
from dataclasses import asdict
from pathlib import Path

from ragstate.campaign.state_pilot_manifest import build
from ragstate.campaign.state_pilot_runtime import (_stability, execute_track_a,
    prepare_prestate, transition_event)
from ragstate.replicas.scientific import ReplicaAuthorizationPlane, ReplicaContentPlane
from ragstate.transitions.events import Transition
from ragstate.world.ledger import WorldLedger
from ragstate.world.models import DocumentVersion, Principal


def _built():
    config = json.loads(Path('configs/state-isolated-pilot-v1.json').read_text())
    return config, build(config)


def test_grant_prestate_is_a_declared_same_tenant_direct_deny():
    config, built = _built()
    scenario = next(s for s in built['track_a'] if s['transition'] == 'GRANT')
    q = built['queries'][scenario['query_id']]
    pre = prepare_prestate(built['world'], q, 'GRANT')
    assert built['world'].allowed(q.principal_id, q.logical_document_ids[0], 9)
    assert not pre.allowed(q.principal_id, q.logical_document_ids[0], 9)
    event, _ = transition_event(pre, built['metadata'], q, 'GRANT')
    assert event.event_type == 'PERMISSION_GRANT'


def test_live_auth_blocks_revoked_context_but_candidate_exposure_remains_visible():
    config, built = _built()
    query = built['queries']['v3-q-engineering-01']
    scenario = {'scenario_id': 'test-revoke-A4', 'block': 'test',
                'transition': 'REVOKE', 'severity': 'S2', 'architecture': 'A4',
                'fault': 'DELAY', 'component_tuple': None, 'partial_fraction': None}
    rows = execute_track_a(scenario, built['world'], built['metadata'], query, config)
    early = next(row for row in rows if row['logical_time'] == 1)
    assert early['ucer_candidate'] > 0
    assert early['ucer_context'] == 0
    assert early['auth_validation_count'] > 0


def test_partial_delete_acts_on_real_chunk_units():
    world = WorldLedger()
    world.add_principal(Principal('p', 't'))
    doc = DocumentVersion('d-v1', 'd', 't', 1, 0, content='word ' * 640)
    world.add_document(doc)
    plane = ReplicaContentPlane(world, 9, chunked=True)
    delete = Transition('delete', 10, 'DOCUMENT_DELETE', 'd-v1',
                        {'logical_document_id': 'd'})
    effect = plane.apply(delete, fraction=.5)
    assert effect == {'applied_units': 2, 'total_units': 4}
    assert len(plane.ordered_chunks()) == 2
    plane.apply(delete)
    assert plane.ordered_chunks() == []


def test_stable_convergence_rejects_one_off_success_and_censors_persistent_failure():
    rows = [{'logical_time': tick, 'state_correct': value}
            for tick, value in [(0, 0), (1, 1), (2, 0), (3, 1), (4, 1)]]
    assert _stability(rows) == (3, False)
    assert _stability(rows[:-2]) == (None, True)


def test_replica_hashes_and_metrics_repeat_independently_of_wall_clock():
    config, built = _built()
    query = built['queries']['v3-q-engineering-01']
    scenario = {'scenario_id': 'repeat-update', 'block': 'test',
                'transition': 'UPDATE', 'severity': 'S2', 'architecture': 'A0',
                'fault': 'DELAY', 'component_tuple': None, 'partial_fraction': None}
    first = execute_track_a(scenario, built['world'], built['metadata'], query, config)
    second = execute_track_a(scenario, built['world'], built['metadata'], query, config)
    fields = ('logical_time', 'canonical_source_state_hash', 'chunk_index_hash',
              'parent_index_hash', 'auth_metadata_hash', 'context_hash',
              'scrs', 'cbr', 'ucer_context', 'dep', 'state_correct')
    assert [tuple(row[k] for k in fields) for row in first] == \
           [tuple(row[k] for k in fields) for row in second]
