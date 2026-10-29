"""Regression coverage for the v3 execution-control policy only."""
import json
from pathlib import Path

import pytest

from ragstate.campaign.state_pilot_manifest import _load_frozen
from ragstate.campaign.state_pilot_v3_campaign import recorded_with_state, validate_scenario
from ragstate.campaign.state_pilot_validity_v3 import (
    COMPOUND_RULES, ISOLATED_RULES, compound_matrix, isolated_matrix,
    validate_components,
)


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def frozen():
    config = json.loads((ROOT / 'configs/state-isolated-pilot-v2.json').read_text())
    config['campaign_version'] = 'state-isolated-pilot-v3'
    world, metadata, queries, _ = _load_frozen(config)
    scenarios = json.loads((ROOT / 'results/state-isolated-pilot-v2/scenario_manifest.json').read_text())['scenarios']
    return config, world, metadata, {q.query_id: q for q in queries}, scenarios


def run_one(frozen, compound, arm):
    config, world, metadata, queries, scenarios = frozen
    scenario = next(s for s in scenarios if s['compound'] == compound
                    and s['component'] == arm and s['architecture'] == 'A0')
    scenario = dict(scenario, campaign_version=config['campaign_version'])
    query = queries[scenario['query_id']]
    rows, accesses, cache_ops, snapshots = recorded_with_state(
        scenario, world, metadata, query, config)
    validity = validate_scenario(scenario, rows, accesses, cache_ops,
                                 snapshots, query, metadata, world, config)
    return scenario, rows, accesses, cache_ops, snapshots, query, validity


def test_compound_component_matrix_has_target_plane_and_distinct_cf6_requirements():
    rows = compound_matrix()
    assert len(rows) == 12
    assert {(r['compound'], r['component']) for r in rows} == {
        (f'CF{i}', arm) for i in range(1, 7) for arm in ('F1', 'F2')}
    assert all(r['target_plane'] and r['required_effect'] and r['test_id'] for r in rows)
    assert COMPOUND_RULES['CF6']['F1'].predicate == 'cross_tenant_cache'
    assert COMPOUND_RULES['CF6']['F2'].predicate == 'stale_acl_cross_tenant'
    assert COMPOUND_RULES['CF6']['F2'].prohibited_false_requirement == 'cache hit'


def test_isolated_faults_map_to_target_replica_planes():
    rows = isolated_matrix()
    assert len(rows) == 9
    assert {r['fault'] for r in rows} == set(ISOLATED_RULES)
    assert ISOLATED_RULES['PARTIAL_CONTENT'].target_plane == 'replica_chunk_units'
    assert ISOLATED_RULES['DROPPED_AUTH'].target_plane == 'replica_acl'


@pytest.mark.parametrize('arm,expected', [('F1', ('CF6/F1',)),
                                          ('F2', ('CF6/F2',)),
                                          ('F1_F2', ('CF6/F1', 'CF6/F2'))])
def test_cf6_arms_validate_only_their_declared_components(frozen, arm, expected):
    _, _, _, cache_ops, _, _, validity = run_one(frozen, 'CF6', arm)
    assert validity['validity'] == 'valid'
    assert tuple(item['component'] for item in validity['component_evidence']) == expected
    if arm == 'F2':
        assert not any(op['operation'] == 'get' and op.get('hit') for op in cache_ops)


def test_cache_hit_does_not_validate_stale_acl(frozen):
    scenario, rows, _, cache_ops, snapshots, query, validity = run_one(frozen, 'CF6', 'F1_F2')
    assert validity['validity'] == 'valid'
    snapshots['auth'] = {key: {'overrides': (), 'epoch': value['epoch']}
                         for key, value in snapshots['auth'].items()}
    evidence = validate_components(scenario, rows, cache_ops, snapshots,
                                   query, frozen[2], frozen[1])
    assert evidence[0]['effective']
    assert not evidence[1]['effective']


def test_stale_acl_without_cache_validates_cf6_f2(frozen):
    _, _, _, cache_ops, _, _, validity = run_one(frozen, 'CF6', 'F2')
    assert not any(op['operation'] == 'get' and op.get('hit') for op in cache_ops)
    assert validity['component_evidence'][0]['effective']


@pytest.mark.parametrize('compound', [f'CF{i}' for i in range(1, 7)])
def test_all_compound_component_arms_effective_on_real_replica_state(frozen, compound):
    for arm in ('F1', 'F2', 'F1_F2'):
        *_, validity = run_one(frozen, compound, arm)
        assert validity['validity'] == 'valid', (compound, arm, validity['reasons'])
        assert all(part['effective'] for part in validity['component_evidence'])
