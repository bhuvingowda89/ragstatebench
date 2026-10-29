"""Focused prospective selection and repeated-measures safeguards."""
import inspect
import json

from ragstate.campaign.partial_seed_robustness import SEEDS, build_pair, select_units, source_pairs
from ragstate.campaign.partial_seed_analysis import classify_query, unique_rows, clustered_ci


def test_selection_is_deterministic_and_identifier_only():
    units = ['a:c0', 'a:c1', 'a:c2', 'a:c3']
    selected = select_units(units, 2, SEEDS['seed_1'])
    assert selected == select_units(units, 2, SEEDS['seed_1'])
    assert set(selected) <= set(units) and len(selected) == 2
    assert select_units(units[::-1], 2, SEEDS['seed_1']) == selected[::-1]


def test_topology_and_horizon_are_preserved():
    _, pairs = source_pairs()
    for source in pairs[:12]:
        generated = build_pair(source, 'seed_2', SEEDS['seed_2'])
        assert generated[0]['selected_unit_ids'] == generated[1]['selected_unit_ids']
        assert generated[0]['h4_final_tick'] == generated[1]['h4_final_tick'] == 5
        assert generated[0]['h4_partial_tick'] == generated[1]['h4_partial_tick'] == 2
        assert generated[1]['total_unit_count'] == source[1]['total_unit_count']
        assert generated[1]['updated_unit_count'] == source[1]['updated_unit_count']
        assert generated[1]['partial_topology'] == source[1]['partial_topology']
        assert len(generated[1]['selected_unit_ids']) == source[1]['updated_unit_count']
        assert len(generated[1]['selected_stale_unit_ids']) == source[1]['total_unit_count']-source[1]['updated_unit_count']


def test_duplicate_subsets_do_not_multiply_query_replication():
    base = {'source_pair_id':'pair','query_id':'query','subset_hash':'same',
            'claim_failure_ticks_difference':-1,'mvcr_ticks_difference':3}
    rows = [dict(base, seed='seed_1'), dict(base, seed='seed_2'),
            dict(base, seed='seed_3', subset_hash='different')]
    assert len(unique_rows(rows)) == 2
    assert {r['query_id'] for r in unique_rows(rows)} == {'query'}


def test_robustness_query_categories():
    def row(claim, version):
        return {'claim_failure_ticks_difference':claim,'mvcr_ticks_difference':version}
    assert classify_query([row(-1,3),row(-1,3)]) == 'seed-stable benefit/tradeoff'
    assert classify_query([row(-1,3),row(0,3)]) == 'magnitude-sensitive'
    assert classify_query([row(-1,3),row(1,3)]) == 'sign-sensitive'
    assert classify_query([row(0,0),row(0,0)]) == 'no effect'


def test_selection_does_not_accept_relevance_or_position_inputs():
    assert tuple(inspect.signature(select_units).parameters) == ('unit_ids','k','seed')
    assert 'gold' not in inspect.getsource(select_units)
    assert 'claim' not in inspect.getsource(select_units)


def test_n2_duplicates_are_counted_by_subset_identity():
    units=['x:c0','x:c1']
    hashes={tuple(select_units(units,1,seed)) for seed in range(100)}
    assert len(hashes)==2


def test_bootstrap_clusters_repeated_subsets_by_query():
    rows=[{'query_id':'q1','effect':-1.0} for _ in range(4)] + [
        {'query_id':'q2','effect':1.0} for _ in range(4)]
    lo,hi=clustered_ci(rows,'effect')
    assert lo<=-1 and hi>=1


def test_frozen_manifest_and_reproduction_artifacts():
    from ragstate.campaign.partial_seed_robustness import ROOT, verify, stratified_pairs
    manifest=verify()
    assert len(manifest['scenarios'])==3216
    strata={(p[1]['ps_seed'],p[1]['transition'],p[1]['total_unit_count'])
            for p in stratified_pairs(manifest)}
    assert len(strata)==4*2*3
    pre=json.loads((ROOT/'preflight_gate_summary.json').read_text())
    assert all(pre['gates'].values())
