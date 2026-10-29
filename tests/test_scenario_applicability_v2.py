"""Focused topology-aware PARTIAL design tests; no scientific outcomes."""
import inspect
import json
from pathlib import Path

import pytest

from ragstate.campaign.scenario_applicability import _transition_facts
from ragstate.campaign.scenario_applicability_v2 import (
    _make_partial_candidates, _uniform_control_id, apply_topology_partial,
    compile_topology_partial, coverage_gate, h4_policy, midpoint_updated,
    selected_unit_ids, topology_states,
)
from ragstate.campaign.state_pilot_manifest import _load_frozen


ROOT = Path(__file__).resolve().parents[1]


def test_topology_state_deduplication_and_midpoint():
    assert topology_states(1) == ()
    assert topology_states(2) == (('P_LOW', 1),)
    assert topology_states(3) == (('P_LOW', 1), ('P_HIGH', 2))
    assert topology_states(4) == (('P_LOW', 1), ('P_MID', 2), ('P_HIGH', 3))
    assert topology_states(5) == (('P_LOW', 1), ('P_MID', 3), ('P_HIGH', 4))
    for n in range(2, 11):
        values = [updated for _, updated in topology_states(n)]
        assert len(values) == len(set(values))
        assert all(1 <= updated < n for updated in values)


@pytest.mark.parametrize('n,expected', [(1, 1), (2, 1), (3, 2),
                                         (4, 2), (5, 3), (6, 3), (7, 4)])
def test_round_half_up_midpoint(n, expected):
    assert midpoint_updated(n) == expected


def test_stable_selection_is_nested_relevance_blind_and_not_first_chunk():
    ids = tuple(f'doc-v2:c{i}' for i in range(8))
    kwargs = {'seed': 505, 'query_id': 'q1', 'transition': 'UPDATE',
              'severity': 'S2', 'target_plane': 'content_chunk'}
    low = selected_unit_ids(ids, 1, **kwargs)
    high = selected_unit_ids(ids, 7, **kwargs)
    assert low == selected_unit_ids(ids, 1, **kwargs)
    assert set(low) <= set(high)
    assert len(high) == 7
    assert 'gold' not in inspect.signature(selected_unit_ids).parameters
    assert 'claim' not in inspect.signature(selected_unit_ids).parameters


@pytest.fixture(scope='module')
def candidates():
    config = json.loads((ROOT / 'configs/state-isolated-pilot-v2.json').read_text())
    world, metadata, queries, _ = _load_frozen(config)
    by_id = {query.query_id: query for query in queries}
    source = json.loads((ROOT / 'results/state-isolated-pilot-v3/scenario_manifest.json').read_text())['scenarios']
    return config, world, metadata, by_id, _make_partial_candidates(
        source, world, metadata, by_id, config), source


def test_one_chunk_target_excluded_and_all_emitted_plans_have_controls(candidates):
    config, world, metadata, by_id, generated, source = candidates
    target = next(s for s in generated if s['query_id'] == 'v3-q-finance-01'
                  and s['transition'] == 'DELETE')
    facts = _transition_facts(world, metadata, by_id[target['query_id']], 'DELETE')
    result = compile_topology_partial(target, by_id[target['query_id']], facts, config)
    assert target['total_unit_count'] == 1
    assert not result.applicable
    assert 'partial_split_not_representable' in result.reasons
    controls = {s['scenario_id'] for s in source if s['block'] == 'TA1'}
    assert all(s['matched_uniform_scenario_id'] == _uniform_control_id(s)
               and s['matched_uniform_scenario_id'] in controls for s in generated)


def test_realized_fraction_and_plane_specific_state_operator(candidates):
    config, world, metadata, by_id, generated, _ = candidates
    for transition in ('DELETE', 'REVOKE'):
        scenario = next(s for s in generated if s['transition'] == transition
                        and s['total_unit_count'] >= 3
                        and s['partial_topology'] == 'P_HIGH')
        query = by_id[scenario['query_id']]
        facts = _transition_facts(world, metadata, query, transition)
        plan = compile_topology_partial(scenario, query, facts, config)
        actual = apply_topology_partial(scenario, query, facts)
        assert plan.applicable
        assert actual['updated_units'] == scenario['updated_unit_count']
        assert actual['stale_units'] == scenario['total_unit_count'] - scenario['updated_unit_count']
        assert actual['realized_fraction'] == scenario['updated_unit_count'] / scenario['total_unit_count']


def test_cross_domain_coverage_rule_rejects_sparse_partial_cells():
    cells = [{'transition': transition, 'domain': domain,
              'matched_distinct_queries': 5} for transition in
             ('UPDATE', 'SUPERSEDE', 'DELETE') for domain in ('hr', 'finance',
             'engineering', 'legal')]
    summary = {'partial': {'transition_domain': cells,
        'content_domain_coverage': {transition: 4 for transition in
                                    ('UPDATE', 'SUPERSEDE', 'DELETE')},
        'unique_query_counts_by_transition': {transition: 30 for transition in
                                              ('UPDATE', 'SUPERSEDE', 'DELETE')}}}
    assert coverage_gate(summary)
    cells[0]['matched_distinct_queries'] = 4
    assert not coverage_gate(summary)


def test_h4_policy_is_machine_readable_and_freezes_primary_comparison():
    policy = json.loads(json.dumps(h4_policy(), sort_keys=True))
    assert policy['primary_exposure'] == {'uniform_stale': 0, 'mixed_partial': 1}
    assert policy['matching_keys'] == ['query_id', 'transition', 'architecture', 'severity']
    assert 'realized_fraction' in policy['secondary_explanatory_variables']
    assert policy['frozen_before_transition_outcomes']


def test_candidate_generation_repeats_identically(candidates):
    config, world, metadata, by_id, generated, source = candidates
    repeated = _make_partial_candidates(source, world, metadata, by_id, config)
    assert json.dumps(generated, sort_keys=True) == json.dumps(repeated, sort_keys=True)
