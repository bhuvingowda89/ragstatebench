"""Full-trajectory selected-unit and boundary tests before v4 freezing."""
import json
from pathlib import Path

import pytest

from ragstate.campaign.scenario_applicability import _transition_facts
from ragstate.campaign.state_pilot_manifest import _load_frozen
from ragstate.campaign.state_pilot_v4_campaign import (
    _enrich_partial, _sample_scenarios, execute_recorded, validate_scenario,
)
from ragstate.campaign.topology_partial_runtime import verify_plan


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def inputs():
    config = json.loads((ROOT / 'configs/state-isolated-pilot-v2.json').read_text())
    config['campaign_version'] = 'state-isolated-pilot-v4'
    world, metadata, queries, _ = _load_frozen(config)
    scenarios = json.loads((ROOT / 'results/scenario-applicability-v2/applicability_manifest.json').read_text())['executable_scenarios']
    return config, world, metadata, {q.query_id: q for q in queries}, scenarios


@pytest.mark.parametrize('transition', ('UPDATE', 'SUPERSEDE', 'DELETE', 'REVOKE'))
def test_selected_units_survive_full_runtime_trajectory(inputs, transition):
    config, world, metadata, queries, scenarios = inputs
    source = next(s for s in scenarios if s.get('partial_topology')
                  and s['transition'] == transition and s['total_unit_count'] in (2, 3))
    query = queries[source['query_id']]
    facts = _transition_facts(world, metadata, query, transition)
    scenario = _enrich_partial(source, facts)
    scenario['campaign_version'] = config['campaign_version']
    rows, accesses, cache_ops, snapshots = execute_recorded(
        scenario, world, metadata, query, config)
    validity = validate_scenario(scenario, rows, accesses, cache_ops,
                                 snapshots, query, metadata, world, config)
    assert validity['validity'] == 'valid', validity['reasons']
    delivered = next(row for row in rows if row['partial_event_trace'])
    assert delivered['partial_actual_updated_unit_ids'] == scenario['selected_unit_ids']
    assert delivered['partial_actual_stale_unit_ids'] == scenario['selected_stale_unit_ids']
    assert delivered['partial_matched_uniform_scenario_id'] == scenario['matched_uniform_scenario_id']
    assert len(rows) == len(config['observation_relative_ticks'])
    assert all(event['allowed'] for event in accesses)
    assert all(stage['valid'] for op in cache_ops for stage in op['stage_events'])


def test_tampered_selected_unit_plan_fails_before_state_mutation(inputs):
    config, world, metadata, queries, scenarios = inputs
    source = next(s for s in scenarios if s.get('partial_topology')
                  and s['transition'] == 'DELETE' and s['total_unit_count'] == 3)
    query = queries[source['query_id']]
    facts = _transition_facts(world, metadata, query, 'DELETE')
    scenario = _enrich_partial(source, facts)
    scenario['selected_stale_unit_ids'] = list(reversed(scenario['selected_stale_unit_ids']))
    with pytest.raises(ValueError, match='selected/stale unit IDs'):
        verify_plan(scenario, facts['event'], query.principal_id,
                    facts['lid'], facts['chunk'])


def test_wiring_sample_has_frozen_structural_coverage(inputs):
    selected = _sample_scenarios(inputs[4])
    labels = {item['stratum'] for item in selected}
    assert len(selected) == 12
    assert {'N2_LOW_UPDATE_engineering', 'N3_LOW_SUPERSEDE_finance',
            'N3_HIGH_DELETE_hr', 'N4_LOW_REVOKE_legal',
            'N4_MID_UPDATE_sales', 'N4_HIGH_SUPERSEDE_security',
            'candidate_cache', 'out_of_order', 'CF3', 'CF6',
            'live_auth_A4', 'combined_A5'} == labels


def test_full_trajectory_is_deterministic_except_latency(inputs):
    config, world, metadata, queries, scenarios = inputs
    source = next(s for s in scenarios if s.get('partial_topology')
                  and s['transition'] == 'UPDATE' and s['total_unit_count'] == 2)
    query = queries[source['query_id']]
    scenario = _enrich_partial(source, _transition_facts(world, metadata, query, 'UPDATE'))
    scenario['campaign_version'] = config['campaign_version']
    outputs = []
    for _ in range(2):
        rows, accesses, cache_ops, snapshots = execute_recorded(
            scenario, world, metadata, query, config)
        outputs.append(([
            {key: value for key, value in row.items()
             if key not in ('retrieval_latency_ms', 'parent_latency_ms')}
            for row in rows], accesses, cache_ops))
    assert outputs[0] == outputs[1]
