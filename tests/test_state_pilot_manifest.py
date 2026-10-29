"""The two-track selection policy is fixed without transition outcomes."""
import json
from pathlib import Path

from ragstate.campaign.state_pilot_manifest import build, digest


CONFIG = Path('configs/state-isolated-pilot-v1.json')


def test_manifest_is_deterministic_and_transfer_population_is_frozen():
    config = json.loads(CONFIG.read_text())
    first, second = build(config), build(config)
    for key in ('applicability', 'track_a', 'track_b', 'component_factorial'):
        assert digest(first[key]) == digest(second[key])
    cohort = set(json.loads(Path(config['track_b_cohort_ids']).read_text()))
    cohort.update(json.loads(Path(config['track_b_negative_ids']).read_text()))
    assert all(row['query_id'] in cohort for row in first['track_b'])
    assert not any('scrs' in row or 'rank' in row for row in first['applicability'])


def test_primary_population_does_not_condition_on_clean_retrieval():
    built = build(json.loads(CONFIG.read_text()))
    applicable_ids = {row['query_id'] for row in built['applicability']
                      if row['applicable']}
    cohort = set(json.loads(Path('data/transition-cohort-v2/eligible_answerable_query_ids.json').read_text()))
    assert applicable_ids - cohort  # clean PA2 failures remain in Track A
    assert set(row['query_id'] for row in built['track_a']) == applicable_ids


def test_component_factorial_is_structurally_orthogonal():
    built = build(json.loads(CONFIG.read_text()))
    groups = {}
    for row in built['component_factorial']:
        groups.setdefault((row['component'], row['query_id']), set()).add(tuple(row['component_tuple']))
    assert groups
    for (block, _), combos in groups.items():
        if block == 'authorization_revoke_S2':
            assert combos == {(0, 0, 0), (1, 0, 0), (0, 0, 1), (1, 0, 1)}
        else:
            assert combos == {(0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)}
