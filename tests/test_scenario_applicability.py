"""Small exhaustive and fault-specific tests for pre-execution applicability."""
from dataclasses import replace
import inspect
import json
from pathlib import Path

import pytest

from ragstate.campaign.scenario_applicability import (
    COMPOUNDS, OPERATORS, _operator_check, _transition_check,
    _transition_facts, compile_candidate, dry_run, partial_updated_units,
)
from ragstate.campaign.state_pilot_manifest import _load_frozen


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def inputs():
    config = json.loads((ROOT / 'configs/state-isolated-pilot-v2.json').read_text())
    world, metadata, queries, _ = _load_frozen(config)
    scenarios = json.loads((ROOT / 'results/state-isolated-pilot-v3/scenario_manifest.json').read_text())['scenarios']
    return config, world, metadata, {query.query_id: query for query in queries}, scenarios


@pytest.mark.parametrize('n', range(11))
@pytest.mark.parametrize('fraction', (.25, .5, .75))
def test_partial_floor_rule_never_emits_an_impossible_split(n, fraction):
    value = partial_updated_units(n, fraction)
    expected = int(n * fraction)
    if n < 2 or expected < 1 or expected >= n:
        assert value is None
    else:
        assert value == expected
        assert 1 <= value < n


def test_single_unit_partial_is_excluded_and_two_unit_quarter_is_excluded():
    assert all(partial_updated_units(1, fraction) is None for fraction in (.25, .5, .75))
    assert [partial_updated_units(2, fraction) for fraction in (.25, .5, .75)] == [None, 1, 1]


def _scenario(inputs, *, fault, transition, compound=None, component=None):
    return next(s for s in inputs[4] if s['fault'] == fault and
                s['transition'] == transition and s['compound'] == compound and
                s['component'] == component and s['architecture'] == 'A0')


def _facts(inputs, scenario):
    query = inputs[3][scenario['query_id']]
    return query, _transition_facts(inputs[1], inputs[2], query, scenario['transition'])


def test_frozen_finance_partial_quarter_is_excluded_before_runtime(inputs):
    scenario = next(s for s in inputs[4] if s['scenario_id'] ==
        'TA2__v3-q-finance-01__DELETE__S2__A0__PARTIAL__0.25')
    query, facts = _facts(inputs, scenario)
    result = compile_candidate(scenario, query, facts, inputs[0])
    assert not result.applicable
    assert result.structural_facts['old_chunk_units'] == 1
    assert 'PARTIAL:partial_split_not_representable' in result.reasons


@pytest.mark.parametrize('transition', ('GRANT', 'REVOKE'))
def test_noop_authorization_transition_is_excluded(inputs, transition):
    scenario = _scenario(inputs, fault='DELAY', transition=transition)
    query, facts = _facts(inputs, scenario)
    noop = dict(facts, after_allowed=facts['before_allowed'])
    assert not _transition_check(scenario, query, noop)['effective']
    assert not compile_candidate(scenario, query, noop, inputs[0]).applicable


def test_out_of_order_requires_two_ordered_events(inputs):
    scenario = _scenario(inputs, fault='OUT_OF_ORDER', transition='UPDATE')
    query, facts = _facts(inputs, scenario)
    missing_prior = dict(facts, old=None)
    result = _operator_check('OUT_OF_ORDER', scenario, query, missing_prior)
    assert not result['applicable']
    assert 'fewer_than_two_ordered_events' in result['reasons']


def test_cache_fault_requires_population(inputs):
    scenario = _scenario(inputs, fault='STALE_CACHE', transition='UPDATE')
    query, facts = _facts(inputs, scenario)
    empty = dict(facts, cacheable_candidates=False)
    result = _operator_check('STALE_CACHE', scenario, query, empty)
    assert not result['applicable']
    assert 'cache_population_impossible' in result['reasons']


def test_dropped_event_requires_event(inputs):
    scenario = _scenario(inputs, fault='DROPPED', transition='DELETE')
    query, facts = _facts(inputs, scenario)
    without_event = dict(facts, event=None)
    result = _operator_check('DROPPED_CONTENT', scenario, query, without_event)
    assert not result['applicable']
    assert 'target_event_absent' in result['reasons']


def test_compound_joint_precondition_is_not_just_component_names(inputs):
    scenario = _scenario(inputs, fault='COMPOUND', transition='REVOKE',
                         compound='CF6', component='F1_F2')
    query, facts = _facts(inputs, scenario)
    missing_cross = dict(facts, cross_lid=None)
    result = compile_candidate(scenario, query, missing_cross, inputs[0])
    assert not result.applicable
    assert 'joint:component_precondition_failed' in result.reasons


def test_compiler_uses_structure_not_outcome_fields(inputs):
    scenario = _scenario(inputs, fault='DELAY', transition='UPDATE')
    query, facts = _facts(inputs, scenario)
    original = compile_candidate(scenario, query, facts, inputs[0])
    contaminated = dict(scenario, scrs=0, ucer_context=1, t_rsc=999)
    assert compile_candidate(contaminated, query, facts, inputs[0]) == original
    assert 'rows' not in inspect.signature(compile_candidate).parameters


def test_compound_specs_have_two_explicit_operators_and_effect_predicates():
    assert set(COMPOUNDS) == {f'CF{i}' for i in range(1, 7)}
    assert all(len(pair) == 2 and all(name in OPERATORS for name in pair)
               for pair in COMPOUNDS.values())
    assert all(spec.effectiveness_predicate and spec.replica_unit_type
               for spec in OPERATORS.values())


def test_compiler_and_replica_operator_agree_for_partial_smoke(inputs):
    scenario = next(s for s in inputs[4] if s['fault'] == 'PARTIAL'
                    and s['transition'] == 'UPDATE' and s['partial_fraction'] == .5
                    and s['query_id'] == 'v3-q-engineering-00')
    query, facts = _facts(inputs, scenario)
    result = compile_candidate(scenario, query, facts, inputs[0])
    assert result.applicable
    assert dry_run(scenario, query, facts, result)['realizable']


def test_repeated_compilation_is_deterministic_for_representative_cases(inputs):
    scenarios = [s for s in inputs[4] if s['fault'] in ('PARTIAL', 'COMPOUND')][:10]
    for scenario in scenarios:
        query, facts = _facts(inputs, scenario)
        assert compile_candidate(scenario, query, facts, inputs[0]) == \
               compile_candidate(scenario, query, facts, inputs[0])
