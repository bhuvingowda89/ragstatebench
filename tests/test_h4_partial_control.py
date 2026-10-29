"""H4 controls are selected from frozen topology, before scientific outcomes."""

from ragstate.campaign.h4_partial_campaign import TICKS, _balance, _pair, _source
from ragstate.campaign.h4_partial_analysis import _continuous_paired, _pair_measure


def test_h4_frozen_horizons_and_intermediate_ticks():
    assert TICKS == {'S1': (1, 2), 'S2': (2, 5), 'S3': (5, 10)}
    assert all(0 < partial < final for partial, final in TICKS.values())


def test_h4_pairs_retain_compiler_selected_units_and_balance():
    _, sources = _source()
    assert len(sources) == 752
    for source in sources:
        uniform, partial = _pair(source)
        assert _balance((uniform, partial))['balanced']
        assert partial['selected_unit_ids'] == uniform['selected_unit_ids']
        assert partial['selected_stale_unit_ids'] == uniform['selected_stale_unit_ids']
        assert partial['h4_source_plan_hash'] == source['selected_unit_plan_hash']
        assert partial['h4_final_tick'] == uniform['h4_final_tick']
        assert 0 < partial['updated_unit_count'] < partial['total_unit_count']


def test_h4_mixed_window_uses_same_ticks_and_separates_integrity_metrics():
    _, source = _source()
    uniform, partial = _pair(source[0])
    def row(tick, *, mixed):
        return {'logical_time':tick,'domain':'engineering','state_correct':0,
                'scrs':0,'all_claims_success':0,'srr':0,'dep':0,
                'ucer_context':0,'ardr':0 if mixed else 1,
                'mvcr':1 if mixed else 0}
    by_id={uniform['scenario_id']:[row(tick,mixed=False) for tick in (2,3,4)],
           partial['scenario_id']:[row(tick,mixed=True) for tick in (2,3,4)]}
    result=_pair_measure((uniform,partial),by_id)
    assert result['interval_ticks']==3
    assert result['any_failure_difference']==0
    assert result['mvcr_ticks_difference']==3
    assert result['ardr_ticks_difference']==-3


def test_h4_continuous_paired_interval_is_deterministic():
    assert _continuous_paired([-3,-3,0]) == _continuous_paired([-3,-3,0])
