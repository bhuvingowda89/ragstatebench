"""Focused checks for the post-validity Track-A v4 analysis helpers."""

from ragstate.campaign.state_pilot_v4_analysis import _km, _paired


def test_paired_difference_preserves_match_and_direction():
    result = _paired([0, 1, 0, 1], [1, 1, 0, 0])
    assert result['n'] == 4
    assert result['risk_difference'] == 0
    assert result['discordant_a_only'] == 1
    assert result['discordant_b_only'] == 1
    assert result['mcnemar_exact_p'] == 1
    assert result['ci95_low'] <= 0 <= result['ci95_high']


def test_kaplan_meier_retains_right_censoring():
    trajectories = [[{'t_rsc': 2}], [{'t_rsc': 5}], [{'t_rsc': None}]]
    result = _km(trajectories)
    assert result['n'] == 3
    assert result['events'] == 2
    assert result['censored'] == 1
    assert result['km_median'] == 5
    assert result['rmst_to_14'] > 5
