from dataclasses import replace

import numpy as np
import pytest

from meanfield_execution import MarketConfig, default_types, simulate
from meanfield_execution.population import population_audit
from meanfield_execution.simulation import STRATEGIES, exogenous_path
from meanfield_execution.statistics import mean_interval


@pytest.mark.parametrize("strategy", STRATEGIES)
def test_all_strategies_preserve_accounting(strategy):
    c = MarketConfig(steps=12)
    r = simulate(c, strategy=strategy, seed=15)
    np.testing.assert_allclose(np.diff(r.inventories, axis=0), -c.dt * r.rates, atol=1e-13)
    assert r.metrics["accounting_error"] < 1e-10
    assert np.isfinite(r.metrics["objective"])


def test_seed_reproducibility_and_common_random_numbers():
    c = MarketConfig(steps=10)
    a, b = simulate(c, strategy="twap", seed=8), simulate(c, strategy="filtered_mfg", seed=8)
    np.testing.assert_array_equal(a.states, b.states)
    np.testing.assert_array_equal(a.fundamental, b.fundamental)
    again = simulate(c, strategy="filtered_mfg", seed=8)
    np.testing.assert_array_equal(b.rates, again.rates)


def test_no_lookahead_prefix_invariance():
    c = MarketConfig(steps=12)
    z, dF = exogenous_path(c, 12)
    changed = dF.copy()
    changed[7:] += 10
    a = simulate(c, path=(z, dF))
    b = simulate(c, path=(z, changed))
    # Decision 7 is made before interval-7 return is observed.
    np.testing.assert_array_equal(a.rates[:8], b.rates[:8])
    np.testing.assert_array_equal(a.beliefs[:8], b.beliefs[:8])


def test_current_state_information_has_zero_decision_brier():
    result = simulate(strategy="full_info_mfg", seed=29)
    assert result.metrics["decision_brier"] == 0.0
    assert result.metrics["filter_prior_brier"] > 0
    expected = np.mean((result.beliefs[:, 1] - result.states) ** 2)
    np.testing.assert_allclose(result.metrics["filter_prior_brier"], expected)
    assert result.metrics["posterior_brier"] > 0  # Auxiliary filter, not the known state.


@pytest.mark.parametrize("strategy", ["filtered_mfg", "naive_filter_mfg", "misspecified_mfg"])
def test_filter_driven_policy_scores_its_actual_prior(strategy):
    result = simulate(strategy=strategy, seed=29)
    expected = np.mean((result.beliefs[:, 1] - result.states) ** 2)
    np.testing.assert_allclose(result.metrics["decision_brier"], expected)
    np.testing.assert_allclose(result.metrics["filter_prior_brier"], expected)


@pytest.mark.parametrize("strategy", ["twap", "ac", "blind_mfg"])
def test_policy_without_state_belief_has_no_decision_brier(strategy):
    result = simulate(strategy=strategy, seed=29)
    assert np.isnan(result.metrics["decision_brier"])
    assert np.isfinite(result.metrics["filter_prior_brier"])
    assert np.isfinite(result.metrics["posterior_brier"])


def test_integer_valued_float_state_path_matches_integer_path():
    config = MarketConfig(steps=12)
    states, changes = exogenous_path(config, 17)
    integer = simulate(config, path=(states, changes))
    floating = simulate(config, path=(states.astype(float), changes))
    np.testing.assert_array_equal(integer.rates, floating.rates)
    np.testing.assert_array_equal(integer.inventories, floating.inventories)
    assert np.issubdtype(floating.states.dtype, np.integer)


def test_fractional_state_path_is_rejected():
    config = MarketConfig(steps=4)
    with pytest.raises(ValueError, match="Path does not match"):
        simulate(config, path=(np.full(4, 0.5), np.zeros(4)))


def test_twap_completes_inventory():
    r = simulate(MarketConfig(steps=12), strategy="twap")
    np.testing.assert_allclose(r.inventories[-1], 0, atol=1e-14)
    np.testing.assert_allclose(r.rates, np.tile(r.inventories[0], (12, 1)))


def test_zero_impact_naive_and_correct_filters_are_identical():
    c = MarketConfig(steps=12, permanent_impact=0)
    a = simulate(c, strategy="filtered_mfg", seed=5)
    b = simulate(c, strategy="naive_filter_mfg", seed=5)
    np.testing.assert_array_equal(a.rates, b.rates)


def test_sell_flow_pushes_price_down():
    c = MarketConfig(steps=12)
    r = simulate(c, strategy="twap")
    w = np.array([a.weight for a in default_types()])
    expected = -c.permanent_impact * c.dt * np.cumsum(r.rates @ w)
    np.testing.assert_allclose(r.price[1:] - r.fundamental[1:], expected, atol=1e-12)


def test_no_drift_information_means_ac_when_no_coupling():
    c = MarketConfig(steps=12, permanent_impact=0, negative_drift=0, positive_drift=0)
    a = simulate(c, strategy="ac", seed=5)
    b = simulate(c, strategy="filtered_mfg", seed=5)
    np.testing.assert_allclose(a.rates, b.rates)


def test_finite_population_gains_nonnegative_and_reproducible():
    c = MarketConfig(steps=12)
    a = population_audit(c, default_types(), 20, 91)
    b = population_audit(c, default_types(), 20, 91)
    assert a == b
    assert a["mean_deviation_gain"] >= 0
    assert a["max_deviation_gain"] >= a["mean_deviation_gain"]


def test_no_coupling_finite_deviation_gain_is_zero():
    c = MarketConfig(steps=12, permanent_impact=0)
    r = population_audit(c, default_types(), 20, 9)
    assert r["max_deviation_gain"] < 1e-25


def test_ci_has_trial_count_and_expected_center():
    out = mean_interval([1, 2, 3, 4, 5])
    assert out["mean"] == 3
    assert out["n"] == 5
    assert out["lower"] < 3 < out["upper"]


@pytest.mark.parametrize("kw", [{"volatility": 0}, {"steps": 0}, {"permanent_impact": -1}])
def test_invalid_market_rejected(kw):
    with pytest.raises(ValueError):
        replace(MarketConfig(), **kw)

