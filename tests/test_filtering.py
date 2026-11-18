import numpy as np
import pytest

from meanfield_execution import MarketConfig, RegimeFilter
from meanfield_execution.filtering import forecast_drift


def make_filter():
    c = MarketConfig()
    return RegimeFilter(c.transition, c.drifts, c.volatility, c.dt, c.prior)


def test_filter_matches_bayes_rule():
    f = make_filter()
    y = 0.01
    likelihood = np.exp(-0.5 * (y - f.drifts * f.dt) ** 2 / (f.volatility**2 * f.dt))
    expected = f.probability * likelihood
    expected /= expected.sum()
    posterior = f.observe(y)
    np.testing.assert_allclose(posterior, expected)
    np.testing.assert_allclose(f.probability, expected @ f.transition)


def test_impact_correction_recovers_identical_beliefs():
    correct, unaffected = make_filter(), make_filter()
    for fundamental, impact in zip([.01, -.02, .03], [-.03, -.02, -.01]):
        np.testing.assert_allclose(correct.observe(fundamental + impact, impact),
                                   unaffected.observe(fundamental), atol=1e-14)


def test_extreme_finite_observation_stays_normalized():
    f = make_filter()
    for y in [100, -100, 50, -50]:
        p = f.observe(y)
        assert np.all(np.isfinite(p))
        np.testing.assert_allclose(p.sum(), 1)


def test_forecast_is_prediction_not_future_observation():
    c = MarketConfig()
    f = make_filter()
    expected = np.array([c.prior @ np.linalg.matrix_power(c.transition, n) @ c.drifts
                         for n in range(6)])
    np.testing.assert_allclose(f.forecast(6), expected, atol=1e-15)


def test_transition_semigroup_under_grid_refinement():
    c, fine = MarketConfig(steps=20), MarketConfig(steps=40)
    np.testing.assert_allclose(c.transition, fine.transition @ fine.transition)


def test_perfect_persistence_forecast():
    np.testing.assert_equal(forecast_drift([0, 1], np.eye(2), [-1, 1], 4), np.ones(4))


def test_invalid_transition_rejected():
    with pytest.raises(ValueError):
        RegimeFilter([[1, 1], [0, 1]], [-1, 1], .1, .1, [.5, .5])

