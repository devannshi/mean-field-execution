import numpy as np
import pytest
from scipy.optimize import minimize

from meanfield_execution.config import AgentType, default_types
from meanfield_execution.optimization import LQProblem


def test_inventory_conservation():
    p = LQProblem(default_types()[0], 8, 0.1)
    v = np.arange(8) / 3
    q = p.inventory(v, 2.0)
    np.testing.assert_allclose(np.diff(np.r_[2, q]), -0.1 * v)


def test_risk_neutral_soft_liquidation_analytic_solution():
    a = AgentType("test", 1, 2, 0.07, 0, 1.5)
    p = LQProblem(a, 20, 0.05)
    expected = a.terminal_penalty * 2 / (a.temporary_impact + a.terminal_penalty)
    np.testing.assert_allclose(p.best_response(2, np.zeros(20)), expected, atol=1e-12)


def test_one_step_solution():
    a = default_types()[0]
    p = LQProblem(a, 1, 0.1)
    q0, h, dt = 1.4, 0.3, 0.1
    expected = (2 * (a.inventory_penalty * dt + a.terminal_penalty) * q0 - dt * h)
    expected /= 2 * (a.temporary_impact + (a.inventory_penalty * dt + a.terminal_penalty) * dt)
    np.testing.assert_allclose(p.best_response(q0, [h]), [expected])


def test_no_penalty_or_drift_means_no_trade():
    a = AgentType("test", 1, 1, 0.1, 0, 0)
    p = LQProblem(a, 8, 0.1)
    np.testing.assert_equal(p.best_response(1, np.zeros(8)), np.zeros(8))


def test_gradient_finite_difference():
    p = LQProblem(default_types()[1], 7, 0.1)
    rng = np.random.default_rng(9)
    v, h = rng.normal(size=(2, 7))
    numeric = np.array([(p.cost(v + np.eye(7)[i] * 1e-5, 1.2, h)
                         - p.cost(v - np.eye(7)[i] * 1e-5, 1.2, h)) / 2e-5
                        for i in range(7)])
    np.testing.assert_allclose(p.gradient(v, 1.2, h), numeric, atol=1e-9)


def test_solution_agrees_with_independent_numerical_optimizer():
    p = LQProblem(default_types()[2], 12, 1 / 12)
    h = np.linspace(-0.2, 0.3, 12)
    result = minimize(lambda v: p.cost(v, 1.1, h), np.ones(12), method="BFGS",
                      options={"gtol": 1e-9})
    np.testing.assert_allclose(p.best_response(1.1, h), result.x, atol=2e-5)


def test_positive_drift_delays_sales():
    p = LQProblem(default_types()[0], 20, 0.05)
    flat = p.best_response(1, np.zeros(20))
    rising = p.best_response(1, np.full(20, 0.2))
    assert rising[0] < flat[0]


def test_finite_player_internalization_matches_numeric_minimum():
    p = LQProblem(default_types()[0], 10, 0.1)
    other, drift, N, kappa = np.linspace(5, 8, 10), np.zeros(10), 10, 0.08
    optimum = p.finite_player_best_response(1, drift, other, kappa, N)
    fun = lambda v: p.cost(v, 1, drift - kappa * (other + v) / N)
    result = minimize(fun, np.ones(10), method="BFGS", options={"gtol": 1e-9})
    np.testing.assert_allclose(optimum, result.x, atol=2e-5)
    assert fun(optimum) <= fun(np.ones(10))


def test_nonconvex_finite_player_parameters_are_rejected():
    p = LQProblem(AgentType("fragile", 1, 1, 0.001, 0, 0), 10, 0.1)
    with pytest.raises(ValueError, match="not strictly convex"):
        p.finite_player_hessian(10, 2)


@pytest.mark.parametrize("drift", [[1, 2], [np.nan] * 4])
def test_invalid_forecast_rejected(drift):
    with pytest.raises(ValueError):
        LQProblem(default_types()[0], 4, 0.25).best_response(1, drift)

