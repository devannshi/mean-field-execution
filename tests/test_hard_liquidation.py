from dataclasses import replace

import numpy as np
import pytest
from numpy.testing import assert_allclose
from scipy.optimize import minimize

from meanfield_execution.config import AgentType, MarketConfig, default_types
from meanfield_execution.equilibrium import EquilibriumSystem, fixed_point
from meanfield_execution.optimization import LQProblem
from meanfield_execution.population import population_audit
from meanfield_execution.simulation import STRATEGIES, simulate


def hard_types():
    return tuple(replace(t, terminal_penalty=0, hard_liquidation=True) for t in default_types())


def test_equal_completion_no_risk_no_drift_is_twap():
    a = AgentType('test', 1, 2, .1, 0, 0, hard_liquidation=True)
    p = LQProblem(a, 20, .05)
    v = p.best_response(2, np.zeros(20))
    assert_allclose(v, 2, atol=1e-12)
    assert abs(p.inventory(v, 2)[-1]) < 1e-12


def test_constrained_response_matches_independent_optimizer():
    a = hard_types()[0]
    p = LQProblem(a, 8, .125)
    h = np.linspace(-.2, .1, 8)
    optimum = p.best_response(a.initial_inventory, h)
    other = minimize(lambda v: 100*p.cost(v, a.initial_inventory, h), np.ones(8),
                     jac=lambda v: 100*p.gradient(v, a.initial_inventory, h),
                     constraints={'type': 'eq', 'fun': lambda v: .125*v.sum()-a.initial_inventory,
                                  'jac': lambda v: np.full(8, .125)},
                     method='SLSQP', options={'ftol': 1e-13, 'maxiter': 1000})
    assert other.success
    assert_allclose(optimum, other.x, atol=2e-6)
    # Gradient is normal to the equality manifold, not necessarily zero.
    gradient = p.gradient(optimum, a.initial_inventory, h)
    assert_allclose(gradient, gradient[0], atol=1e-12)


def test_hard_equilibrium_preserves_completion_and_fixed_point():
    types = hard_types()
    system = EquilibriumSystem(types, 20, .05, .08)
    q0 = np.array([t.initial_inventory for t in types])
    drift = np.linspace(.1, -.1, 20)
    mean, plans = system.solve(q0, drift)
    assert_allclose(.05*plans.sum(1), q0, atol=1e-12)
    assert_allclose(system.weights @ plans, mean, atol=1e-10)
    result = fixed_point(system, q0, drift)
    assert result.converged
    assert_allclose(result.mean_rate, mean, atol=1e-8)


@pytest.mark.parametrize('strategy', STRATEGIES)
def test_all_policies_complete_and_conserve_accounting(strategy):
    result = simulate(types=hard_types(), strategy=strategy, seed=29)
    assert_allclose(result.inventories[-1], 0, atol=1e-12)
    assert result.metrics['accounting_error'] < 1e-10


def test_finite_player_constrained_deviation_is_feasible():
    p = LQProblem(hard_types()[1], 15, 1/15)
    v = p.finite_player_best_response(1.2, np.zeros(15), np.ones(15)*49, .08, 50)
    assert_allclose(v.sum()/15, 1.2, atol=1e-12)
    result = population_audit(MarketConfig(steps=15), hard_types(), 50, 91)
    assert result['mean_deviation_gain'] >= 0
    assert np.isfinite(result['max_deviation_gain'])


@pytest.mark.parametrize("type_index", [0, 1, 2])
def test_constrained_finite_player_matches_independent_optimizer(type_index):
    problem = LQProblem(hard_types()[type_index], 8, 0.125)
    q0, population_size, impact = 1.2, 20, 0.08
    drift = np.linspace(-0.1, 0.15, 8)
    other_rates = np.linspace(12, 22, 8)
    actual = problem.finite_player_best_response(q0, drift, other_rates,
                                                impact, population_size)

    def objective(rate):
        actual_drift = drift - impact * (other_rates + rate) / population_size
        return problem.cost(rate, q0, actual_drift)

    numerical = minimize(objective, np.full(8, q0), method="SLSQP",
                         constraints={"type": "eq", "fun": lambda rate: rate.sum() / 8 - q0},
                         options={"ftol": 1e-13, "maxiter": 500})
    assert numerical.success
    assert_allclose(actual, numerical.x, atol=3e-5)
    assert_allclose(actual.sum() / 8, q0, atol=1e-12)
    assert objective(actual) <= objective(numerical.x) + 1e-10


def test_invalid_exogenous_path_is_rejected():
    cfg = MarketConfig(steps=10)
    with pytest.raises(ValueError):
        simulate(cfg, path=(np.zeros(10), np.full(10, np.nan)))
