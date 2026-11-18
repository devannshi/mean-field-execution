import numpy as np
import pytest

from meanfield_execution import EquilibriumSystem, default_types, fixed_point


@pytest.fixture
def system():
    return EquilibriumSystem(default_types(), 20, 0.05, 0.08)


def test_direct_equilibrium_is_consistent(system):
    q, drift = np.array([1, 1.15, 0.8]), np.linspace(0.1, 0, 20)
    m, plans = system.solve(q, drift)
    np.testing.assert_allclose(system.weights @ plans, m, atol=1e-12)
    for p, plan, q0 in zip(system.problems, plans, q):
        np.testing.assert_allclose(p.gradient(plan, q0, drift - 0.08 * m), 0, atol=1e-12)


def test_fixed_point_matches_direct_solver(system):
    q, drift = [1, 1.15, 0.8], np.zeros(20)
    result = fixed_point(system, q, drift, damping=0.6)
    direct, _ = system.solve(q, drift)
    assert result.converged
    np.testing.assert_allclose(result.mean_rate, direct, atol=1e-9)


def test_tiny_damping_does_not_fake_convergence(system):
    result = fixed_point(system, [1, 1, 1], np.zeros(20), damping=1e-12, max_iterations=3)
    assert not result.converged
    assert result.residuals[-1] > 1


def test_iteration_cap_preserves_audited_residual(system):
    q, drift = [1, 1, 1], np.zeros(20)
    result = fixed_point(system, q, drift, max_iterations=2)
    residual = np.linalg.norm(system.response_map(result.mean_rate, q, drift) - result.mean_rate)
    residual /= max(1, np.linalg.norm(result.mean_rate))
    np.testing.assert_allclose(result.residuals[-1], residual)


def test_zero_coupling_decouples_types():
    s = EquilibriumSystem(default_types(), 10, 0.1, 0)
    q, a = [1, 2, 3], np.zeros(10)
    _, plans = s.solve(q, a)
    for k, p in enumerate(s.problems):
        np.testing.assert_allclose(plans[k], p.best_response(q[k], a))
    assert s.spectral_radius() == 0


def test_direct_solution_scaling(system):
    m, plans = system.solve([1, 1, 1], np.zeros(20))
    m2, plans2 = system.solve([2, 2, 2], np.zeros(20))
    np.testing.assert_allclose(m2, 2 * m)
    np.testing.assert_allclose(plans2, 2 * plans)


@pytest.mark.parametrize("damping", [0, -1, 2])
def test_bad_damping_rejected(system, damping):
    with pytest.raises(ValueError):
        fixed_point(system, [1, 1, 1], np.zeros(20), damping=damping)

