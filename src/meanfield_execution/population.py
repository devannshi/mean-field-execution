"""Finite-player transfer and unilateral deviation audit for open-loop plans."""
import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .config import validate_population
from .equilibrium import EquilibriumSystem


def population_audit(config, types, population_size, seed, inventory_std=0.20):
    """Return sampling error and exact deterministic unilateral deviation gains.

    Sampled finite agents use their own inventory and the limiting mean-flow
    forecast. Others are frozen when testing a deviation. This is not an audit
    of all adaptive policies, and no asymptotic epsilon-Nash theorem is claimed.
    """
    if population_size < 2 or inventory_std < 0:
        raise ValueError("Use population_size >= 2 and nonnegative inventory_std")
    types, N = tuple(types), int(population_size)
    weights = validate_population(types)
    system = EquilibriumSystem(types, config.steps, config.dt, config.permanent_impact)
    q0_types = np.array([a.initial_inventory for a in types])
    drift = np.zeros(config.steps)
    mean_limit, _ = system.solve(q0_types, drift)
    rng = np.random.default_rng(seed)
    labels = rng.choice(len(types), size=N, p=weights)
    # Mean-zero bounded variation preserves positive initial holdings.
    q0 = q0_types[labels] + rng.uniform(-np.sqrt(3) * inventory_std,
                                      np.sqrt(3) * inventory_std, N)
    plans = np.empty((N, config.steps))
    h_limit = drift - config.permanent_impact * mean_limit
    for k, problem in enumerate(system.problems):
        take = labels == k
        plans[take] = (q0[take, None] * problem.inventory_response
                       + problem.drift_response @ h_limit)
    mean_finite = plans.mean(axis=0)
    gains = np.empty(N)
    kappa, dt = config.permanent_impact, config.dt
    for k, problem in enumerate(system.problems):
        take = labels == k
        if not np.any(take):
            continue
        others = N * mean_finite - plans[take]
        h_exogenous = drift - kappa * others / N
        f = (q0[take, None] * problem.inventory_linear
             + dt * h_exogenous @ problem.D
             + kappa * dt / N * q0[take, None])
        H = problem.finite_player_hessian(kappa, N)
        optimum = -cho_solve(cho_factor(H), f.T).T
        if problem.agent.hard_liquidation:
            last = problem.D[-1]
            hinv_last = cho_solve(cho_factor(H), last)
            adjustment = (q0[take] - optimum @ last) / (last @ hinv_last)
            optimum += adjustment[:, None] * hinv_last
        delta = plans[take] - optimum
        gains[take] = 0.5 * np.einsum("ij,jk,ik->i", delta, H, delta)
    return {
        "population_size": N, "seed": int(seed),
        "mean_flow_rmse": float(np.sqrt(np.mean((mean_finite - mean_limit) ** 2))),
        "terminal_impact_error": float(kappa * dt * np.sum(mean_finite - mean_limit)),
        "mean_deviation_gain": float(gains.mean()),
        "max_deviation_gain": float(gains.max()),
        "sample_initial_inventory": float(q0.mean()),
    }

