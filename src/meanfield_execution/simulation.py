"""Receding-horizon execution on paired, seeded exogenous market paths.

The strategy repeatedly solves a deterministic conditional-mean game. This is
an implementable certainty-equivalent approximation, not the continuous-time
stochastic MFG solution or a proof of an adaptive Nash equilibrium.
"""
from dataclasses import dataclass
from functools import lru_cache

import numpy as np

from .config import MarketConfig, default_types, validate_population
from .equilibrium import EquilibriumSystem
from .filtering import RegimeFilter, forecast_drift

STRATEGIES = ("twap", "ac", "filtered_mfg", "full_info_mfg", "naive_filter_mfg",
              "misspecified_mfg", "blind_mfg")


@lru_cache(maxsize=1024)
def _system(types, steps, dt, kappa):
    return EquilibriumSystem(types, steps, dt, kappa)


def exogenous_path(config, seed):
    state_seed, noise_seed = np.random.SeedSequence(seed).spawn(2)
    states_rng, noise_rng = np.random.default_rng(state_seed), np.random.default_rng(noise_seed)
    z = np.empty(config.steps, dtype=int)
    z[0] = states_rng.choice(2, p=config.prior)
    for t in range(1, config.steps):
        z[t] = states_rng.choice(2, p=config.transition[z[t - 1]])
    increments = (config.drifts[z] * config.dt
                  + config.volatility * np.sqrt(config.dt) * noise_rng.normal(size=config.steps))
    return z, increments


@dataclass
class SimulationResult:
    strategy: str
    seed: int
    states: np.ndarray
    fundamental: np.ndarray
    price: np.ndarray
    inventories: np.ndarray
    rates: np.ndarray
    cash: np.ndarray
    beliefs: np.ndarray  # Auxiliary filter priors, including policies that ignore the filter.
    posteriors: np.ndarray  # Auxiliary filter posteriors after observing each interval.
    metrics: dict


def simulate(config=None, types=None, strategy="filtered_mfg", seed=20261117,
             path=None):
    config = config or MarketConfig()
    types = tuple(types or default_types())
    w = validate_population(types)
    if strategy not in STRATEGIES:
        raise ValueError(f"Unknown strategy {strategy!r}; choose from {STRATEGIES}")
    z, dF = exogenous_path(config, seed) if path is None else path
    z, dF = np.asarray(z), np.asarray(dF, dtype=float)
    n, K, dt = config.steps, len(types), config.dt
    if (z.shape != (n,) or dF.shape != (n,) or not np.all(np.isfinite(z))
            or np.any((z < 0) | (z > 1))
            or not np.all(np.isfinite(dF)) or not np.all(z == z.astype(int))):
        raise ValueError("Path does not match the simulation grid")
    # Integer-valued user-supplied arrays are valid even when stored as floats.
    z = z.astype(int, copy=False)
    kappa_assumed = config.permanent_impact
    transition, assumed_sigma = config.transition.copy(), config.volatility
    if strategy == "misspecified_mfg":
        # Explicit, prespecified combined misspecification stress.
        kappa_assumed *= 0.5
        transition = 0.75 * np.eye(2) + 0.25 * transition
        assumed_sigma *= 0.65
    filt = RegimeFilter(transition, config.drifts, assumed_sigma, dt, config.prior)
    q, v = np.zeros((n + 1, K)), np.zeros((n, K))
    q[0] = [a.initial_inventory for a in types]
    price, cash = np.zeros(n + 1), np.zeros((n + 1, K))
    price[0] = config.initial_price
    beliefs, posteriors = np.zeros((n, 2)), np.zeros((n, 2))
    eta = np.array([a.temporary_impact for a in types])
    phi = np.array([a.inventory_penalty for a in types])
    terminal = np.array([a.terminal_penalty for a in types])
    for t in range(n):
        remaining = n - t
        beliefs[t] = filt.probability
        if strategy == "twap":
            v[t] = q[t] / (remaining * dt)
        elif strategy == "ac":
            _, plans = _system(types, remaining, dt, 0.0).solve(q[t], np.zeros(remaining))
            v[t] = plans[:, 0]
        else:
            if strategy == "full_info_mfg":
                p = np.eye(2)[z[t]]
                forecast = forecast_drift(p, config.transition, config.drifts, remaining)
            elif strategy == "blind_mfg":
                forecast = np.zeros(remaining)
            else:
                forecast = filt.forecast(remaining)
            _, plans = _system(types, remaining, dt, kappa_assumed).solve(q[t], forecast)
            v[t] = plans[:, 0]
        q[t + 1] = q[t] - dt * v[t]
        cash[t + 1] = cash[t] + dt * v[t] * (price[t] - eta * v[t])
        mean_rate = w @ v[t]
        impact_change = -config.permanent_impact * mean_rate * dt
        price[t + 1] = price[t] + dF[t] + impact_change
        correction = 0.0 if strategy == "naive_filter_mfg" else -kappa_assumed * mean_rate * dt
        posteriors[t] = filt.observe(price[t + 1] - price[t], correction)
    # Actual accounting, including the terminal marked inventory, is checked
    # against the discrete integration-by-parts decomposition below.
    shortfall = q[0] * config.initial_price - cash[-1] - q[-1] * price[-1]
    temporary = dt * eta * np.sum(v * v, axis=0)
    drift_component = -dt * np.sum(q[1:] * config.drifts[z, None], axis=0)
    impact_component = dt * config.permanent_impact * np.sum(q[1:] * (v @ w)[:, None], axis=0)
    noise_component = -np.sum(q[1:] * (dF - config.drifts[z] * dt)[:, None], axis=0)
    risk = dt * phi * np.sum(q[1:] ** 2, axis=0)
    terminal_cost = terminal * q[-1] ** 2
    error = np.max(np.abs(shortfall - temporary - drift_component - impact_component - noise_component))
    filter_prior_brier = float(np.mean((beliefs[:, 1] - z) ** 2))
    if strategy == "full_info_mfg":
        # The planner observes the current state; its belief is a point mass.
        decision_brier = 0.0
    elif strategy in {"filtered_mfg", "naive_filter_mfg", "misspecified_mfg"}:
        decision_brier = filter_prior_brier
    else:
        # These planners do not use a regime probability, so the score is undefined.
        decision_brier = float("nan")
    metrics = {
        "objective": float(w @ (shortfall + risk + terminal_cost)),
        "shortfall": float(w @ shortfall),
        "temporary_cost": float(w @ temporary),
        "drift_cost": float(w @ drift_component),
        "permanent_impact_cost": float(w @ impact_component),
        "noise_cost": float(w @ noise_component),
        "inventory_penalty": float(w @ risk),
        "terminal_penalty": float(w @ terminal_cost),
        "terminal_inventory_rms": float(np.sqrt(w @ q[-1] ** 2)),
        "integrated_inventory": float(dt * np.sum(q[1:] ** 2 @ w)),
        "gross_turnover": float(dt * np.sum(np.abs(v) @ w)),
        "negative_rate_fraction": float(np.mean(v < -1e-12)),
        "posterior_brier": float(np.mean((posteriors[:, 1] - z) ** 2)),
        "filter_prior_brier": filter_prior_brier,
        "decision_brier": decision_brier,
        "posterior_log_loss": float(-np.mean(np.log(np.maximum(posteriors[np.arange(n), z], 1e-300)))),
        "accounting_error": float(error),
    }
    fundamental = config.initial_price + np.r_[0, np.cumsum(dF)]
    return SimulationResult(strategy, int(seed), z, fundamental, price, q, v, cash,
                            beliefs, posteriors, metrics)

