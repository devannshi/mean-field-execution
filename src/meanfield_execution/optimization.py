"""Exact linear-quadratic best responses with optional terminal equality.

For selling rate v, post-trade inventory is q = q0 - D v. The cost is
eta dt ||v||² + phi dt ||q||² + A q[-1]² - dt q·h.
Here h is a forecast of the impacted price drift, held fixed in a nonatomic
agent's best response. No optimizer tolerances are hidden in the solution.
Hard liquidation adds the exact KKT constraint dt * sum(v) = q0.
"""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import cho_factor, cho_solve

from .config import AgentType


@dataclass
class LQProblem:
    agent: AgentType
    steps: int
    dt: float

    def __post_init__(self):
        if self.steps < 1 or self.dt <= 0:
            raise ValueError("steps and dt must be positive")
        n, dt, a = self.steps, self.dt, self.agent
        self.D = dt * np.tril(np.ones((n, n)))
        last = self.D[-1]
        self.H = (2 * a.temporary_impact * dt * np.eye(n)
                  + 2 * a.inventory_penalty * dt * self.D.T @ self.D
                  + 2 * a.terminal_penalty * np.outer(last, last))
        self.inventory_linear = (-2 * a.inventory_penalty * dt * self.D.T @ np.ones(n)
                                 - 2 * a.terminal_penalty * last)
        self.factor = cho_factor(self.H, lower=True, check_finite=False)
        self.inventory_response = -cho_solve(self.factor, self.inventory_linear)
        self.drift_response = -cho_solve(self.factor, dt * self.D.T)
        if a.hard_liquidation:
            # Exact KKT projection onto dt * sum(v) = q0. This is an equality
            # constraint, not a large penalty or a last-period forced trade.
            hinv_last = cho_solve(self.factor, last)
            direction = hinv_last / (last @ hinv_last)
            self.inventory_response += direction * (1 - last @ self.inventory_response)
            self.drift_response -= np.outer(direction, last @ self.drift_response)

    def _vector(self, value):
        value = np.asarray(value, dtype=float)
        if value.shape != (self.steps,) or not np.all(np.isfinite(value)):
            raise ValueError(f"Expected a finite vector of length {self.steps}")
        return value

    def inventory(self, rate, q0):
        return float(q0) - self.D @ self._vector(rate)

    def linear_term(self, q0, drift):
        return self.inventory_linear * q0 + self.dt * self.D.T @ self._vector(drift)

    def best_response(self, q0, drift):
        if not np.isfinite(q0):
            raise ValueError("Initial inventory must be finite")
        return self.inventory_response * q0 + self.drift_response @ self._vector(drift)

    def cost(self, rate, q0, drift):
        v, h = self._vector(rate), self._vector(drift)
        q, a, dt = q0 - self.D @ v, self.agent, self.dt
        return float(a.temporary_impact * dt * (v @ v)
                     + a.inventory_penalty * dt * (q @ q)
                     + a.terminal_penalty * q[-1] ** 2 - dt * (q @ h))

    def gradient(self, rate, q0, drift):
        return self.H @ self._vector(rate) + self.linear_term(q0, drift)

    def finite_player_hessian(self, permanent_impact, population_size):
        if population_size < 1 or permanent_impact < 0:
            raise ValueError("Invalid population size or permanent impact")
        H = self.H - permanent_impact * self.dt / population_size * (self.D + self.D.T)
        try:
            cho_factor(H)
        except np.linalg.LinAlgError as exc:
            raise ValueError("Finite-player cost is not strictly convex at these parameters") from exc
        return H

    def finite_player_best_response(self, q0, fundamental_drift, other_rate_sum,
                                    permanent_impact, population_size):
        """Others' rates are held fixed; own permanent impact is internalized."""
        drift = self._vector(fundamental_drift)
        others = self._vector(other_rate_sum)
        k, N = permanent_impact, population_size
        H = self.finite_player_hessian(k, N)
        f = (self.linear_term(q0, drift - k * others / N)
             + k * self.dt / N * q0 * np.ones(self.steps))
        optimum = np.linalg.solve(H, -f)
        if self.agent.hard_liquidation:
            last = self.D[-1]
            hinv_last = np.linalg.solve(H, last)
            optimum += hinv_last * (q0 - last @ optimum) / (last @ hinv_last)
        return optimum

