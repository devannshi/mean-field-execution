"""Deterministic forecast mean-field equilibrium and auditable iteration."""
from dataclasses import dataclass

import numpy as np
from scipy.linalg import lu_factor, lu_solve

from .config import validate_population
from .optimization import LQProblem


class EquilibriumSystem:
    def __init__(self, types, steps, dt, permanent_impact):
        self.types = tuple(types)
        self.weights = validate_population(types)
        if permanent_impact < 0 or not np.isfinite(permanent_impact):
            raise ValueError("Permanent impact must be finite and nonnegative")
        self.steps, self.dt, self.permanent_impact = steps, dt, permanent_impact
        self.problems = [LQProblem(t, steps, dt) for t in types]
        self.G = sum(w * p.drift_response for w, p in zip(self.weights, self.problems))
        self.jacobian = -permanent_impact * self.G
        self.equilibrium_matrix = np.eye(steps) - self.jacobian
        self.factor = lu_factor(self.equilibrium_matrix)

    def _inputs(self, inventories, drift):
        q, h = np.asarray(inventories, dtype=float), np.asarray(drift, dtype=float)
        if q.shape != (len(self.types),) or h.shape != (self.steps,):
            raise ValueError("Inventory or forecast dimensions do not match the system")
        if not np.all(np.isfinite(q)) or not np.all(np.isfinite(h)):
            raise ValueError("Inventories and drift must be finite")
        return q, h

    def offset(self, inventories, drift):
        q, a = self._inputs(inventories, drift)
        return sum(w * p.inventory_response * qk
                   for w, p, qk in zip(self.weights, self.problems, q)) + self.G @ a

    def response_map(self, mean_rate, inventories, drift):
        return self.offset(inventories, drift) + self.jacobian @ np.asarray(mean_rate)

    def plans(self, mean_rate, inventories, drift):
        q, a = self._inputs(inventories, drift)
        h = a - self.permanent_impact * np.asarray(mean_rate)
        return np.array([p.best_response(qk, h) for p, qk in zip(self.problems, q)])

    def solve(self, inventories, drift):
        m = lu_solve(self.factor, self.offset(inventories, drift))
        plans = self.plans(m, inventories, drift)
        residual = np.linalg.norm(self.weights @ plans - m) / max(1, np.linalg.norm(m))
        if not np.isfinite(residual) or residual > 1e-7:
            raise RuntimeError(f"Equilibrium linear solve failed residual check: {residual:g}")
        return m, plans

    def spectral_radius(self, damping=1.0):
        J = (1 - damping) * np.eye(self.steps) + damping * self.jacobian
        return float(np.max(np.abs(np.linalg.eigvals(J))))


@dataclass
class FixedPointResult:
    mean_rate: np.ndarray
    plans: np.ndarray
    residuals: np.ndarray
    converged: bool
    iterations: int
    damping: float


def fixed_point(system, inventories, drift, damping=0.6, tolerance=1e-10,
                max_iterations=500, initial=None):
    """Stop on the UNDAMPED best-response residual, never on the small update."""
    if not 0 < damping <= 1 or tolerance <= 0 or max_iterations < 1:
        raise ValueError("Invalid iteration controls")
    offset = system.offset(inventories, drift)
    m = np.zeros(system.steps) if initial is None else np.asarray(initial, dtype=float).copy()
    if m.shape != (system.steps,) or not np.all(np.isfinite(m)):
        raise ValueError("Invalid initial mean flow")
    residuals, converged = [], False
    for _ in range(max_iterations):
        target = offset + system.jacobian @ m
        residual = np.linalg.norm(target - m) / max(1, np.linalg.norm(m))
        residuals.append(float(residual))
        if residual < tolerance:
            converged = True
            break
        # Return the final audited iterate. Do not apply an unmeasured last step.
        if len(residuals) < max_iterations:
            m += damping * (target - m)
    return FixedPointResult(m, system.plans(m, inventories, drift), np.array(residuals),
                            converged, len(residuals), damping)

