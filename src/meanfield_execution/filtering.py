"""Causal HMM filtering of interval drift from impact-corrected price changes."""
import numpy as np
from scipy.special import logsumexp


class RegimeFilter:
    def __init__(self, transition, drifts, volatility, dt, prior):
        self.transition = np.asarray(transition, dtype=float)
        self.drifts = np.asarray(drifts, dtype=float)
        self.probability = np.asarray(prior, dtype=float).copy()
        self.volatility, self.dt = float(volatility), float(dt)
        n = len(self.drifts)
        if (self.transition.shape != (n, n) or self.probability.shape != (n,)
                or np.any(self.transition < 0) or np.any(self.probability < 0)
                or not np.allclose(self.transition.sum(axis=1), 1)
                or not np.isclose(self.probability.sum(), 1)
                or self.volatility <= 0 or self.dt <= 0
                or not np.all(np.isfinite(self.drifts))):
            raise ValueError("Invalid hidden-state model")

    def forecast(self, steps):
        return forecast_drift(self.probability, self.transition, self.drifts, steps)

    def observe(self, price_change, known_impact_change=0.0):
        """Update interval z_t, then predict z_{t+1}; return posterior for z_t.

        known_impact_change = -kappa * mean_selling_rate * dt.
        Subtracting it isolates the fundamental price change.
        """
        if not np.isfinite(price_change) or not np.isfinite(known_impact_change):
            raise ValueError("Observation and impact correction must be finite")
        innovation = (price_change - known_impact_change - self.drifts * self.dt)
        variance = self.volatility ** 2 * self.dt
        with np.errstate(divide="ignore"):
            logw = np.log(self.probability) - 0.5 * innovation ** 2 / variance
        posterior = np.exp(logw - logsumexp(logw))
        self.probability = posterior @ self.transition
        self.probability /= self.probability.sum()
        return posterior


def forecast_drift(probability, transition, drifts, steps):
    if steps < 1:
        raise ValueError("Forecast horizon must be positive")
    p = np.asarray(probability, dtype=float).copy()
    means = np.empty(steps)
    for j in range(steps):
        means[j] = p @ drifts
        p = p @ transition
    return means

