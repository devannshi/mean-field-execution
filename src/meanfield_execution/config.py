"""Validated model parameters in normalized price / inventory / horizon units."""
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AgentType:
    name: str
    weight: float
    initial_inventory: float
    temporary_impact: float
    inventory_penalty: float
    terminal_penalty: float
    hard_liquidation: bool = False

    def __post_init__(self):
        numbers = [self.weight, self.initial_inventory, self.temporary_impact,
                   self.inventory_penalty, self.terminal_penalty]
        if not np.all(np.isfinite(numbers)):
            raise ValueError("Agent parameters must be finite")
        if self.weight <= 0 or self.temporary_impact <= 0:
            raise ValueError("Weight and temporary impact must be positive")
        if self.inventory_penalty < 0 or self.terminal_penalty < 0:
            raise ValueError("Inventory penalties must be nonnegative")


@dataclass(frozen=True)
class MarketConfig:
    steps: int = 40
    horizon: float = 1.0
    permanent_impact: float = 0.08
    volatility: float = 0.12
    negative_drift: float = -0.18
    positive_drift: float = 0.18
    switching_rate: float = 0.8
    initial_positive_probability: float = 0.5
    initial_price: float = 100.0

    def __post_init__(self):
        if not isinstance(self.steps, int) or self.steps < 1:
            raise ValueError("steps must be a positive integer")
        vals = [self.horizon, self.permanent_impact, self.volatility, self.negative_drift,
                self.positive_drift, self.switching_rate, self.initial_price,
                self.initial_positive_probability]
        if not np.all(np.isfinite(vals)):
            raise ValueError("Market parameters must be finite")
        if self.horizon <= 0 or self.volatility <= 0:
            raise ValueError("Horizon and volatility must be positive")
        if self.permanent_impact < 0 or self.switching_rate < 0:
            raise ValueError("Impact and switching rate must be nonnegative")
        if not 0 <= self.initial_positive_probability <= 1:
            raise ValueError("Initial regime probability must be in [0, 1]")

    @property
    def dt(self):
        return self.horizon / self.steps

    @property
    def drifts(self):
        return np.array([self.negative_drift, self.positive_drift], dtype=float)

    @property
    def transition(self):
        # Exact transition of a symmetric two-state continuous-time chain,
        # sampled on the decision grid. Drift is piecewise constant on each interval.
        switch = -0.5 * np.expm1(-2 * self.switching_rate * self.dt)
        return np.array([[1 - switch, switch], [switch, 1 - switch]])

    @property
    def prior(self):
        p = self.initial_positive_probability
        return np.array([1 - p, p])


def default_types():
    return (
        AgentType("patient", 0.45, 1.0, 0.035, 0.012, 2.0),
        AgentType("balanced", 0.35, 1.15, 0.045, 0.06, 2.0),
        AgentType("urgent", 0.20, 0.8, 0.055, 0.22, 2.0),
    )


def validate_population(types):
    if not types:
        raise ValueError("At least one agent type is required")
    weights = np.array([t.weight for t in types])
    if not np.isclose(weights.sum(), 1.0, atol=1e-12):
        raise ValueError("Population weights must sum to one")
    if len({t.name for t in types}) != len(types):
        raise ValueError("Agent type names must be unique")
    return weights

