"""Mean-field execution research tools.

Positive trading rates SELL inventory. All public costs are minimized.
"""

from .config import AgentType, MarketConfig, default_types
from .equilibrium import EquilibriumSystem, fixed_point
from .filtering import RegimeFilter
from .simulation import simulate

__all__ = [
           "AgentType",
           "EquilibriumSystem",
           "MarketConfig",
           "RegimeFilter",
           "default_types",
           "fixed_point",
           "simulate",
]
__version__ = "0.1.0"

