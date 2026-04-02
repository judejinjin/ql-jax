"""Quanto options — options with FX adjustment."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class QuantoVanillaOption:
    """Quanto vanilla option.

    European option on a foreign asset, settled in domestic currency
    at a predetermined exchange rate.

    Parameters
    ----------
    strike : strike price (in foreign currency)
    is_call : True for call
    maturity : time to maturity
    fx_rate : fixed exchange rate for conversion
    """
    strike: float
    is_call: bool
    maturity: float
    fx_rate: float = 1.0

    def payoff(self, spot):
        if self.is_call:
            return self.fx_rate * jnp.maximum(spot - self.strike, 0.0)
        return self.fx_rate * jnp.maximum(self.strike - spot, 0.0)


@dataclass(frozen=True)
class QuantoBarrierOption:
    """Quanto barrier option — a barrier option with FX quanto adjustment."""
    strike: float
    barrier: float
    is_call: bool
    is_knock_in: bool
    maturity: float
    fx_rate: float = 1.0
    rebate: float = 0.0
