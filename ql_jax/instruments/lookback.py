"""Lookback options — fixed and floating strike.

ContinuousFixedLookbackOption: payoff based on max/min of underlying vs fixed strike.
ContinuousFloatingLookbackOption: payoff based on spot vs running max/min.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import jax.numpy as jnp


class LookbackType(Enum):
    FIXED = "fixed"
    FLOATING = "floating"


@dataclass(frozen=True)
class ContinuousFixedLookbackOption:
    """Fixed-strike lookback option.

    For a call: payoff = max(S_max - K, 0)
    For a put:  payoff = max(K - S_min, 0)
    """
    strike: float
    is_call: bool
    current_minmax: float  # running max (call) or min (put) observed so far
    maturity: float

    def payoff(self, spot_minmax):
        if self.is_call:
            return jnp.maximum(spot_minmax - self.strike, 0.0)
        return jnp.maximum(self.strike - spot_minmax, 0.0)


@dataclass(frozen=True)
class ContinuousFloatingLookbackOption:
    """Floating-strike lookback option.

    For a call: payoff = S_T - S_min
    For a put:  payoff = S_max - S_T
    """
    is_call: bool
    current_minmax: float
    maturity: float

    def payoff(self, spot_final, spot_minmax):
        if self.is_call:
            return jnp.maximum(spot_final - spot_minmax, 0.0)
        return jnp.maximum(spot_minmax - spot_final, 0.0)


@dataclass(frozen=True)
class ContinuousPartialFloatingLookbackOption:
    """Partial-time floating strike lookback."""
    is_call: bool
    current_minmax: float
    lambda_: float  # strike multiplier
    lookback_period_end: float  # time at which lookback monitoring ends
    maturity: float


@dataclass(frozen=True)
class ContinuousPartialFixedLookbackOption:
    """Partial-time fixed strike lookback."""
    strike: float
    is_call: bool
    current_minmax: float
    lookback_period_start: float  # time at which lookback monitoring starts
    maturity: float
