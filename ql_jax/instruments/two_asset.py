"""Two-asset options — barrier and correlation options on two underlyings."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class TwoAssetBarrierOption:
    """Two-asset barrier option.

    European option on asset 1, with a barrier monitored on asset 2.

    Parameters
    ----------
    strike : strike price for asset 1
    barrier : barrier level on asset 2
    is_call : True for call on asset 1
    is_knock_in : True for knock-in, False for knock-out
    barrier_above : True if barrier is above current level of asset 2
    maturity : time to maturity
    """
    strike: float
    barrier: float
    is_call: bool
    is_knock_in: bool
    barrier_above: bool
    maturity: float


@dataclass(frozen=True)
class TwoAssetCorrelationOption:
    """Two-asset correlation option.

    Payoff depends on both assets at expiry.
    Call payoff: max(S1 - K1, 0) if S2 > K2
    Put payoff:  max(K1 - S1, 0) if S2 < K2

    Parameters
    ----------
    strike1 : strike for asset 1
    strike2 : threshold for asset 2
    is_call : True for call
    maturity : time to maturity
    """
    strike1: float
    strike2: float
    is_call: bool
    maturity: float

    def payoff(self, s1, s2):
        if self.is_call:
            condition = s2 > self.strike2
            intrinsic = jnp.maximum(s1 - self.strike1, 0.0)
        else:
            condition = s2 < self.strike2
            intrinsic = jnp.maximum(self.strike1 - s1, 0.0)
        return jnp.where(condition, intrinsic, 0.0)
