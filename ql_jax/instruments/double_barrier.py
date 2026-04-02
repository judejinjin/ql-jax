"""Double barrier options."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import jax.numpy as jnp


class DoubleBarrierType(Enum):
    KNOCK_IN = "knock_in"
    KNOCK_OUT = "knock_out"
    KIKO = "knock_in_knock_out"   # knock in at one, knock out at other
    KOKI = "knock_out_knock_in"


@dataclass(frozen=True)
class DoubleBarrierOption:
    """Double barrier option with upper and lower barriers.

    Parameters
    ----------
    barrier_lo : lower barrier level
    barrier_hi : upper barrier level
    barrier_type : DoubleBarrierType
    strike : strike price
    is_call : True for call, False for put
    rebate : rebate paid if knocked out (default 0)
    maturity : time to maturity
    """
    barrier_lo: float
    barrier_hi: float
    barrier_type: DoubleBarrierType
    strike: float
    is_call: bool
    rebate: float = 0.0
    maturity: float = 1.0

    def vanilla_payoff(self, spot):
        if self.is_call:
            return jnp.maximum(spot - self.strike, 0.0)
        return jnp.maximum(self.strike - spot, 0.0)

    def is_alive(self, spot_path):
        """Check if option is still alive (not knocked out) along a path.

        Parameters
        ----------
        spot_path : array of spot prices along the path
        """
        above_lo = jnp.all(spot_path > self.barrier_lo)
        below_hi = jnp.all(spot_path < self.barrier_hi)

        if self.barrier_type == DoubleBarrierType.KNOCK_OUT:
            return above_lo & below_hi
        elif self.barrier_type == DoubleBarrierType.KNOCK_IN:
            return ~(above_lo & below_hi)
        return above_lo & below_hi  # default to knock-out behavior
