"""Cliquet (ratchet) options — options on periodic returns."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import jax.numpy as jnp


@dataclass(frozen=True)
class CliquetOption:
    """Cliquet option — option on capped/floored periodic returns.

    The payoff is based on the sum of capped/floored returns over
    a set of reset dates.

    Parameters
    ----------
    local_cap : cap on each period return (None = no cap)
    local_floor : floor on each period return (None = no floor)
    global_cap : cap on total accumulated return (None = no cap)
    global_floor : floor on total accumulated return (None = no floor)
    reset_dates : sequence of reset times
    maturity : final maturity
    is_call : True for call, False for put
    """
    local_cap: float = None
    local_floor: float = None
    global_cap: float = None
    global_floor: float = None
    reset_dates: tuple = ()
    maturity: float = 1.0
    is_call: bool = True

    def payoff(self, periodic_returns):
        """Compute payoff from periodic returns.

        Parameters
        ----------
        periodic_returns : array of returns for each period (S_{i+1}/S_i - 1)
        """
        capped = periodic_returns
        if self.local_cap is not None:
            capped = jnp.minimum(capped, self.local_cap)
        if self.local_floor is not None:
            capped = jnp.maximum(capped, self.local_floor)

        total = jnp.sum(capped)

        if self.global_cap is not None:
            total = jnp.minimum(total, self.global_cap)
        if self.global_floor is not None:
            total = jnp.maximum(total, self.global_floor)

        if self.is_call:
            return jnp.maximum(total, 0.0)
        return jnp.maximum(-total, 0.0)
