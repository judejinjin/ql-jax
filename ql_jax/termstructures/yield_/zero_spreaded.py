"""Piecewise zero-spreaded term structure.

Adds piecewise-constant or interpolated zero-rate spreads to a base curve.
"""

import jax.numpy as jnp
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class PiecewiseZeroSpreadedTermStructure:
    """Base curve + interpolated zero-rate spread.

    Parameters
    ----------
    base_curve_fn : callable(t) -> discount factor
    spread_times : array of spread breakpoint times
    spreads : array of zero-rate spreads at each breakpoint
    interpolation : 'linear' or 'flat_forward'
    """
    base_curve_fn: Callable
    spread_times: jnp.ndarray
    spreads: jnp.ndarray
    interpolation: str = 'linear'

    def spread_at(self, t):
        """Interpolated spread at time t."""
        if self.interpolation == 'flat_forward':
            idx = jnp.searchsorted(self.spread_times, t, side='right') - 1
            idx = jnp.clip(idx, 0, len(self.spreads) - 1)
            return self.spreads[idx]
        else:  # linear
            return jnp.interp(t, self.spread_times, self.spreads)

    def discount(self, t):
        base_df = self.base_curve_fn(t)
        spread = self.spread_at(t)
        return base_df * jnp.exp(-spread * t)

    def zero_rate(self, t):
        return -jnp.log(self.discount(t)) / jnp.maximum(t, 1e-10)

    def forward_rate(self, t1, t2):
        return -jnp.log(self.discount(t2) / self.discount(t1)) / (t2 - t1)
