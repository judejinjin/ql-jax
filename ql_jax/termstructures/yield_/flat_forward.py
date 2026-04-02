"""Flat forward yield curve.

The simplest yield term structure — a constant continuously compounded rate.
"""

import jax.numpy as jnp

from ql_jax.time.date import Date
from ql_jax.termstructures.yield_.base import YieldTermStructure


class FlatForward(YieldTermStructure):
    """Flat forward rate term structure.

    P(0, t) = exp(-r * t) for all t.
    """

    def __init__(self, reference_date: Date, rate, day_counter: str = 'Actual365Fixed'):
        super().__init__(reference_date, day_counter)
        self._rate = jnp.asarray(rate, dtype=jnp.float64)

    @property
    def rate(self):
        return self._rate

    def discount_impl(self, t):
        return jnp.exp(-self._rate * t)

    def zero_rate_impl(self, t):
        return self._rate

    def forward_rate_impl(self, t):
        return self._rate
