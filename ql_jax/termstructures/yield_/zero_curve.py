"""Interpolated zero-rate curve.

Given (time, zero_rate) pillars, interpolates zero rates and
derives discount factors: P(0,t) = exp(-z(t) * t).
"""

import jax.numpy as jnp

from ql_jax.time.date import Date
from ql_jax.termstructures.yield_.base import YieldTermStructure
from ql_jax.math.interpolations.linear import build as linear_build, evaluate as linear_eval


class ZeroCurve(YieldTermStructure):
    """Yield curve from interpolated zero rates."""

    def __init__(
        self,
        reference_date: Date,
        dates: list[Date],
        zero_rates: list[float],
        day_counter: str = 'Actual365Fixed',
    ):
        super().__init__(reference_date, day_counter)
        times = [float(self.time_from_reference(d)) for d in dates]
        self._times = jnp.array(times, dtype=jnp.float64)
        self._zero_rates = jnp.array(zero_rates, dtype=jnp.float64)
        self._interp_state = linear_build(self._times, self._zero_rates)

    def zero_rate_impl(self, t):
        return linear_eval(self._interp_state, t)

    def discount_impl(self, t):
        r = self.zero_rate_impl(t)
        return jnp.exp(-r * t)
