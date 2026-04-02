"""Interpolated forward-rate curve.

Given (time, instantaneous_forward) pillars, integrates to get zero rates
and discount factors.
"""

import jax.numpy as jnp

from ql_jax.time.date import Date
from ql_jax.termstructures.yield_.base import YieldTermStructure
from ql_jax.math.interpolations.linear import build as linear_build, evaluate as linear_eval


class ForwardCurve(YieldTermStructure):
    """Yield curve from interpolated instantaneous forward rates.

    discount(t) = exp(-∫₀ᵗ f(s) ds)
    """

    def __init__(
        self,
        reference_date: Date,
        dates: list[Date],
        forwards: list[float],
        day_counter: str = 'Actual365Fixed',
    ):
        super().__init__(reference_date, day_counter)
        times = [float(self.time_from_reference(d)) for d in dates]
        self._times = jnp.array(times, dtype=jnp.float64)
        self._forwards = jnp.array(forwards, dtype=jnp.float64)
        self._interp_state = linear_build(self._times, self._forwards)

    def forward_rate_impl(self, t):
        return linear_eval(self._interp_state, t)

    def discount_impl(self, t):
        # Integrate forward rates numerically using trapezoidal rule
        n = 200
        dt = t / jnp.maximum(n, 1.0)
        ts = jnp.linspace(0.0, t, n + 1)
        fs = jnp.array([
            float(linear_eval(self._interp_state, ti))
            for ti in ts
        ])
        integral = jnp.trapezoid(fs, ts)
        return jnp.exp(-integral)

    def zero_rate_impl(self, t):
        df = self.discount_impl(t)
        return jnp.where(t > 0, -jnp.log(df) / t, self._forwards[0])
