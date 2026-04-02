"""Interpolated discount curve.

Given a set of (time, discount_factor) pillars, interpolates between them.
Differentiable through the interpolation layer.
"""

import jax.numpy as jnp

from ql_jax.time.date import Date
from ql_jax.termstructures.yield_.base import YieldTermStructure
from ql_jax.math.interpolations.linear import build as linear_build, evaluate as linear_eval
from ql_jax.math.interpolations.log import build as log_build, evaluate as log_eval


class DiscountCurve(YieldTermStructure):
    """Yield curve from interpolated discount factors.

    Default interpolation is log-linear on discount factors.
    """

    def __init__(
        self,
        reference_date: Date,
        dates: list[Date],
        discounts: list[float],
        day_counter: str = 'Actual365Fixed',
        interpolation: str = 'log_linear',
    ):
        super().__init__(reference_date, day_counter)
        times = [float(self.time_from_reference(d)) for d in dates]
        self._times = jnp.array(times, dtype=jnp.float64)
        self._discounts = jnp.array(discounts, dtype=jnp.float64)
        self._interpolation = interpolation

        if interpolation == 'log_linear':
            self._interp_state = log_build(self._times, self._discounts)
        else:
            self._interp_state = linear_build(self._times, self._discounts)

    def discount_impl(self, t):
        if self._interpolation == 'log_linear':
            return log_eval(self._interp_state, t)
        return linear_eval(self._interp_state, t)
