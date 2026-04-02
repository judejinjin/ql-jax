"""Default density term structures."""

from __future__ import annotations

import jax.numpy as jnp

from ql_jax.termstructures.credit.default_curves import DefaultProbabilityTermStructure


class InterpolatedDefaultDensityCurve(DefaultProbabilityTermStructure):
    """Default probability curve built from interpolated default densities.

    The default density d(t) is related to survival probability S(t) by:
        S(t) = 1 - integral_0^t d(s) ds

    Parameters
    ----------
    reference_date_ : Date
    times : array of pillar times
    densities : array of default density values
    day_counter_ : DayCountConvention, optional
    """

    def __init__(self, reference_date_, times, densities, day_counter_=None):
        self._ref_date = reference_date_
        self._times = jnp.asarray(times, dtype=jnp.float64)
        self._densities = jnp.asarray(densities, dtype=jnp.float64)
        self._day_counter = day_counter_
        # Precompute cumulative integral (trapezoid) for survival prob
        dt = jnp.diff(self._times, prepend=0.0)
        avg_d = 0.5 * (
            jnp.concatenate([jnp.array([self._densities[0]]), self._densities[:-1]])
            + self._densities
        )
        self._cum_integral = jnp.cumsum(avg_d * dt)

    @property
    def reference_date(self):
        return self._ref_date

    @property
    def day_counter(self):
        return self._day_counter

    def survival_probability(self, t):
        t = jnp.float64(t)
        integral = jnp.interp(t, self._times, self._cum_integral)
        return jnp.maximum(1.0 - integral, 0.0)

    def default_density(self, t):
        t = jnp.float64(t)
        return jnp.interp(t, self._times, self._densities)

    def hazard_rate(self, t):
        d = self.default_density(t)
        s = self.survival_probability(t)
        return d / jnp.maximum(s, 1e-15)
