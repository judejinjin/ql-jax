"""Spread-based yield curve wrappers."""

from __future__ import annotations

import jax.numpy as jnp

from ql_jax.termstructures.yield_.base import YieldTermStructure


class SpreadDiscountCurve(YieldTermStructure):
    """Discount curve defined as base_discount(t) * exp(-spread * t).

    Parameters
    ----------
    base : YieldTermStructure
        Underlying discount curve.
    spread : float
        Constant spread applied on top.
    """

    def __init__(self, base: YieldTermStructure, spread: float):
        super().__init__(base.reference_date, base.day_counter)
        self._base = base
        self._spread = jnp.float64(spread)

    @property
    def reference_date(self):
        return self._base.reference_date

    @property
    def day_counter(self):
        return self._base.day_counter

    def discount_impl(self, t):
        return self._base.discount(t) * jnp.exp(-self._spread * t)


class PiecewiseForwardSpreadedTermStructure(YieldTermStructure):
    """Yield curve = base curve + piecewise-constant forward spread.

    Parameters
    ----------
    base : YieldTermStructure
    times : array of breakpoint times
    spreads : array of forward spreads between breakpoints
    """

    def __init__(self, base: YieldTermStructure, times, spreads):
        self._base = base
        self._times = jnp.asarray(times, dtype=jnp.float64)
        self._spreads = jnp.asarray(spreads, dtype=jnp.float64)

    @property
    def reference_date(self):
        return self._base.reference_date

    @property
    def day_counter(self):
        return self._base.day_counter

    def discount_impl(self, t):
        # Integrate piecewise-constant spread from 0 to t
        # For segment [t_{i}, t_{i+1}], spread = s_i
        # Integral = sum of s_i * (min(t, t_{i+1}) - t_i) for all applicable
        t = jnp.float64(t)
        n = len(self._times)
        # Prepend 0 to times
        ts = jnp.concatenate([jnp.array([0.0]), self._times])
        # For each segment, contribution = spread_i * max(0, min(t, ts[i+1]) - ts[i])
        # Pad spreads if needed: last spread continues beyond last breakpoint
        segments_start = ts[:-1]
        segments_end = ts[1:]
        contributions = self._spreads[:n] * jnp.maximum(
            0.0, jnp.minimum(t, segments_end) - segments_start
        )
        # If t extends beyond last breakpoint, use last spread
        extra = jnp.where(
            t > self._times[-1],
            self._spreads[-1] * (t - self._times[-1]),
            0.0,
        )
        total_spread = jnp.sum(contributions) + extra
        return self._base.discount(t) * jnp.exp(-total_spread)


class InterpolatedSimpleZeroCurve(YieldTermStructure):
    """Zero curve with simple (annual) compounding and linear interpolation.

    Parameters
    ----------
    reference_date_ : Date
    times : array of times
    rates : array of simple zero rates
    day_counter_ : DayCountConvention, optional
    """

    def __init__(self, reference_date_, times, rates, day_counter_=None):
        self._ref_date = reference_date_
        self._times = jnp.asarray(times, dtype=jnp.float64)
        self._rates = jnp.asarray(rates, dtype=jnp.float64)
        self._day_counter = day_counter_

    @property
    def reference_date(self):
        return self._ref_date

    @property
    def day_counter(self):
        return self._day_counter

    def discount_impl(self, t):
        # Simple compounding: D(t) = 1 / (1 + r * t)
        t = jnp.float64(t)
        r = jnp.interp(t, self._times, self._rates)
        return 1.0 / (1.0 + r * t)
