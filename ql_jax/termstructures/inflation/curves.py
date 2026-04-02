"""Inflation term structures.

Zero-coupon and year-on-year inflation curves.
"""

from __future__ import annotations

import jax.numpy as jnp

from ql_jax.time.date import Date
from ql_jax.time.daycounter import year_fraction
from ql_jax.math.interpolations.linear import build as linear_build, evaluate as linear_eval


class InflationTermStructure:
    """Base class for inflation term structures."""

    def __init__(self, reference_date: Date, base_rate: float,
                 observation_lag_months: int = 3,
                 day_counter: str = 'Actual365Fixed'):
        self._reference_date = reference_date
        self._base_rate = base_rate
        self._observation_lag = observation_lag_months
        self._day_counter = day_counter

    @property
    def reference_date(self):
        return self._reference_date

    @property
    def base_rate(self):
        return self._base_rate

    def time_from_reference(self, d: Date):
        return year_fraction(self._reference_date, d, self._day_counter)


class ZeroInflationTermStructure(InflationTermStructure):
    """Base for zero-coupon inflation term structures.

    Returns the zero-inflation rate: the annualized rate such that
    I(t) / I(0) = (1 + z(t))^t
    """

    def zero_rate(self, t):
        """Zero-coupon inflation rate at time t. Override in subclass."""
        raise NotImplementedError


class InterpolatedZeroInflationCurve(ZeroInflationTermStructure):
    """Interpolated zero-coupon inflation curve."""

    def __init__(self, reference_date: Date, dates, rates,
                 base_rate: float = 0.02,
                 observation_lag_months: int = 3,
                 day_counter: str = 'Actual365Fixed'):
        super().__init__(reference_date, base_rate, observation_lag_months, day_counter)
        times = [year_fraction(reference_date, d, day_counter) for d in dates]
        self._times = jnp.array(times, dtype=jnp.float64)
        self._rates = jnp.array(rates, dtype=jnp.float64)
        self._interp_state = linear_build(self._times, self._rates)

    def zero_rate(self, t):
        t = jnp.asarray(t, dtype=jnp.float64)
        return linear_eval(self._interp_state, t)


class YoYInflationTermStructure(InflationTermStructure):
    """Base for year-on-year inflation term structures."""

    def yoy_rate(self, t):
        """Year-on-year inflation rate. Override in subclass."""
        raise NotImplementedError


class InterpolatedYoYInflationCurve(YoYInflationTermStructure):
    """Interpolated year-on-year inflation curve."""

    def __init__(self, reference_date: Date, dates, rates,
                 base_rate: float = 0.02,
                 observation_lag_months: int = 3,
                 day_counter: str = 'Actual365Fixed'):
        super().__init__(reference_date, base_rate, observation_lag_months, day_counter)
        times = [year_fraction(reference_date, d, day_counter) for d in dates]
        self._times = jnp.array(times, dtype=jnp.float64)
        self._rates = jnp.array(rates, dtype=jnp.float64)
        self._interp_state = linear_build(self._times, self._rates)

    def yoy_rate(self, t):
        t = jnp.asarray(t, dtype=jnp.float64)
        return linear_eval(self._interp_state, t)


class Seasonality:
    """Monthly seasonality adjustment factors.

    Multiplied into the inflation index to account for seasonal patterns.
    """

    def __init__(self, monthly_factors=None):
        if monthly_factors is None:
            self._factors = jnp.ones(12, dtype=jnp.float64)
        else:
            self._factors = jnp.array(monthly_factors, dtype=jnp.float64)
            assert len(self._factors) == 12

    def correction(self, month: int) -> float:
        """Seasonality factor for month (1-12)."""
        return float(self._factors[month - 1])

    def corrected_rate(self, rate: float, month: int) -> float:
        """Apply seasonality correction to an inflation rate."""
        return rate * float(self._factors[month - 1])
