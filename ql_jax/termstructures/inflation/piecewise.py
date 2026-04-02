"""Piecewise zero-inflation and year-on-year inflation curves."""

import jax.numpy as jnp
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class PiecewiseZeroInflationCurve:
    """Bootstrapped zero-coupon inflation curve.

    Parameters
    ----------
    base_date : float – base date (year fraction)
    observation_lag : float – CPI observation lag (e.g. 0.25 for 3 months)
    times : array – pillar maturities
    zero_rates : array – zero inflation rates at each pillar
    """
    base_date: float
    observation_lag: float
    times: jnp.ndarray
    zero_rates: jnp.ndarray

    def zero_rate(self, t):
        """Interpolated zero inflation rate at time t."""
        return jnp.interp(t, self.times, self.zero_rates)

    def cpi_ratio(self, t):
        """CPI(t) / CPI(base) = (1 + zero_rate)^t."""
        zr = self.zero_rate(t)
        return (1.0 + zr) ** t

    def discount_factor(self, t, nominal_discount_fn):
        """Real discount factor = nominal / CPI ratio."""
        return nominal_discount_fn(t) / self.cpi_ratio(t)


@dataclass(frozen=True)
class PiecewiseYoYInflationCurve:
    """Bootstrapped year-on-year inflation curve.

    Parameters
    ----------
    base_date : float
    observation_lag : float
    times : array – pillar maturities
    yoy_rates : array – YoY inflation rates at each pillar
    """
    base_date: float
    observation_lag: float
    times: jnp.ndarray
    yoy_rates: jnp.ndarray

    def yoy_rate(self, t):
        """Interpolated YoY inflation rate at time t."""
        return jnp.interp(t, self.times, self.yoy_rates)


def bootstrap_zero_inflation(helpers, nominal_discount_fn, base_cpi,
                               observation_lag=0.25):
    """Bootstrap zero-coupon inflation curve from swap helpers.

    Parameters
    ----------
    helpers : list of dicts with 'maturity' and 'zero_rate' (market quote)
    nominal_discount_fn : callable(t) -> nominal discount factor
    base_cpi : float – base CPI level

    Returns PiecewiseZeroInflationCurve.
    """
    maturities = jnp.array([h['maturity'] for h in helpers])
    zero_rates = jnp.array([h['zero_rate'] for h in helpers])

    return PiecewiseZeroInflationCurve(
        base_date=0.0,
        observation_lag=observation_lag,
        times=maturities,
        zero_rates=zero_rates,
    )


def bootstrap_yoy_inflation(helpers, nominal_discount_fn, observation_lag=0.25):
    """Bootstrap YoY inflation curve from swap helpers.

    Parameters
    ----------
    helpers : list of dicts with 'maturity' and 'yoy_rate'

    Returns PiecewiseYoYInflationCurve.
    """
    maturities = jnp.array([h['maturity'] for h in helpers])
    yoy_rates = jnp.array([h['yoy_rate'] for h in helpers])

    return PiecewiseYoYInflationCurve(
        base_date=0.0,
        observation_lag=observation_lag,
        times=maturities,
        yoy_rates=yoy_rates,
    )
