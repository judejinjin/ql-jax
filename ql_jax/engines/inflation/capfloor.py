"""Inflation cap/floor instruments and pricing."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax.scipy.stats import norm


@dataclass(frozen=True)
class InflationCapFloor:
    """Year-on-year inflation cap or floor.

    Parameters
    ----------
    cap_or_floor : 'cap' or 'floor'
    strike : inflation strike rate
    notional : notional
    payment_dates : array of payment dates (year fractions)
    base_cpi : base CPI value
    """
    cap_or_floor: str
    strike: float
    notional: float
    payment_dates: jnp.ndarray
    base_cpi: float = 100.0

    @property
    def is_cap(self):
        return self.cap_or_floor == 'cap'

    @property
    def n_periods(self):
        return self.payment_dates.shape[0]


def black_yoy_capfloor_price(capfloor, discount_fn, forward_yoy_rates, volatilities):
    """Price a YoY inflation cap/floor using Black's formula.

    Parameters
    ----------
    capfloor : InflationCapFloor
    discount_fn : callable(t) -> P(0,t)
    forward_yoy_rates : array of forward year-on-year inflation rates
    volatilities : array of Black vols per caplet

    Returns
    -------
    price
    """
    K = capfloor.strike
    N = capfloor.notional
    omega = 1.0 if capfloor.is_cap else -1.0
    n = capfloor.n_periods

    total = 0.0
    for i in range(n):
        T = float(capfloor.payment_dates[i])
        F = forward_yoy_rates[i]
        sigma = volatilities[i]
        df = discount_fn(T)

        # Black's formula on inflation rate
        safe_T = jnp.maximum(T, 1e-10)
        sqrt_T = jnp.sqrt(safe_T)
        d1 = (jnp.log(F / K) + 0.5 * sigma**2 * safe_T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T

        caplet = df * N * omega * (F * norm.cdf(omega * d1) - K * norm.cdf(omega * d2))
        caplet = jnp.where(T < 1e-8, df * N * jnp.maximum(omega * (F - K), 0.0), caplet)
        total = total + caplet

    return total


def zero_coupon_inflation_swap_npv(
    notional, fixed_rate, maturity,
    discount_fn, forward_cpi_fn, base_cpi=100.0,
):
    """NPV of a zero-coupon inflation swap (pay fixed, receive inflation).

    Inflation leg pays: N * (CPI(T)/CPI(0) - 1) at maturity.
    Fixed leg pays: N * ((1+K)^T - 1) at maturity.

    Parameters
    ----------
    notional : notional
    fixed_rate : fixed inflation rate
    maturity : swap maturity
    discount_fn : P(0,T)
    forward_cpi_fn : callable(t) -> expected CPI at time t
    base_cpi : CPI at inception

    Returns
    -------
    npv : from payer (fixed) perspective
    """
    df = discount_fn(maturity)
    cpi_T = forward_cpi_fn(maturity)

    inflation_leg = notional * (cpi_T / base_cpi - 1.0) * df
    fixed_leg = notional * ((1.0 + fixed_rate)**maturity - 1.0) * df

    return inflation_leg - fixed_leg
