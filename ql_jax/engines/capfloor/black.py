"""Black cap/floor pricing engine.

Prices each caplet/floorlet using Black's formula for interest rate options.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.stats import norm


def black_capfloor_price(cap_floor, discount_curve_fn, forward_rates, volatilities):
    """Price a cap or floor using Black's formula.

    Parameters
    ----------
    cap_floor : CapFloor instrument
    discount_curve_fn : callable(t) -> discount factor P(0,t)
    forward_rates : array of forward rates for each period
    volatilities : array of Black volatilities for each caplet/floorlet

    Returns
    -------
    price : total cap/floor price
    """
    n = cap_floor.n_periods
    K = cap_floor.strike
    N = cap_floor.notional
    omega = 1.0 if cap_floor.is_cap else -1.0

    total = 0.0
    for i in range(n):
        T_reset = float(cap_floor.reset_dates[i])
        T_pay = float(cap_floor.payment_dates[i])
        tau = float(cap_floor.accrual_fractions[i])
        F = forward_rates[i]
        sigma = volatilities[i]

        df = discount_curve_fn(T_pay)
        caplet = _black_caplet(F, K, T_reset, sigma, tau, df, omega, N)
        total = total + caplet

    return total


def black_capfloor_price_flat_vol(cap_floor, discount_curve_fn, forward_rates, flat_vol):
    """Price with a single flat volatility for all caplets."""
    n = cap_floor.n_periods
    vols = jnp.full(n, flat_vol)
    return black_capfloor_price(cap_floor, discount_curve_fn, forward_rates, vols)


def _black_caplet(F, K, T, sigma, tau, df, omega, notional):
    """Price a single caplet/floorlet using Black's formula.

    Parameters
    ----------
    F : forward rate
    K : strike
    T : option expiry (reset date)
    sigma : Black volatility
    tau : accrual fraction
    df : discount factor to payment date
    omega : +1 for caplet, -1 for floorlet
    notional : notional

    Returns
    -------
    caplet/floorlet price
    """
    F = jnp.asarray(F, dtype=jnp.float64)
    K = jnp.asarray(K, dtype=jnp.float64)

    # Handle T=0 case (first caplet at inception has zero time value)
    safe_T = jnp.maximum(T, 1e-10)
    sqrt_T = jnp.sqrt(safe_T)
    d1 = (jnp.log(F / K) + 0.5 * sigma**2 * safe_T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    price = df * tau * notional * omega * (F * norm.cdf(omega * d1) - K * norm.cdf(omega * d2))
    # At T=0, caplet is just intrinsic value
    intrinsic = df * tau * notional * jnp.maximum(omega * (F - K), 0.0)
    return jnp.where(T < 1e-8, intrinsic, price)


def implied_black_vol_caplet(market_price, F, K, T, tau, df, omega, notional):
    """Implied Black vol for a single caplet via Newton's method.

    Parameters
    ----------
    market_price : observed caplet price
    F, K, T, tau, df, omega, notional : as in _black_caplet

    Returns
    -------
    implied vol
    """
    sigma = 0.2  # initial guess
    for _ in range(50):
        price = _black_caplet(F, K, T, sigma, tau, df, omega, notional)
        vega = _black_caplet_vega(F, K, T, sigma, tau, df, notional)
        if abs(float(vega)) < 1e-15:
            break
        sigma = sigma - (price - market_price) / vega
        sigma = jnp.clip(sigma, 1e-4, 5.0)
    return sigma


def _black_caplet_vega(F, K, T, sigma, tau, df, notional):
    """Vega of a caplet (derivative w.r.t. sigma)."""
    sqrt_T = jnp.sqrt(T)
    d1 = (jnp.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
    return df * tau * notional * F * norm.pdf(d1) * sqrt_T
