"""Perturbative barrier engine — asymptotic expansion for barrier options.

Uses a perturbative (asymptotic series) approach to price barrier options
with time-dependent volatility.

Reference: Davydov, D. & Linetsky, V. (2001), "Pricing and Hedging
Path-Dependent Options Under the CEV Process", Management Science.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.stats import norm

from ql_jax.engines.analytic.barrier import barrier_price


def perturbative_barrier_price(
    S, K, T, r, q, sigma, barrier, rebate,
    option_type, barrier_type,
    vol_slope=0.0, vol_curv=0.0,
):
    """Barrier option price with perturbative correction for vol skew.

    Computes the flat-vol barrier price and adds corrections due to
    local volatility slope (skew) and curvature (smile).

    Parameters
    ----------
    S : spot
    K : strike
    T : maturity
    r : risk-free rate
    q : dividend yield
    sigma : ATM volatility
    barrier : barrier level
    rebate : rebate
    option_type : 'call' or 'put'
    barrier_type : e.g. 'down_and_out'
    vol_slope : d(sigma)/d(log S) at barrier (skew)
    vol_curv : d^2(sigma)/d(log S)^2 at barrier (smile curvature)

    Returns
    -------
    price : barrier option price with perturbative corrections
    """
    S, K, T, r, q, sigma = (
        jnp.asarray(x, dtype=jnp.float64)
        for x in (S, K, T, r, q, sigma)
    )
    vol_slope = jnp.float64(vol_slope)
    vol_curv = jnp.float64(vol_curv)
    barrier = jnp.float64(barrier)

    # Flat-vol base price
    base = barrier_price(S, K, T, r, q, sigma, barrier, rebate,
                          option_type, barrier_type)

    # Perturbative corrections
    H = barrier
    sqrt_T = jnp.sqrt(T)
    x0 = jnp.log(S / H) / (sigma * sqrt_T)
    y0 = jnp.log(H / K) / (sigma * sqrt_T)

    # First-order correction: proportional to vol_slope * sigma * sqrt(T)
    n_x0 = jnp.exp(-0.5 * x0**2) / jnp.sqrt(2.0 * jnp.pi)
    correction1 = vol_slope * sigma * T * n_x0 * S * jnp.exp(-q * T) * (x0 / sigma)

    # Second-order correction: proportional to vol_curv * sigma^2 * T
    correction2 = 0.5 * vol_curv * sigma**2 * T * n_x0 * S * jnp.exp(-q * T) * (x0**2 - 1.0) / sigma

    return base + correction1 + correction2
