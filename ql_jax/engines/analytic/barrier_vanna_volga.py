"""Vanna-volga barrier option engines.

Uses the vanna-volga method (Castagna & Mercurio, 2007) to price single
and double barrier options by replicating vega, vanna, and volga risks
with three vanilla pillars.

Reference: Castagna, A. & Mercurio, F. (2007), "The vanna-volga method
for implied volatilities".
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.stats import norm

from ql_jax.engines.analytic.barrier import barrier_price
from ql_jax.engines.analytic.black_formula import black_scholes_price


def vanna_volga_barrier_price(
    S, K, T, r, q, sigma_atm,
    barrier, rebate,
    option_type, barrier_type,
    sigma_25d_put=None, sigma_25d_call=None,
):
    """Single barrier price via vanna-volga method.

    Parameters
    ----------
    S : spot
    K : strike
    T : maturity
    r : domestic rate
    q : foreign rate (or dividend yield)
    sigma_atm : ATM volatility
    barrier : barrier level
    rebate : rebate paid if knocked out
    option_type : 'call' or 'put'
    barrier_type : 'down_and_out', 'down_and_in', 'up_and_out', 'up_and_in'
    sigma_25d_put : 25-delta put vol (if None, uses flat vol)
    sigma_25d_call : 25-delta call vol (if None, uses flat vol)

    Returns
    -------
    price : barrier option price
    """
    S, K, T, r, q, sigma_atm = (
        jnp.asarray(x, dtype=jnp.float64)
        for x in (S, K, T, r, q, sigma_atm)
    )

    if sigma_25d_put is None or sigma_25d_call is None:
        return barrier_price(S, K, T, r, q, sigma_atm, barrier, rebate,
                            option_type, barrier_type)

    sigma_p = jnp.float64(sigma_25d_put)
    sigma_c = jnp.float64(sigma_25d_call)

    # Three pillar strikes (25d put, ATM, 25d call)
    F = S * jnp.exp((r - q) * T)
    sqrt_T = jnp.sqrt(T)

    K1 = F * jnp.exp(-0.67449 * sigma_p * sqrt_T + 0.5 * sigma_p**2 * T)
    K2 = F * jnp.exp(0.5 * sigma_atm**2 * T)  # ATM DNS
    K3 = F * jnp.exp(0.67449 * sigma_c * sqrt_T + 0.5 * sigma_c**2 * T)

    # BS prices and barrier prices at ATM vol
    bs_K = black_scholes_price(S, K, T, r, q, sigma_atm, 1)
    bar_atm = barrier_price(S, K, T, r, q, sigma_atm, barrier, rebate,
                            option_type, barrier_type)

    # Vanna-volga adjustment weights
    log_K = jnp.log(K)
    log_K1 = jnp.log(K1)
    log_K2 = jnp.log(K2)
    log_K3 = jnp.log(K3)

    x1 = (log_K - log_K2) * (log_K - log_K3) / ((log_K1 - log_K2) * (log_K1 - log_K3))
    x2 = (log_K - log_K1) * (log_K - log_K3) / ((log_K2 - log_K1) * (log_K2 - log_K3))
    x3 = (log_K - log_K1) * (log_K - log_K2) / ((log_K3 - log_K1) * (log_K3 - log_K2))

    # Cost of hedging = sum xi * (C_mkt(Ki) - C_bs(Ki))
    cost1 = x1 * (black_scholes_price(S, K1, T, r, q, sigma_p, 1) -
                   black_scholes_price(S, K1, T, r, q, sigma_atm, 1))
    cost2 = x2 * jnp.float64(0.0)  # ATM pillar has zero cost
    cost3 = x3 * (black_scholes_price(S, K3, T, r, q, sigma_c, 1) -
                   black_scholes_price(S, K3, T, r, q, sigma_atm, 1))

    # VV adjustment
    adjustment = cost1 + cost2 + cost3

    # Survival probability approx for the barrier
    eta = jnp.where(barrier < S, 1.0, -1.0)
    mu = (r - q - 0.5 * sigma_atm**2)
    touch_prob = norm.cdf(eta * (jnp.log(barrier / S) - mu * T) / (sigma_atm * sqrt_T))

    # Scale adjustment by no-touch probability
    no_touch = 1.0 - touch_prob
    price = bar_atm + no_touch * adjustment

    return jnp.maximum(price, 0.0)


def vanna_volga_double_barrier_price(
    S, K, T, r, q, sigma_atm,
    lower_barrier, upper_barrier,
    rebate, option_type,
    sigma_25d_put=None, sigma_25d_call=None,
):
    """Double barrier price via vanna-volga method.

    Parameters
    ----------
    S : spot
    K : strike
    T : maturity
    r, q : rates
    sigma_atm : ATM vol
    lower_barrier, upper_barrier : barrier levels
    rebate : knock-out rebate
    option_type : 'call' or 'put'
    sigma_25d_put, sigma_25d_call : smile vols

    Returns
    -------
    price : double knock-out option price
    """
    from ql_jax.engines.analytic.double_barrier import double_barrier_price

    S, K, T, r, q, sigma_atm = (
        jnp.asarray(x, dtype=jnp.float64)
        for x in (S, K, T, r, q, sigma_atm)
    )

    db_atm = double_barrier_price(S, K, T, r, q, sigma_atm,
                                   lower_barrier, upper_barrier, rebate, option_type)

    if sigma_25d_put is None or sigma_25d_call is None:
        return db_atm

    # Simplified vanna-volga adjustment
    sigma_p = jnp.float64(sigma_25d_put)
    sigma_c = jnp.float64(sigma_25d_call)
    skew_adj = 0.5 * ((sigma_p - sigma_atm) + (sigma_c - sigma_atm))
    sigma_adj = sigma_atm + skew_adj

    db_adj = double_barrier_price(S, K, T, r, q, sigma_adj,
                                   lower_barrier, upper_barrier, rebate, option_type)

    return db_adj
