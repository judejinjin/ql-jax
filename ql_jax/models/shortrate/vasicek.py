"""Vasicek short-rate model.

dr = a(b - r)dt + sigma dW

Closed-form zero-coupon bond price P(t,T) available.
"""

from __future__ import annotations

import jax.numpy as jnp


def vasicek_bond_price(r, a, b, sigma, t, T):
    """Zero-coupon bond price P(t,T) under Vasicek model.

    Parameters
    ----------
    r : current short rate
    a : mean reversion speed
    b : long-run mean rate
    sigma : volatility
    t : current time
    T : maturity time

    Returns
    -------
    P(t,T) : bond price
    """
    tau = T - t
    B = (1.0 - jnp.exp(-a * tau)) / a
    A = jnp.exp(
        (B - tau) * (a**2 * b - 0.5 * sigma**2) / a**2
        - sigma**2 * B**2 / (4.0 * a)
    )
    return A * jnp.exp(-B * r)


def vasicek_discount(r, a, b, sigma, T):
    """Discount factor from 0 to T under Vasicek. Shorthand for P(0,T)."""
    return vasicek_bond_price(r, a, b, sigma, 0.0, T)


def vasicek_zero_rate(r, a, b, sigma, T):
    """Continuously compounded zero rate R(0,T) such that P(0,T)=exp(-R*T)."""
    P = vasicek_discount(r, a, b, sigma, T)
    return -jnp.log(P) / T


def vasicek_caplet_price(r, a, b, sigma, K, T_reset, T_pay, notional=1.0):
    """Price of a caplet paying max(L-K,0)*tau at T_pay.

    Uses Jamshidian decomposition: caplet = put on ZCB.

    Parameters
    ----------
    r : current short rate
    a : mean reversion speed
    b : long-run mean
    sigma : volatility
    K : cap strike rate
    T_reset : reset date
    T_pay : payment date (= T_reset + tau)
    notional : notional amount

    Returns
    -------
    caplet price
    """
    tau = T_pay - T_reset
    X = (1.0 + K * tau)  # strike for the ZCB put

    P_reset = vasicek_bond_price(r, a, b, sigma, 0.0, T_reset)
    P_pay = vasicek_bond_price(r, a, b, sigma, 0.0, T_pay)

    # Bond option volatility
    B_val = (1.0 - jnp.exp(-a * tau)) / a
    sigma_p = sigma * B_val * jnp.sqrt((1.0 - jnp.exp(-2.0 * a * T_reset)) / (2.0 * a))

    from jax.scipy.stats import norm
    d1 = jnp.log(P_pay / (X * P_reset)) / sigma_p + 0.5 * sigma_p
    d2 = d1 - sigma_p

    # Put on ZCB = call on rate
    put = X * P_reset * norm.cdf(-d2) - P_pay * norm.cdf(-d1)
    return notional * put
