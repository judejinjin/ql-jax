"""Closed-form American payoff utilities.

Analytic formulas for American digital/barrier payoffs at hit and at expiry,
following the approach of Haug (2006) / QuantLib ``americanpayoffathit.hpp``
and ``americanpayoffatexpiry.hpp``.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.stats import norm


def american_payoff_at_hit(
    spot: float,
    strike: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    option_type: int = 1,
) -> float:
    """Value of an American cash-or-nothing that pays 1 at first hit of strike.

    For a "down" barrier (strike < spot and option_type = -1) or
    "up" barrier (strike > spot and option_type = +1).

    Parameters
    ----------
    spot : current price
    strike : barrier level
    r, q, sigma, T : standard BS parameters
    option_type : +1 up, -1 down

    Returns
    -------
    price : present value of 1 paid at first passage time
    """
    sigma2 = sigma * sigma
    mu = (r - q) / sigma2 - 0.5
    lam = jnp.sqrt(mu * mu + 2.0 * r / sigma2)
    log_ratio = jnp.log(spot / strike)

    if option_type > 0:
        # Up: strike > spot
        alpha = -mu + lam
        price = jnp.where(
            spot >= strike,
            1.0,
            (spot / strike) ** alpha,
        )
    else:
        # Down: strike < spot
        alpha = -mu - lam
        price = jnp.where(
            spot <= strike,
            1.0,
            (spot / strike) ** alpha,
        )

    return float(price)


def american_payoff_at_expiry(
    spot: float,
    strike: float,
    r: float,
    q: float,
    sigma: float,
    T: float,
    option_type: int = 1,
) -> float:
    """Value of an American cash-or-nothing that pays 1 *at expiry*
    if the barrier has been hit during [0, T].

    Uses the reflection principle / image solution approach.

    Parameters
    ----------
    spot, strike, r, q, sigma, T : standard parameters
    option_type : +1 up, -1 down

    Returns
    -------
    price : present value
    """
    if T <= 0.0:
        return float(jnp.where(option_type * (spot - strike) >= 0, 1.0, 0.0))

    sigma2 = sigma * sigma
    sqrt_T = jnp.sqrt(T)
    mu = (r - q) / sigma2 - 0.5
    log_HS = jnp.log(strike / spot)

    d1 = -log_HS / (sigma * sqrt_T) + mu * sigma * sqrt_T
    d2 = -log_HS / (sigma * sqrt_T) - mu * sigma * sqrt_T

    discount = jnp.exp(-r * T)

    if option_type > 0:
        # Up type: barrier above spot
        price = discount * (
            norm.cdf(d1)
            + (strike / spot) ** (2.0 * mu) * norm.cdf(d2)
        )
    else:
        # Down type
        price = discount * (
            norm.cdf(-d1)
            + (strike / spot) ** (2.0 * mu) * norm.cdf(-d2)
        )

    return float(price)
