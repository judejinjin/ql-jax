"""Cox-Ingersoll-Ross (CIR) short-rate model.

dr = a(b - r)dt + sigma*sqrt(r)*dW

Closed-form bond price available when 2*a*b > sigma^2 (Feller condition).
"""

from __future__ import annotations

import jax.numpy as jnp


def cir_bond_price(r, a, b, sigma, t, T):
    """Zero-coupon bond price P(t,T) under CIR model.

    Parameters
    ----------
    r : current short rate
    a : mean reversion speed
    b : long-run mean rate
    sigma : volatility
    t : current time
    T : maturity

    Returns
    -------
    P(t,T)
    """
    tau = T - t
    gamma = jnp.sqrt(a**2 + 2.0 * sigma**2)

    num = 2.0 * gamma * jnp.exp((a + gamma) * tau / 2.0)
    denom = (gamma + a) * (jnp.exp(gamma * tau) - 1.0) + 2.0 * gamma

    A = (num / denom) ** (2.0 * a * b / sigma**2)
    B = 2.0 * (jnp.exp(gamma * tau) - 1.0) / denom

    return A * jnp.exp(-B * r)


def cir_discount(r, a, b, sigma, T):
    """Discount factor from 0 to T."""
    return cir_bond_price(r, a, b, sigma, 0.0, T)


def cir_zero_rate(r, a, b, sigma, T):
    """Continuously compounded zero rate."""
    P = cir_discount(r, a, b, sigma, T)
    return -jnp.log(P) / T
