"""Barrier option instrument and analytic pricing."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
from jax.scipy.stats import norm

from ql_jax._util.types import OptionType, BarrierType


@dataclass(frozen=True)
class BarrierOption:
    """Barrier option (knock-in or knock-out)."""
    strike: float
    option_type: int       # OptionType.Call or Put
    barrier_type: int      # BarrierType.DownIn/UpIn/DownOut/UpOut
    barrier: float
    rebate: float = 0.0
    exercise: object = None


def analytic_barrier_price(
    S, K, T, r, q, sigma,
    option_type: int,
    barrier_type: int,
    barrier: float,
    rebate: float = 0.0,
):
    """Analytic European barrier option price (Reiner-Rubinstein).

    Parameters
    ----------
    S : spot price
    K : strike
    T : time to expiry
    r : risk-free rate
    q : dividend yield
    sigma : volatility
    option_type : OptionType.Call or Put
    barrier_type : BarrierType
    barrier : barrier level
    rebate : rebate paid on knock-out

    Returns
    -------
    price : float
    """
    S, K, T, r, q, sigma, barrier = (
        jnp.asarray(x, dtype=jnp.float64)
        for x in (S, K, T, r, q, sigma, barrier)
    )

    phi = jnp.where(option_type == OptionType.Call, 1.0, -1.0)
    # eta = +1 for down barriers, -1 for up barriers
    eta = jnp.where(
        (barrier_type == BarrierType.DownIn) | (barrier_type == BarrierType.DownOut),
        1.0, -1.0,
    )
    H = barrier
    vol_sqrt_T = sigma * jnp.sqrt(T)

    mu = (r - q - 0.5 * sigma**2) / sigma**2
    lam = jnp.sqrt(mu**2 + 2.0 * r / sigma**2)
    z = jnp.log(H / S) / vol_sqrt_T + lam * vol_sqrt_T

    x1 = jnp.log(S / K) / vol_sqrt_T + (1.0 + mu) * vol_sqrt_T
    x2 = jnp.log(S / H) / vol_sqrt_T + (1.0 + mu) * vol_sqrt_T
    y1 = jnp.log(H**2 / (S * K)) / vol_sqrt_T + (1.0 + mu) * vol_sqrt_T
    y2 = jnp.log(H / S) / vol_sqrt_T + (1.0 + mu) * vol_sqrt_T

    df = jnp.exp(-r * T)
    dq = jnp.exp(-q * T)

    # Standard European components
    A = phi * S * dq * norm.cdf(phi * x1) - phi * K * df * norm.cdf(phi * (x1 - vol_sqrt_T))
    B = phi * S * dq * norm.cdf(phi * x2) - phi * K * df * norm.cdf(phi * (x2 - vol_sqrt_T))
    C = (phi * S * dq * (H / S) ** (2.0 * (mu + 1.0)) * norm.cdf(eta * y1)
         - phi * K * df * (H / S) ** (2.0 * mu) * norm.cdf(eta * (y1 - vol_sqrt_T)))
    D = (phi * S * dq * (H / S) ** (2.0 * (mu + 1.0)) * norm.cdf(eta * y2)
         - phi * K * df * (H / S) ** (2.0 * mu) * norm.cdf(eta * (y2 - vol_sqrt_T)))

    # Rebate terms
    E = (rebate * df * (
        (H / S) ** (mu + lam) * norm.cdf(eta * z)
        + (H / S) ** (mu - lam) * norm.cdf(eta * (z - 2.0 * lam * vol_sqrt_T))
    ))

    # Combine based on barrier type
    # Down-and-in call (H < S)
    # Down-and-out call
    # Up-and-in put (H > S)
    # Up-and-out put
    is_call = option_type == OptionType.Call

    if barrier_type == BarrierType.DownIn:
        price = jnp.where(
            is_call,
            jnp.where(K > H, C, A - B + D),
            jnp.where(K > H, B - C + D, A),
        )
    elif barrier_type == BarrierType.DownOut:
        price = jnp.where(
            is_call,
            jnp.where(K > H, A - C + E, B - D + E),
            jnp.where(K > H, A - B + C - D + E, E),
        )
    elif barrier_type == BarrierType.UpIn:
        price = jnp.where(
            is_call,
            jnp.where(K > H, A, B - C + D),
            jnp.where(K > H, A - B + D, C),
        )
    elif barrier_type == BarrierType.UpOut:
        price = jnp.where(
            is_call,
            jnp.where(K > H, E, A - B + C - D + E),
            jnp.where(K > H, B - D + E, A - C + E),
        )
    else:
        price = 0.0

    return price
