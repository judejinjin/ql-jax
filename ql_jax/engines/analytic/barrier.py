"""Analytic barrier option pricing – closed-form formulas."""

import jax.numpy as jnp
from jax.scipy.stats import norm


def barrier_price(S, K, T, r, q, sigma, barrier, rebate=0.0,
                  option_type='call', barrier_type='down_and_out'):
    """Analytical barrier option price (Merton/Reiner-Rubinstein).

    Parameters
    ----------
    S, K, T, r, q, sigma : float
    barrier : float – barrier level H
    rebate : float – paid if knocked out
    option_type : str – 'call' or 'put'
    barrier_type : str – 'down_and_out', 'down_and_in', 'up_and_out', 'up_and_in'

    Returns
    -------
    price : float
    """
    S, K, T, r, q, sigma, barrier = (
        jnp.asarray(x, dtype=jnp.float64) for x in (S, K, T, r, q, sigma, barrier)
    )

    phi = 1.0 if option_type == 'call' else -1.0
    mu = (r - q - 0.5 * sigma**2) / sigma**2
    lam = jnp.sqrt(mu**2 + 2.0 * r / sigma**2)
    H = barrier

    x1 = jnp.log(S / K) / (sigma * jnp.sqrt(T)) + (1.0 + mu) * sigma * jnp.sqrt(T)
    x2 = jnp.log(S / H) / (sigma * jnp.sqrt(T)) + (1.0 + mu) * sigma * jnp.sqrt(T)
    y1 = jnp.log(H**2 / (S * K)) / (sigma * jnp.sqrt(T)) + (1.0 + mu) * sigma * jnp.sqrt(T)
    y2 = jnp.log(H / S) / (sigma * jnp.sqrt(T)) + (1.0 + mu) * sigma * jnp.sqrt(T)
    z = jnp.log(H / S) / (sigma * jnp.sqrt(T)) + lam * sigma * jnp.sqrt(T)

    # Standard terms
    A = phi * S * jnp.exp(-q * T) * norm.cdf(phi * x1) - \
        phi * K * jnp.exp(-r * T) * norm.cdf(phi * (x1 - sigma * jnp.sqrt(T)))

    B = phi * S * jnp.exp(-q * T) * norm.cdf(phi * x2) - \
        phi * K * jnp.exp(-r * T) * norm.cdf(phi * (x2 - sigma * jnp.sqrt(T)))

    C = phi * S * jnp.exp(-q * T) * (H / S)**(2 * (mu + 1)) * norm.cdf(phi * y1) - \
        phi * K * jnp.exp(-r * T) * (H / S)**(2 * mu) * norm.cdf(phi * (y1 - sigma * jnp.sqrt(T)))

    D = phi * S * jnp.exp(-q * T) * (H / S)**(2 * (mu + 1)) * norm.cdf(phi * y2) - \
        phi * K * jnp.exp(-r * T) * (H / S)**(2 * mu) * norm.cdf(phi * (y2 - sigma * jnp.sqrt(T)))

    # Rebate terms
    E = rebate * jnp.exp(-r * T) * (
        norm.cdf(phi * (x2 - sigma * jnp.sqrt(T))) -
        (H / S)**(2 * mu) * norm.cdf(phi * (y2 - sigma * jnp.sqrt(T)))
    )

    F = rebate * (
        (H / S)**(mu + lam) * norm.cdf(phi * z) +
        (H / S)**(mu - lam) * norm.cdf(phi * (z - 2.0 * lam * sigma * jnp.sqrt(T)))
    )

    if barrier_type == 'down_and_out':
        if option_type == 'call':
            return jnp.where(K > H, A - C + F, B - D + F)
        else:
            return jnp.where(K > H, F, A - B + D + F)

    elif barrier_type == 'down_and_in':
        vanilla = _bs_price(S, K, T, r, q, sigma, phi)
        out = barrier_price(S, K, T, r, q, sigma, barrier, rebate,
                           option_type, 'down_and_out')
        return vanilla - out + rebate * jnp.exp(-r * T) - F + F  # In = Vanilla - Out

    elif barrier_type == 'up_and_out':
        if option_type == 'call':
            return jnp.where(K > H, F, A - B + D + F)
        else:
            return jnp.where(K > H, A - C + F, B - D + F)

    elif barrier_type == 'up_and_in':
        vanilla = _bs_price(S, K, T, r, q, sigma, phi)
        out = barrier_price(S, K, T, r, q, sigma, barrier, rebate,
                           option_type, 'up_and_out')
        return vanilla - out + rebate * jnp.exp(-r * T) - F + F

    return 0.0


def _bs_price(S, K, T, r, q, sigma, phi):
    """Plain Black-Scholes price."""
    d1 = (jnp.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)
    return phi * (S * jnp.exp(-q * T) * norm.cdf(phi * d1) -
                  K * jnp.exp(-r * T) * norm.cdf(phi * d2))
