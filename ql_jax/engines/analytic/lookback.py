"""Analytic lookback option pricing."""

import jax.numpy as jnp
from jax.scipy.stats import norm


def floating_lookback_price(S, S_min, S_max, T, r, q, sigma,
                             option_type='call'):
    """Floating-strike lookback option (Goldman-Sosin-Gatto).

    Call pays S_T - S_min, Put pays S_max - S_T.

    Parameters
    ----------
    S : float – current spot
    S_min : float – running minimum (for call)
    S_max : float – running maximum (for put)
    T : float – time to maturity
    r, q, sigma : float
    option_type : 'call' or 'put'

    Returns
    -------
    price : float
    """
    S, T, r, q, sigma = (jnp.asarray(x, dtype=jnp.float64)
                          for x in (S, T, r, q, sigma))

    b = r - q  # cost of carry
    sigma2 = sigma**2

    if option_type == 'call':
        S_ext = jnp.asarray(S_min, dtype=jnp.float64)
        a1 = (jnp.log(S / S_ext) + (b + sigma2 / 2.0) * T) / (sigma * jnp.sqrt(T))
        a2 = a1 - sigma * jnp.sqrt(T)

        price = (S * jnp.exp((b - r) * T) * norm.cdf(a1) -
                 S_ext * jnp.exp(-r * T) * norm.cdf(a2) +
                 S * jnp.exp(-r * T) * sigma2 / (2.0 * b) *
                 (-(S / S_ext)**(-2.0 * b / sigma2) * norm.cdf(a1 - 2.0 * b * jnp.sqrt(T) / sigma) +
                  jnp.exp(b * T) * norm.cdf(a1)))
    else:
        S_ext = jnp.asarray(S_max, dtype=jnp.float64)
        b1 = (jnp.log(S / S_ext) + (b + sigma2 / 2.0) * T) / (sigma * jnp.sqrt(T))
        b2 = b1 - sigma * jnp.sqrt(T)

        price = (S_ext * jnp.exp(-r * T) * norm.cdf(-b2) -
                 S * jnp.exp((b - r) * T) * norm.cdf(-b1) +
                 S * jnp.exp(-r * T) * sigma2 / (2.0 * b) *
                 ((S / S_ext)**(-2.0 * b / sigma2) * norm.cdf(-b1 + 2.0 * b * jnp.sqrt(T) / sigma) -
                  jnp.exp(b * T) * norm.cdf(-b1)))

    return price


def fixed_lookback_price(S, K, S_min, S_max, T, r, q, sigma,
                          option_type='call'):
    """Fixed-strike lookback option.

    Call pays max(S_max - K, 0), Put pays max(K - S_min, 0).

    Parameters
    ----------
    S, K : float
    S_min : float – running min
    S_max : float – running max
    T, r, q, sigma : float
    option_type : 'call' or 'put'

    Returns
    -------
    price : float
    """
    S, K, T, r, q, sigma = (jnp.asarray(x, dtype=jnp.float64)
                             for x in (S, K, T, r, q, sigma))
    b = r - q

    if option_type == 'call':
        S_ext = jnp.maximum(jnp.asarray(S_max, dtype=jnp.float64), K)
        d1 = (jnp.log(S / S_ext) + (b + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
        d2 = d1 - sigma * jnp.sqrt(T)

        price = (S * jnp.exp((b - r) * T) * norm.cdf(d1) -
                 S_ext * jnp.exp(-r * T) * norm.cdf(d2) +
                 jnp.exp(-r * T) * sigma**2 / (2.0 * b) * S *
                 (-(S / S_ext)**(-2.0 * b / sigma**2) *
                  norm.cdf(d1 - 2.0 * b * jnp.sqrt(T) / sigma) +
                  jnp.exp(b * T) * norm.cdf(d1)))
    else:
        S_ext = jnp.minimum(jnp.asarray(S_min, dtype=jnp.float64), K)
        e1 = (jnp.log(S / S_ext) + (b + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
        e2 = e1 - sigma * jnp.sqrt(T)

        price = (-S * jnp.exp((b - r) * T) * norm.cdf(-e1) +
                 S_ext * jnp.exp(-r * T) * norm.cdf(-e2) +
                 jnp.exp(-r * T) * sigma**2 / (2.0 * b) * S *
                 ((S / S_ext)**(-2.0 * b / sigma**2) *
                  norm.cdf(-e1 + 2.0 * b * jnp.sqrt(T) / sigma) -
                  jnp.exp(b * T) * norm.cdf(-e1)))

    return price
