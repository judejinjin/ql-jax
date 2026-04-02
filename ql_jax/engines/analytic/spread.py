"""Analytic spread option pricing – Kirk's approximation."""

import jax.numpy as jnp
from jax.scipy.stats import norm


def kirk_spread_price(S1, S2, K, T, r, q1, q2, sigma1, sigma2, rho,
                       option_type='call'):
    """Kirk's approximation for spread option price.

    Payoff: max(S1 - S2 - K, 0) for call, max(K - S1 + S2, 0) for put.

    Parameters
    ----------
    S1, S2 : float – spot prices of two assets
    K : float – strike
    T : float – maturity
    r : float – risk-free rate
    q1, q2 : float – dividend yields
    sigma1, sigma2 : float – volatilities
    rho : float – correlation
    option_type : 'call' or 'put'

    Returns
    -------
    price : float
    """
    S1, S2, K, T, r = (jnp.asarray(x, dtype=jnp.float64)
                        for x in (S1, S2, K, T, r))
    sigma1, sigma2, rho = (jnp.asarray(x, dtype=jnp.float64)
                           for x in (sigma1, sigma2, rho))

    F1 = S1 * jnp.exp((r - q1) * T)
    F2 = S2 * jnp.exp((r - q2) * T)

    # Kirk's approximation: treat (S2 + K) as a single asset
    F2K = F2 + K * jnp.exp(-r * T)  # Forward of S2 + K
    sigma_eff = jnp.sqrt(sigma1**2 - 2.0 * rho * sigma1 * sigma2 * F2 / F2K +
                          (sigma2 * F2 / F2K)**2)

    d1 = (jnp.log(F1 / F2K) + 0.5 * sigma_eff**2 * T) / (sigma_eff * jnp.sqrt(T))
    d2 = d1 - sigma_eff * jnp.sqrt(T)

    phi = 1.0 if option_type == 'call' else -1.0
    price = phi * jnp.exp(-r * T) * (F1 * norm.cdf(phi * d1) -
                                       F2K * norm.cdf(phi * d2))
    return price


def bjerksund_stensland_spread(S1, S2, K, T, r, q1, q2, sigma1, sigma2, rho):
    """Bjerksund-Stensland spread option approximation.

    More accurate than Kirk for large spreads.

    Parameters
    ----------
    Same as kirk_spread_price but only for calls.

    Returns
    -------
    price : float
    """
    S1, S2, K, T, r = (jnp.asarray(x, dtype=jnp.float64)
                        for x in (S1, S2, K, T, r))

    F1 = S1 * jnp.exp((r - q1) * T)
    F2 = S2 * jnp.exp((r - q2) * T)

    # Effective vol using Bjerksund-Stensland adjustment
    alpha = F2 / (F2 + K)
    sigma_eff = jnp.sqrt(sigma1**2 - 2.0 * rho * sigma1 * sigma2 * alpha +
                          sigma2**2 * alpha**2)

    d1 = (jnp.log(F1 / (F2 + K)) + 0.5 * sigma_eff**2 * T) / (sigma_eff * jnp.sqrt(T))
    d2 = d1 - sigma_eff * jnp.sqrt(T)

    price = jnp.exp(-r * T) * (F1 * norm.cdf(d1) - (F2 + K) * norm.cdf(d2))
    return price


def margrabe_exchange_price(S1, S2, T, q1, q2, sigma1, sigma2, rho):
    """Margrabe's formula for exchange option: max(S1 - S2, 0).

    Parameters
    ----------
    S1, S2 : float
    T : float
    q1, q2 : float
    sigma1, sigma2 : float
    rho : float

    Returns
    -------
    price : float
    """
    S1, S2 = jnp.asarray(S1, jnp.float64), jnp.asarray(S2, jnp.float64)
    sigma = jnp.sqrt(sigma1**2 - 2.0 * rho * sigma1 * sigma2 + sigma2**2)

    d1 = (jnp.log(S1 / S2) + 0.5 * sigma**2 * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)

    return S1 * jnp.exp(-q1 * T) * norm.cdf(d1) - S2 * jnp.exp(-q2 * T) * norm.cdf(d2)
