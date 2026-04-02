"""Analytic digital option pricing."""

import jax.numpy as jnp
from jax.scipy.stats import norm


def digital_price(S, K, T, r, q, sigma, option_type='call',
                   payout_type='cash', payout=1.0):
    """Digital (binary) option price under Black-Scholes.

    Parameters
    ----------
    S : float – spot
    K : float – strike
    T : float – maturity
    r, q, sigma : float
    option_type : 'call' or 'put'
    payout_type : 'cash' (cash-or-nothing) or 'asset' (asset-or-nothing)
    payout : float – cash payout amount (for cash-or-nothing)

    Returns
    -------
    price : float
    """
    S, K, T, r, q, sigma = (jnp.asarray(x, dtype=jnp.float64)
                             for x in (S, K, T, r, q, sigma))

    d1 = (jnp.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)

    phi = 1.0 if option_type == 'call' else -1.0

    if payout_type == 'cash':
        # Cash-or-nothing: pays fixed amount if ITM
        return payout * jnp.exp(-r * T) * norm.cdf(phi * d2)
    else:
        # Asset-or-nothing: pays S_T if ITM
        return S * jnp.exp(-q * T) * norm.cdf(phi * d1)


def digital_gap_price(S, K1, K2, T, r, q, sigma, option_type='call'):
    """Gap option price.

    Pays (S_T - K1) when S_T > K2 (for call).
    K1 = payment strike, K2 = trigger strike.

    Parameters
    ----------
    S : float – spot
    K1 : float – payment strike
    K2 : float – trigger strike
    T, r, q, sigma : float
    option_type : 'call' or 'put'

    Returns
    -------
    price : float
    """
    S, K1, K2, T = (jnp.asarray(x, dtype=jnp.float64) for x in (S, K1, K2, T))

    d1 = (jnp.log(S / K2) + (r - q + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)

    phi = 1.0 if option_type == 'call' else -1.0

    return phi * (S * jnp.exp(-q * T) * norm.cdf(phi * d1) -
                  K1 * jnp.exp(-r * T) * norm.cdf(phi * d2))


def double_digital_price(S, K_lower, K_upper, T, r, q, sigma, payout=1.0):
    """Double digital (range binary): pays if K_lower < S_T < K_upper.

    Parameters
    ----------
    S : float
    K_lower, K_upper : float
    T, r, q, sigma : float
    payout : float

    Returns
    -------
    price : float
    """
    S, K_lower, K_upper, T = (
        jnp.asarray(x, dtype=jnp.float64) for x in (S, K_lower, K_upper, T)
    )

    d2_lower = ((jnp.log(S / K_lower) + (r - q - 0.5 * sigma**2) * T) /
                (sigma * jnp.sqrt(T)))
    d2_upper = ((jnp.log(S / K_upper) + (r - q - 0.5 * sigma**2) * T) /
                (sigma * jnp.sqrt(T)))

    return payout * jnp.exp(-r * T) * (norm.cdf(d2_lower) - norm.cdf(d2_upper))
