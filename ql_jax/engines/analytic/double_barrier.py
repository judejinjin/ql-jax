"""Analytic double barrier option pricing – Ikeda-Kunitomo formula."""

import jax.numpy as jnp
from jax.scipy.stats import norm


def double_barrier_price(S, K, T, r, q, sigma, lower_barrier, upper_barrier,
                          option_type='call', n_terms=20):
    """Double knock-out barrier option (Ikeda-Kunitomo).

    Option is knocked out if S hits either barrier.

    Parameters
    ----------
    S : float – spot
    K : float – strike
    T : float – maturity
    r, q, sigma : float
    lower_barrier : float – lower knock-out barrier (L)
    upper_barrier : float – upper knock-out barrier (U)
    option_type : 'call' or 'put'
    n_terms : int – number of series terms

    Returns
    -------
    price : float
    """
    S, K, T, r, q, sigma = (jnp.asarray(x, dtype=jnp.float64)
                             for x in (S, K, T, r, q, sigma))
    L = jnp.asarray(lower_barrier, dtype=jnp.float64)
    U = jnp.asarray(upper_barrier, dtype=jnp.float64)

    mu = (r - q - 0.5 * sigma**2) / sigma**2
    log_ratio = jnp.log(U / L)

    phi = 1.0 if option_type == 'call' else -1.0

    price = 0.0
    for n in range(-n_terms, n_terms + 1):
        # Image charges method
        d = 2.0 * n * log_ratio
        S_n = S * jnp.exp(d)

        # Contribution from the n-th image
        factor = (U / S)**(2.0 * n * mu) if n != 0 else 1.0
        if n != 0:
            factor = jnp.exp(2.0 * mu * n * log_ratio)

        x1 = (jnp.log(S_n / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
        x2 = (jnp.log(S_n / U) + (r - q + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
        x3 = (jnp.log(L**2 / (S_n * K)) + (r - q + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
        x4 = (jnp.log(L**2 / (S_n * U)) + (r - q + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))

        if option_type == 'call':
            term = (factor * (
                _bs_call_d(S_n, K, T, r, q, sigma, x1) -
                _bs_call_d(S_n, U, T, r, q, sigma, x2)
            ))
        else:
            term = (factor * (
                _bs_put_d(S_n, K, T, r, q, sigma, x1) -
                _bs_put_d(S_n, L, T, r, q, sigma, x4)
            ))

        price += term

    return jnp.maximum(price, 0.0)


def _bs_call_d(S, K, T, r, q, sigma, d1):
    """BS call given d1."""
    d2 = d1 - sigma * jnp.sqrt(T)
    return (S * jnp.exp(-q * T) * norm.cdf(d1) -
            K * jnp.exp(-r * T) * norm.cdf(d2))


def _bs_put_d(S, K, T, r, q, sigma, d1):
    """BS put given d1."""
    d2 = d1 - sigma * jnp.sqrt(T)
    return (K * jnp.exp(-r * T) * norm.cdf(-d2) -
            S * jnp.exp(-q * T) * norm.cdf(-d1))


def double_knock_in_price(S, K, T, r, q, sigma, L, U,
                            option_type='call', n_terms=20):
    """Double knock-in = Vanilla - Double knock-out.

    Parameters
    ----------
    Same as double_barrier_price.

    Returns
    -------
    price : float
    """
    # Vanilla BS price
    d1 = (jnp.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * jnp.sqrt(T))
    d2 = d1 - sigma * jnp.sqrt(T)

    phi = 1.0 if option_type == 'call' else -1.0
    vanilla = phi * (S * jnp.exp(-q * T) * norm.cdf(phi * d1) -
                     K * jnp.exp(-r * T) * norm.cdf(phi * d2))

    dko = double_barrier_price(S, K, T, r, q, sigma, L, U, option_type, n_terms)
    return vanilla - dko
