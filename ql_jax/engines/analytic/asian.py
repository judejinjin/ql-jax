"""Analytic Asian option pricing – geometric average closed-form."""

import jax.numpy as jnp
from jax.scipy.stats import norm


def geometric_asian_price(S, K, T, r, q, sigma, n_fixings,
                           option_type='call'):
    """Geometric average Asian option (closed-form).

    Uses the result that the geometric average of log-normal prices
    is itself log-normal.

    Parameters
    ----------
    S : float – spot
    K : float – strike
    T : float – maturity
    r, q : float – rates
    sigma : float – volatility
    n_fixings : int – number of averaging fixings
    option_type : str – 'call' or 'put'

    Returns
    -------
    price : float
    """
    S, K, T, r, q, sigma = (jnp.asarray(x, dtype=jnp.float64)
                             for x in (S, K, T, r, q, sigma))
    n = n_fixings

    # Adjusted parameters for geometric average
    sigma_a = sigma * jnp.sqrt((2.0 * n + 1.0) / (6.0 * (n + 1.0)))
    mu_a = 0.5 * sigma_a**2 + (r - q - 0.5 * sigma**2) * (n + 1.0) / (2.0 * n)

    d1 = (jnp.log(S / K) + (mu_a + 0.5 * sigma_a**2) * T) / (sigma_a * jnp.sqrt(T))
    d2 = d1 - sigma_a * jnp.sqrt(T)

    phi = 1.0 if option_type == 'call' else -1.0

    price = phi * (S * jnp.exp((mu_a - r) * T) * norm.cdf(phi * d1) -
                   K * jnp.exp(-r * T) * norm.cdf(phi * d2))
    return price


def turnbull_wakeman_price(S, K, T, r, q, sigma, n_fixings,
                            option_type='call'):
    """Turnbull-Wakeman approximation for arithmetic average Asian.

    Matches first two moments of the arithmetic average
    to a log-normal distribution.

    Parameters
    ----------
    S, K, T, r, q, sigma : float
    n_fixings : int
    option_type : 'call' or 'put'

    Returns
    -------
    price : float
    """
    S, K, T, r, q, sigma = (jnp.asarray(x, dtype=jnp.float64)
                             for x in (S, K, T, r, q, sigma))
    n = jnp.float64(n_fixings)
    dt = T / n

    # First moment of arithmetic average
    M1 = S * jnp.exp((r - q) * dt) * (1.0 - jnp.exp((r - q) * T)) / \
         (n * (1.0 - jnp.exp((r - q) * dt)))

    # Second moment
    t_sum = jnp.sum(jnp.arange(1, n_fixings + 1) * dt)
    sigma2_eff = sigma**2 * T * (2.0 * n + 1.0) / (6.0 * (n + 1.0))

    # Match to log-normal
    sigma_a = jnp.sqrt(sigma2_eff / T)
    d1 = (jnp.log(M1 / K) + 0.5 * sigma_a**2 * T) / (sigma_a * jnp.sqrt(T))
    d2 = d1 - sigma_a * jnp.sqrt(T)

    phi = 1.0 if option_type == 'call' else -1.0
    price = phi * jnp.exp(-r * T) * (M1 * norm.cdf(phi * d1) -
                                       K * norm.cdf(phi * d2))
    return price
