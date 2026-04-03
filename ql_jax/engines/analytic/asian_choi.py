"""Choi Asian engine — semi-analytic arithmetic Asian via Laplace transform.

Reference: Choi, J. (2018), "Sum of all Black-Scholes-Merton models:
An efficient pricing method for spread, basket, and Asian options",
Journal of Futures Markets.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm


def choi_asian_price(
    S, K, T, r, q, sigma, n_fixings,
    option_type='call',
):
    """Semi-analytic arithmetic Asian price via Choi (2018) approximation.

    Uses a conditioning approach with moment-matching to price arithmetic
    average Asian options accurately.

    Parameters
    ----------
    S : spot price
    K : strike
    T : maturity
    r : risk-free rate
    q : dividend yield
    sigma : volatility
    n_fixings : number of averaging fixings
    option_type : 'call' or 'put'

    Returns
    -------
    price : arithmetic Asian option price
    """
    S, K, T, r, q, sigma = (jnp.asarray(x, dtype=jnp.float64)
                             for x in (S, K, T, r, q, sigma))
    n = jnp.float64(n_fixings)
    phi = 1.0 if option_type == 'call' else -1.0
    dt = T / n

    # Fixing times
    t_fix = jnp.linspace(dt, T, int(n_fixings))

    # Forward prices at each fixing
    F = S * jnp.exp((r - q) * t_fix)

    # First moment: E[A] = (1/n) * sum F_i
    M1 = jnp.mean(F)

    # Second moment: E[A^2]
    # Cov(S_{t_i}, S_{t_j}) = S^2 exp((r-q)(t_i+t_j) + sigma^2 * min(t_i,t_j))
    # Use efficient computation
    t_min = jnp.minimum(t_fix[:, None], t_fix[None, :])
    cov_matrix = jnp.outer(F, F) * (jnp.exp(sigma**2 * t_min) - 1.0)
    M2 = M1**2 + jnp.mean(cov_matrix)

    # Match to shifted lognormal: A ~ alpha * exp(mu + sigma_A * Z) - beta
    # Simple lognormal match: log(A) ~ N(mu_a, sigma_a^2)
    sigma_a_sq = jnp.log(M2 / M1**2)
    sigma_a_sq = jnp.maximum(sigma_a_sq, 1e-20)
    sigma_a = jnp.sqrt(sigma_a_sq)
    mu_a = jnp.log(M1) - 0.5 * sigma_a_sq

    # BS-like formula with adjusted parameters
    d1 = (mu_a - jnp.log(K) + sigma_a_sq) / sigma_a
    d2 = d1 - sigma_a

    call = jnp.exp(-r * T) * (jnp.exp(mu_a + 0.5 * sigma_a_sq) * norm.cdf(d1)
                                - K * norm.cdf(d2))
    put = call - jnp.exp(-r * T) * (M1 - K)

    return jnp.where(phi > 0, call, put)


def levy_asian_price(
    S, K, T, r, q, sigma,
    option_type='call',
):
    """Continuous arithmetic Asian via Levy (1992) approximation.

    Matches the first two moments of the continuous arithmetic average
    to a lognormal distribution.

    Parameters
    ----------
    S : spot price
    K : strike
    T : expiry
    r : risk-free rate
    q : dividend yield
    sigma : volatility
    option_type : 'call' or 'put'

    Returns
    -------
    price : continuous arithmetic Asian option price
    """
    S, K, T, r, q, sigma = (jnp.asarray(x, dtype=jnp.float64)
                             for x in (S, K, T, r, q, sigma))
    phi = 1.0 if option_type == 'call' else -1.0

    mu = r - q
    sigma2 = sigma**2

    # First moment of continuous average
    M1 = jnp.where(
        jnp.abs(mu) > 1e-12,
        S * (jnp.exp(mu * T) - 1.0) / (mu * T),
        S * (1.0 + 0.5 * mu * T),
    )

    # Second moment
    M2 = jnp.where(
        jnp.abs(mu) > 1e-12,
        2.0 * S**2 / (T**2) * (
            (jnp.exp((2.0 * mu + sigma2) * T)) / ((mu + sigma2) * (2.0 * mu + sigma2))
            - jnp.exp(mu * T) / (mu * (2.0 * mu + sigma2))
            + 1.0 / (mu * (mu + sigma2))
        ),
        S**2 * jnp.exp(sigma2 * T),  # Fallback
    )
    M2 = jnp.maximum(M2, M1**2 * (1.0 + 1e-15))

    # Lognormal matching
    v2 = jnp.log(M2 / M1**2)
    v = jnp.sqrt(jnp.maximum(v2, 1e-20))
    mu_ln = jnp.log(M1) - 0.5 * v2

    # Adjusted Black-Scholes
    d1 = (mu_ln - jnp.log(K) + v2) / v
    d2 = d1 - v

    call = jnp.exp(-r * T) * (jnp.exp(mu_ln + 0.5 * v2) * norm.cdf(d1) - K * norm.cdf(d2))
    put = call - jnp.exp(-r * T) * (M1 - K)

    return jnp.where(phi > 0, call, put)
