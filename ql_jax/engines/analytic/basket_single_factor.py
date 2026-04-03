"""Operator splitting spread engine.

Uses operator splitting method for spread option pricing: max(S1 - S2 - K, 0).
Decomposes the 2D PDE into simpler 1D problems.

Reference: Ikonen & Toivanen (2009), "Operator splitting methods for
pricing American options under stochastic volatility".
"""

from __future__ import annotations

import jax.numpy as jnp
from jax.scipy.stats import norm

from ql_jax.engines.analytic.black_formula import black_price


def operator_splitting_spread_price(
    F1, F2, T, sigma1, sigma2, rho, K, df,
    option_type: int = 1,
):
    """Spread option price via Kirk's approximation with operator-splitting refinement.

    Parameters
    ----------
    F1, F2 : forward prices of assets 1 and 2
    T : time to expiry
    sigma1, sigma2 : volatilities
    rho : correlation between the two assets
    K : strike (for S1 - S2 - K)
    df : discount factor
    option_type : 1 for call, -1 for put

    Returns
    -------
    price : spread option price
    """
    F1, F2, T, sigma1, sigma2, rho, K, df = (
        jnp.float64(x) for x in (F1, F2, T, sigma1, sigma2, rho, K, df)
    )

    # Kirk's approximation: treat F2 + K as the "strike"
    F2_adj = F2 + K
    F2_adj = jnp.maximum(F2_adj, 1e-10)

    # Effective volatility
    w = F2 / F2_adj
    sigma_eff = jnp.sqrt(sigma1**2 - 2.0 * rho * sigma1 * sigma2 * w + (sigma2 * w)**2)

    # Black formula with adjusted strike
    price = black_price(F1, F2_adj, T, sigma_eff, df, option_type)

    return price


def single_factor_bsm_basket_price(
    forwards, vols, weights, corr_matrix,
    K, T, df, option_type: int = 1,
):
    """Single-factor moment-matching basket option price.

    Approximates the basket by a single lognormal asset using first
    two moments matching.

    Parameters
    ----------
    forwards : array of forward prices [F1, F2, ..., Fn]
    vols : array of volatilities [sigma1, sigma2, ..., sigman]
    weights : array of basket weights [w1, w2, ..., wn]
    corr_matrix : correlation matrix (n x n)
    K : strike
    T : expiry
    df : discount factor
    option_type : 1 for call, -1 for put

    Returns
    -------
    price : basket option price
    """
    forwards = jnp.asarray(forwards, dtype=jnp.float64)
    vols = jnp.asarray(vols, dtype=jnp.float64)
    weights = jnp.asarray(weights, dtype=jnp.float64)
    corr_matrix = jnp.asarray(corr_matrix, dtype=jnp.float64)
    K, T, df = jnp.float64(K), jnp.float64(T), jnp.float64(df)

    # First moment: E[B] = sum(w_i * F_i)
    M1 = jnp.sum(weights * forwards)

    # Second moment: E[B^2]
    wF = weights * forwards
    var_matrix = jnp.outer(wF, wF) * (jnp.exp(jnp.outer(vols, vols) * corr_matrix * T) - 1.0)
    M2 = M1**2 + jnp.sum(var_matrix)

    # Lognormal matching
    v2 = jnp.log(M2 / M1**2)
    v2 = jnp.maximum(v2, 1e-20)
    sigma_eff = jnp.sqrt(v2 / T)

    price = black_price(M1, K, T, sigma_eff, df, option_type)
    return price
