"""Basket and multi-asset option engines.

Includes Stulz, moment-matching, and MC basket engines.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm as jnorm

from ql_jax.engines.analytic.black_formula import black_scholes_price
from ql_jax.math.distributions.bivariate import bivariate_normal_cdf


# ---------------------------------------------------------------------------
# Stulz 2-asset basket option (1982)
# ---------------------------------------------------------------------------

def stulz_basket_price(
    S1: float, S2: float, K: float, T: float, r: float,
    q1: float, q2: float, sigma1: float, sigma2: float, rho: float,
    option_type: int = 1,
) -> float:
    """Stulz (1982) analytic price for a 2-asset European basket option.

    The basket is max(S1, S2) for call, min(S1, S2) for put.

    Parameters
    ----------
    S1, S2 : spot prices
    K : strike
    T : time to expiry
    r : risk-free rate
    q1, q2 : dividend yields
    sigma1, sigma2 : volatilities
    rho : correlation
    option_type : 1 for call on max, -1 for put on min
    """
    S1, S2, K, T = (jnp.float64(x) for x in (S1, S2, K, T))
    r, q1, q2 = jnp.float64(r), jnp.float64(q1), jnp.float64(q2)
    sigma1, sigma2, rho = jnp.float64(sigma1), jnp.float64(sigma2), jnp.float64(rho)

    sqrtT = jnp.sqrt(T)
    sigma_d = jnp.sqrt(sigma1 ** 2 - 2.0 * rho * sigma1 * sigma2 + sigma2 ** 2)

    d1 = (jnp.log(S1 / K) + (r - q1 + 0.5 * sigma1 ** 2) * T) / (sigma1 * sqrtT)
    d2 = d1 - sigma1 * sqrtT
    e1 = (jnp.log(S2 / K) + (r - q2 + 0.5 * sigma2 ** 2) * T) / (sigma2 * sqrtT)
    e2 = e1 - sigma2 * sqrtT

    # Correlation adjustments
    rho1 = (sigma1 - rho * sigma2) / sigma_d
    rho2 = (sigma2 - rho * sigma1) / sigma_d
    f1 = (jnp.log(S1 / S2) + 0.5 * sigma_d ** 2 * T) / (sigma_d * sqrtT)
    f2 = f1 - sigma_d * sqrtT

    # Call on max
    call_max = (
        S1 * jnp.exp(-q1 * T) * bivariate_normal_cdf(d1, f1, -rho1)
        + S2 * jnp.exp(-q2 * T) * bivariate_normal_cdf(e1, -f2, -rho2)
        - K * jnp.exp(-r * T) * bivariate_normal_cdf(d2, e2, rho)
    )

    # Put-call parity for min/max basket
    # put on min = call on max + K*exp(-rT) - S1*exp(-q1*T) - S2*exp(-q2*T)
    put_min = (
        call_max + K * jnp.exp(-r * T)
        - S1 * jnp.exp(-q1 * T) - S2 * jnp.exp(-q2 * T)
    )

    return jnp.where(option_type == 1, jnp.maximum(call_max, 0.0),
                     jnp.maximum(put_min, 0.0))


# ---------------------------------------------------------------------------
# Moment-matching (Levy 1992) for N-asset basket
# ---------------------------------------------------------------------------

def moment_matching_basket_price(
    spots: jnp.ndarray, K: float, T: float, r: float,
    divs: jnp.ndarray, sigmas: jnp.ndarray, corr: jnp.ndarray,
    weights: jnp.ndarray = None, option_type: int = 1,
) -> float:
    """Moment-matching basket option price (Levy 1992).

    Matches first two moments of the basket to a lognormal.

    Parameters
    ----------
    spots : array of N spot prices
    K : strike
    T : expiry
    r : risk-free rate
    divs : array of N dividend yields
    sigmas : array of N volatilities
    corr : NxN correlation matrix
    weights : portfolio weights (default equal)
    option_type : 1 for call, -1 for put
    """
    spots = jnp.asarray(spots, dtype=jnp.float64)
    n = spots.shape[0]
    if weights is None:
        weights = jnp.ones(n, dtype=jnp.float64) / n

    divs = jnp.asarray(divs, dtype=jnp.float64)
    sigmas = jnp.asarray(sigmas, dtype=jnp.float64)
    corr = jnp.asarray(corr, dtype=jnp.float64)

    # Forward prices
    F = spots * jnp.exp((r - divs) * T)
    wF = weights * F

    # First moment of basket
    M1 = jnp.sum(wF)

    # Second moment
    M2 = 0.0
    for i in range(n):
        for j in range(n):
            M2 = M2 + wF[i] * wF[j] * jnp.exp(corr[i, j] * sigmas[i] * sigmas[j] * T)

    # Match to lognormal
    sigma_B = jnp.sqrt(jnp.log(M2 / M1 ** 2) / T)
    sigma_B = jnp.maximum(sigma_B, 1e-6)
    F_B = M1

    # Black price
    d1 = (jnp.log(F_B / K) + 0.5 * sigma_B ** 2 * T) / (sigma_B * jnp.sqrt(T))
    d2 = d1 - sigma_B * jnp.sqrt(T)

    df = jnp.exp(-r * T)
    call = df * (F_B * jnorm.cdf(d1) - K * jnorm.cdf(d2))
    put = df * (K * jnorm.cdf(-d2) - F_B * jnorm.cdf(-d1))

    return jnp.where(option_type == 1, jnp.maximum(call, 0.0), jnp.maximum(put, 0.0))


# ---------------------------------------------------------------------------
# MC European basket
# ---------------------------------------------------------------------------

def mc_european_basket(
    spots: jnp.ndarray, K: float, T: float, r: float,
    divs: jnp.ndarray, sigmas: jnp.ndarray, corr: jnp.ndarray,
    weights: jnp.ndarray = None, option_type: int = 1,
    n_paths: int = 100_000, seed: int = 42,
) -> float:
    """Monte Carlo European basket option price.

    Parameters
    ----------
    spots : array of N spot prices
    K : strike
    T : expiry
    r : risk-free rate
    divs : dividend yields
    sigmas : volatilities
    corr : NxN correlation matrix
    weights : portfolio weights
    option_type : 1 for call, -1 for put
    n_paths, seed : MC parameters
    """
    spots = jnp.asarray(spots, dtype=jnp.float64)
    n = spots.shape[0]
    if weights is None:
        weights = jnp.ones(n, dtype=jnp.float64) / n

    divs = jnp.asarray(divs, dtype=jnp.float64)
    sigmas = jnp.asarray(sigmas, dtype=jnp.float64)
    corr = jnp.asarray(corr, dtype=jnp.float64)

    # Cholesky decomposition
    L = jnp.linalg.cholesky(corr)

    key = jax.random.PRNGKey(seed)
    Z = jax.random.normal(key, shape=(n_paths, n))
    Z_corr = Z @ L.T

    # Simulate terminal prices
    drift = (r - divs - 0.5 * sigmas ** 2) * T
    diffusion = sigmas * jnp.sqrt(T) * Z_corr
    S_T = spots * jnp.exp(drift + diffusion)

    # Basket value
    basket = jnp.sum(weights * S_T, axis=1)

    # Payoffs
    payoffs = jnp.where(option_type == 1,
                        jnp.maximum(basket - K, 0.0),
                        jnp.maximum(K - basket, 0.0))

    return jnp.exp(-r * T) * jnp.mean(payoffs)
