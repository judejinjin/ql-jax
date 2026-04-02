"""Heston Black volatility surface: implied vol grid from Heston model."""

import jax.numpy as jnp
from ql_jax.engines.analytic.heston import heston_price
from ql_jax._util.types import OptionType


def heston_implied_vol(S, K, T, r, q, v0, kappa, theta, xi, rho, tol=1e-6):
    """Compute Black implied vol from Heston model price.

    Uses Newton's method on the BS formula.
    """
    from jax.scipy.stats import norm

    target = heston_price(S, K, T, r, q, v0, kappa, theta, xi, rho, OptionType.Call)
    F = S * jnp.exp((r - q) * T)
    discount = jnp.exp(-r * T)

    # Initial guess from ATM formula
    sigma = jnp.sqrt(v0)

    for _ in range(20):
        sqrt_T = jnp.sqrt(T)
        d1 = (jnp.log(F / K) + 0.5 * sigma**2 * T) / (sigma * sqrt_T)
        d2 = d1 - sigma * sqrt_T
        bs_price = discount * (F * norm.cdf(d1) - K * norm.cdf(d2))
        vega = F * discount * norm.pdf(d1) * sqrt_T

        diff = bs_price - target
        sigma = sigma - diff / jnp.maximum(vega, 1e-10)
        sigma = jnp.maximum(sigma, 1e-4)

        if jnp.abs(diff) < tol:
            break

    return sigma


def build_heston_vol_surface(S, r, q, v0, kappa, theta, xi, rho,
                              strikes, maturities):
    """Build a grid of Black implied vols from Heston parameters.

    Parameters
    ----------
    strikes : array (n_k,)
    maturities : array (n_t,)

    Returns 2D array (n_t, n_k) of implied vols.
    """
    n_t = len(maturities)
    n_k = len(strikes)
    surface = jnp.zeros((n_t, n_k))

    for i in range(n_t):
        for j in range(n_k):
            surface = surface.at[i, j].set(
                heston_implied_vol(S, strikes[j], maturities[i], r, q,
                                    v0, kappa, theta, xi, rho)
            )
    return surface
