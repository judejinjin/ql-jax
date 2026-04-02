"""Local volatility surface derived from market implied vols (Dupire formula).

sigma_local^2(T, K) = (dC/dT + (r-q)*K*dC/dK + q*C) / (0.5*K^2*d^2C/dK^2)
"""

import jax
import jax.numpy as jnp
from jax.scipy.stats import norm


def dupire_local_vol(T, K, S, r, q, implied_vol_fn):
    """Compute local volatility using Dupire formula.

    Parameters
    ----------
    T : float – time to expiry
    K : float – strike
    S : float – spot price
    r : float – risk-free rate
    q : float – dividend yield
    implied_vol_fn : callable(T, K) -> implied Black vol

    Returns local volatility at (T, K).
    """
    eps_T = 1e-4
    eps_K = S * 1e-3

    sigma = implied_vol_fn(T, K)
    C = _bs_call(S, K, T, r, q, sigma)

    # dC/dT by central difference
    sigma_up = implied_vol_fn(T + eps_T, K)
    sigma_dn = implied_vol_fn(jnp.maximum(T - eps_T, 1e-6), K)
    C_T_up = _bs_call(S, K, T + eps_T, r, q, sigma_up)
    C_T_dn = _bs_call(S, K, jnp.maximum(T - eps_T, 1e-6), r, q, sigma_dn)
    dC_dT = (C_T_up - C_T_dn) / (2.0 * eps_T)

    # dC/dK
    sigma_Kup = implied_vol_fn(T, K + eps_K)
    sigma_Kdn = implied_vol_fn(T, jnp.maximum(K - eps_K, 1e-6))
    C_K_up = _bs_call(S, K + eps_K, T, r, q, sigma_Kup)
    C_K_dn = _bs_call(S, jnp.maximum(K - eps_K, 1e-6), T, r, q, sigma_Kdn)
    dC_dK = (C_K_up - C_K_dn) / (2.0 * eps_K)

    # d^2C/dK^2
    d2C_dK2 = (C_K_up - 2.0 * C + C_K_dn) / (eps_K**2)

    numerator = dC_dT + (r - q) * K * dC_dK + q * C
    denominator = 0.5 * K**2 * d2C_dK2

    local_var = numerator / jnp.maximum(denominator, 1e-10)
    return jnp.sqrt(jnp.maximum(local_var, 1e-10))


def _bs_call(S, K, T, r, q, sigma):
    """Black-Scholes call price."""
    F = S * jnp.exp((r - q) * T)
    total_vol = sigma * jnp.sqrt(T)
    d1 = (jnp.log(F / K) + 0.5 * total_vol**2) / total_vol
    d2 = d1 - total_vol
    return jnp.exp(-r * T) * (F * norm.cdf(d1) - K * norm.cdf(d2))


def build_local_vol_surface(spots_grid, times_grid, S, r, q, implied_vol_fn):
    """Build local vol on a grid.

    Parameters
    ----------
    spots_grid : array of strikes/spots
    times_grid : array of maturities
    S : spot
    r, q : rates
    implied_vol_fn : callable(T, K) -> sigma_impl

    Returns 2D array (n_times, n_spots) of local vols.
    """
    n_t = len(times_grid)
    n_s = len(spots_grid)
    lv = jnp.zeros((n_t, n_s))

    for i in range(n_t):
        for j in range(n_s):
            lv = lv.at[i, j].set(
                dupire_local_vol(times_grid[i], spots_grid[j], S, r, q, implied_vol_fn)
            )
    return lv
