"""Heston SLV model — Monte Carlo calibration variant."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def heston_slv_mc_calibrate(
    spot, r, q,
    v0, kappa, theta, sigma, rho,
    market_local_vol_fn,
    T, n_steps=100, n_paths=50000, key=None,
):
    """Calibrate the leverage function L(t, S) for Heston SLV via MC.

    The SLV model: dS/S = (r-q)dt + L(t,S) * sqrt(V) dW_1
                   dV   = kappa*(theta - V)dt + sigma*sqrt(V) dW_2

    L(t,S) is chosen so that E[V | S_t = S] * L(t,S)^2 = sigma_local(t,S)^2

    Parameters
    ----------
    spot : float
    r, q : float risk-free rate and dividend yield
    v0, kappa, theta, sigma, rho : Heston parameters
    market_local_vol_fn : callable (t, S) -> local vol
    T : float maturity
    n_steps : int
    n_paths : int
    key : JAX PRNG key

    Returns
    -------
    dict with 'leverage_grid' (n_steps x n_bins), 'time_grid', 'spot_grid'
    """
    if key is None:
        key = jax.random.PRNGKey(42)

    dt = T / n_steps
    sqrt_dt = jnp.sqrt(dt)

    # Simulate Heston paths
    S = jnp.full(n_paths, spot, dtype=jnp.float64)
    V = jnp.full(n_paths, v0, dtype=jnp.float64)

    n_bins = 50
    time_grid = jnp.linspace(0, T, n_steps + 1)
    s_lo, s_hi = spot * 0.3, spot * 3.0
    spot_grid = jnp.linspace(s_lo, s_hi, n_bins)
    leverage_grid = jnp.ones((n_steps, n_bins), dtype=jnp.float64)

    for i in range(n_steps):
        t = i * dt
        key, k1, k2 = jax.random.split(key, 3)
        z1 = jax.random.normal(k1, shape=(n_paths,))
        z2 = rho * z1 + jnp.sqrt(1 - rho ** 2) * jax.random.normal(k2, shape=(n_paths,))

        V_pos = jnp.maximum(V, 1e-8)
        sqrt_V = jnp.sqrt(V_pos)

        # Compute conditional expectation E[V | S in bin]
        bin_idx = jnp.clip(
            ((S - s_lo) / (s_hi - s_lo) * n_bins).astype(jnp.int32),
            0, n_bins - 1,
        )
        # For each bin, average V
        cond_v = jnp.zeros(n_bins, dtype=jnp.float64)
        counts = jnp.zeros(n_bins, dtype=jnp.float64)
        for b in range(n_bins):
            mask = bin_idx == b
            cond_v = cond_v.at[b].set(jnp.where(jnp.sum(mask) > 0, jnp.mean(jnp.where(mask, V_pos, 0.0)) * n_paths / jnp.maximum(jnp.sum(mask), 1.0), v0))
            counts = counts.at[b].set(jnp.sum(mask))

        # Leverage = market_local_vol / sqrt(E[V|S])
        for b in range(n_bins):
            S_b = spot_grid[b]
            lv = market_local_vol_fn(t, S_b)
            L = lv / jnp.sqrt(jnp.maximum(cond_v[b], 1e-8))
            leverage_grid = leverage_grid.at[i, b].set(jnp.clip(L, 0.01, 10.0))

        # Get leverage for each path
        L_path = jnp.interp(S, spot_grid, leverage_grid[i])

        # Euler step
        S = S * jnp.exp((r - q - 0.5 * L_path ** 2 * V_pos) * dt + L_path * sqrt_V * sqrt_dt * z1)
        V = V + kappa * (theta - V_pos) * dt + sigma * sqrt_V * sqrt_dt * z2
        V = jnp.maximum(V, 0.0)

    return {
        'leverage_grid': leverage_grid,
        'time_grid': time_grid,
        'spot_grid': spot_grid,
    }
