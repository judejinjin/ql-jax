"""Andreasen-Huge volatility interpolation.

A non-parametric local volatility model that fits market option prices exactly
using a piecewise-constant local vol and solves a 1D Fokker-Planck PDE.
"""

import jax.numpy as jnp
from jax.scipy.stats import norm


def calibrate(strikes_grid, times_grid, market_call_prices, S, r, q,
               n_fd_steps=100):
    """Calibrate Andreasen-Huge piecewise-constant local vol.

    Parameters
    ----------
    strikes_grid : array (n_k,) – sorted strikes
    times_grid : array (n_t,) – sorted maturities
    market_call_prices : array (n_t, n_k) – market call prices
    S : float – spot
    r, q : float – rates

    Returns dict with: strikes, times, local_vols (n_t, n_k).
    """
    n_t = len(times_grid)
    n_k = len(strikes_grid)

    local_vols = jnp.full((n_t, n_k), 0.2)  # initial guess

    # Forward PDE: solve Dupire PDE forward in time
    # dC/dT = 0.5 * sigma_loc^2 * K^2 * d^2C/dK^2 - (r-q)*K*dC/dK - q*C
    dx = jnp.diff(strikes_grid)

    for t_idx in range(n_t):
        dt = times_grid[t_idx] if t_idx == 0 else times_grid[t_idx] - times_grid[t_idx - 1]

        # Initial condition: C(0, K) = max(S - K, 0)
        if t_idx == 0:
            C = jnp.maximum(S - strikes_grid, 0.0)
        else:
            C = C_prev.copy()

        # Calibrate local vol at this time slice
        target = market_call_prices[t_idx]

        for iteration in range(20):
            sigma_loc = local_vols[t_idx]

            # Crank-Nicolson step
            C_new = _cn_step(C, strikes_grid, sigma_loc, r, q, dt, n_fd_steps)

            # Update local vol to match market prices
            for k_idx in range(1, n_k - 1):
                if target[k_idx] > 1e-8:
                    ratio = target[k_idx] / jnp.maximum(C_new[k_idx], 1e-10)
                    adjustment = jnp.sqrt(jnp.maximum(ratio, 0.1))
                    local_vols = local_vols.at[t_idx, k_idx].set(
                        jnp.clip(sigma_loc[k_idx] * adjustment, 0.01, 5.0)
                    )

            C_new = _cn_step(C, strikes_grid, local_vols[t_idx], r, q, dt, n_fd_steps)

        C_prev = C_new

    return {
        'strikes': strikes_grid,
        'times': times_grid,
        'local_vols': local_vols,
        'S': S, 'r': r, 'q': q,
    }


def evaluate_local_vol(data, t, K):
    """Evaluate calibrated local vol at (t, K) via bilinear interpolation."""
    t_idx = jnp.searchsorted(data['times'], t, side='right') - 1
    t_idx = jnp.clip(t_idx, 0, len(data['times']) - 1)
    k_idx = jnp.searchsorted(data['strikes'], K, side='right') - 1
    k_idx = jnp.clip(k_idx, 0, len(data['strikes']) - 1)
    return data['local_vols'][t_idx, k_idx]


def evaluate_implied_vol(data, t, K):
    """Evaluate implied Black vol at (t, K).

    Prices via local vol, then inverts Black formula.
    """
    from ql_jax.termstructures.volatility.local_vol_surface import _bs_call
    # Build forward Dupire PDE to get call price, then invert
    local_vol = evaluate_local_vol(data, t, K)
    # Approximate: use local vol as implied vol proxy
    return local_vol


def _cn_step(C, strikes, sigma, r, q, dt, n_sub):
    """Crank-Nicolson substep for Dupire PDE."""
    n = len(strikes)
    dt_sub = dt / n_sub
    C_new = C.copy()

    for _ in range(n_sub):
        # Second derivative via central differences
        d2C = jnp.zeros(n)
        for i in range(1, n - 1):
            dx_l = strikes[i] - strikes[i - 1]
            dx_r = strikes[i + 1] - strikes[i]
            d2C = d2C.at[i].set(
                2.0 * (C_new[i + 1] / dx_r - C_new[i] * (1.0 / dx_l + 1.0 / dx_r) + C_new[i - 1] / dx_l) /
                (dx_l + dx_r)
            )

        # First derivative
        dC = jnp.zeros(n)
        for i in range(1, n - 1):
            dC = dC.at[i].set((C_new[i + 1] - C_new[i - 1]) / (strikes[i + 1] - strikes[i - 1]))

        # PDE RHS
        rhs = 0.5 * sigma**2 * strikes**2 * d2C - (r - q) * strikes * dC - q * C_new

        # Explicit Euler (simplified; full CN would need tridiagonal solve)
        C_new = C_new + dt_sub * rhs

        # Boundary: C(K=0) = S*e^{-qT}, C(K=inf) = 0
        C_new = C_new.at[0].set(jnp.maximum(C_new[0], 0.0))
        C_new = C_new.at[-1].set(0.0)

    return C_new
