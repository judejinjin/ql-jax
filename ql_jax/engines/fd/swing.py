"""Finite-difference engine for swing (virtual storage) options."""

from __future__ import annotations

import jax.numpy as jnp

from ql_jax.methods.finitedifferences.operators import (
    TridiagonalOperator, d_zero, d_plus_d_minus,
)
from ql_jax.methods.finitedifferences.schemes import theta_step


def fd_swing_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    q: float,
    sigma: float,
    n_exercises: int,
    *,
    n_x: int = 200,
    n_t: int = 200,
    n_sd: float = 4.0,
    min_rest: int = 1,
) -> float:
    """Price a swing option using finite differences.

    A swing option allows the holder to exercise (buy/sell at strike K)
    up to ``n_exercises`` times over the life.  Between exercises there is
    a mandatory rest period of ``min_rest`` time-steps.

    Parameters
    ----------
    S0, K, T, r, q, sigma : standard BS parameters
    n_exercises : maximum number of exercise rights
    n_x : spatial grid points
    n_t : time steps
    n_sd : standard deviations for log-spot grid
    min_rest : minimum time-steps between exercises

    Returns
    -------
    price : float
    """
    dt = T / n_t
    x_min = jnp.log(S0) - n_sd * sigma * jnp.sqrt(T)
    x_max = jnp.log(S0) + n_sd * sigma * jnp.sqrt(T)
    x = jnp.linspace(x_min, x_max, n_x)
    S = jnp.exp(x)
    dx = x[1] - x[0]

    # BS operator in log-spot coordinates
    nu = r - q - 0.5 * sigma**2
    diff_coef = 0.5 * sigma**2
    D1 = d_zero(x)
    D2 = d_plus_d_minus(x)
    lower = diff_coef * D2.lower + nu * D1.lower
    diag = diff_coef * D2.diag + nu * D1.diag - r
    upper = diff_coef * D2.upper + nu * D1.upper
    L = TridiagonalOperator(lower, diag, upper)

    # Intrinsic payoff per exercise
    payoff = jnp.maximum(S - K, 0.0)

    # V[j] = value of option with j remaining exercise rights
    V = jnp.zeros((n_exercises + 1, n_x))

    for step in range(n_t - 1, -1, -1):
        # Time-step the continuation for each rights-level
        V_new = jnp.zeros_like(V)
        for j in range(n_exercises + 1):
            V_new = V_new.at[j].set(theta_step(V[j], dt, L, theta=0.5))

        # Exercise decision: exercising moves from j rights → j-1 rights
        for j in range(1, n_exercises + 1):
            exercise_val = payoff + V_new[j - 1]
            V_new = V_new.at[j].set(jnp.maximum(V_new[j], exercise_val))

        V = V_new

    # Interpolate at S0
    idx = jnp.searchsorted(x, jnp.log(S0))
    idx = jnp.clip(idx, 1, n_x - 1)
    w = (jnp.log(S0) - x[idx - 1]) / dx
    price = V[n_exercises, idx - 1] * (1 - w) + V[n_exercises, idx] * w
    return price
