"""Finite-difference Black-Scholes engine for European and American options.

Uses Crank-Nicolson time-stepping with a tridiagonal solver (Thomas algorithm).
All loops use jax.lax.fori_loop / jax.lax.scan for efficient JIT compilation.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax

from ql_jax._util.types import OptionType


def fd_black_scholes_price(
    S, K, T, r, q, sigma, option_type: int,
    n_space: int = 200,
    n_time: int = 200,
    american: bool = False,
    s_max_mult: float = 4.0,
    theta_fd: float = 0.5,  # 0.5 = Crank-Nicolson, 1.0 = implicit Euler
):
    """Price a vanilla option using finite differences.

    Solves the Black-Scholes PDE on a uniform log-spot grid (Crank-Nicolson).

    Parameters
    ----------
    S, K, T, r, q, sigma : option parameters
    option_type : OptionType.Call or OptionType.Put
    n_space : number of spatial grid points
    n_time : number of time steps
    american : if True, enforce early exercise constraint
    s_max_mult : max spot / K ratio for the grid
    theta_fd : 0.5 for Crank-Nicolson, 1.0 for fully implicit

    Returns
    -------
    price : float
    """
    S, K, T, r, q, sigma = (jnp.asarray(x, dtype=jnp.float64)
                             for x in (S, K, T, r, q, sigma))

    # Grid in log-spot space: x = log(S)
    x_min = jnp.log(K / s_max_mult)
    x_max = jnp.log(K * s_max_mult)
    dx = (x_max - x_min) / n_space
    dt = T / n_time

    x_grid = jnp.linspace(x_min, x_max, n_space + 1)
    s_grid = jnp.exp(x_grid)

    # PDE coefficients for Black-Scholes in log-spot
    alpha = 0.5 * sigma**2
    beta = r - q - 0.5 * sigma**2

    # Spatial operator L V_i = al * V_{i-1} + ad * V_i + au * V_{i+1}
    al = alpha / dx**2 - beta / (2.0 * dx)
    ad = -2.0 * alpha / dx**2 - r
    au = alpha / dx**2 + beta / (2.0 * dx)

    # Terminal condition (payoff at T)
    phi = jnp.where(option_type == OptionType.Call, 1.0, -1.0)
    V = jnp.maximum(phi * (s_grid - K), 0.0)

    # Precompute tridiagonal system: (I - theta*dt*L) on LHS
    lower_vec = (-theta_fd * dt * al) * jnp.ones(n_space - 2)
    diag_vec = (1.0 - theta_fd * dt * ad) * jnp.ones(n_space - 1)
    upper_vec = (-theta_fd * dt * au) * jnp.ones(n_space - 2)

    is_call = (option_type == OptionType.Call)
    exercise_values = jnp.maximum(phi * (s_grid - K), 0.0)

    def time_step(i, V):
        remaining = T - (i + 1) * dt

        # Explicit RHS: V + (1-theta)*dt*L*V
        rhs_inner = V[1:-1] + (1.0 - theta_fd) * dt * (
            al * V[:-2] + ad * V[1:-1] + au * V[2:]
        )

        # Boundary conditions (time-dependent)
        if is_call:
            V_left = 0.0
            V_right = s_grid[-1] - K * jnp.exp(-r * remaining)
        else:
            V_left = K * jnp.exp(-r * remaining) - s_grid[0]
            V_right = 0.0

        # Boundary correction: move known boundary terms from LHS to RHS
        rhs_inner = rhs_inner.at[0].add(theta_fd * dt * al * V_left)
        rhs_inner = rhs_inner.at[-1].add(theta_fd * dt * au * V_right)

        # Solve tridiagonal system
        V_inner = _tridiag_solve(lower_vec, diag_vec, upper_vec, rhs_inner)

        V = V.at[0].set(V_left)
        V = V.at[-1].set(V_right)
        V = V.at[1:-1].set(V_inner)

        # American exercise constraint
        if american:
            V = jnp.maximum(V, exercise_values)

        return V

    V = jax.lax.fori_loop(0, n_time, time_step, V)

    # Interpolate to get price at spot S
    x_target = jnp.log(S)
    price = jnp.interp(x_target, x_grid, V)
    return price


def _tridiag_solve(lower, diag, upper, rhs):
    """Solve a tridiagonal system using the Thomas algorithm via jax.lax.scan.

    Parameters
    ----------
    lower : sub-diagonal (n-1 elements)
    diag : main diagonal (n elements)
    upper : super-diagonal (n-1 elements)
    rhs : right-hand side (n elements)

    Returns
    -------
    solution : (n,) array
    """
    # Forward sweep
    c0 = upper[0] / diag[0]
    d0 = rhs[0] / diag[0]

    # Scan inputs for i=1..n-1 (n-1 iterations)
    upper_scan = jnp.concatenate([upper[1:], jnp.array([0.0])])

    def forward_step(carry, inputs):
        c_prev, d_prev = carry
        l_i, diag_i, up_i, rhs_i = inputs
        denom = diag_i - l_i * c_prev
        c_new = up_i / denom
        d_new = (rhs_i - l_i * d_prev) / denom
        return (c_new, d_new), (c_new, d_new)

    (_, _), (c_arr, d_arr) = jax.lax.scan(
        forward_step, (c0, d0),
        (lower, diag[1:], upper_scan, rhs[1:])
    )

    c_prime = jnp.concatenate([jnp.array([c0]), c_arr])
    d_prime = jnp.concatenate([jnp.array([d0]), d_arr])

    # Back substitution (reverse scan)
    def backward_step(x_next, inputs):
        c_i, d_i = inputs
        x_i = d_i - c_i * x_next
        return x_i, x_i

    x_last = d_prime[-1]
    _, x_rev = jax.lax.scan(
        backward_step, x_last,
        (c_prime[:-1], d_prime[:-1]),
        reverse=True,
    )

    x = jnp.concatenate([x_rev, jnp.array([x_last])])
    return x
