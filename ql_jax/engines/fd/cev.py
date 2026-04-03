"""FD CEV vanilla engine.

Solves the CEV PDE on a log-spot grid using Crank-Nicolson time
stepping with jax.lax.fori_loop, modeled after the BS FD engine.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ql_jax.methods.finitedifferences.operators import tridiag_solve


def fd_cev_price(S, K, T, r, q, sigma, beta, option_type=1,
                 n_x=200, n_t=200, is_american=False, s_max_mult=4.0):
    """CEV model FD price.

    Parameters
    ----------
    S, K, T, r, q : standard params
    sigma, beta : CEV params (dS = (r-q)S dt + sigma*S^beta dW)
    option_type : 1=call, -1=put
    n_x : spatial grid points
    n_t : time steps
    is_american : early exercise
    s_max_mult : grid range (K/mult to K*mult)

    Returns
    -------
    price : option price
    """
    S, K, T, r, q, sigma = (jnp.asarray(x, dtype=jnp.float64)
                             for x in (S, K, T, r, q, sigma))
    phi = jnp.asarray(jnp.where(option_type == 1, 1.0, -1.0), dtype=jnp.float64)

    # Log-spot grid
    x_min = jnp.log(K / s_max_mult)
    x_max = jnp.log(K * s_max_mult)
    dx = (x_max - x_min) / n_x
    dt = T / n_t

    x_grid = jnp.linspace(x_min, x_max, n_x + 1)
    s_grid = jnp.exp(x_grid)

    # CEV PDE coefficients in log-spot space
    a_coeff = 0.5 * sigma**2 * s_grid**(2.0 * (beta - 1.0))
    b_coeff = (r - q) - a_coeff

    a_int = a_coeff[1:-1]
    b_int = b_coeff[1:-1]
    al = a_int / dx**2 - b_int / (2.0 * dx)
    ad = -2.0 * a_int / dx**2 - r
    au = a_int / dx**2 + b_int / (2.0 * dx)

    # Terminal condition
    V = jnp.maximum(phi * (s_grid - K), 0.0)
    exercise_values = V

    # Tridiagonal LHS
    theta_fd = 0.5
    lower_vec = -theta_fd * dt * al[1:]
    diag_vec = 1.0 - theta_fd * dt * ad
    upper_vec = -theta_fd * dt * au[:-1]

    # Precompute boundary values for call and put at each step
    # Use jnp.where to avoid Python if inside fori_loop
    def time_step(i, V):
        remaining = T - (i + 1) * dt

        rhs_inner = V[1:-1] + (1.0 - theta_fd) * dt * (
            al * V[:-2] + ad * V[1:-1] + au * V[2:]
        )

        # Use jnp.where for call/put boundary
        V_left = jnp.where(phi > 0, 0.0,
                           K * jnp.exp(-r * remaining) - s_grid[0] * jnp.exp(-q * remaining))
        V_right = jnp.where(phi > 0,
                            s_grid[-1] * jnp.exp(-q * remaining) - K * jnp.exp(-r * remaining),
                            0.0)

        rhs_inner = rhs_inner.at[0].add(theta_fd * dt * al[0] * V_left)
        rhs_inner = rhs_inner.at[-1].add(theta_fd * dt * au[-1] * V_right)

        V_inner = tridiag_solve(lower_vec, diag_vec, upper_vec, rhs_inner)

        V = V.at[0].set(V_left)
        V = V.at[-1].set(V_right)
        V = V.at[1:-1].set(V_inner)

        # American early exercise via jnp.where
        V = jnp.where(is_american, jnp.maximum(V, exercise_values), V)

        return V

    V = jax.lax.fori_loop(0, n_t, time_step, V)

    x_target = jnp.log(S)
    return float(jnp.interp(x_target, x_grid, V))
