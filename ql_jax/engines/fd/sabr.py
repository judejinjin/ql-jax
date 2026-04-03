"""FD SABR vanilla engine.

Solves the SABR PDE in frozen-alpha approximation using
finite differences on a log-forward grid with jax.lax.fori_loop.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ql_jax.methods.finitedifferences.operators import tridiag_solve


def fd_sabr_price(F, K, T, r, alpha, beta, rho, nu, option_type=1,
                  n_x=200, n_t=200, is_american=False, s_max_mult=4.0):
    """SABR model FD price with frozen-alpha approximation.

    Parameters
    ----------
    F : forward price
    K : strike
    T : maturity
    r : discount rate
    alpha, beta, rho, nu : SABR parameters
    option_type : 1=call, -1=put
    n_x : spatial grid
    n_t : time steps
    is_american : early exercise
    s_max_mult : grid range factor

    Returns
    -------
    price : option price
    """
    F, K, T, r, alpha = (jnp.asarray(x, dtype=jnp.float64)
                          for x in (F, K, T, r, alpha))
    phi = jnp.asarray(jnp.where(option_type == 1, 1.0, -1.0), dtype=jnp.float64)

    # Log-forward grid
    x_min = jnp.log(K / s_max_mult)
    x_max = jnp.log(K * s_max_mult)
    dx = (x_max - x_min) / n_x
    dt = T / n_t

    x_grid = jnp.linspace(x_min, x_max, n_x + 1)
    f_grid = jnp.exp(x_grid)

    # SABR local vol: sigma_loc = alpha * F^beta
    a_coeff = 0.5 * alpha**2 * f_grid**(2.0 * (beta - 1.0))
    b_coeff = -a_coeff

    a_int = a_coeff[1:-1]
    b_int = b_coeff[1:-1]
    al = a_int / dx**2 - b_int / (2.0 * dx)
    ad = -2.0 * a_int / dx**2
    au = a_int / dx**2 + b_int / (2.0 * dx)

    # Terminal condition
    V = jnp.maximum(phi * (f_grid - K), 0.0)
    exercise_values = V

    # Tridiagonal LHS
    theta_fd = 0.5
    lower_vec = -theta_fd * dt * al[1:]
    diag_vec = 1.0 - theta_fd * dt * ad
    upper_vec = -theta_fd * dt * au[:-1]

    def time_step(i, V):
        rhs_inner = V[1:-1] + (1.0 - theta_fd) * dt * (
            al * V[:-2] + ad * V[1:-1] + au * V[2:]
        )

        V_left = jnp.where(phi > 0, 0.0, K - f_grid[0])
        V_right = jnp.where(phi > 0, f_grid[-1] - K, 0.0)

        rhs_inner = rhs_inner.at[0].add(theta_fd * dt * al[0] * V_left)
        rhs_inner = rhs_inner.at[-1].add(theta_fd * dt * au[-1] * V_right)

        V_inner = tridiag_solve(lower_vec, diag_vec, upper_vec, rhs_inner)

        V = V.at[0].set(V_left)
        V = V.at[-1].set(V_right)
        V = V.at[1:-1].set(V_inner)

        V = jnp.where(is_american, jnp.maximum(V, exercise_values), V)

        return V

    V = jax.lax.fori_loop(0, n_t, time_step, V)

    x_target = jnp.log(F)
    return float(jnp.exp(-r * T) * jnp.interp(x_target, x_grid, V))
