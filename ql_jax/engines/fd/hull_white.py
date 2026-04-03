"""FD Hull-White short-rate engine for bond options, swaptions, caps/floors."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ql_jax.methods.finitedifferences.operators import tridiag_solve


def fd_hull_white_bond_option(face, K, T_option, T_bond, r0,
                              a, sigma, discount_curve_fn,
                              option_type=1, n_r=200, n_t=200):
    """Price a European bond option under Hull-White using FD.

    Parameters
    ----------
    face : bond face value
    K : option strike (price)
    T_option : option expiry
    T_bond : bond maturity (T_bond > T_option)
    r0 : current short rate
    a : mean reversion speed
    sigma : short-rate volatility
    discount_curve_fn : P(0,t) market discount function
    option_type : 1=call (right to buy bond), -1=put
    n_r : rate grid points
    n_t : time steps

    Returns
    -------
    price : bond option price
    """
    phi = jnp.asarray(jnp.where(option_type == 1, 1.0, -1.0), dtype=jnp.float64)
    dt = T_option / n_t

    # Instantaneous forward rate for theta(t) calibration
    eps = 1e-4

    # Vectorize discount curve
    disc_vec_fn = jax.vmap(lambda t: discount_curve_fn(t))

    # Rate grid centered around r0
    r_range = 6.0 * sigma / jnp.sqrt(2.0 * a)
    r_min = r0 - float(r_range)
    r_max = r0 + float(r_range)
    r_grid = jnp.linspace(r_min, r_max, n_r + 1)
    dr = (r_max - r_min) / n_r

    # Terminal condition: bond price at T_option as function of r
    B_val = (1.0 - jnp.exp(-a * (T_bond - T_option))) / a
    f_Te = -(jnp.log(discount_curve_fn(T_option + eps))
             - jnp.log(discount_curve_fn(T_option))) / eps
    lnA = (jnp.log(discount_curve_fn(T_bond) / discount_curve_fn(T_option))
           + B_val * float(f_Te)
           - sigma**2 / (4.0 * a) * (1.0 - jnp.exp(-2.0 * a * T_option)) * B_val**2)
    A_val = jnp.exp(lnA)

    bond_price = face * A_val * jnp.exp(-B_val * r_grid)
    V = jnp.maximum(phi * (bond_price - K), 0.0)

    # Precompute theta(t) values at each backward time step
    t_steps = jnp.maximum(T_option - jnp.arange(n_t) * dt, eps)
    f_ts = -(jnp.log(disc_vec_fn(t_steps + eps))
             - jnp.log(disc_vec_fn(t_steps))) / eps
    df_ts = -(jnp.log(disc_vec_fn(t_steps + 2*eps))
              - 2*jnp.log(disc_vec_fn(t_steps + eps))
              + jnp.log(disc_vec_fn(t_steps))) / eps**2
    theta_vals = df_ts + a * f_ts + sigma**2 / (2.0 * a) * (1.0 - jnp.exp(-2.0 * a * t_steps))

    # Constant PDE coefficients (interior points only)
    r_int = r_grid[1:-1]
    diffusion = 0.5 * sigma**2

    # Time-independent parts of al, ad, au
    # drift = theta_t - a*r_int (only -a*r_int part is constant)
    al_const = diffusion / dr**2 + a * r_int / (2.0 * dr)
    ad_const = -2.0 * diffusion / dr**2 - r_int
    au_const = diffusion / dr**2 - a * r_int / (2.0 * dr)

    # theta_t adds: -theta_t/(2dr) to al, +theta_t/(2dr) to au, 0 to ad
    theta_al_factor = -1.0 / (2.0 * dr)
    theta_au_factor = 1.0 / (2.0 * dr)

    # Constant part of LHS tridiag: (I - theta_fd*dt*L_const)
    theta_fd = 0.5
    lower_const = -theta_fd * dt * al_const[1:]
    diag_const = 1.0 - theta_fd * dt * ad_const
    upper_const = -theta_fd * dt * au_const[:-1]

    # Initial boundary values
    V_init_left = float(V[0])
    V_init_right = float(V[-1])

    def time_step(i, V):
        theta_i = theta_vals[i]
        remaining = T_option - (i + 1) * dt

        al = al_const + theta_i * theta_al_factor
        au = au_const + theta_i * theta_au_factor
        ad = ad_const

        rhs_inner = V[1:-1] + (1.0 - theta_fd) * dt * (
            al * V[:-2] + ad * V[1:-1] + au * V[2:]
        )

        # Bond price at boundary rates
        bp_left = face * A_val * jnp.exp(-B_val * r_grid[0])
        bp_right = face * A_val * jnp.exp(-B_val * r_grid[-1])

        # Use jnp.where for call/put boundary conditions
        V_left = jnp.where(
            phi > 0,
            jnp.maximum(bp_left - K, 0.0) * jnp.exp(-r_grid[0] * remaining),
            0.0
        )
        V_right = jnp.where(
            phi > 0,
            0.0,
            jnp.maximum(K - bp_right, 0.0) * jnp.exp(-r_grid[-1] * remaining)
        )

        rhs_inner = rhs_inner.at[0].add(theta_fd * dt * al[0] * V_left)
        rhs_inner = rhs_inner.at[-1].add(theta_fd * dt * au[-1] * V_right)

        lower_vec = lower_const - theta_fd * dt * theta_i * theta_al_factor
        upper_vec = upper_const - theta_fd * dt * theta_i * theta_au_factor

        V_inner = tridiag_solve(lower_vec, diag_const, upper_vec, rhs_inner)

        V = V.at[0].set(V_left)
        V = V.at[-1].set(V_right)
        V = V.at[1:-1].set(V_inner)

        return V

    V = jax.lax.fori_loop(0, n_t, time_step, V)

    return float(jnp.interp(r0, r_grid, V))
