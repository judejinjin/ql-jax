"""FD Heston engine – vectorized 2D ADI finite-difference for Heston model."""

import jax
import jax.numpy as jnp

from ql_jax._util.types import OptionType
from ql_jax.methods.finitedifferences.meshers import Concentrating1dMesher
from ql_jax.methods.finitedifferences.heston_operator import (
    build_heston_operator, heston_douglas_step,
)


def fd_heston_price(S, K, T, r, q, v0, kappa, theta, sigma_v, rho,
                     option_type: int, n_x=100, n_v=50, n_t=100,
                     american=False):
    """Price a vanilla option under Heston using vectorized 2D FD (ADI)."""
    S, K, T = (jnp.asarray(x, dtype=jnp.float64) for x in (S, K, T))

    # Log-spot grid
    x_mesher = Concentrating1dMesher(
        low=float(jnp.log(K * 0.1)), high=float(jnp.log(K * 10.0)),
        size=n_x, center=float(jnp.log(S)), density=5.0,
    )
    x_grid = x_mesher.locations()

    # Variance grid
    v_max = max(5.0 * theta, 3.0 * v0)
    v_mesher = Concentrating1dMesher(
        low=0.001, high=v_max, size=n_v, center=v0, density=3.0,
    )
    v_grid = v_mesher.locations()

    # Build operator (vectorized, done once)
    op = build_heston_operator(
        r=float(r), q=float(q), kappa=kappa, theta=theta,
        sigma_v=sigma_v, rho=rho, x_grid=x_grid, v_grid=v_grid,
    )

    # Terminal condition
    s_grid = jnp.exp(x_grid)
    phi = jnp.where(option_type == 1, 1.0, -1.0)
    payoff = jnp.maximum(phi * (s_grid - K), 0.0)
    V_2d = jnp.outer(payoff, jnp.ones(n_v))

    # Time stepping (backward) with Python loop (each step has different BCs)
    dt = float(T) / n_t
    for step in range(n_t):
        V_2d = heston_douglas_step(V_2d, dt, op, theta_fd=0.5)

        # Boundary conditions
        tau = (step + 1) * dt
        disc = jnp.exp(-r * tau)
        V_2d = V_2d.at[:, 0].set(jnp.maximum(phi * (s_grid - K * disc), 0.0))
        V_2d = V_2d.at[:, -1].set(2.0 * V_2d[:, -2] - V_2d[:, -3])
        # S boundaries
        call_lo = 0.0
        call_hi = s_grid[-1] - K * disc
        put_lo = K * disc - s_grid[0]
        put_hi = 0.0
        lo_val = jnp.where(phi > 0, call_lo, put_lo)
        hi_val = jnp.where(phi > 0, call_hi, put_hi)
        V_2d = V_2d.at[0, :].set(lo_val)
        V_2d = V_2d.at[-1, :].set(hi_val)

        # American exercise
        if american:
            V_2d = jnp.maximum(V_2d, jnp.outer(payoff, jnp.ones(n_v)))

    # Interpolate to (S, v0) with bilinear interpolation
    x_target = jnp.log(S)
    # x-interpolation for each v-column, then v-interpolation
    prices_at_v = jax.vmap(lambda col: jnp.interp(x_target, x_grid, col))(V_2d.T)
    price = jnp.interp(v0, v_grid, prices_at_v)
    return price
