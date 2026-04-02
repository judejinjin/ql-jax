"""FD Heston engine – 2D ADI finite-difference for Heston model."""

import jax.numpy as jnp

from ql_jax._util.types import OptionType
from ql_jax.methods.finitedifferences.meshers import (
    Concentrating1dMesher, Uniform1dMesher,
)
from ql_jax.methods.finitedifferences.heston_operator import (
    HestonFdmOperator, heston_douglas_step,
)


def fd_heston_price(S, K, T, r, q, v0, kappa, theta, sigma_v, rho,
                     option_type: int, n_x=100, n_v=50, n_time=100,
                     american=False):
    """Price a vanilla option under Heston using 2D FD (ADI).

    Parameters
    ----------
    S, K, T, r, q : float – market params
    v0 : float – initial variance
    kappa, theta, sigma_v, rho : float – Heston params
    option_type : int
    n_x, n_v : int – grid sizes
    n_time : int
    american : bool

    Returns
    -------
    price : float
    """
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

    # Build operator
    op = HestonFdmOperator(
        r=float(r), q=float(q), kappa=kappa, theta=theta,
        sigma_v=sigma_v, rho=rho, x_grid=x_grid, v_grid=v_grid,
    )

    # Terminal condition
    s_grid = jnp.exp(x_grid)
    phi = jnp.where(option_type == OptionType.Call, 1.0, -1.0)
    payoff = jnp.maximum(phi * (s_grid - K), 0.0)
    V_2d = jnp.outer(payoff, jnp.ones(n_v))

    # Time stepping (backward)
    dt = float(T) / n_time
    for step in range(n_time):
        V_2d = heston_douglas_step(V_2d, dt, op, theta_fd=0.5)

        # Boundary conditions
        # v=0: BS with sigma=0 => intrinsic value
        V_2d = V_2d.at[:, 0].set(jnp.maximum(phi * (s_grid - K * jnp.exp(-r * (step + 1) * dt)), 0.0))
        # v=v_max: linear extrapolation
        V_2d = V_2d.at[:, -1].set(2.0 * V_2d[:, -2] - V_2d[:, -3])
        # S boundaries
        if option_type == OptionType.Call:
            V_2d = V_2d.at[0, :].set(0.0)
            V_2d = V_2d.at[-1, :].set(s_grid[-1] - K * jnp.exp(-r * (step + 1) * dt))
        else:
            V_2d = V_2d.at[0, :].set(K * jnp.exp(-r * (step + 1) * dt) - s_grid[0])
            V_2d = V_2d.at[-1, :].set(0.0)

        # American exercise
        if american:
            V_2d = jnp.maximum(V_2d, jnp.outer(payoff, jnp.ones(n_v)))

    # Interpolate to (S, v0)
    x_target = jnp.log(S)
    # Find closest v-grid index for v0
    v_idx = jnp.argmin(jnp.abs(v_grid - v0))

    # Bilinear interpolation
    price_at_v = jnp.interp(x_target, x_grid, V_2d[:, v_idx])
    return price_at_v
