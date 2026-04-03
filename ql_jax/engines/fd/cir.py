"""Finite-difference engine for vanilla options under the CIR process."""

from __future__ import annotations

import jax.numpy as jnp

from ql_jax.methods.finitedifferences.operators import (
    TridiagonalOperator, d_zero, d_plus_d_minus,
)
from ql_jax.methods.finitedifferences.schemes import theta_step


def fd_cir_vanilla_price(
    S0: float,
    K: float,
    T: float,
    r: float,
    kappa: float,
    theta: float,
    sigma: float,
    option_type: int = 1,
    american: bool = False,
    *,
    n_x: int = 200,
    n_t: int = 200,
) -> float:
    """Price a vanilla option when the underlying follows a CIR process.

    dS = kappa*(theta - S)*dt + sigma*sqrt(S)*dW

    Used for interest-rate or volatility-linked derivatives where the
    underlying is mean-reverting with a square-root diffusion.

    The option payoff at maturity is max(option_type * (S - K), 0).

    Parameters
    ----------
    S0 : initial level
    K : strike
    T : maturity
    r : risk-free rate for discounting
    kappa, theta, sigma : CIR parameters
    option_type : +1 call, -1 put
    american : if True, allow early exercise
    n_x, n_t : grid sizes
    """
    dt = T / n_t

    # Build grid: CIR stays positive, concentrate near theta
    S_max = max(theta * 5, S0 * 3, K * 3)
    S_grid = jnp.linspace(1e-6, S_max, n_x)

    # CIR drift and diffusion
    drift = kappa * (theta - S_grid)
    diffusion = 0.5 * sigma**2 * S_grid

    D1 = d_zero(S_grid)
    D2 = d_plus_d_minus(S_grid)

    lower = diffusion[1:] * D2.lower + drift[1:] * D1.lower
    diag = diffusion * D2.diag + drift * D1.diag - r
    upper = diffusion[:-1] * D2.upper + drift[:-1] * D1.upper

    L = TridiagonalOperator(lower, diag, upper)

    # Terminal condition
    intrinsic = jnp.maximum(option_type * (S_grid - K), 0.0)
    V = intrinsic.copy()

    for step in range(n_t):
        V = theta_step(V, dt, L, theta=0.5)
        if american:
            V = jnp.maximum(V, intrinsic)

    # Interpolate at S0
    price = float(jnp.interp(S0, S_grid, V))
    return price
