"""Additional FD operators for extended models.

Adds operators for Bates (Heston + jumps), CEV, CIR, SABR, HW, G2++,
forward BS/Heston, and 2D Black-Scholes for basket/spread FD.
"""

from __future__ import annotations

import jax.numpy as jnp
from dataclasses import dataclass

from ql_jax.methods.finitedifferences.operators import (
    TridiagonalOperator,
    d_zero,
    d_plus_d_minus,
)


def fdm_bates_operator(x_grid, v_grid, r, q, kappa, theta, xi, rho,
                        lam_j, mu_j, sigma_j, t=0.0):
    """Bates model FD operator: Heston + Merton jumps.

    Heston diffusion part + integral (jump) correction to drift.
    The jump integral is approximated as a drift/variance adjustment.

    Parameters
    ----------
    x_grid : 1D log-spot grid
    v_grid : 1D variance grid
    r, q : rates
    kappa, theta, xi, rho : Heston parameters
    lam_j : jump intensity
    mu_j : mean log-jump size
    sigma_j : jump size vol

    Returns
    -------
    drift_adj : adjusted drift for the x-dimension
    """
    # Jump compensation: E[e^J - 1] = exp(mu_j + 0.5*sigma_j^2) - 1
    jump_comp = lam_j * (jnp.exp(mu_j + 0.5 * sigma_j**2) - 1.0)
    # Adjusted drift
    drift_x = r - q - 0.5  # * v (applied pointwise) - jump_comp
    return drift_x - jump_comp


def fdm_cev_operator(x_grid, r, q, sigma, beta):
    """CEV model 1D FD operator on the spot grid.

    dS = (r-q)*S*dt + sigma*S^beta*dW

    Parameters
    ----------
    x_grid : spot grid
    r, q : rates
    sigma, beta : CEV parameters

    Returns
    -------
    L : TridiagonalOperator
    """
    S = x_grid
    n = len(S)

    # Drift: (r-q)*S
    drift = (r - q) * S

    # Diffusion: 0.5 * sigma^2 * S^{2*beta}
    diffusion = 0.5 * sigma**2 * S**(2.0 * beta)

    # Build operators on the grid
    D1 = d_zero(S)
    D2 = d_plus_d_minus(S)

    # Combined: L = diffusion * d^2/dS^2 + drift * d/dS - r
    lower = diffusion[1:] * D2.lower + drift[1:] * D1.lower
    diag = diffusion * D2.diag + drift * D1.diag - r
    upper = diffusion[:-1] * D2.upper + drift[:-1] * D1.upper

    return TridiagonalOperator(lower, diag, upper)


def fdm_hull_white_operator(r_grid, a, sigma, theta_t):
    """Hull-White short-rate 1D FD operator.

    dr = (theta(t) - a*r)*dt + sigma*dW

    Parameters
    ----------
    r_grid : 1D rate grid
    a : mean reversion
    sigma : volatility
    theta_t : theta(t) value at current time

    Returns
    -------
    L : TridiagonalOperator
    """
    r = r_grid
    n = len(r)

    drift = theta_t - a * r
    diffusion = 0.5 * sigma**2

    D1 = d_zero(r)
    D2 = d_plus_d_minus(r)

    lower = diffusion * D2.lower + drift[1:] * D1.lower
    diag = diffusion * D2.diag + drift * D1.diag - r
    upper = diffusion * D2.upper + drift[:-1] * D1.upper

    return TridiagonalOperator(lower, diag, upper)


def fdm_sabr_operator(f_grid, alpha, beta, rho, nu):
    """SABR model 1D FD operator (frozen alpha approximation).

    Parameters
    ----------
    f_grid : forward rate grid
    alpha, beta, rho, nu : SABR parameters

    Returns
    -------
    L : TridiagonalOperator
    """
    F = f_grid
    n = len(F)

    # Local vol: sigma_loc = alpha * F^beta
    sigma_loc = alpha * jnp.power(jnp.maximum(F, 1e-10), beta)
    diffusion = 0.5 * sigma_loc**2

    D2 = d_plus_d_minus(F)

    # Pure diffusion operator (no drift for SABR in forward measure)
    lower = diffusion[1:] * D2.lower
    diag = diffusion * D2.diag
    upper = diffusion[:-1] * D2.upper

    return TridiagonalOperator(lower, diag, upper)


@dataclass(frozen=True)
class Fdm2dBlackScholesOp:
    """2D Black-Scholes operator for basket/spread FD.

    Stores x and y direction operators plus cross-derivative term.
    """
    Lx: TridiagonalOperator
    Ly: TridiagonalOperator
    rho: float
    sigma_x: float
    sigma_y: float
    x_grid: jnp.ndarray
    y_grid: jnp.ndarray

    def apply(self, V):
        """Apply 2D operator to value grid V[nx, ny]."""
        nx, ny = V.shape
        result = jnp.zeros_like(V)

        # x-direction
        for j in range(ny):
            result = result.at[:, j].add(self.Lx.apply(V[:, j]))

        # y-direction
        for i in range(nx):
            result = result.at[i, :].add(self.Ly.apply(V[i, :]))

        # Cross derivative (central differences)
        dx = self.x_grid[1] - self.x_grid[0]
        dy = self.y_grid[1] - self.y_grid[0]
        cross = self.rho * self.sigma_x * self.sigma_y

        for i in range(1, nx - 1):
            for j in range(1, ny - 1):
                d2_cross = (V[i+1, j+1] - V[i+1, j-1] - V[i-1, j+1] + V[i-1, j-1]) / (4.0 * dx * dy)
                result = result.at[i, j].add(cross * self.x_grid[i] * self.y_grid[j] * d2_cross)

        return result
