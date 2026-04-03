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


def fdm_cir_operator(x_grid, kappa, theta_cir, sigma, r_discount=None):
    """CIR (Cox-Ingersoll-Ross) 1D FD operator.

    dX = kappa*(theta - X)*dt + sigma*sqrt(X)*dW

    Parameters
    ----------
    x_grid : 1D grid for the CIR variable (must be positive)
    kappa : mean reversion speed
    theta_cir : long-run mean
    sigma : vol-of-vol
    r_discount : discount rate (if None, uses the CIR variable itself)

    Returns
    -------
    L : TridiagonalOperator
    """
    X = jnp.maximum(x_grid, 1e-10)
    drift = kappa * (theta_cir - X)
    diffusion = 0.5 * sigma**2 * X

    D1 = d_zero(x_grid)
    D2 = d_plus_d_minus(x_grid)

    r = X if r_discount is None else r_discount
    lower = diffusion[1:] * D2.lower + drift[1:] * D1.lower
    diag = diffusion * D2.diag + drift * D1.diag - r
    upper = diffusion[:-1] * D2.upper + drift[:-1] * D1.upper

    return TridiagonalOperator(lower, diag, upper)


def fdm_g2pp_operator(x_grid, y_grid, a, b, sigma_x, sigma_y, rho):
    """G2++ two-factor short-rate FD operator.

    dx = -a*x*dt + sigma_x*dW1
    dy = -b*y*dt + sigma_y*dW2
    dW1*dW2 = rho*dt
    r(t) = phi(t) + x(t) + y(t)

    Returns 1D operators for each factor plus the 2D composite.

    Parameters
    ----------
    x_grid, y_grid : 1D grids for each factor
    a, b : mean-reversion speeds
    sigma_x, sigma_y : volatilities
    rho : correlation

    Returns
    -------
    Fdm2dBlackScholesOp with x-direction and y-direction operators
    """
    # x-direction: OU process
    drift_x = -a * x_grid
    diff_x = 0.5 * sigma_x**2
    D1x = d_zero(x_grid)
    D2x = d_plus_d_minus(x_grid)
    lower_x = diff_x * D2x.lower + drift_x[1:] * D1x.lower
    diag_x = diff_x * D2x.diag + drift_x * D1x.diag
    upper_x = diff_x * D2x.upper + drift_x[:-1] * D1x.upper
    Lx = TridiagonalOperator(lower_x, diag_x, upper_x)

    # y-direction: OU process
    drift_y = -b * y_grid
    diff_y = 0.5 * sigma_y**2
    D1y = d_zero(y_grid)
    D2y = d_plus_d_minus(y_grid)
    lower_y = diff_y * D2y.lower + drift_y[1:] * D1y.lower
    diag_y = diff_y * D2y.diag + drift_y * D1y.diag
    upper_y = diff_y * D2y.upper + drift_y[:-1] * D1y.upper
    Ly = TridiagonalOperator(lower_y, diag_y, upper_y)

    return Fdm2dBlackScholesOp(
        Lx=Lx, Ly=Ly, rho=rho,
        sigma_x=sigma_x, sigma_y=sigma_y,
        x_grid=x_grid, y_grid=y_grid,
    )


def fdm_bs_forward_operator(x_grid, r, q, sigma):
    """Black-Scholes forward (Fokker-Planck) operator.

    The forward PDE is the adjoint of the backward BS PDE,
    used for density/probability evolution:

    dp/dt = -d/dx[(r-q-0.5*sigma^2)*p] + 0.5*sigma^2 * d^2p/dx^2

    with x = log(S).

    Parameters
    ----------
    x_grid : log-spot grid
    r, q : rates
    sigma : volatility

    Returns
    -------
    L : TridiagonalOperator (forward operator)
    """
    nu = r - q - 0.5 * sigma**2
    diff = 0.5 * sigma**2

    D1 = d_zero(x_grid)
    D2 = d_plus_d_minus(x_grid)

    # Forward (adjoint): sign flip on drift term
    lower = diff * D2.lower - nu * D1.lower
    diag = diff * D2.diag - nu * D1.diag
    upper = diff * D2.upper - nu * D1.upper

    return TridiagonalOperator(lower, diag, upper)


def fdm_heston_forward_operator(x_grid, v_grid, r, q, kappa, theta, xi, rho):
    """Heston forward (Fokker-Planck) operator for density evolution.

    Forward PDE for the joint (log-S, v) density.

    Returns per-dimension operators for use in ADI schemes.

    Parameters
    ----------
    x_grid : log-spot grid
    v_grid : variance grid
    r, q : rates
    kappa, theta, xi, rho : Heston parameters

    Returns
    -------
    (Lx, Lv) : tuple of 1D forward operators
    """
    # x-direction forward operator
    nu = r - q - 0.5  # * v applied pointwise
    D1x = d_zero(x_grid)
    D2x = d_plus_d_minus(x_grid)
    # Average variance for the 1D slice
    v_mid = jnp.mean(v_grid)
    diff_x = 0.5 * v_mid
    drift_x = -(r - q - 0.5 * v_mid)  # negated for forward

    lower_x = diff_x * D2x.lower + drift_x * D1x.lower
    diag_x = diff_x * D2x.diag + drift_x * D1x.diag
    upper_x = diff_x * D2x.upper + drift_x * D1x.upper
    Lx = TridiagonalOperator(lower_x, diag_x, upper_x)

    # v-direction forward operator
    D1v = d_zero(v_grid)
    D2v = d_plus_d_minus(v_grid)
    v_safe = jnp.maximum(v_grid, 1e-10)
    diff_v = 0.5 * xi**2 * v_safe
    drift_v = -(kappa * (theta - v_grid))  # negated for forward

    lower_v = diff_v[1:] * D2v.lower + drift_v[1:] * D1v.lower
    diag_v = diff_v * D2v.diag + drift_v * D1v.diag
    upper_v = diff_v[:-1] * D2v.upper + drift_v[:-1] * D1v.upper
    Lv = TridiagonalOperator(lower_v, diag_v, upper_v)

    return Lx, Lv
