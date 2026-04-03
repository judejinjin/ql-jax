"""Experimental FD operators and engines for energy/commodity models."""

import jax
import jax.numpy as jnp
from dataclasses import dataclass

from ql_jax.methods.finitedifferences.operators import (
    TridiagonalOperator, tridiag_solve, d_plus_d_minus, d_zero,
)


def build_extended_ou_operator(kappa, theta, sigma, r, grid):
    """FD operator for extended Ornstein-Uhlenbeck process.

    dX = kappa*(theta - X) dt + sigma dW

    L = sigma^2/2 * d²/dx² + kappa*(theta-x) * d/dx - r

    Parameters
    ----------
    kappa : mean-reversion speed
    theta : long-term mean
    sigma : diffusion volatility
    r : discount rate
    grid : (n,) spatial grid

    Returns
    -------
    TridiagonalOperator
    """
    n = len(grid)
    dx = jnp.diff(grid)

    # Build operator directly for interior points i=1..n-2
    dx_m = dx[:-1]   # dx[i-1] for interior point i
    dx_p = dx[1:]    # dx[i] for interior point i
    h = 0.5 * (dx_m + dx_p)

    diff_coeff = 0.5 * sigma**2
    drift = kappa * (theta - grid[1:-1])

    # Second derivative coefficients: diff_coeff/(dx_m*h), etc.
    # First derivative coefficients (central): -drift/(dx_m+dx_p), drift/(dx_m+dx_p)
    h2 = dx_m + dx_p
    l_coeff = diff_coeff / (dx_m * h) - drift / h2  # A[i, i-1]
    d_coeff = -diff_coeff / (dx_p * h) - diff_coeff / (dx_m * h) - r  # A[i, i]
    u_coeff = diff_coeff / (dx_p * h) + drift / h2  # A[i, i+1]

    diag = jnp.zeros(n)
    diag = diag.at[1:-1].set(d_coeff)

    lower_full = jnp.zeros(n - 1)
    lower_full = lower_full.at[:-1].set(l_coeff)  # lower[k] = A[k+1, k]
    upper_full = jnp.zeros(n - 1)
    upper_full = upper_full.at[1:].set(u_coeff)   # upper[k] = A[k, k+1]

    return TridiagonalOperator(lower=lower_full, diag=diag, upper=upper_full)


def build_dupire_operator(local_vol_func, r, q, grid, t):
    """FD operator for Dupire local vol model.

    L = sigma_loc(S,t)^2/2 * S^2 * d²/dS² + (r-q)*S * d/dS - r

    In log-spot x = ln(S):
    L = sigma_loc^2/2 * d²/dx² + (r-q-sigma_loc^2/2) * d/dx - r

    Parameters
    ----------
    local_vol_func : callable(S, t) -> local vol
    r, q : rates
    grid : (n,) log-spot grid
    t : current time

    Returns
    -------
    TridiagonalOperator
    """
    n = len(grid)
    dx = jnp.diff(grid)
    S = jnp.exp(grid[1:-1])

    # Local vol at each interior point
    sigma_loc = jnp.array([local_vol_func(float(s), t) for s in S])
    diff_coeff = 0.5 * sigma_loc**2
    drift = r - q - diff_coeff

    dx_m = dx[:-1]
    dx_p = dx[1:]
    h = 0.5 * (dx_m + dx_p)
    h2 = dx_m + dx_p

    l_coeff = diff_coeff / (dx_m * h) - drift / h2
    d_coeff = -diff_coeff / (dx_p * h) - diff_coeff / (dx_m * h) - r
    u_coeff = diff_coeff / (dx_p * h) + drift / h2

    diag = jnp.zeros(n)
    diag = diag.at[1:-1].set(d_coeff)

    lower_full = jnp.zeros(n - 1)
    lower_full = lower_full.at[:-1].set(l_coeff)
    upper_full = jnp.zeros(n - 1)
    upper_full = upper_full.at[1:].set(u_coeff)

    return TridiagonalOperator(lower=lower_full, diag=diag, upper=upper_full)


def fd_ou_vanilla_price(S0, K, T, r, kappa, theta, sigma, option_type=1,
                         n_x=100, n_t=100):
    """FD engine for vanilla option on OU process.

    Parameters
    ----------
    S0 : initial price (=X0 for OU)
    K : strike
    T : maturity
    r : risk-free rate
    kappa, theta, sigma : OU parameters
    option_type : 1=call, -1=put
    n_x : spatial grid points
    n_t : time steps

    Returns
    -------
    price : option price
    """
    # Grid around the mean
    std = sigma / jnp.sqrt(2.0 * kappa)
    x_low = min(S0, theta) - 4.0 * float(std)
    x_high = max(S0, theta) + 4.0 * float(std)
    grid = jnp.linspace(x_low, x_high, n_x)

    op = build_extended_ou_operator(kappa, theta, sigma, r, grid)
    dt = T / n_t

    # Terminal condition
    phi = jnp.where(option_type == 1, 1.0, -1.0)
    V = jnp.maximum(phi * (grid - K), 0.0)

    # Crank-Nicolson time stepping
    for step in range(n_t):
        # Explicit part
        LV = op.apply(V)
        rhs = V + 0.5 * dt * LV

        # Implicit part
        V = tridiag_solve(
            -0.5 * dt * op.lower,
            jnp.ones(n_x) - 0.5 * dt * op.diag,
            -0.5 * dt * op.upper,
            rhs,
        )

        # Boundary conditions
        tau = (step + 1) * dt
        disc = jnp.exp(-r * tau)
        V = V.at[0].set(jnp.maximum(phi * (grid[0] - K * disc), 0.0))
        V = V.at[-1].set(jnp.maximum(phi * (grid[-1] - K * disc), 0.0))

    return float(jnp.interp(S0, grid, V))


@dataclass
class Glued1dMesher:
    """Mesher combining two 1D meshers at a junction point.

    Parameters
    ----------
    grid_left : (n_left,) left grid (ascending, ends at junction)
    grid_right : (n_right,) right grid (ascending, starts at junction)
    """
    grid_left: jnp.ndarray
    grid_right: jnp.ndarray

    def locations(self):
        """Combined grid locations without duplicate at junction."""
        return jnp.concatenate([self.grid_left, self.grid_right[1:]])

    @property
    def size(self):
        return len(self.grid_left) + len(self.grid_right) - 1

    @staticmethod
    def create(low, junction, high, n_left, n_right):
        """Create a glued mesher from specifications."""
        left = jnp.linspace(low, junction, n_left)
        right = jnp.linspace(junction, high, n_right)
        return Glued1dMesher(grid_left=left, grid_right=right)
