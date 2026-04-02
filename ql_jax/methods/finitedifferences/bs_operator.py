"""Black-Scholes FD operator – builds spatial operator for BS PDE."""

import jax.numpy as jnp
from dataclasses import dataclass

from ql_jax.methods.finitedifferences.operators import (
    TridiagonalOperator, d_plus_d_minus, d_zero,
)


@dataclass(frozen=True)
class BSMOperator:
    """Black-Scholes-Merton operator on a log-spot grid.

    PDE: dV/dt + (r-q-σ²/2) dV/dx + σ²/2 d²V/dx² - rV = 0
    where x = log(S).

    Spatial operator L such that dV/dt = -L V:
        L = σ²/2 D+D- + (r-q-σ²/2) D0 - r I

    Parameters
    ----------
    r : float – risk-free rate
    q : float – dividend yield
    sigma : float – volatility
    x_grid : array – log-spot grid points
    """
    r: float
    q: float
    sigma: float
    x_grid: jnp.ndarray

    def build(self):
        """Build the tridiagonal spatial operator.

        Returns
        -------
        TridiagonalOperator
        """
        alpha = 0.5 * self.sigma**2
        beta = self.r - self.q - alpha

        # Second derivative
        d2 = d_plus_d_minus(self.x_grid)
        # First derivative (central)
        d1 = d_zero(self.x_grid)

        # L = alpha * D+D- + beta * D0 - r * I
        op = d2.scale(alpha).add(d1.scale(beta))
        op = TridiagonalOperator(
            op.lower, op.diag - self.r, op.upper
        )
        return op


@dataclass(frozen=True)
class LocalVolOperator:
    """Local volatility FD operator.

    PDE: dV/dt + (r-q-σ(x,t)²/2) dV/dx + σ(x,t)²/2 d²V/dx² - rV = 0

    Parameters
    ----------
    r : float
    q : float
    local_vol_fn : callable(x, t) -> σ(x,t)
    x_grid : array – log-spot grid
    """
    r: float
    q: float
    local_vol_fn: object
    x_grid: jnp.ndarray

    def build(self, t):
        """Build the operator at time t.

        Parameters
        ----------
        t : float

        Returns
        -------
        TridiagonalOperator
        """
        x = self.x_grid
        n = len(x)
        sigma = self.local_vol_fn(x, t)
        alpha = 0.5 * sigma**2
        beta = self.r - self.q - alpha

        dx = jnp.diff(x)

        # Build tridiagonal coefficients directly for non-uniform grid
        lower = jnp.zeros(n - 1)
        diag = jnp.zeros(n)
        upper = jnp.zeros(n - 1)

        # Interior points
        dx_m = dx[:-1]  # x[i] - x[i-1]
        dx_p = dx[1:]   # x[i+1] - x[i]
        h = 0.5 * (dx_m + dx_p)

        alpha_i = alpha[1:-1]
        beta_i = beta[1:-1]

        lower = lower.at[1:].set(alpha_i / (dx_m * h) - beta_i / (dx_m + dx_p))
        diag = diag.at[1:-1].set(-alpha_i / (dx_p * h) - alpha_i / (dx_m * h) - self.r)
        upper = upper.at[:-1].set(alpha_i / (dx_p * h) + beta_i / (dx_m + dx_p))

        return TridiagonalOperator(lower, diag, upper)
