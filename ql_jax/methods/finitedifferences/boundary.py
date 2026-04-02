"""Boundary conditions for finite-difference PDEs."""

import jax.numpy as jnp
from dataclasses import dataclass


@dataclass(frozen=True)
class DirichletBC:
    """Dirichlet boundary condition: V(boundary) = value(t).

    Parameters
    ----------
    lower_value_fn : callable(t) -> float – value at lower boundary
    upper_value_fn : callable(t) -> float – value at upper boundary
    """
    lower_value_fn: object
    upper_value_fn: object

    def apply(self, V, t):
        """Apply boundary conditions to solution vector."""
        V = V.at[0].set(self.lower_value_fn(t))
        V = V.at[-1].set(self.upper_value_fn(t))
        return V


@dataclass(frozen=True)
class NeumannBC:
    """Neumann boundary condition: dV/dx(boundary) = value.

    Parameters
    ----------
    lower_deriv : float – derivative at lower boundary
    upper_deriv : float – derivative at upper boundary
    dx_lower : float – grid spacing at lower boundary
    dx_upper : float – grid spacing at upper boundary
    """
    lower_deriv: float = 0.0
    upper_deriv: float = 0.0
    dx_lower: float = 1.0
    dx_upper: float = 1.0

    def apply(self, V, t=None):
        """Apply Neumann BCs by linear extrapolation."""
        V = V.at[0].set(V[1] - self.lower_deriv * self.dx_lower)
        V = V.at[-1].set(V[-2] + self.upper_deriv * self.dx_upper)
        return V


@dataclass(frozen=True)
class LinearBC:
    """Linear boundary condition (for far-field approximation).

    At boundaries, V extrapolates linearly from interior.
    """
    def apply(self, V, t=None):
        V = V.at[0].set(2.0 * V[1] - V[2])
        V = V.at[-1].set(2.0 * V[-2] - V[-3])
        return V


def european_call_bc(s_grid, K, r):
    """Create Dirichlet BCs for European call on BS PDE.

    Lower BC: V(0,t) = 0
    Upper BC: V(S_max,t) = S_max - K*exp(-r*tau)
    """
    S_max = s_grid[-1]
    return DirichletBC(
        lower_value_fn=lambda t: 0.0,
        upper_value_fn=lambda t: S_max - K * jnp.exp(-r * t),
    )


def european_put_bc(s_grid, K, r):
    """Create Dirichlet BCs for European put on BS PDE.

    Lower BC: V(0,t) = K*exp(-r*tau)
    Upper BC: V(S_max,t) = 0
    """
    return DirichletBC(
        lower_value_fn=lambda t: K * jnp.exp(-r * t),
        upper_value_fn=lambda t: 0.0,
    )


def barrier_bc(s_grid, barrier_level, rebate=0.0):
    """Create BCs for barrier option (knocked out at barrier).

    Parameters
    ----------
    s_grid : array – spot grid
    barrier_level : float – barrier level
    rebate : float – rebate paid on knock-out
    """
    is_lower = barrier_level <= s_grid[0]
    if is_lower:
        return DirichletBC(
            lower_value_fn=lambda t: rebate,
            upper_value_fn=lambda t: 0.0,  # placeholder, set by payoff
        )
    else:
        return DirichletBC(
            lower_value_fn=lambda t: 0.0,  # placeholder
            upper_value_fn=lambda t: rebate,
        )
