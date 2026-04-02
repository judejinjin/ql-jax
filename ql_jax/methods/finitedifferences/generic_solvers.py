"""Generic FD solvers for 1D, 2D, and N-dimensional PDEs.

These wrap the time-stepping schemes and spatial operators into
complete PDE solution workflows.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ql_jax.methods.finitedifferences.schemes import (
    theta_step, crank_nicolson_step, implicit_euler_step,
)


def fdm_1d_solve(
    grid: jnp.ndarray,
    payoff_fn,
    operator_fn,
    T: float,
    n_steps: int,
    bc_fn=None,
    step_conditions=None,
    theta: float = 0.5,
) -> jnp.ndarray:
    """Generic 1D finite-difference PDE solver.

    Backward-in-time from terminal payoff to t=0.

    Parameters
    ----------
    grid : 1D spatial grid
    payoff_fn : callable(grid) -> terminal values
    operator_fn : callable(t) -> TridiagonalOperator
    T : time to maturity
    n_steps : number of time steps
    bc_fn : boundary condition function(V, t)
    step_conditions : list of (time, callable(V)) for exercise conditions
    theta : scheme parameter (0.5 = Crank-Nicolson)

    Returns
    -------
    V : solution at t=0
    """
    dt = T / n_steps
    V = payoff_fn(grid)

    for i in range(n_steps):
        t = T - i * dt
        L = operator_fn(t)
        V = theta_step(V, dt, L, theta=theta, bc_fn=bc_fn, t=t)

        # Apply step conditions (e.g., American exercise)
        if step_conditions is not None:
            for (t_cond, cond_fn) in step_conditions:
                if abs(t - dt - t_cond) < dt * 0.5:
                    V = cond_fn(V)

    return V


def fdm_2d_solve(
    grid_x: jnp.ndarray,
    grid_y: jnp.ndarray,
    payoff_fn,
    operator_fn,
    T: float,
    n_steps: int,
    bc_fn=None,
    scheme: str = "douglas",
    theta: float = 0.5,
) -> jnp.ndarray:
    """Generic 2D finite-difference PDE solver.

    Uses ADI splitting for efficiency.

    Parameters
    ----------
    grid_x, grid_y : 1D spatial grids for each dimension
    payoff_fn : callable(X, Y) -> 2D terminal values
    operator_fn : callable(t) -> list of operators [Lx, Ly, (Lxy)]
    T : time to maturity
    n_steps : time steps
    bc_fn : boundary conditions
    scheme : 'douglas', 'craig_sneyd', 'hundsdorfer'
    theta : implicitness

    Returns
    -------
    V : 2D solution array at t=0
    """
    from ql_jax.methods.finitedifferences.adi_schemes import (
        douglas_step, craig_sneyd_step, hundsdorfer_verwer_step,
    )

    dt = T / n_steps
    X, Y = jnp.meshgrid(grid_x, grid_y, indexing='ij')
    V = payoff_fn(X, Y)
    V_flat = V.ravel()

    step_fn = {
        'douglas': douglas_step,
        'craig_sneyd': craig_sneyd_step,
        'hundsdorfer': hundsdorfer_verwer_step,
    }.get(scheme, douglas_step)

    for i in range(n_steps):
        t = T - i * dt
        ops = operator_fn(t)
        V_flat = step_fn(V_flat, dt, ops, theta=theta, bc_fn=bc_fn, t=t)

    return V_flat.reshape(V.shape)


def fdm_backward_solve(
    V: jnp.ndarray,
    operator_fn,
    T: float,
    n_steps: int,
    bc_fn=None,
    theta: float = 0.5,
) -> jnp.ndarray:
    """Generic backward PDE solve from given terminal condition.

    Parameters
    ----------
    V : terminal condition (any shape, flattened internally)
    operator_fn : callable(t) -> TridiagonalOperator
    T : time horizon
    n_steps : time steps
    bc_fn : boundary conditions
    theta : scheme parameter

    Returns
    -------
    V0 : solution at t=0
    """
    dt = T / n_steps

    for i in range(n_steps):
        t = T - i * dt
        L = operator_fn(t)
        V = theta_step(V, dt, L, theta=theta, bc_fn=bc_fn, t=t)

    return V
