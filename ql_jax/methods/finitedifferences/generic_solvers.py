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


def fdm_3d_solve(
    grid_x: jnp.ndarray,
    grid_y: jnp.ndarray,
    grid_z: jnp.ndarray,
    payoff_fn,
    operator_fn,
    T: float,
    n_steps: int,
    bc_fn=None,
    theta: float = 0.5,
) -> jnp.ndarray:
    """Generic 3D finite-difference PDE solver.

    Uses operator-splitting (LOD) for 3-factor models like
    Heston-Hull-White (spot, variance, short-rate).

    Parameters
    ----------
    grid_x, grid_y, grid_z : 1D spatial grids
    payoff_fn : callable(X, Y, Z) -> 3D terminal values
    operator_fn : callable(t) -> (Lx, Ly, Lz) operators
    T : maturity
    n_steps : time steps
    bc_fn : boundary conditions
    theta : implicitness

    Returns
    -------
    V : 3D solution array at t=0
    """
    dt = T / n_steps
    X, Y, Z = jnp.meshgrid(grid_x, grid_y, grid_z, indexing='ij')
    V = payoff_fn(X, Y, Z)
    shape = V.shape
    nx, ny, nz = shape

    for step in range(n_steps):
        t = T - step * dt
        Lx, Ly, Lz = operator_fn(t)

        # x-sweep: solve along x for each (j, k) slice
        for j in range(ny):
            for k in range(nz):
                v_line = V[:, j, k]
                v_line = theta_step(v_line, dt / 3.0, Lx, theta=theta, t=t)
                V = V.at[:, j, k].set(v_line)

        # y-sweep
        for i in range(nx):
            for k in range(nz):
                v_line = V[i, :, k]
                v_line = theta_step(v_line, dt / 3.0, Ly, theta=theta, t=t)
                V = V.at[i, :, k].set(v_line)

        # z-sweep
        for i in range(nx):
            for j in range(ny):
                v_line = V[i, j, :]
                v_line = theta_step(v_line, dt / 3.0, Lz, theta=theta, t=t)
                V = V.at[i, j, :].set(v_line)

        if bc_fn is not None:
            V = bc_fn(V, t)

    return V


def fdm_nd_solve(
    grids: list[jnp.ndarray],
    payoff_fn,
    operator_fn,
    T: float,
    n_steps: int,
    bc_fn=None,
    theta: float = 0.5,
) -> jnp.ndarray:
    """Generic N-dimensional FD solver using LOD splitting.

    Parameters
    ----------
    grids : list of 1D arrays, one per dimension
    payoff_fn : callable(*meshgrids) -> N-D terminal values
    operator_fn : callable(t) -> list of N 1D operators
    T : maturity
    n_steps : time steps
    bc_fn, theta : as usual

    Returns
    -------
    V : N-D solution array
    """
    dt = T / n_steps
    meshes = jnp.meshgrid(*grids, indexing='ij')
    V = payoff_fn(*meshes)
    ndim = len(grids)
    shape = V.shape

    for step in range(n_steps):
        t = T - step * dt
        ops = operator_fn(t)

        for dim in range(ndim):
            # Build index slicing for this dimension
            n_along = shape[dim]
            # For each line along dim, apply 1D step
            other_dims = [range(shape[d]) for d in range(ndim) if d != dim]
            import itertools
            for idx_combo in itertools.product(*other_dims):
                # Build full index
                full_idx = list(idx_combo)
                full_idx.insert(dim, slice(None))
                full_idx = tuple(full_idx)
                v_line = V[full_idx]
                v_line = theta_step(v_line, dt / ndim, ops[dim],
                                    theta=theta, t=t)
                V = V.at[full_idx].set(v_line)

        if bc_fn is not None:
            V = bc_fn(V, t)

    return V
