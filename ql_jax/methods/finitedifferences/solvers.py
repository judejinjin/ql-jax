"""Generic FD solver – orchestrates time-stepping for 1D and 2D PDEs."""

import jax.numpy as jnp
import jax
from dataclasses import dataclass

from ql_jax.methods.finitedifferences.schemes import (
    theta_step, crank_nicolson_step, implicit_euler_step,
    explicit_euler_step, trbdf2_step,
)


@dataclass(frozen=True)
class FdmSolverConfig:
    """Configuration for the FDM solver.

    Parameters
    ----------
    n_time : int – number of time steps
    scheme : str – 'cn', 'implicit', 'explicit', 'trbdf2'
    theta : float – theta parameter (for theta scheme)
    """
    n_time: int = 200
    scheme: str = 'cn'
    theta: float = 0.5


def fdm_solve_1d(T, operator, terminal_values, config, bc_fn=None,
                  step_condition=None):
    """Solve a 1D PDE backwards in time.

    Parameters
    ----------
    T : float – maturity (time to solve backwards from T to 0)
    operator : TridiagonalOperator or callable(t) -> TridiagonalOperator
    terminal_values : array – V(T, x) for all grid points
    config : FdmSolverConfig
    bc_fn : callable(V, t) -> V – boundary condition (or None)
    step_condition : object with .apply(V, t) method (or None)

    Returns
    -------
    V : array – solution at t=0
    """
    dt = T / config.n_time
    V = terminal_values

    is_time_dependent = callable(operator) and not hasattr(operator, 'apply')

    scheme_fn = _get_scheme_fn(config)

    def step_fn(i, V):
        t = T - i * dt  # current time (stepping backwards)

        L = operator(t) if is_time_dependent else operator

        V_new = scheme_fn(V, dt, L, bc_fn=bc_fn, t=t - dt)

        if step_condition is not None:
            V_new = step_condition.apply(V_new, t - dt)

        return V_new

    V = jax.lax.fori_loop(0, config.n_time, step_fn, V)
    return V


def fdm_solve_1d_full(T, operator, terminal_values, config, bc_fn=None,
                        step_condition=None):
    """Solve 1D PDE and return full time-space solution.

    Parameters
    ----------
    Same as fdm_solve_1d.

    Returns
    -------
    V_all : array(n_time+1, n_space) – full solution grid
    """
    dt = T / config.n_time
    n = len(terminal_values)

    is_time_dependent = callable(operator) and not hasattr(operator, 'apply')
    scheme_fn = _get_scheme_fn(config)

    def step_fn(carry, i):
        V = carry
        t = T - i * dt

        L = operator(t) if is_time_dependent else operator
        V_new = scheme_fn(V, dt, L, bc_fn=bc_fn, t=t - dt)

        if step_condition is not None:
            V_new = step_condition.apply(V_new, t - dt)

        return V_new, V_new

    _, V_history = jax.lax.scan(
        step_fn, terminal_values, jnp.arange(config.n_time)
    )

    V_all = jnp.concatenate([terminal_values[None, :], V_history], axis=0)
    return V_all


def interpolate_solution(x_grid, V, x_target):
    """Interpolate solution at target point.

    Parameters
    ----------
    x_grid : array
    V : array – solution values on grid
    x_target : float

    Returns
    -------
    float – interpolated value
    """
    return jnp.interp(x_target, x_grid, V)


def _get_scheme_fn(config):
    """Select time-stepping scheme."""
    if config.scheme == 'cn':
        return crank_nicolson_step
    elif config.scheme == 'implicit':
        return implicit_euler_step
    elif config.scheme == 'explicit':
        return explicit_euler_step
    elif config.scheme == 'trbdf2':
        return trbdf2_step
    elif config.scheme == 'theta':
        from functools import partial
        return partial(theta_step, theta=config.theta)
    else:
        return crank_nicolson_step
