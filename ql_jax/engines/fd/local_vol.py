"""FD local volatility engine – finite-difference with Dupire local vol."""

import jax.numpy as jnp

from ql_jax._util.types import OptionType
from ql_jax.methods.finitedifferences.meshers import Concentrating1dMesher
from ql_jax.methods.finitedifferences.bs_operator import LocalVolOperator
from ql_jax.methods.finitedifferences.solvers import FdmSolverConfig
from ql_jax.methods.finitedifferences.schemes import theta_step
from ql_jax.methods.finitedifferences.boundary import DirichletBC
from ql_jax.methods.finitedifferences.step_conditions import AmericanStepCondition


def fd_local_vol_price(S, K, T, r, q, local_vol_fn, option_type: int,
                        n_space=200, n_time=200, american=False,
                        s_max_mult=4.0, theta_fd=0.5):
    """Price a vanilla option under local volatility using FD.

    Parameters
    ----------
    S, K, T, r, q : float
    local_vol_fn : callable(x, t) -> sigma(x, t) where x is log-spot
    option_type : int
    n_space, n_time : int
    american : bool
    s_max_mult : float
    theta_fd : float

    Returns
    -------
    price : float
    """
    S, K, T = (jnp.asarray(x, dtype=jnp.float64) for x in (S, K, T))

    x_min = jnp.log(K / s_max_mult)
    x_max = jnp.log(K * s_max_mult)

    mesher = Concentrating1dMesher(
        low=float(x_min), high=float(x_max), size=n_space + 1,
        center=float(jnp.log(S)), density=5.0,
    )
    x_grid = mesher.locations()
    s_grid = jnp.exp(x_grid)

    phi = jnp.where(option_type == OptionType.Call, 1.0, -1.0)
    terminal = jnp.maximum(phi * (s_grid - K), 0.0)

    # Build time-dependent operator
    op_builder = LocalVolOperator(
        r=float(r), q=float(q), local_vol_fn=local_vol_fn, x_grid=x_grid,
    )

    # BCs
    is_call = option_type == OptionType.Call
    bc = DirichletBC(
        lower_value_fn=lambda t: jnp.where(is_call, 0.0, K * jnp.exp(-r * t) - s_grid[0]),
        upper_value_fn=lambda t: jnp.where(is_call, s_grid[-1] - K * jnp.exp(-r * t), 0.0),
    )

    # Step condition for American
    step_cond = None
    if american:
        step_cond = AmericanStepCondition(exercise_values=terminal)

    # Time stepping
    dt = float(T) / n_time
    V = terminal
    for step in range(n_time):
        t = T - step * dt
        L = op_builder.build(t)
        V = theta_step(V, dt, L, theta=theta_fd, bc_fn=bc.apply, t=t - dt)
        if step_cond is not None:
            V = step_cond.apply(V, t - dt)

    return jnp.interp(jnp.log(S), x_grid, V)
