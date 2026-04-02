"""FD barrier option engine – using the generic FD framework."""

import jax.numpy as jnp

from ql_jax._util.types import OptionType
from ql_jax.methods.finitedifferences.meshers import Concentrating1dMesher
from ql_jax.methods.finitedifferences.bs_operator import BSMOperator
from ql_jax.methods.finitedifferences.solvers import fdm_solve_1d, FdmSolverConfig
from ql_jax.methods.finitedifferences.step_conditions import BarrierStepCondition
from ql_jax.methods.finitedifferences.boundary import DirichletBC


def fd_barrier_price(S, K, T, r, q, sigma, barrier, option_type: int,
                      barrier_type='down_and_out', rebate=0.0,
                      n_space=200, n_time=200, scheme='cn'):
    """Price a barrier option using finite differences.

    Parameters
    ----------
    S, K, T, r, q, sigma : float
    barrier : float – barrier level
    option_type : int – OptionType.Call or Put
    barrier_type : str
    rebate : float
    n_space, n_time : int
    scheme : str

    Returns
    -------
    price : float
    """
    S, K, T, r, q, sigma, barrier = (
        jnp.asarray(x, dtype=jnp.float64)
        for x in (S, K, T, r, q, sigma, barrier)
    )

    # Grid in log-spot
    x_min = jnp.log(jnp.minimum(barrier * 0.5, K * 0.25))
    x_max = jnp.log(jnp.maximum(barrier * 2.0, K * 4.0))

    mesher = Concentrating1dMesher(
        low=float(x_min), high=float(x_max), size=n_space + 1,
        center=float(jnp.log(S)), density=5.0,
    )
    x_grid = mesher.locations()

    # Build operator
    op = BSMOperator(r=float(r), q=float(q), sigma=float(sigma), x_grid=x_grid)
    L = op.build()

    # Terminal condition
    s_grid = jnp.exp(x_grid)
    phi = jnp.where(option_type == OptionType.Call, 1.0, -1.0)
    terminal = jnp.maximum(phi * (s_grid - K), 0.0)

    # Apply barrier at terminal
    is_up = 'up' in barrier_type
    if is_up:
        terminal = jnp.where(s_grid >= barrier, rebate, terminal)
    else:
        terminal = jnp.where(s_grid <= barrier, rebate, terminal)

    # Boundary conditions
    if 'down' in barrier_type:
        bc = DirichletBC(
            lower_value_fn=lambda t: rebate * jnp.exp(-r * t),
            upper_value_fn=lambda t: jnp.where(
                option_type == OptionType.Call,
                s_grid[-1] - K * jnp.exp(-r * t),
                0.0
            ),
        )
    else:
        bc = DirichletBC(
            lower_value_fn=lambda t: jnp.where(
                option_type == OptionType.Put,
                K * jnp.exp(-r * t) - s_grid[0],
                0.0
            ),
            upper_value_fn=lambda t: rebate * jnp.exp(-r * t),
        )

    # Step condition for barrier
    step_cond = BarrierStepCondition(
        grid=s_grid, barrier=float(barrier), rebate=rebate, is_up=is_up,
    )

    config = FdmSolverConfig(n_time=n_time, scheme=scheme)
    V = fdm_solve_1d(float(T), L, terminal, config, bc_fn=bc.apply,
                      step_condition=step_cond)

    return jnp.interp(jnp.log(S), x_grid, V)
