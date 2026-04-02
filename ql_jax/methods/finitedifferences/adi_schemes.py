"""ADI (Alternating Direction Implicit) schemes for multi-dimensional FD.

Implements Craig-Sneyd, Douglas, Hundsdorfer, and Modified Craig-Sneyd.
These are used for 2D+ PDE problems (e.g., Heston, multi-asset).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from ql_jax.methods.finitedifferences.operators import TridiagonalOperator


def douglas_step(V, dt, ops, theta=0.5, bc_fn=None, t=None):
    """Douglas ADI scheme for 2D problems.

    Given operators L = L1 + L2 (split by dimension):
    Y0 = V + dt * L * V
    (I - theta*dt*L1) Y1 = Y0 - theta*dt*L1*V
    (I - theta*dt*L2) Y2 = Y1 - theta*dt*L2*V

    Parameters
    ----------
    V : 2D array (n1 x n2) flattened or structured
    dt : time step
    ops : list of [L1, L2] TridiagonalOperator per dimension
    theta : implicitness parameter
    bc_fn : boundary condition function
    t : current time
    """
    L1, L2 = ops

    # Explicit predictor
    Y0 = V + dt * (L1.apply(V) + L2.apply(V))

    # Implicit correction in dimension 1
    n = L1.size
    lhs1 = TridiagonalOperator(
        -theta * dt * L1.lower,
        jnp.ones(n) - theta * dt * L1.diag,
        -theta * dt * L1.upper,
    )
    Y1 = lhs1.solve(Y0 - theta * dt * L1.apply(V))

    # Implicit correction in dimension 2
    n2 = L2.size
    lhs2 = TridiagonalOperator(
        -theta * dt * L2.lower,
        jnp.ones(n2) - theta * dt * L2.diag,
        -theta * dt * L2.upper,
    )
    Y2 = lhs2.solve(Y1 - theta * dt * L2.apply(V))

    if bc_fn is not None:
        Y2 = bc_fn(Y2, t)
    return Y2


def craig_sneyd_step(V, dt, ops, theta=0.5, bc_fn=None, t=None):
    """Craig-Sneyd ADI scheme.

    Second-order accurate, handles mixed derivatives.

    Parameters
    ----------
    V : solution vector
    dt : time step
    ops : list of [L1, L2, L_mix] operators
        L1, L2 are tridiagonal per dimension
        L_mix is the mixed derivative operator (can be None)
    theta : implicitness
    bc_fn : boundary conditions
    t : current time
    """
    if len(ops) == 2:
        L1, L2 = ops
        L_mix = None
    else:
        L1, L2, L_mix = ops

    # Full explicit RHS
    L_V = L1.apply(V) + L2.apply(V)
    if L_mix is not None:
        L_V = L_V + L_mix.apply(V)

    Y0 = V + dt * L_V

    # First sweep: dimension 1
    n1 = L1.size
    lhs1 = TridiagonalOperator(
        -theta * dt * L1.lower,
        jnp.ones(n1) - theta * dt * L1.diag,
        -theta * dt * L1.upper,
    )
    Y1 = lhs1.solve(Y0 - theta * dt * L1.apply(V))

    # Second sweep: dimension 2
    n2 = L2.size
    lhs2 = TridiagonalOperator(
        -theta * dt * L2.lower,
        jnp.ones(n2) - theta * dt * L2.diag,
        -theta * dt * L2.upper,
    )
    Y_hat = lhs2.solve(Y1 - theta * dt * L2.apply(V))

    if bc_fn is not None:
        Y_hat = bc_fn(Y_hat, t)
    return Y_hat


def modified_craig_sneyd_step(V, dt, ops, theta=0.5, mu=0.5, bc_fn=None, t=None):
    """Modified Craig-Sneyd ADI scheme.

    Adds correction step for improved stability.

    Parameters
    ----------
    V : solution
    dt : time step
    ops : [L1, L2] or [L1, L2, L_mix]
    theta, mu : scheme parameters
    bc_fn : boundary conditions
    t : time
    """
    if len(ops) == 2:
        L1, L2 = ops
        L_mix = None
    else:
        L1, L2, L_mix = ops

    L_V = L1.apply(V) + L2.apply(V)
    if L_mix is not None:
        L_V = L_V + L_mix.apply(V)

    # Step 1: predictor
    Y0 = V + dt * L_V

    # Step 2: first sweep
    n1 = L1.size
    lhs1 = TridiagonalOperator(
        -theta * dt * L1.lower,
        jnp.ones(n1) - theta * dt * L1.diag,
        -theta * dt * L1.upper,
    )
    Y1 = lhs1.solve(Y0 - theta * dt * L1.apply(V))

    # Step 3: second sweep
    n2 = L2.size
    lhs2 = TridiagonalOperator(
        -theta * dt * L2.lower,
        jnp.ones(n2) - theta * dt * L2.diag,
        -theta * dt * L2.upper,
    )
    Y_tilde = lhs2.solve(Y1 - theta * dt * L2.apply(V))

    # Step 4: correction
    L_Y_tilde = L1.apply(Y_tilde) + L2.apply(Y_tilde)
    if L_mix is not None:
        L_Y_tilde = L_Y_tilde + L_mix.apply(Y_tilde)
    Y0_corr = Y0 + mu * dt * (L_Y_tilde - L_V)

    # Step 5-6: repeat sweeps on corrected
    Y1_corr = lhs1.solve(Y0_corr - theta * dt * L1.apply(V))
    Y_new = lhs2.solve(Y1_corr - theta * dt * L2.apply(V))

    if bc_fn is not None:
        Y_new = bc_fn(Y_new, t)
    return Y_new


def hundsdorfer_verwer_step(V, dt, ops, theta=0.5, bc_fn=None, t=None):
    """Hundsdorfer-Verwer ADI scheme.

    Second-order stabilizing correction with better mixed-derivative handling.
    """
    if len(ops) == 2:
        L1, L2 = ops
        L_mix = None
    else:
        L1, L2, L_mix = ops

    L_V = L1.apply(V) + L2.apply(V)
    if L_mix is not None:
        L_V = L_V + L_mix.apply(V)

    # Predictor
    Y0 = V + dt * L_V

    # Sweep 1
    n1 = L1.size
    lhs1 = TridiagonalOperator(
        -theta * dt * L1.lower,
        jnp.ones(n1) - theta * dt * L1.diag,
        -theta * dt * L1.upper,
    )
    Y1 = lhs1.solve(Y0 - theta * dt * L1.apply(V))

    # Sweep 2
    n2 = L2.size
    lhs2 = TridiagonalOperator(
        -theta * dt * L2.lower,
        jnp.ones(n2) - theta * dt * L2.diag,
        -theta * dt * L2.upper,
    )
    Y_tilde = lhs2.solve(Y1 - theta * dt * L2.apply(V))

    # Corrector
    L_Y_tilde = L1.apply(Y_tilde) + L2.apply(Y_tilde)
    if L_mix is not None:
        L_Y_tilde = L_Y_tilde + L_mix.apply(Y_tilde)

    Y0_hat = Y_tilde + 0.5 * dt * (L_Y_tilde - L_V)

    # Final sweeps
    Y1_hat = lhs1.solve(Y0_hat - theta * dt * L1.apply(Y_tilde))
    Y_new = lhs2.solve(Y1_hat - theta * dt * L2.apply(Y_tilde))

    if bc_fn is not None:
        Y_new = bc_fn(Y_new, t)
    return Y_new
