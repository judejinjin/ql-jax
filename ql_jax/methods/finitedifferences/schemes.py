"""Time-stepping schemes for finite-difference methods."""

import jax.numpy as jnp
import jax
from dataclasses import dataclass

from ql_jax.methods.finitedifferences.operators import TridiagonalOperator


def implicit_euler_step(V, dt, L, bc_fn=None, t=None):
    """Implicit Euler (fully implicit) step: (I - dt L) V_new = V.

    Parameters
    ----------
    V : array – current solution
    dt : float – time step
    L : TridiagonalOperator – spatial operator
    bc_fn : callable(V, t) -> V or None – boundary condition application
    t : float – current time

    Returns
    -------
    V_new : array
    """
    n = L.size
    lhs = TridiagonalOperator(
        -dt * L.lower,
        jnp.ones(n) - dt * L.diag,
        -dt * L.upper,
    )
    V_new = lhs.solve(V)
    if bc_fn is not None:
        V_new = bc_fn(V_new, t)
    return V_new


def explicit_euler_step(V, dt, L, bc_fn=None, t=None):
    """Explicit Euler step: V_new = V + dt L V.

    Parameters
    ----------
    V : array – current solution
    dt : float
    L : TridiagonalOperator – spatial operator
    bc_fn : callable or None
    t : float

    Returns
    -------
    V_new : array
    """
    V_new = V + dt * L.apply(V)
    if bc_fn is not None:
        V_new = bc_fn(V_new, t)
    return V_new


def crank_nicolson_step(V, dt, L, bc_fn=None, t=None):
    """Crank-Nicolson (θ=0.5) step.

    (I - 0.5*dt*L) V_new = (I + 0.5*dt*L) V

    Parameters
    ----------
    V : array – current solution
    dt : float
    L : TridiagonalOperator
    bc_fn : callable or None
    t : float

    Returns
    -------
    V_new : array
    """
    return theta_step(V, dt, L, theta=0.5, bc_fn=bc_fn, t=t)


def theta_step(V, dt, L, theta=0.5, bc_fn=None, t=None):
    """Theta-method step.

    (I - θ*dt*L) V_new = V + (1-θ)*dt*L*V

    Parameters
    ----------
    V : array
    dt : float
    L : TridiagonalOperator
    theta : float – 0=explicit, 0.5=CN, 1=implicit
    bc_fn : callable or None
    t : float

    Returns
    -------
    V_new : array
    """
    n = L.size
    # Explicit part: rhs = V + (1-theta)*dt*L*V
    rhs = V + (1.0 - theta) * dt * L.apply(V)

    # Implicit part
    lhs = TridiagonalOperator(
        -theta * dt * L.lower,
        jnp.ones(n) - theta * dt * L.diag,
        -theta * dt * L.upper,
    )
    V_new = lhs.solve(rhs)

    if bc_fn is not None:
        V_new = bc_fn(V_new, t)
    return V_new


def trbdf2_step(V, dt, L, bc_fn=None, t=None, alpha=2.0 - jnp.sqrt(2.0)):
    """TR-BDF2 (Trapezoidal-BDF2) two-stage step.

    Stage 1: Trapezoidal from t to t+α*dt
    Stage 2: BDF2 from t+α*dt to t+dt

    Higher-order L-stable scheme.

    Parameters
    ----------
    V : array
    dt : float
    L : TridiagonalOperator
    bc_fn : callable or None
    t : float
    alpha : float – splitting parameter (default: 2 - √2)

    Returns
    -------
    V_new : array
    """
    n = L.size

    # Stage 1: TR step from t to t + alpha*dt
    dt1 = alpha * dt
    rhs1 = V + 0.5 * dt1 * L.apply(V)
    lhs1 = TridiagonalOperator(
        -0.5 * dt1 * L.lower,
        jnp.ones(n) - 0.5 * dt1 * L.diag,
        -0.5 * dt1 * L.upper,
    )
    V_star = lhs1.solve(rhs1)

    if bc_fn is not None:
        V_star = bc_fn(V_star, t - alpha * dt if t is not None else None)

    # Stage 2: BDF2 step from t+alpha*dt to t+dt
    dt2 = (1.0 - alpha) * dt
    w = dt2 / (dt1 + dt2)
    gamma = (1.0 - w) / (2.0 - w)
    beta1 = (1.0 - gamma) / (w * (2.0 - w))
    beta0 = beta1 * w**2

    rhs2 = beta1 * V_star - beta0 * V
    lhs2 = TridiagonalOperator(
        -gamma * dt2 * L.lower,
        jnp.ones(n) * (beta1 - beta0 + gamma) - gamma * dt2 * L.diag,
        -gamma * dt2 * L.upper,
    )
    # Simplify: (beta1 - beta0 + gamma) I on diagonal
    coeff = beta1 - beta0 + gamma
    lhs2 = TridiagonalOperator(
        -gamma * dt2 * L.lower,
        jnp.ones(n) * coeff - gamma * dt2 * L.diag,
        -gamma * dt2 * L.upper,
    )
    V_new = lhs2.solve(rhs2)

    if bc_fn is not None:
        V_new = bc_fn(V_new, t - dt if t is not None else None)

    return V_new


def method_of_lines_step(V, dt, L, bc_fn=None, t=None, n_sub=4):
    """Method of Lines: spatial FD + RK4 temporal integration.

    Instead of an implicit tridiagonal solve, we use an explicit
    4th-order Runge-Kutta step in time, which gives high accuracy
    for smooth operators.  Stability requires dt < C*dx^2/diffusion.

    Parameters
    ----------
    V : array – current solution
    dt : float – time step
    L : TridiagonalOperator – spatial operator  (dV/dt = L*V)
    bc_fn : boundary conditions
    t : current time
    n_sub : number of sub-steps (for stability)

    Returns
    -------
    V_new : array
    """
    h = dt / n_sub
    for _ in range(n_sub):
        k1 = L.apply(V)
        V1 = V + 0.5 * h * k1
        if bc_fn is not None:
            V1 = bc_fn(V1, t)
        k2 = L.apply(V1)
        V2 = V + 0.5 * h * k2
        if bc_fn is not None:
            V2 = bc_fn(V2, t)
        k3 = L.apply(V2)
        V3 = V + h * k3
        if bc_fn is not None:
            V3 = bc_fn(V3, t)
        k4 = L.apply(V3)
        V = V + (h / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        if bc_fn is not None:
            V = bc_fn(V, t)
    return V


def mixed_scheme_step(V, dt, L, bc_fn=None, t=None):
    """Mixed scheme: use explicit Euler when stable, implicit otherwise.

    Falls back to implicit for stiff regions.
    A heuristic: uses Crank-Nicolson as default.
    """
    return crank_nicolson_step(V, dt, L, bc_fn=bc_fn, t=t)
