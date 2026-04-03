"""Adaptive Runge-Kutta ODE solver (Dormand-Prince RK45).

Solves dy/dt = f(t, y) with adaptive step-size control.
"""

from __future__ import annotations

import jax.numpy as jnp


def adaptive_runge_kutta(
    f,
    y0,
    t0: float,
    t1: float,
    dt0: float = 0.01,
    tol: float = 1e-8,
    max_steps: int = 10000,
):
    """Solve an ODE system using the Dormand-Prince (RK45) method.

    Parameters
    ----------
    f : callable(t, y) -> array
        Right-hand side of ODE: dy/dt = f(t, y).
    y0 : array
        Initial condition at t0.
    t0, t1 : float
        Integration interval.
    dt0 : float
        Initial step size.
    tol : float
        Error tolerance for adaptive stepping.
    max_steps : int
        Maximum number of steps.

    Returns
    -------
    y : array
        Solution at t1.
    """
    y0 = jnp.asarray(y0, dtype=jnp.float64)
    scalar_input = y0.ndim == 0
    if scalar_input:
        y0 = y0.reshape(1)
        f_orig = f
        f = lambda t, y: jnp.atleast_1d(f_orig(t, y[0]))

    # Dormand-Prince coefficients
    a2, a3, a4, a5, a6 = 1/5, 3/10, 4/5, 8/9, 1.0
    b21 = 1/5
    b31, b32 = 3/40, 9/40
    b41, b42, b43 = 44/45, -56/15, 32/9
    b51, b52, b53, b54 = 19372/6561, -25360/2187, 64448/6561, -212/729
    b61, b62, b63, b64, b65 = 9017/3168, -355/33, 46732/5247, 49/176, -5103/18656

    # 5th order weights
    c1, c3, c4, c5, c6 = 35/384, 500/1113, 125/192, -2187/6784, 11/84

    # 4th order weights (for error estimate)
    d1, d3, d4, d5, d6 = 5179/57600, 7571/16695, 393/640, -92097/339200, 187/2100
    d7 = 1/40

    t = float(t0)
    y = y0
    dt = float(dt0)
    sign = 1.0 if t1 > t0 else -1.0

    for _ in range(max_steps):
        if sign * (t - t1) >= 0:
            break
        # Clamp dt so we don't overshoot t1
        if sign * (t + sign * dt - t1) > 0:
            dt = sign * (t1 - t)

        h = sign * abs(dt)

        k1 = f(t, y)
        k2 = f(t + a2 * h, y + h * b21 * k1)
        k3 = f(t + a3 * h, y + h * (b31 * k1 + b32 * k2))
        k4 = f(t + a4 * h, y + h * (b41 * k1 + b42 * k2 + b43 * k3))
        k5 = f(t + a5 * h, y + h * (b51 * k1 + b52 * k2 + b53 * k3 + b54 * k4))
        k6 = f(t + a6 * h, y + h * (b61 * k1 + b62 * k2 + b63 * k3 + b64 * k4 + b65 * k5))

        # 5th order solution
        y5 = y + h * (c1 * k1 + c3 * k3 + c4 * k4 + c5 * k5 + c6 * k6)
        # 4th order solution
        k7 = f(t + h, y5)
        y4 = y + h * (d1 * k1 + d3 * k3 + d4 * k4 + d5 * k5 + d6 * k6 + d7 * k7)

        # Error estimate
        err = jnp.max(jnp.abs(y5 - y4))
        err = float(err)

        if err <= tol or abs(dt) < 1e-15:
            # Accept step
            t = t + h
            y = y5
            # Increase step size
            if err > 0:
                dt = abs(dt) * min(5.0, max(0.2, 0.9 * (tol / err) ** 0.2))
            else:
                dt = abs(dt) * 5.0
        else:
            # Reject step, reduce step size
            dt = abs(dt) * max(0.1, 0.9 * (tol / err) ** 0.25)

    if scalar_input:
        return float(y[0])
    return y
