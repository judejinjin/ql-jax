"""BFGS optimizer using JAX automatic differentiation."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def minimize(f, x0, max_iter=100, tol=1e-8):
    """Minimize a scalar function using BFGS with line search.

    Parameters
    ----------
    f : callable(x) -> scalar
    x0 : initial guess (1D array)
    max_iter : max iterations
    tol : gradient norm tolerance

    Returns
    -------
    dict with 'x', 'fun', 'n_iter'
    """
    grad_f = jax.grad(f)

    x = jnp.asarray(x0, dtype=jnp.float64)
    n = x.shape[0]
    H = jnp.eye(n, dtype=jnp.float64)  # inverse Hessian approximation

    g = grad_f(x)

    for i in range(max_iter):
        if jnp.linalg.norm(g) < tol:
            break

        # Search direction
        p = -H @ g

        # Backtracking line search (Armijo)
        alpha = 1.0
        fx = f(x)
        c1 = 1e-4
        for _ in range(30):
            if f(x + alpha * p) <= fx + c1 * alpha * jnp.dot(g, p):
                break
            alpha = alpha * 0.5

        # Update
        s = alpha * p
        x_new = x + s
        g_new = grad_f(x_new)
        y = g_new - g

        # BFGS update of inverse Hessian
        sy = jnp.dot(s, y)
        if sy > 1e-15:
            rho = 1.0 / sy
            I = jnp.eye(n, dtype=jnp.float64)
            V = I - rho * jnp.outer(s, y)
            H = V @ H @ V.T + rho * jnp.outer(s, s)

        x = x_new
        g = g_new

    return {'x': x, 'fun': f(x), 'n_iter': i + 1 if 'i' in dir() else 0}
