"""Steepest Descent optimizer."""

import jax
import jax.numpy as jnp
from ql_jax.math.optimization.line_search import armijo_backtracking


def minimize(f, x0, max_iterations=1000, tol=1e-8, alpha_init=1.0):
    """Minimize f using steepest descent with Armijo line search.

    Parameters
    ----------
    f : callable(x) -> float (must be JAX-differentiable)
    x0 : array – initial guess
    max_iterations : int
    tol : float – gradient norm tolerance

    Returns dict with: x, fun, n_iterations, converged.
    """
    x = jnp.asarray(x0, dtype=jnp.float64)
    grad_f = jax.grad(f)

    for iteration in range(max_iterations):
        g = grad_f(x)
        g_norm = jnp.linalg.norm(g)

        if g_norm < tol:
            return {
                'x': x, 'fun': f(x),
                'n_iterations': iteration, 'converged': True,
            }

        direction = -g
        alpha = armijo_backtracking(f, x, direction, gradient=g,
                                    alpha_init=alpha_init)
        x = x + alpha * direction

    return {
        'x': x, 'fun': f(x),
        'n_iterations': max_iterations, 'converged': False,
    }
