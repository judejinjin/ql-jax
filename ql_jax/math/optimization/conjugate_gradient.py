"""Conjugate Gradient optimizer (nonlinear, Fletcher-Reeves / Polak-Ribière)."""

import jax
import jax.numpy as jnp
from ql_jax.math.optimization.line_search import armijo_backtracking


def minimize(f, x0, max_iterations=1000, tol=1e-8,
             method='polak_ribiere', alpha_init=1.0):
    """Minimize f using nonlinear conjugate gradient.

    Parameters
    ----------
    f : callable(x) -> float (JAX-differentiable)
    x0 : array – initial guess
    method : str – 'fletcher_reeves' or 'polak_ribiere'
    """
    x = jnp.asarray(x0, dtype=jnp.float64)
    grad_f = jax.grad(f)

    g = grad_f(x)
    d = -g  # initial search direction
    g_norm_sq = jnp.dot(g, g)

    for iteration in range(max_iterations):
        if jnp.sqrt(g_norm_sq) < tol:
            return {
                'x': x, 'fun': f(x),
                'n_iterations': iteration, 'converged': True,
            }

        alpha = armijo_backtracking(f, x, d, gradient=g, alpha_init=alpha_init)
        x_new = x + alpha * d
        g_new = grad_f(x_new)
        g_new_norm_sq = jnp.dot(g_new, g_new)

        if method == 'fletcher_reeves':
            beta = g_new_norm_sq / (g_norm_sq + 1e-30)
        else:  # polak_ribiere
            beta = jnp.maximum(jnp.dot(g_new, g_new - g) / (g_norm_sq + 1e-30), 0.0)

        d = -g_new + beta * d

        # Reset if direction is not descent
        if jnp.dot(d, g_new) > 0:
            d = -g_new

        x = x_new
        g = g_new
        g_norm_sq = g_new_norm_sq

    return {
        'x': x, 'fun': f(x),
        'n_iterations': max_iterations, 'converged': False,
    }
