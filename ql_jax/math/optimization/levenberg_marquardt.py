"""Levenberg-Marquardt optimizer for nonlinear least squares.

Uses jax.jacfwd or jax.jacrev for automatic Jacobian computation.
"""

import jax
import jax.numpy as jnp

from ql_jax.math.optimization.end_criteria import EndCriteria


def minimize(
    residuals_fn,
    x0,
    end_criteria=None,
    lambda_init=1e-3,
    lambda_factor=10.0,
):
    """Minimize sum(residuals(x)**2) using Levenberg-Marquardt.

    Args:
        residuals_fn: Function mapping x (1D array) -> residuals (1D array).
        x0: Initial parameter vector.
        end_criteria: EndCriteria instance.
        lambda_init: Initial damping parameter.
        lambda_factor: Factor to adjust lambda.

    Returns:
        Tuple of (optimal_x, final_cost, iterations).
    """
    if end_criteria is None:
        end_criteria = EndCriteria()

    x = jnp.asarray(x0, dtype=jnp.float64)
    lam = lambda_init

    r = residuals_fn(x)
    cost = float(jnp.sum(r ** 2))

    for iteration in range(end_criteria.max_iterations):
        if end_criteria.check_root(cost):
            return x, cost, iteration

        # Jacobian: J[i,j] = d(residual_i) / d(x_j)
        J = jax.jacfwd(residuals_fn)(x)

        JtJ = J.T @ J
        Jtr = J.T @ r

        # Solve (JtJ + lambda * diag(JtJ)) * delta = -Jtr
        diag = jnp.diag(jnp.diag(JtJ))
        A = JtJ + lam * diag
        delta = -jnp.linalg.solve(A, Jtr)

        x_new = x + delta
        r_new = residuals_fn(x_new)
        cost_new = float(jnp.sum(r_new ** 2))

        if cost_new < cost:
            x = x_new
            r = r_new
            if end_criteria.check_stationary_point(cost, cost_new):
                return x, cost_new, iteration
            cost = cost_new
            lam /= lambda_factor
        else:
            lam *= lambda_factor

    return x, cost, end_criteria.max_iterations
