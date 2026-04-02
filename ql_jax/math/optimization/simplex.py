"""Simplex (Nelder-Mead) optimizer."""

import jax.numpy as jnp

from ql_jax.math.optimization.end_criteria import EndCriteria


def minimize(f, x0, end_criteria=None, simplex_size=0.1):
    """Minimize f(x) using the Nelder-Mead simplex method.

    Args:
        f: Objective function mapping x (1D array) -> scalar.
        x0: Initial point.
        end_criteria: EndCriteria instance.
        simplex_size: Initial simplex edge length.

    Returns:
        Tuple of (optimal_x, final_value, iterations).
    """
    if end_criteria is None:
        end_criteria = EndCriteria()

    x0 = jnp.asarray(x0, dtype=jnp.float64)
    n = len(x0)

    # Build initial simplex
    simplex = [x0]
    for i in range(n):
        xi = x0.at[i].add(simplex_size)
        simplex.append(xi)

    f_values = [float(f(x)) for x in simplex]

    alpha = 1.0   # reflection
    gamma = 2.0   # expansion
    rho = 0.5     # contraction
    sigma = 0.5   # shrink

    for iteration in range(end_criteria.max_iterations):
        # Sort
        order = sorted(range(n + 1), key=lambda i: f_values[i])
        simplex = [simplex[i] for i in order]
        f_values = [f_values[i] for i in order]

        # Convergence check
        f_range = f_values[-1] - f_values[0]
        if f_range < end_criteria.function_epsilon:
            return simplex[0], f_values[0], iteration

        # Centroid (excluding worst)
        centroid = sum(simplex[:-1]) / n

        # Reflection
        xr = centroid + alpha * (centroid - simplex[-1])
        fr = float(f(xr))

        if f_values[0] <= fr < f_values[-2]:
            simplex[-1] = xr
            f_values[-1] = fr
        elif fr < f_values[0]:
            # Expansion
            xe = centroid + gamma * (xr - centroid)
            fe = float(f(xe))
            if fe < fr:
                simplex[-1] = xe
                f_values[-1] = fe
            else:
                simplex[-1] = xr
                f_values[-1] = fr
        else:
            # Contraction
            xc = centroid + rho * (simplex[-1] - centroid)
            fc = float(f(xc))
            if fc < f_values[-1]:
                simplex[-1] = xc
                f_values[-1] = fc
            else:
                # Shrink
                for i in range(1, n + 1):
                    simplex[i] = simplex[0] + sigma * (simplex[i] - simplex[0])
                    f_values[i] = float(f(simplex[i]))

    return simplex[0], f_values[0], end_criteria.max_iterations
