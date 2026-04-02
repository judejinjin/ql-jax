"""Lagrange interpolation."""

import jax.numpy as jnp


def build(xs, ys):
    """Build Lagrange interpolation state.

    Parameters
    ----------
    xs : interpolation nodes
    ys : function values at nodes

    Returns
    -------
    dict with nodes, values, and barycentric weights
    """
    xs = jnp.asarray(xs, dtype=jnp.float64)
    ys = jnp.asarray(ys, dtype=jnp.float64)
    n = len(xs)

    # Compute barycentric weights
    weights = jnp.ones(n)
    for j in range(n):
        for k in range(n):
            if k != j:
                weights = weights.at[j].set(weights[j] / (xs[j] - xs[k]))

    return {"xs": xs, "ys": ys, "weights": weights}


def evaluate(state, x):
    """Evaluate using barycentric form of Lagrange interpolation."""
    xs = state["xs"]
    ys = state["ys"]
    w = state["weights"]

    diffs = x - xs
    # Handle exact node case
    exact = jnp.abs(diffs) < 1e-15
    has_exact = jnp.any(exact)

    # Barycentric formula
    terms = w / diffs
    result_bary = jnp.sum(terms * ys) / jnp.sum(terms)

    # If x is exactly a node, return that value
    exact_result = jnp.sum(jnp.where(exact, ys, 0.0))
    return jnp.where(has_exact, exact_result, result_bary)
