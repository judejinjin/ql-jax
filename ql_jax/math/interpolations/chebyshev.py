"""Chebyshev interpolation."""

import jax.numpy as jnp


def chebyshev_nodes(n, a=-1.0, b=1.0):
    """Generate Chebyshev nodes of the first kind on [a, b].

    x_k = (a+b)/2 + (b-a)/2 * cos((2k+1)/(2n) * pi), k=0,...,n-1
    """
    k = jnp.arange(n)
    nodes = jnp.cos((2.0 * k + 1.0) / (2.0 * n) * jnp.pi)
    return (a + b) / 2.0 + (b - a) / 2.0 * nodes


def build(f, n, a=-1.0, b=1.0):
    """Build Chebyshev interpolation.

    Parameters
    ----------
    f : callable to interpolate
    n : number of Chebyshev nodes
    a, b : interval endpoints

    Returns
    -------
    dict with nodes, values, and coefficients
    """
    nodes = chebyshev_nodes(n, a, b)
    values = jnp.array([f(float(x)) for x in nodes])

    # Compute Chebyshev coefficients via DCT-like transform
    coeffs = jnp.zeros(n)
    for j in range(n):
        s = 0.0
        for k in range(n):
            s += values[k] * jnp.cos(j * (2.0 * k + 1.0) / (2.0 * n) * jnp.pi)
        coeffs = coeffs.at[j].set(2.0 / n * s)
    coeffs = coeffs.at[0].set(coeffs[0] / 2.0)

    return {"nodes": nodes, "values": values, "coeffs": coeffs, "a": a, "b": b, "n": n}


def evaluate(state, x):
    """Evaluate Chebyshev interpolation using Clenshaw's algorithm."""
    a, b = state["a"], state["b"]
    coeffs = state["coeffs"]
    n = state["n"]

    # Map x to [-1, 1]
    t = (2.0 * x - a - b) / (b - a)

    # Clenshaw recurrence
    b_prev = 0.0
    b_curr = 0.0
    for j in range(n - 1, 0, -1):
        b_next = coeffs[j] + 2.0 * t * b_curr - b_prev
        b_prev = b_curr
        b_curr = b_next

    return coeffs[0] + t * b_curr - b_prev
