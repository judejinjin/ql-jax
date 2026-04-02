"""Linear interpolation: y = y[i] + (x - x[i]) * s[i] where s[i] = (y[i+1]-y[i])/(x[i+1]-x[i]).

Fully differentiable via JAX.
"""

import jax.numpy as jnp


def build(xs, ys):
    """Build linear interpolation state from sorted x, y arrays.

    Returns a dict with xs, ys, and precomputed slopes.
    """
    xs = jnp.asarray(xs, dtype=jnp.float64)
    ys = jnp.asarray(ys, dtype=jnp.float64)
    slopes = (ys[1:] - ys[:-1]) / (xs[1:] - xs[:-1])
    return {"xs": xs, "ys": ys, "slopes": slopes}


def evaluate(state, x):
    """Evaluate the linear interpolation at point x."""
    xs = state["xs"]
    ys = state["ys"]
    slopes = state["slopes"]

    # Find interval index via searchsorted
    idx = jnp.searchsorted(xs, x, side="right") - 1
    idx = jnp.clip(idx, 0, len(xs) - 2)

    return ys[idx] + slopes[idx] * (x - xs[idx])


def evaluate_many(state, x_array):
    """Evaluate linear interpolation at multiple points (vectorized)."""
    xs = state["xs"]
    ys = state["ys"]
    slopes = state["slopes"]

    x_array = jnp.asarray(x_array, dtype=jnp.float64)
    idx = jnp.searchsorted(xs, x_array, side="right") - 1
    idx = jnp.clip(idx, 0, len(xs) - 2)

    return ys[idx] + slopes[idx] * (x_array - xs[idx])


def derivative(state, x):
    """Derivative of linear interpolation at point x (piecewise constant)."""
    xs = state["xs"]
    slopes = state["slopes"]

    idx = jnp.searchsorted(xs, x, side="right") - 1
    idx = jnp.clip(idx, 0, len(xs) - 2)

    return slopes[idx]


def second_derivative(state, x):
    """Second derivative of linear interpolation (always zero)."""
    return jnp.float64(0.0)
