"""Backward-flat interpolation: y(x) = y[i] where x[i] <= x < x[i+1]."""

import jax.numpy as jnp


def build(xs, ys):
    """Build backward-flat interpolation state."""
    xs = jnp.asarray(xs, dtype=jnp.float64)
    ys = jnp.asarray(ys, dtype=jnp.float64)
    return {"xs": xs, "ys": ys}


def evaluate(state, x):
    """Evaluate backward-flat interpolation at point x.

    Returns y[i] where x[i] is the largest x[i] <= x.
    """
    xs = state["xs"]
    ys = state["ys"]
    idx = jnp.searchsorted(xs, x, side="right") - 1
    idx = jnp.clip(idx, 0, len(xs) - 1)
    return ys[idx]


def evaluate_many(state, x_array):
    """Evaluate at multiple points."""
    xs = state["xs"]
    ys = state["ys"]
    x_array = jnp.asarray(x_array, dtype=jnp.float64)
    idx = jnp.searchsorted(xs, x_array, side="right") - 1
    idx = jnp.clip(idx, 0, len(xs) - 1)
    return ys[idx]
