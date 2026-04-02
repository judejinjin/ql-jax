"""Forward-flat interpolation: y(x) = y[i+1] where x[i] < x <= x[i+1]."""

import jax.numpy as jnp


def build(xs, ys):
    """Build forward-flat interpolation state."""
    xs = jnp.asarray(xs, dtype=jnp.float64)
    ys = jnp.asarray(ys, dtype=jnp.float64)
    return {"xs": xs, "ys": ys}


def evaluate(state, x):
    """Evaluate forward-flat interpolation at point x."""
    xs = state["xs"]
    ys = state["ys"]
    idx = jnp.searchsorted(xs, x, side="left")
    idx = jnp.clip(idx, 0, len(xs) - 1)
    return ys[idx]


def evaluate_many(state, x_array):
    """Evaluate at multiple points."""
    xs = state["xs"]
    ys = state["ys"]
    x_array = jnp.asarray(x_array, dtype=jnp.float64)
    idx = jnp.searchsorted(xs, x_array, side="left")
    idx = jnp.clip(idx, 0, len(xs) - 1)
    return ys[idx]
