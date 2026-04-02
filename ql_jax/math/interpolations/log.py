"""Log-linear interpolation: interpolate in log space, exponentiate back.

y(x) = exp( linear_interp(x, xs, log(ys)) )
"""

import jax.numpy as jnp

from ql_jax.math.interpolations import linear as _linear


def build(xs, ys):
    """Build log-linear interpolation state."""
    ys = jnp.asarray(ys, dtype=jnp.float64)
    return _linear.build(xs, jnp.log(ys))


def evaluate(state, x):
    """Evaluate log-linear interpolation at point x."""
    return jnp.exp(_linear.evaluate(state, x))


def evaluate_many(state, x_array):
    """Evaluate log-linear interpolation at multiple points."""
    return jnp.exp(_linear.evaluate_many(state, x_array))
