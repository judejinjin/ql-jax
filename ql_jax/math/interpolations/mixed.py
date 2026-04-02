"""Mixed interpolation: different methods for different segments.

Combines two interpolation methods — e.g. linear in the short end and
cubic in the long end — with a switchover index.
"""

import jax.numpy as jnp
from ql_jax.math.interpolations import linear, cubic


def build(xs, ys, switch_index, method_left='linear', method_right='cubic'):
    """Build a mixed interpolant.

    Parameters
    ----------
    xs : array – nodes (sorted)
    ys : array – values
    switch_index : int – index at which to switch methods
    method_left : str – 'linear' or 'cubic' for x <= xs[switch_index]
    method_right : str – 'linear' or 'cubic' for x > xs[switch_index]
    """
    xs = jnp.asarray(xs, dtype=jnp.float64)
    ys = jnp.asarray(ys, dtype=jnp.float64)

    builders = {'linear': linear.build, 'cubic': cubic.build}

    left_data = builders[method_left](xs[:switch_index + 1], ys[:switch_index + 1])
    right_data = builders[method_right](xs[switch_index:], ys[switch_index:])

    return {
        'xs': xs,
        'ys': ys,
        'switch_index': switch_index,
        'switch_x': xs[switch_index],
        'left_data': left_data,
        'right_data': right_data,
        'method_left': method_left,
        'method_right': method_right,
    }


def evaluate(data, x):
    """Evaluate mixed interpolant at x."""
    evaluators = {
        'linear': linear.evaluate,
        'cubic': cubic.evaluate,
    }
    in_left = x <= data['switch_x']
    val_left = evaluators[data['method_left']](data['left_data'], x)
    val_right = evaluators[data['method_right']](data['right_data'], x)
    return jnp.where(in_left, val_left, val_right)
