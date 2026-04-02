"""Flat extrapolation wrapper for 2D interpolations."""

import jax.numpy as jnp


def build(interp_2d_data, x_min, x_max, y_min, y_max, eval_fn):
    """Wrap a 2D interpolation with flat extrapolation outside the grid.

    Parameters
    ----------
    interp_2d_data : dict – data from interp2d.bilinear_build or bicubic_build
    x_min, x_max : float – x boundaries
    y_min, y_max : float – y boundaries
    eval_fn : callable(data, x, y) -> float – the underlying 2D evaluator
    """
    return {
        'inner_data': interp_2d_data,
        'x_min': x_min, 'x_max': x_max,
        'y_min': y_min, 'y_max': y_max,
        'eval_fn': eval_fn,
    }


def evaluate(data, x, y):
    """Evaluate with flat extrapolation: clamp coordinates to grid bounds."""
    x_clamped = jnp.clip(x, data['x_min'], data['x_max'])
    y_clamped = jnp.clip(y, data['y_min'], data['y_max'])
    return data['eval_fn'](data['inner_data'], x_clamped, y_clamped)
