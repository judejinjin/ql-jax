"""Kernel interpolation (1D and 2D) using radial basis functions."""

import jax.numpy as jnp
from jax.numpy.linalg import solve


# ── Kernel functions ──

def gaussian_kernel(r, epsilon=1.0):
    """Gaussian RBF: exp(-epsilon^2 * r^2)."""
    return jnp.exp(-(epsilon * r) ** 2)


def multiquadric_kernel(r, epsilon=1.0):
    """Multiquadric RBF: sqrt(1 + (epsilon*r)^2)."""
    return jnp.sqrt(1.0 + (epsilon * r) ** 2)


def inverse_multiquadric_kernel(r, epsilon=1.0):
    """Inverse multiquadric: 1/sqrt(1 + (epsilon*r)^2)."""
    return 1.0 / jnp.sqrt(1.0 + (epsilon * r) ** 2)


# ── 1D Kernel Interpolation ──

def build(xs, ys, kernel_fn=None, epsilon=1.0):
    """Build 1D kernel interpolant.

    Parameters
    ----------
    xs : array – nodes
    ys : array – values
    kernel_fn : callable(r, epsilon) -> float, default Gaussian
    epsilon : float – shape parameter

    Returns dict with: xs, weights, kernel_fn, epsilon.
    """
    if kernel_fn is None:
        kernel_fn = gaussian_kernel

    xs = jnp.asarray(xs, dtype=jnp.float64)
    ys = jnp.asarray(ys, dtype=jnp.float64)
    n = len(xs)

    # Build kernel matrix
    K = jnp.zeros((n, n))
    for i in range(n):
        for j in range(n):
            K = K.at[i, j].set(kernel_fn(jnp.abs(xs[i] - xs[j]), epsilon))

    weights = solve(K, ys)
    return {'xs': xs, 'weights': weights, 'kernel_fn': kernel_fn, 'epsilon': epsilon}


def evaluate(data, x):
    """Evaluate 1D kernel interpolant at x."""
    xs = data['xs']
    weights = data['weights']
    kernel_fn = data['kernel_fn']
    epsilon = data['epsilon']

    k_vec = jnp.array([kernel_fn(jnp.abs(x - xi), epsilon) for xi in xs])
    return jnp.dot(k_vec, weights)


# ── 2D Kernel Interpolation ──

def build_2d(xs, ys, zs, kernel_fn=None, epsilon=1.0):
    """Build 2D kernel interpolant.

    Parameters
    ----------
    xs, ys : arrays – node coordinates (flattened)
    zs : array – values at each (x, y) point
    kernel_fn : callable(r, epsilon), default Gaussian
    epsilon : float – shape parameter
    """
    if kernel_fn is None:
        kernel_fn = gaussian_kernel

    xs = jnp.asarray(xs, dtype=jnp.float64)
    ys = jnp.asarray(ys, dtype=jnp.float64)
    zs = jnp.asarray(zs, dtype=jnp.float64)
    n = len(xs)

    K = jnp.zeros((n, n))
    for i in range(n):
        for j in range(n):
            r = jnp.sqrt((xs[i] - xs[j]) ** 2 + (ys[i] - ys[j]) ** 2)
            K = K.at[i, j].set(kernel_fn(r, epsilon))

    weights = solve(K, zs)
    return {'xs': xs, 'ys': ys, 'weights': weights,
            'kernel_fn': kernel_fn, 'epsilon': epsilon}


def evaluate_2d(data, x, y):
    """Evaluate 2D kernel interpolant at (x, y)."""
    xs = data['xs']
    ys_data = data['ys']
    weights = data['weights']
    kernel_fn = data['kernel_fn']
    epsilon = data['epsilon']

    k_vec = jnp.array([
        kernel_fn(jnp.sqrt((x - xi) ** 2 + (y - yi) ** 2), epsilon)
        for xi, yi in zip(xs, ys_data)
    ])
    return jnp.dot(k_vec, weights)
