"""General and weighted linear least squares regression."""

from __future__ import annotations

import jax.numpy as jnp


def general_linear_least_squares(X, y):
    """General linear least squares: find beta minimizing ||X beta - y||^2.

    Parameters
    ----------
    X : array [n_samples, n_features] — design matrix
    y : array [n_samples] — target values

    Returns
    -------
    array [n_features] — fitted coefficients beta
    """
    X = jnp.asarray(X, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    # Normal equations: beta = (X^T X)^{-1} X^T y
    return jnp.linalg.lstsq(X, y, rcond=None)[0]


def weighted_linear_least_squares(X, y, weights):
    """Weighted linear least squares.

    Minimizes sum_i w_i * (X_i . beta - y_i)^2.

    Parameters
    ----------
    X : array [n_samples, n_features]
    y : array [n_samples]
    weights : array [n_samples] (positive)

    Returns
    -------
    array [n_features]
    """
    X = jnp.asarray(X, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    w = jnp.sqrt(jnp.asarray(weights, dtype=jnp.float64))
    Xw = X * w[:, None]
    yw = y * w
    return jnp.linalg.lstsq(Xw, yw, rcond=None)[0]


def polynomial_regression(x, y, degree, weights=None):
    """Polynomial regression of given degree.

    Parameters
    ----------
    x : array [n_samples]
    y : array [n_samples]
    degree : int
    weights : optional array [n_samples]

    Returns
    -------
    array [degree + 1] — coefficients [c_0, c_1, ..., c_degree] where y ≈ sum c_k * x^k
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    y = jnp.asarray(y, dtype=jnp.float64)
    X = jnp.column_stack([x ** k for k in range(degree + 1)])
    if weights is not None:
        return weighted_linear_least_squares(X, y, weights)
    return general_linear_least_squares(X, y)
