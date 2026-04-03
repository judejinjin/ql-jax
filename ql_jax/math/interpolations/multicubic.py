"""Multi-dimensional cubic spline interpolation.

Extends 1-D cubic spline to N dimensions via successive interpolation
along each axis (tensor-product approach).
"""

from __future__ import annotations

import jax.numpy as jnp
from functools import partial


def _cubic_spline_1d(x, y):
    """Build natural cubic spline coefficients for (x, y) data.

    Returns (a, b, c, d) arrays where, for interval i:
        s_i(t) = a[i] + b[i]*(t-x[i]) + c[i]*(t-x[i])^2 + d[i]*(t-x[i])^3
    """
    n = len(x) - 1
    h = jnp.diff(x)

    # Set up tridiagonal system for second derivatives (natural BC: c[0]=c[n]=0)
    alpha = jnp.zeros(n + 1)
    for i in range(1, n):
        alpha = alpha.at[i].set(
            3.0 / h[i] * (y[i + 1] - y[i]) - 3.0 / h[i - 1] * (y[i] - y[i - 1])
        )

    # Solve tridiagonal
    l = jnp.ones(n + 1)
    mu = jnp.zeros(n + 1)
    z = jnp.zeros(n + 1)

    for i in range(1, n):
        l_val = 2.0 * (x[i + 1] - x[i - 1]) - h[i - 1] * mu[i - 1]
        l = l.at[i].set(l_val)
        mu = mu.at[i].set(h[i] / l_val)
        z = z.at[i].set((alpha[i] - h[i - 1] * z[i - 1]) / l_val)

    c = jnp.zeros(n + 1)
    b = jnp.zeros(n)
    d = jnp.zeros(n)

    for j in range(n - 1, -1, -1):
        c = c.at[j].set(z[j] - mu[j] * c[j + 1])
        b = b.at[j].set(
            (y[j + 1] - y[j]) / h[j] - h[j] * (c[j + 1] + 2.0 * c[j]) / 3.0
        )
        d = d.at[j].set((c[j + 1] - c[j]) / (3.0 * h[j]))

    return y[:n], b, c[:n], d


def _cubic_spline_eval(x_data, coeffs, x_query):
    """Evaluate 1D cubic spline at query points."""
    a, b, c, d = coeffs
    idx = jnp.searchsorted(x_data[1:], x_query)
    idx = jnp.clip(idx, 0, len(a) - 1)
    dx = x_query - x_data[idx]
    return a[idx] + b[idx] * dx + c[idx] * dx**2 + d[idx] * dx**3


class MultiCubicSpline:
    """Multi-dimensional cubic spline interpolation.

    Uses tensor-product approach: successive 1-D cubic spline
    interpolation along each axis.

    Parameters
    ----------
    grids : list of 1-D arrays, one per dimension
    values : N-D array of shape (len(g) for g in grids)
    """

    def __init__(self, grids: list[jnp.ndarray], values: jnp.ndarray):
        self.grids = [jnp.asarray(g) for g in grids]
        self.values = jnp.asarray(values)
        self.ndim = len(grids)
        assert self.values.ndim == self.ndim

    def __call__(self, *points: float) -> float:
        """Evaluate at a single point (x0, x1, ..., x_{n-1})."""
        assert len(points) == self.ndim
        return self._interpolate_recursive(
            self.values, list(range(self.ndim)), list(points)
        )

    def _interpolate_recursive(self, data, dims, points):
        """Recursively interpolate along each dimension."""
        if len(dims) == 0:
            return float(data)

        dim = dims[0]
        grid = self.grids[dim]
        x_q = points[0]

        # Interpolate along first remaining axis
        n_slices = data.shape[0]
        if len(dims) == 1:
            # Last dimension: just do 1D cubic spline
            y = data
            coeffs = _cubic_spline_1d(grid, y)
            return float(_cubic_spline_eval(grid, coeffs, jnp.array(x_q)))

        # For each point along this axis, recursively interpolate the rest
        sub_values = []
        for i in range(n_slices):
            sub = self._interpolate_recursive(
                data[i], dims[1:], points[1:]
            )
            sub_values.append(sub)

        y = jnp.array(sub_values)
        coeffs = _cubic_spline_1d(grid, y)
        return float(_cubic_spline_eval(grid, coeffs, jnp.array(x_q)))
