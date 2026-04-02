"""B-spline basis functions and Bernstein polynomials."""

from __future__ import annotations

import jax.numpy as jnp


def bspline_basis(knots, degree, i, x):
    """Evaluate the i-th B-spline basis function of given degree at x.

    Uses the Cox-de Boor recursion.

    Parameters
    ----------
    knots : array of knot values
    degree : int (0, 1, 2, 3, ...)
    i : int, basis function index
    x : float, evaluation point

    Returns
    -------
    float
    """
    knots = jnp.asarray(knots, dtype=jnp.float64)
    x = jnp.float64(x)

    if degree == 0:
        return jnp.where((knots[i] <= x) & (x < knots[i + 1]), 1.0, 0.0)

    # Recursive definition
    denom1 = knots[i + degree] - knots[i]
    denom2 = knots[i + degree + 1] - knots[i + 1]

    w1 = jnp.where(jnp.abs(denom1) > 1e-15,
                    (x - knots[i]) / denom1, 0.0)
    w2 = jnp.where(jnp.abs(denom2) > 1e-15,
                    (knots[i + degree + 1] - x) / denom2, 0.0)

    return w1 * bspline_basis(knots, degree - 1, i, x) + w2 * bspline_basis(knots, degree - 1, i + 1, x)


def bspline_values(knots, degree, x):
    """Evaluate all B-spline basis functions at x.

    Parameters
    ----------
    knots : array of knots (length n + degree + 1)
    degree : int
    x : float

    Returns
    -------
    array of length n (number of basis functions)
    """
    knots = jnp.asarray(knots, dtype=jnp.float64)
    n = len(knots) - degree - 1
    return jnp.array([bspline_basis(knots, degree, i, x) for i in range(n)])


def bernstein_polynomial(n, k, x):
    """Evaluate the (n, k)-th Bernstein basis polynomial at x.

    B_{k,n}(x) = C(n,k) * x^k * (1-x)^{n-k}

    Parameters
    ----------
    n : degree
    k : index (0 <= k <= n)
    x : evaluation point in [0, 1]

    Returns
    -------
    float
    """
    from jax.scipy.special import gammaln
    n, k, x = jnp.float64(n), jnp.float64(k), jnp.float64(x)
    log_coeff = gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)
    log_val = log_coeff + k * jnp.log(jnp.maximum(x, 1e-30)) + (n - k) * jnp.log(jnp.maximum(1 - x, 1e-30))
    return jnp.exp(log_val)


def bernstein_expansion(coeffs, x):
    """Evaluate a polynomial in Bernstein form.

    f(x) = sum_{k=0}^{n} c_k * B_{k,n}(x)

    Parameters
    ----------
    coeffs : array of Bernstein coefficients
    x : evaluation point in [0, 1]

    Returns
    -------
    float
    """
    coeffs = jnp.asarray(coeffs, dtype=jnp.float64)
    n = len(coeffs) - 1
    basis = jnp.array([bernstein_polynomial(n, k, x) for k in range(n + 1)])
    return jnp.dot(coeffs, basis)
