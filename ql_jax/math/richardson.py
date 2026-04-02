"""Richardson extrapolation for convergence acceleration."""

from __future__ import annotations

import jax.numpy as jnp


def richardson_extrapolation(f, x, h=0.1, order=1, ratio=2.0):
    """Richardson extrapolation to improve convergence.

    Given a function f(h) that approximates a limit as h->0,
    uses evaluations at h and h/ratio to extrapolate.

    Parameters
    ----------
    f : callable(h) -> value
        Function to extrapolate.
    x : not used (for signature compatibility), pass None
    h : initial step size
    order : order of the leading error term
    ratio : step size reduction factor

    Returns
    -------
    float : extrapolated value
    """
    f_h = f(h)
    f_h2 = f(h / ratio)
    p = ratio ** order
    return (p * f_h2 - f_h) / (p - 1.0)


def richardson_extrapolation_table(f, h, n_levels, ratio=2.0):
    """Full Richardson extrapolation table (Romberg-like).

    Parameters
    ----------
    f : callable(h) -> value
    h : initial step size
    n_levels : number of extrapolation levels
    ratio : step size reduction ratio

    Returns
    -------
    array of shape (n_levels + 1,) — extrapolated values at each level
    """
    # Compute function values at decreasing step sizes
    hs = jnp.array([h / ratio ** i for i in range(n_levels + 1)])
    values = jnp.array([f(float(hi)) for hi in hs])

    # Build Richardson table
    table = jnp.zeros((n_levels + 1, n_levels + 1), dtype=jnp.float64)
    table = table.at[:, 0].set(values)

    for j in range(1, n_levels + 1):
        p = ratio ** j
        for i in range(j, n_levels + 1):
            table = table.at[i, j].set(
                (p * table[i, j - 1] - table[i - 1, j - 1]) / (p - 1.0)
            )

    return jnp.diag(table)  # Best estimate at each level


def numerical_differentiation(f, x, h=1e-5, order=1, method='central'):
    """Numerical differentiation using finite differences.

    Parameters
    ----------
    f : callable(x) -> value
    x : evaluation point
    h : step size
    order : derivative order (1 or 2)
    method : 'forward', 'backward', or 'central'

    Returns
    -------
    float : approximate derivative
    """
    x = jnp.float64(x)
    if order == 1:
        if method == 'central':
            return (f(x + h) - f(x - h)) / (2 * h)
        elif method == 'forward':
            return (f(x + h) - f(x)) / h
        else:  # backward
            return (f(x) - f(x - h)) / h
    elif order == 2:
        return (f(x + h) - 2 * f(x) + f(x - h)) / h ** 2
    else:
        raise ValueError(f"Order {order} not supported")
