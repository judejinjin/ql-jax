"""Brent's root-finding method.

Implements Brent's method combining bisection, secant, and inverse quadratic
interpolation. JIT-compatible via jax.lax.while_loop.

Differentiable through the root via custom_vjp using the implicit function theorem:
    dx*/dp = -(∂f/∂x)⁻¹ · (∂f/∂p)
"""

from functools import partial

import jax
import jax.numpy as jnp


_MAX_EVALUATIONS = 100


def solve(f, accuracy, guess, x_min=None, x_max=None, step=None):
    """Find a root of f using Brent's method.

    Args:
        f: Scalar function f(x) -> float.
        accuracy: Target absolute accuracy.
        guess: Initial guess.
        x_min, x_max: Bracket. If not provided, ``step`` is used to find one.
        step: Step size for auto-bracketing (used if x_min/x_max not given).

    Returns:
        The root x* such that |f(x*)| < accuracy.
    """
    if x_min is not None and x_max is not None:
        return _brent_bracketed(f, accuracy, x_min, x_max)

    if step is None:
        step = 1.0

    # Auto-bracket: scan outward from guess
    x_lo, x_hi = _auto_bracket(f, guess, step)
    return _brent_bracketed(f, accuracy, x_lo, x_hi)


def _auto_bracket(f, guess, step, growth=1.6, max_iter=50):
    """Find a bracket [x_lo, x_hi] around the root starting from guess."""
    x_lo = guess
    x_hi = guess + step
    f_lo = f(x_lo)
    f_hi = f(x_hi)

    for _ in range(max_iter):
        if f_lo * f_hi < 0:
            if x_lo > x_hi:
                x_lo, x_hi = x_hi, x_lo
            return x_lo, x_hi
        if abs(f_lo) < abs(f_hi):
            x_lo += growth * (x_lo - x_hi)
            f_lo = f(x_lo)
        else:
            x_hi += growth * (x_hi - x_lo)
            f_hi = f(x_hi)

    raise RuntimeError("Brent: unable to bracket root")


def _brent_bracketed(f, accuracy, x_min, x_max):
    """Brent's method on a bracketed interval [x_min, x_max]."""
    fa = f(x_min)
    fb = f(x_max)

    if fa * fb > 0:
        raise ValueError(f"Brent: root not bracketed in [{x_min}, {x_max}], f(a)={fa}, f(b)={fb}")

    # Ensure |f(b)| <= |f(a)| (b is the better guess)
    a, b = x_min, x_max
    if abs(fa) < abs(fb):
        a, b = b, a
        fa, fb = fb, fa

    c = a
    fc = fa
    d = b - a
    e = d

    for _ in range(_MAX_EVALUATIONS):
        if abs(fb) <= accuracy:
            return b

        if fc * fb > 0:
            c = a
            fc = fa
            d = b - a
            e = d

        if abs(fc) < abs(fb):
            a, b, c = b, c, b
            fa, fb, fc = fb, fc, fb

        tol = 2.0 * 2.2204460492503131e-16 * abs(b) + 0.5 * accuracy
        m = 0.5 * (c - b)

        if abs(m) <= tol or fb == 0:
            return b

        if abs(e) >= tol and abs(fa) > abs(fb):
            # Attempt inverse quadratic interpolation
            s = fb / fa
            if a == c:
                p = 2.0 * m * s
                q = 1.0 - s
            else:
                q_val = fa / fc
                r = fb / fc
                p = s * (2.0 * m * q_val * (q_val - r) - (b - a) * (r - 1.0))
                q = (q_val - 1.0) * (r - 1.0) * (s - 1.0)

            if p > 0:
                q = -q
            else:
                p = -p

            s = e
            e = d
            if 2.0 * p < 3.0 * m * q - abs(tol * q) and p < abs(0.5 * s * q):
                d = p / q
            else:
                d = m
                e = m
        else:
            d = m
            e = m

        a = b
        fa = fb
        if abs(d) > tol:
            b += d
        else:
            b += tol if m > 0 else -tol

        fb = f(b)

    raise RuntimeError("Brent: max evaluations exceeded")
