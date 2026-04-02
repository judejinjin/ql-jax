"""Ridder's root-finding method."""

import jax.numpy as jnp


def solve(f, lower, upper, tol=1e-12, max_iter=100):
    """Find root of f in [lower, upper] using Ridder's method.

    Ridder's method achieves order ~1.84 convergence using a combination
    of bisection and false position with an exponential transformation.
    """
    fl = f(lower)
    fu = f(upper)

    x = lower
    for _ in range(max_iter):
        mid = 0.5 * (lower + upper)
        fm = f(mid)

        s = jnp.sqrt(fm**2 - fl * fu)
        if s == 0.0:
            return mid

        sign = jnp.where(fl - fu > 0.0, 1.0, -1.0)
        x_new = mid + (mid - lower) * sign * fm / s

        fx = f(x_new)

        if jnp.abs(fx) < tol:
            return x_new

        # Update bracket
        if jnp.sign(fm) != jnp.sign(fx):
            lower = mid
            fl = fm
            upper = x_new
            fu = fx
        elif jnp.sign(fl) != jnp.sign(fx):
            upper = x_new
            fu = fx
        else:
            lower = x_new
            fl = fx

        if jnp.abs(upper - lower) < tol:
            return x_new

        x = x_new

    return x
