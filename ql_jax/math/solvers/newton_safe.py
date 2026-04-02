"""Newton-safe root finding — Newton with bisection fallback."""

import jax
import jax.numpy as jnp


def solve(f, lower, upper, x0=None, tol=1e-12, max_iter=100):
    """Newton's method with bisection safeguard.

    Uses Newton when it stays within brackets, falls back to bisection
    when Newton steps outside the bracket.
    """
    if x0 is None:
        x0 = 0.5 * (lower + upper)

    df = jax.grad(f)

    fl = f(lower)
    fu = f(upper)

    x = x0
    for _ in range(max_iter):
        fx = f(x)
        if jnp.abs(fx) < tol:
            return x

        fpx = df(x)

        # Newton step
        if jnp.abs(fpx) > 1e-30:
            x_newton = x - fx / fpx
        else:
            x_newton = 0.5 * (lower + upper)

        # Check if Newton stays in bracket
        in_bracket = (x_newton > lower) & (x_newton < upper)
        x_bisect = 0.5 * (lower + upper)
        x_new = jnp.where(in_bracket, x_newton, x_bisect)

        # Update bracket
        fx_new = f(x_new)
        if jnp.sign(fx_new) == jnp.sign(fl):
            lower = x_new
            fl = fx_new
        else:
            upper = x_new
            fu = fx_new

        x = x_new

        if jnp.abs(upper - lower) < tol:
            return x

    return x
