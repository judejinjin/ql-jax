"""Halley's root-finding method (cubic convergence)."""

import jax
import jax.numpy as jnp


def solve(f, x0, tol=1e-12, max_iter=50):
    """Find root near x0 using Halley's method.

    Requires f, f', and f''. Uses JAX AD to compute derivatives.
    Achieves cubic convergence.
    """
    df = jax.grad(f)
    ddf = jax.grad(df)

    x = x0
    for _ in range(max_iter):
        fx = f(x)
        if jnp.abs(fx) < tol:
            return x

        fpx = df(x)
        fppx = ddf(x)

        # Halley's formula: x_{n+1} = x_n - 2*f*f' / (2*f'^2 - f*f'')
        denom = 2.0 * fpx**2 - fx * fppx
        if jnp.abs(denom) < 1e-30:
            # Fallback to Newton
            x = x - fx / fpx
        else:
            x = x - 2.0 * fx * fpx / denom

    return x
