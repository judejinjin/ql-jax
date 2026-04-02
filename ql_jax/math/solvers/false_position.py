"""False position (Regula Falsi) root-finding method."""

import jax.numpy as jnp


def solve(f, lower, upper, tol=1e-12, max_iter=100):
    """Find root of f in [lower, upper] using the false position method.

    Illinois variant for improved convergence.
    """
    fl = f(lower)
    fu = f(upper)

    for _ in range(max_iter):
        # False position estimate
        x = (lower * fu - upper * fl) / (fu - fl)
        fx = f(x)

        if jnp.abs(fx) < tol:
            return x

        if jnp.sign(fx) == jnp.sign(fl):
            lower = x
            fl = fx
            # Illinois modification: halve the other function value
            fu = fu / 2.0
        else:
            upper = x
            fu = fx
            fl = fl / 2.0

        if jnp.abs(upper - lower) < tol:
            return x

    return (lower * fu - upper * fl) / (fu - fl)
