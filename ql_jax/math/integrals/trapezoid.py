"""Trapezoid rule for numerical integration."""

import jax.numpy as jnp


def integrate(f, a, b, n=100):
    """Integrate f over [a, b] using composite trapezoid rule with n subintervals."""
    x = jnp.linspace(a, b, n + 1)
    y = f(x)
    h = (b - a) / n
    return h * (0.5 * y[0] + jnp.sum(y[1:-1]) + 0.5 * y[-1])
