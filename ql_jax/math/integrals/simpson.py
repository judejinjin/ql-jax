"""Simpson's rule for numerical integration."""

import jax.numpy as jnp


def integrate(f, a, b, n=100):
    """Integrate f over [a, b] using composite Simpson's rule with n subintervals.

    n must be even. If odd, n is incremented by 1.
    """
    if n % 2 != 0:
        n += 1

    h = (b - a) / n
    x = jnp.linspace(a, b, n + 1)
    y = f(x)

    # Simpson's 1/3 rule: (h/3) * [f(x0) + 4*f(x1) + 2*f(x2) + 4*f(x3) + ... + f(xn)]
    result = y[0] + y[-1]
    result += 4.0 * jnp.sum(y[1:-1:2])
    result += 2.0 * jnp.sum(y[2:-2:2])
    return result * h / 3.0
