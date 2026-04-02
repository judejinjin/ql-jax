"""Inverse cumulative normal distribution for RNG transform."""

import jax.numpy as jnp
from jax.scipy.stats import norm


def inverse_cumulative_normal(u):
    """Transform uniform [0,1] to standard normal via Moro's algorithm.

    This is a fast rational approximation to the inverse normal CDF,
    accurate to about 9 significant digits.
    """
    # Use JAX built-in for accuracy
    return norm.ppf(u)


def transform_uniform_to_normal(uniforms):
    """Transform array of uniform [0,1) samples to standard normal.

    Clips inputs to avoid infinities at boundaries.
    """
    clipped = jnp.clip(uniforms, 1e-10, 1.0 - 1e-10)
    return norm.ppf(clipped)


def moro_inverse_normal(u):
    """Moro's rational approximation to inverse normal CDF.

    Separate implementation that doesn't depend on scipy for
    environments where a lightweight version is needed.
    """
    # Beasley-Springer-Moro algorithm
    a = jnp.array([
        -3.969683028665376e+01, 2.209460984245205e+02,
        -2.759285104469687e+02, 1.383577518672690e+02,
        -3.066479806614716e+01, 2.506628277459239e+00,
    ])
    b = jnp.array([
        -5.447609879822406e+01, 1.615858368580409e+02,
        -1.556989798598866e+02, 6.680131188771972e+01,
        -1.328068155288572e+01,
    ])
    c = jnp.array([
        -7.784894002430293e-03, -3.223964580411365e-01,
        -2.400758277161838e+00, -2.549732539343734e+00,
        4.374664141464968e+00, 2.938163982698783e+00,
    ])
    d = jnp.array([
        7.784695709041462e-03, 3.224671290700398e-01,
        2.445134137142996e+00, 3.754408661907416e+00,
    ])

    p_low = 0.02425
    p_high = 1.0 - p_low

    # Central region
    q = u - 0.5
    r = q * q
    x_central = (((((a[0] * r + a[1]) * r + a[2]) * r + a[3]) * r + a[4]) * r + a[5]) * q / \
                (((((b[0] * r + b[1]) * r + b[2]) * r + b[3]) * r + 1.0) * r + 1.0)

    # Lower tail
    q_low = jnp.sqrt(-2.0 * jnp.log(jnp.maximum(u, 1e-20)))
    x_low = (((((c[0] * q_low + c[1]) * q_low + c[2]) * q_low + c[3]) * q_low + c[4]) * q_low + c[5]) / \
            ((((d[0] * q_low + d[1]) * q_low + d[2]) * q_low + d[3]) * q_low + 1.0)

    # Upper tail
    q_high = jnp.sqrt(-2.0 * jnp.log(jnp.maximum(1.0 - u, 1e-20)))
    x_high = -(((((c[0] * q_high + c[1]) * q_high + c[2]) * q_high + c[3]) * q_high + c[4]) * q_high + c[5]) / \
              ((((d[0] * q_high + d[1]) * q_high + d[2]) * q_high + d[3]) * q_high + 1.0)

    return jnp.where(u < p_low, x_low, jnp.where(u > p_high, x_high, x_central))
