"""Student-t distribution."""

import jax.numpy as jnp
from jax.scipy.special import gammaln, betainc


def pdf(x, nu):
    """Student-t probability density function with nu degrees of freedom."""
    coeff = jnp.exp(gammaln((nu + 1.0) / 2.0) - gammaln(nu / 2.0))
    return coeff / jnp.sqrt(nu * jnp.pi) * (1.0 + x**2 / nu) ** (-(nu + 1.0) / 2.0)


def cdf(x, nu):
    """Student-t cumulative distribution function."""
    t = nu / (nu + x**2)
    ib = 0.5 * betainc(nu / 2.0, 0.5, t)
    return jnp.where(x >= 0, 1.0 - ib, ib)


def mean(nu):
    """Mean (0 for nu > 1, undefined otherwise)."""
    return jnp.where(nu > 1.0, 0.0, jnp.nan)


def variance(nu):
    """Variance = nu / (nu - 2) for nu > 2."""
    return jnp.where(nu > 2.0, nu / (nu - 2.0), jnp.nan)
