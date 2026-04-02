"""Gamma distribution."""

import jax.numpy as jnp
from jax.scipy.special import gammaln


def pdf(x, alpha, beta=1.0):
    """Gamma PDF: f(x) = beta^alpha / Gamma(alpha) * x^(alpha-1) * exp(-beta*x)."""
    log_p = (alpha * jnp.log(beta) - gammaln(alpha)
             + (alpha - 1.0) * jnp.log(jnp.maximum(x, 1e-300))
             - beta * x)
    return jnp.where(x > 0, jnp.exp(log_p), 0.0)


def cdf(x, alpha, beta=1.0):
    """Gamma CDF via regularized incomplete gamma function."""
    from jax.scipy.special import gammainc
    return jnp.where(x > 0, gammainc(alpha, beta * x), 0.0)


def mean(alpha, beta=1.0):
    return alpha / beta


def variance(alpha, beta=1.0):
    return alpha / beta**2
