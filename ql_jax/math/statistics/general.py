"""General statistics: mean, variance, skewness, kurtosis.

Pure functions operating on JAX arrays. All differentiable.
"""

import jax.numpy as jnp


def mean(samples, weights=None):
    """Weighted or unweighted mean."""
    if weights is not None:
        return jnp.average(samples, weights=weights)
    return jnp.mean(samples)


def variance(samples, weights=None):
    """Weighted or unweighted sample variance."""
    if weights is not None:
        m = jnp.average(samples, weights=weights)
        return jnp.average((samples - m) ** 2, weights=weights)
    return jnp.var(samples, ddof=1)


def standard_deviation(samples, weights=None):
    """Sample standard deviation."""
    return jnp.sqrt(variance(samples, weights))


def skewness(samples):
    """Sample skewness."""
    m = jnp.mean(samples)
    s = jnp.std(samples, ddof=1)
    n = len(samples)
    return jnp.sum(((samples - m) / s) ** 3) * n / ((n - 1) * (n - 2))


def kurtosis(samples):
    """Sample excess kurtosis."""
    m = jnp.mean(samples)
    s = jnp.std(samples, ddof=1)
    n = len(samples)
    k4 = jnp.sum(((samples - m) / s) ** 4) / n
    return k4 - 3.0


def min_value(samples):
    return jnp.min(samples)


def max_value(samples):
    return jnp.max(samples)


def percentile(samples, p):
    """p-th percentile (0–100)."""
    return jnp.percentile(samples, p)


def median(samples):
    return jnp.median(samples)


def downsideVariance(samples):
    """Variance of negative returns only."""
    neg = samples[samples < 0]
    if len(neg) == 0:
        return 0.0
    return jnp.var(neg, ddof=1)
