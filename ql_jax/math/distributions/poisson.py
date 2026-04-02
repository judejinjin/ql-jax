"""Poisson distribution: PMF, CDF."""

import jax.numpy as jnp
import jax.scipy.stats as jstats


def pmf(k, mu):
    """Poisson probability mass function P(X=k)."""
    return jstats.poisson.pmf(k, mu)


def cdf(k, mu):
    """Poisson cumulative distribution function P(X<=k)."""
    return jstats.poisson.cdf(k, mu)


def log_pmf(k, mu):
    """Log of Poisson PMF."""
    return jstats.poisson.logpmf(k, mu)
