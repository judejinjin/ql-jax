"""Binomial distribution."""

import jax.numpy as jnp
from jax.scipy.special import gammaln


def _log_comb(n, k):
    """Log of binomial coefficient C(n, k)."""
    return gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1)


def pmf(k, n, p):
    """Binomial PMF: P(X=k) = C(n,k) * p^k * (1-p)^(n-k)."""
    log_p = _log_comb(n, k) + k * jnp.log(p) + (n - k) * jnp.log(1.0 - p)
    return jnp.exp(log_p)


def cdf(k, n, p):
    """Binomial CDF: P(X <= k) = sum_{i=0}^{k} PMF(i)."""
    # Use regularized incomplete beta for efficiency
    from jax.scipy.special import betainc
    return jnp.where(k >= n, 1.0,
                     betainc(n - k, k + 1, 1.0 - p))


def mean(n, p):
    return n * p


def variance(n, p):
    return n * p * (1.0 - p)
