"""Chi-squared distribution: PDF, CDF."""

import jax.numpy as jnp
import jax.scipy.stats as jstats


def pdf(x, df):
    """Chi-squared probability density function."""
    return jstats.chi2.pdf(x, df)


def cdf(x, df):
    """Chi-squared cumulative distribution function."""
    return jstats.chi2.cdf(x, df)


def inverse_cdf(p, df):
    """Inverse chi-squared CDF (quantile function)."""
    return jstats.chi2.ppf(p, df)
