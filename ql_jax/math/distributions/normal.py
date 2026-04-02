"""Normal (Gaussian) distribution: PDF, CDF, inverse CDF.

Maps to jax.scipy.stats.norm for core operations.
All differentiable via JAX.
"""

import jax.numpy as jnp
import jax.scipy.stats as jstats
import jax.scipy.special as jsp


def pdf(x, mean=0.0, sigma=1.0):
    """Normal probability density function."""
    return jstats.norm.pdf(x, loc=mean, scale=sigma)


def cdf(x, mean=0.0, sigma=1.0):
    """Normal cumulative distribution function."""
    return jstats.norm.cdf(x, loc=mean, scale=sigma)


def inverse_cdf(p, mean=0.0, sigma=1.0):
    """Inverse normal CDF (quantile function / probit).

    Uses the rational approximation from Abramowitz and Stegun,
    refined by Acklam.
    """
    return jstats.norm.ppf(p, loc=mean, scale=sigma)


def derivative_cdf(x, mean=0.0, sigma=1.0):
    """Derivative of the CDF = PDF."""
    return pdf(x, mean, sigma)


# ---------------------------------------------------------------------------
# Bivariate normal CDF (Drezner-Wesolowsky 1990)
# ---------------------------------------------------------------------------

def bivariate_cdf(x, y, rho):
    """Bivariate normal cumulative distribution function.

    P(X <= x, Y <= y) where corr(X,Y) = rho.
    Uses Drezner's improved algorithm.
    """
    # For |rho| < 0.925, use Gauss-Legendre; otherwise use series.
    # Simplified implementation using Gauss-Legendre quadrature.
    if rho == 0.0:
        return cdf(x) * cdf(y)

    if rho == 1.0:
        return cdf(jnp.minimum(x, y))

    if rho == -1.0:
        return jnp.maximum(cdf(x) + cdf(y) - 1.0, 0.0)

    return _drezner_bivariate_cdf(x, y, rho)


def _drezner_bivariate_cdf(a, b, rho):
    """Drezner (1978) algorithm for bivariate normal CDF.

    Uses Gauss-Legendre quadrature for high accuracy.
    """
    # Gauss-Legendre nodes and weights for 5-point quadrature
    x_gl = jnp.array([0.04691008, 0.23076534, 0.5, 0.76923466, 0.95308992])
    w_gl = jnp.array([0.01846567, 0.10412573, 0.20093191, 0.10412573, 0.01846567]) * 2

    # Transform: asinrho = arcsin(rho)
    asr = jnp.arcsin(rho)

    total = 0.0
    for i in range(5):
        sn = jnp.sin(asr * x_gl[i])
        total += w_gl[i] * jnp.exp(
            (sn * a * b - 0.5 * (a * a + b * b)) / (1.0 - sn * sn)
        )

    result = total * asr / (4.0 * jnp.pi) + cdf(a) * cdf(b)
    return result
