"""Special functions: Bessel, incomplete beta/gamma, error function, factorial."""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax.scipy.special import gammaln, gammainc, gammaincc, betaln


def factorial(n):
    """Factorial n! using gamma function for JAX compatibility."""
    return jnp.exp(gammaln(jnp.float64(n) + 1.0))


def double_factorial(n):
    """Double factorial n!! = n * (n-2) * (n-4) * ..."""
    n = jnp.float64(n)
    # n!! = 2^(n/2) * (n/2)!  for even n
    # n!! = n! / (2^((n-1)/2) * ((n-1)/2)!)  for odd n
    k = jnp.floor(n / 2.0)
    is_even = jnp.mod(n, 2.0) < 0.5
    even_result = 2.0 ** k * jnp.exp(gammaln(k + 1.0))
    odd_result = jnp.exp(gammaln(n + 1.0)) / even_result
    return jnp.where(is_even, even_result, jnp.where(n <= 0, 1.0, odd_result))


def modified_bessel_first(nu, x):
    """Modified Bessel function of the first kind I_nu(x).

    Uses the series expansion for moderate x and scipy for large.
    """
    from jax.scipy.special import i0e, i1e

    x = jnp.float64(x)
    nu = jnp.float64(nu)

    # For nu=0 and nu=1, use optimized JAX functions
    i0_val = jnp.exp(jnp.abs(x)) * i0e(x)
    i1_val = jnp.exp(jnp.abs(x)) * i1e(x)

    # For general nu, series expansion
    # I_nu(x) = sum_{m=0}^{inf} (x/2)^{nu+2m} / (m! * Gamma(nu+m+1))
    def _series(nu_, x_):
        half_x = x_ / 2.0
        result = 0.0
        for m in range(50):
            log_term = (nu_ + 2 * m) * jnp.log(jnp.maximum(half_x, 1e-30)) - gammaln(m + 1.0) - gammaln(nu_ + m + 1.0)
            result = result + jnp.exp(log_term)
        return result

    general = _series(nu, x)

    return jnp.where(nu == 0.0, i0_val, jnp.where(nu == 1.0, i1_val, general))


def modified_bessel_second(nu, x):
    """Modified Bessel function of the second kind K_nu(x).

    K_nu(x) = pi/2 * (I_{-nu}(x) - I_nu(x)) / sin(nu*pi)  for non-integer nu
    """
    x = jnp.float64(x)
    nu = jnp.float64(nu)

    i_pos = modified_bessel_first(nu, x)
    i_neg = modified_bessel_first(-nu, x)
    sin_nu = jnp.sin(nu * jnp.pi)

    # For integer nu, use limiting form
    return jnp.where(
        jnp.abs(sin_nu) < 1e-10,
        # Integer nu: use recurrence/asymptotic
        jnp.sqrt(jnp.pi / (2 * jnp.maximum(x, 1e-30))) * jnp.exp(-x),
        jnp.pi / 2.0 * (i_neg - i_pos) / sin_nu,
    )


def incomplete_beta(a, b, x):
    """Regularized incomplete beta function I_x(a, b).

    Parameters
    ----------
    a, b : shape parameters (positive)
    x : value in [0, 1]

    Returns
    -------
    float : regularized incomplete beta
    """
    from jax.scipy.special import betainc
    return betainc(jnp.float64(a), jnp.float64(b), jnp.float64(x))


def incomplete_gamma(a, x):
    """Regularized lower incomplete gamma function P(a, x) = gamma(a,x)/Gamma(a).

    Parameters
    ----------
    a : shape parameter
    x : value

    Returns
    -------
    float
    """
    return gammainc(jnp.float64(a), jnp.float64(x))


def incomplete_gamma_upper(a, x):
    """Regularized upper incomplete gamma Q(a, x) = 1 - P(a, x)."""
    return gammaincc(jnp.float64(a), jnp.float64(x))


def error_function(x):
    """Error function erf(x)."""
    return jax.scipy.special.erf(jnp.float64(x))


def error_function_complement(x):
    """Complementary error function erfc(x) = 1 - erf(x)."""
    return jax.scipy.special.erfc(jnp.float64(x))


def pascal_triangle_row(n):
    """Return row n of Pascal's triangle (binomial coefficients).

    Returns
    -------
    array of length n+1: [C(n,0), C(n,1), ..., C(n,n)]
    """
    n = int(n)
    row = jnp.zeros(n + 1, dtype=jnp.float64)
    row = row.at[0].set(1.0)
    for k in range(1, n + 1):
        row = row.at[k].set(row[k - 1] * (n - k + 1) / k)
    return row


def binomial_coefficient(n, k):
    """Binomial coefficient C(n, k) = n! / (k! * (n-k)!)."""
    n, k = jnp.float64(n), jnp.float64(k)
    return jnp.exp(gammaln(n + 1) - gammaln(k + 1) - gammaln(n - k + 1))
