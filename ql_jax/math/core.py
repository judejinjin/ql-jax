"""Core math functions: special functions, comparison, rounding.

Most map directly to jax.numpy or jax.scipy.special with thin wrappers.
All are differentiable via JAX's built-in AD rules.
"""

import jax.numpy as jnp
import jax.scipy.special as jsp


# ---------------------------------------------------------------------------
# Special functions — thin JAX wrappers
# ---------------------------------------------------------------------------

def beta(a, b):
    """Beta function B(a, b)."""
    return jnp.exp(jsp.betaln(a, b))


def incomplete_beta(a, b, x):
    """Regularized incomplete beta function I_x(a, b)."""
    return jsp.betainc(a, b, x)


def gamma_function(x):
    """Gamma function Γ(x)."""
    return jnp.exp(jsp.gammaln(x))


def incomplete_gamma(a, x):
    """Regularized lower incomplete gamma function P(a, x) = γ(a,x)/Γ(a)."""
    return jsp.gammainc(a, x)


def error_function(x):
    """Error function erf(x)."""
    return jsp.erf(x)


def erfc(x):
    """Complementary error function erfc(x) = 1 - erf(x)."""
    return jsp.erfc(x)


def bessel_i0(x):
    """Modified Bessel function of the first kind, order 0."""
    return jsp.i0(x)


def bessel_i1(x):
    """Modified Bessel function of the first kind, order 1."""
    return jsp.i1(x)


# ---------------------------------------------------------------------------
# Array / Matrix operations (most are trivial via jnp)
# ---------------------------------------------------------------------------

def dot_product(a, b):
    """Dot product of two arrays."""
    return jnp.dot(a, b)


def outer_product(a, b):
    """Outer product of two arrays."""
    return jnp.outer(a, b)


def transpose(m):
    """Matrix transpose."""
    return jnp.transpose(m)


def inverse(m):
    """Matrix inverse."""
    return jnp.linalg.inv(m)


def determinant(m):
    """Matrix determinant."""
    return jnp.linalg.det(m)


def eigenvalues(m):
    """Eigenvalues of a symmetric matrix."""
    return jnp.linalg.eigvalsh(m)


def eigenvectors(m):
    """Eigenvalues and eigenvectors of a symmetric matrix."""
    return jnp.linalg.eigh(m)


def cholesky(m):
    """Cholesky decomposition of a positive-definite matrix."""
    return jnp.linalg.cholesky(m)


def svd(m):
    """Singular value decomposition."""
    return jnp.linalg.svd(m)


def pseudo_sqrt(m):
    """Pseudo square root of a positive semi-definite matrix via eigendecomposition."""
    eigenvals, eigenvecs = jnp.linalg.eigh(m)
    eigenvals = jnp.maximum(eigenvals, 0.0)
    return eigenvecs * jnp.sqrt(eigenvals)[None, :]


# ---------------------------------------------------------------------------
# FFT
# ---------------------------------------------------------------------------

def fft(x):
    """Fast Fourier Transform."""
    return jnp.fft.fft(x)


def ifft(x):
    """Inverse Fast Fourier Transform."""
    return jnp.fft.ifft(x)


# ---------------------------------------------------------------------------
# Rounding
# ---------------------------------------------------------------------------

def round_half_up(x, decimals=0):
    """Round half up (QuantLib default rounding)."""
    multiplier = 10.0 ** decimals
    return jnp.floor(x * multiplier + 0.5) / multiplier


def round_half_down(x, decimals=0):
    """Round half down."""
    multiplier = 10.0 ** decimals
    return jnp.ceil(x * multiplier - 0.5) / multiplier


def round_half_even(x, decimals=0):
    """Banker's rounding (round half to even)."""
    return jnp.round(x, decimals)


def round_up(x, decimals=0):
    """Always round up (ceiling)."""
    multiplier = 10.0 ** decimals
    return jnp.ceil(x * multiplier) / multiplier


def round_down(x, decimals=0):
    """Always round down (floor)."""
    multiplier = 10.0 ** decimals
    return jnp.floor(x * multiplier) / multiplier


def round_closest(x, decimals=0):
    """Round to closest value."""
    return round_half_up(x, decimals)


# ---------------------------------------------------------------------------
# Richardson extrapolation
# ---------------------------------------------------------------------------

def richardson_extrapolation(f, x, h, order=2):
    """Richardson extrapolation for order-`order` convergent approximation.

    Combines f(x, h) and f(x, h/2) to cancel leading error term.
    """
    r = 2 ** order
    return (r * f(x, h / 2) - f(x, h)) / (r - 1)


# ---------------------------------------------------------------------------
# Comparison with tolerance
# ---------------------------------------------------------------------------

def close(a, b, n=42):
    """Check if two values are close (n * epsilon)."""
    eps = n * jnp.finfo(jnp.float64).eps
    return jnp.abs(a - b) <= eps


def close_enough(a, b, n=42):
    """Check if two values are close enough (relative comparison)."""
    eps = n * jnp.finfo(jnp.float64).eps
    diff = jnp.abs(a - b)
    norm = jnp.maximum(jnp.abs(a), jnp.abs(b))
    return jnp.where(norm > 0.0, diff / norm <= eps, diff <= eps)
