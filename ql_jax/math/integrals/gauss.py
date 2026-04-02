"""Gaussian quadrature: Gauss-Legendre, Gauss-Laguerre, Gauss-Hermite, Gauss-Jacobi.

Uses precomputed nodes and weights: integral ≈ dot(weights, f(nodes)).
All differentiable via JAX (Leibniz rule through the integrand).
"""

import jax.numpy as jnp
import numpy as np


def gauss_legendre_nodes_weights(n: int):
    """Gauss-Legendre quadrature nodes and weights on [-1, 1].

    Uses numpy for the eigenvalue computation (one-time cost),
    then returns JAX arrays.
    """
    nodes, weights = np.polynomial.legendre.leggauss(n)
    return jnp.array(nodes, dtype=jnp.float64), jnp.array(weights, dtype=jnp.float64)


def gauss_laguerre_nodes_weights(n: int):
    """Gauss-Laguerre quadrature nodes and weights on [0, ∞)."""
    nodes, weights = np.polynomial.laguerre.laggauss(n)
    return jnp.array(nodes, dtype=jnp.float64), jnp.array(weights, dtype=jnp.float64)


def gauss_hermite_nodes_weights(n: int):
    """Gauss-Hermite quadrature nodes and weights on (-∞, ∞)."""
    nodes, weights = np.polynomial.hermite.hermgauss(n)
    return jnp.array(nodes, dtype=jnp.float64), jnp.array(weights, dtype=jnp.float64)


def integrate_gauss_legendre(f, a, b, n=16):
    """Integrate f over [a, b] using n-point Gauss-Legendre quadrature.

    Transforms from [-1,1] to [a,b]: x = (b-a)/2 * t + (a+b)/2
    """
    nodes, weights = gauss_legendre_nodes_weights(n)
    half = (b - a) / 2.0
    mid = (a + b) / 2.0
    x = half * nodes + mid
    return half * jnp.dot(weights, f(x))


def integrate_gauss_laguerre(f, n=16):
    """Integrate f(x)*exp(-x) over [0, ∞) using n-point Gauss-Laguerre."""
    nodes, weights = gauss_laguerre_nodes_weights(n)
    return jnp.dot(weights, f(nodes))


def integrate_gauss_hermite(f, n=16):
    """Integrate f(x)*exp(-x²) over (-∞, ∞) using n-point Gauss-Hermite."""
    nodes, weights = gauss_hermite_nodes_weights(n)
    return jnp.dot(weights, f(nodes))
