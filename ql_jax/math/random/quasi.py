"""Quasi-random (low-discrepancy) sequences: Sobol, Halton.

These are deterministic sequences with better space-filling properties
than pseudo-random numbers, important for Monte Carlo convergence.
"""

import jax.numpy as jnp
import numpy as np


def sobol_sequence(dimension: int, n_points: int, skip: int = 0):
    """Generate an n_points x dimension Sobol' sequence.

    Uses scipy's Sobol engine via numpy (one-time generation),
    then converts to JAX array for downstream differentiable use.
    """
    try:
        from scipy.stats import qmc
        sampler = qmc.Sobol(d=dimension, scramble=True)
        if skip > 0:
            sampler.fast_forward(skip)
        points = sampler.random(n_points)
        return jnp.array(points, dtype=jnp.float64)
    except ImportError:
        # Fallback: use simple van der Corput in each dimension
        return _halton_sequence(dimension, n_points, skip)


def halton_sequence(dimension: int, n_points: int, skip: int = 0):
    """Generate an n_points x dimension Halton sequence."""
    try:
        from scipy.stats import qmc
        sampler = qmc.Halton(d=dimension, scramble=True)
        if skip > 0:
            sampler.fast_forward(skip)
        points = sampler.random(n_points)
        return jnp.array(points, dtype=jnp.float64)
    except ImportError:
        return _halton_sequence(dimension, n_points, skip)


def _halton_sequence(dimension: int, n_points: int, skip: int = 0):
    """Simple Halton sequence implementation using first `dimension` primes."""
    primes = _first_primes(dimension)
    result = np.zeros((n_points, dimension))
    for j, base in enumerate(primes):
        for i in range(n_points):
            result[i, j] = _van_der_corput(i + skip + 1, base)
    return jnp.array(result, dtype=jnp.float64)


def _van_der_corput(n: int, base: int) -> float:
    """Van der Corput sequence element."""
    result = 0.0
    denom = 1.0
    while n > 0:
        denom *= base
        n, remainder = divmod(n, base)
        result += remainder / denom
    return result


def _first_primes(n: int) -> list[int]:
    """Return the first n prime numbers."""
    primes = []
    candidate = 2
    while len(primes) < n:
        if all(candidate % p != 0 for p in primes):
            primes.append(candidate)
        candidate += 1
    return primes
