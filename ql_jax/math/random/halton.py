"""Halton low-discrepancy sequence generator."""

from __future__ import annotations

import jax.numpy as jnp


def _prime(n):
    """Return the n-th prime number (0-indexed: prime(0)=2)."""
    primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47,
              53, 59, 61, 67, 71, 73, 79, 83, 89, 97, 101, 103, 107, 109, 113,
              127, 131, 137, 139, 149, 151, 157, 163, 167, 173, 179, 181, 191, 193,
              197, 199, 211, 223, 227, 229, 233, 239, 241, 251, 257, 263, 269, 271]
    if n < len(primes):
        return primes[n]
    # for larger n, compute
    p = primes[-1] + 2
    found = list(primes)
    while len(found) <= n:
        is_prime = True
        for d in found:
            if d * d > p:
                break
            if p % d == 0:
                is_prime = False
                break
        if is_prime:
            found.append(p)
        p += 2
    return found[n]


def halton_element(index, base):
    """Compute a single element of the Halton sequence.

    Parameters
    ----------
    index : int (1-based)
    base : int (prime base)

    Returns
    -------
    float in (0, 1)
    """
    result = 0.0
    f = 1.0 / base
    i = index
    while i > 0:
        result += f * (i % base)
        i = i // base
        f /= base
    return result


def halton_sequence(n_points, n_dims, skip=0):
    """Generate a Halton quasi-random sequence.

    Parameters
    ----------
    n_points : int
    n_dims : int (each dimension uses a different prime base)
    skip : int (number of initial points to skip)

    Returns
    -------
    array [n_points, n_dims] in (0, 1)
    """
    bases = [_prime(d) for d in range(n_dims)]
    result = []
    for i in range(skip + 1, skip + n_points + 1):
        row = [halton_element(i, b) for b in bases]
        result.append(row)
    return jnp.array(result, dtype=jnp.float64)


def halton_normal(n_points, n_dims, skip=0):
    """Halton sequence transformed to standard normal via inverse CDF.

    Parameters
    ----------
    n_points, n_dims, skip : as in halton_sequence

    Returns
    -------
    array [n_points, n_dims]
    """
    from jax.scipy.stats import norm as jnorm
    u = halton_sequence(n_points, n_dims, skip)
    return jnorm.ppf(jnp.clip(u, 1e-10, 1.0 - 1e-10))
