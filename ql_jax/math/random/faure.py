"""Faure low-discrepancy sequence generator."""

import jax.numpy as jnp
import math


def _next_prime(n):
    """Find smallest prime >= n."""
    if n <= 2:
        return 2
    candidate = n if n % 2 != 0 else n + 1
    while True:
        is_prime = True
        for d in range(3, int(math.sqrt(candidate)) + 1, 2):
            if candidate % d == 0:
                is_prime = False
                break
        if is_prime:
            return candidate
        candidate += 2


def generate(n_points, dimension, skip=0):
    """Generate Faure low-discrepancy sequence.

    Parameters
    ----------
    n_points : int – number of points
    dimension : int – dimensionality
    skip : int – number of initial points to skip

    Returns array of shape (n_points, dimension) in [0, 1)^d.
    """
    base = _next_prime(dimension)

    # Precompute Pascal matrix modulo base
    max_digits = max(20, int(math.log(n_points + skip + 1) / math.log(base)) + 2)
    C = _pascal_matrix(max_digits, base)

    result = []
    for i in range(skip, skip + n_points):
        point = []
        for d in range(dimension):
            if d == 0:
                # First dimension: Van der Corput in base
                point.append(_van_der_corput(i + 1, base))
            else:
                # Higher dimensions: apply generalized Faure permutation
                digits = _to_base(i + 1, base, max_digits)
                # Apply C^d matrix
                Cd = C
                for _ in range(d - 1):
                    Cd = (Cd @ C) % base
                new_digits = Cd @ digits % base
                val = sum(int(new_digits[k]) * base ** (-(k + 1)) for k in range(max_digits))
                point.append(val)
        result.append(point)

    return jnp.array(result, dtype=jnp.float64)


def _van_der_corput(n, base):
    """Van der Corput sequence element."""
    result = 0.0
    denom = 1.0
    while n > 0:
        denom *= base
        n, remainder = divmod(n, base)
        result += remainder / denom
    return result


def _to_base(n, base, num_digits):
    """Convert n to base representation as array of digits."""
    digits = []
    for _ in range(num_digits):
        digits.append(n % base)
        n //= base
    return jnp.array(digits, dtype=jnp.int32)


def _pascal_matrix(size, modulus):
    """Pascal matrix (mod modulus) for Faure permutation."""
    C = jnp.zeros((size, size), dtype=jnp.int32)
    for i in range(size):
        C = C.at[i, i].set(1)
        for j in range(i + 1, size):
            val = (int(C[i - 1, j - 1]) + int(C[i, j - 1])) % modulus if i > 0 else 0
            C = C.at[i, j].set(val)
    # Fill properly with binomial coefficients mod base
    for i in range(size):
        for j in range(i, size):
            # C(j, i) mod base
            binom = 1
            for k in range(i):
                binom = binom * (j - k) // (k + 1)
            C = C.at[i, j].set(binom % modulus)
    return C
