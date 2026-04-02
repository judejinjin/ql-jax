"""Pseudo-random number generation using JAX's PRNG system.

JAX uses a functional, splittable PRNG (Threefry2x32 by default)
which replaces QuantLib's MT19937 and other stateful generators.
"""

import jax
import jax.numpy as jnp


def make_key(seed: int = 0):
    """Create a JAX PRNG key from an integer seed.

    Equivalent to QuantLib's MersenneTwisterUniformRng(seed).
    """
    return jax.random.PRNGKey(seed)


def split(key, num=2):
    """Split a key into num independent keys."""
    return jax.random.split(key, num)


def uniform(key, shape=(), low=0.0, high=1.0, dtype=jnp.float64):
    """Generate uniform random numbers in [low, high)."""
    return jax.random.uniform(key, shape=shape, minval=low, maxval=high, dtype=dtype)


def normal(key, shape=(), dtype=jnp.float64):
    """Generate standard normal random numbers.

    Replaces QuantLib's BoxMullerGaussianRng / ZigguratGaussianRng.
    """
    return jax.random.normal(key, shape=shape, dtype=dtype)


def multivariate_normal(key, mean, cov, shape=()):
    """Generate multivariate normal random samples."""
    return jax.random.multivariate_normal(key, mean, cov, shape=shape)


def exponential(key, shape=(), dtype=jnp.float64):
    """Generate exponential(1) random numbers."""
    return jax.random.exponential(key, shape=shape, dtype=dtype)


def poisson(key, lam, shape=(), dtype=jnp.int32):
    """Generate Poisson random numbers with rate lam."""
    return jax.random.poisson(key, lam, shape=shape, dtype=dtype)
