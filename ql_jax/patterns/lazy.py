"""LazyObject pattern — functional replacement via jax.jit caching.

QuantLib's LazyObject defers computation until results are needed and caches
results until inputs change. In JAX:
- jax.jit provides compilation caching (traced once, fast re-execution)
- functools.lru_cache provides Python-level memoization
- Pure functions with immutable inputs naturally avoid stale cache issues
"""

import functools


def lazy(fn):
    """Decorator for lazy evaluation with memoization.

    This is the functional replacement for QuantLib's LazyObject pattern.
    The decorated function caches its result based on arguments.
    """
    return functools.lru_cache(maxsize=128)(fn)
