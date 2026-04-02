"""QL-JAX Interpolation methods.

All interpolation methods are JAX-differentiable.
"""

from ql_jax.math.interpolations import (
    linear,
    cubic,
    log,
    backward_flat,
    forward_flat,
    convex_monotone,
    sabr,
    chebyshev,
    lagrange,
    kernel,
    mixed,
    interp2d,
    flat_extrap_2d,
)
