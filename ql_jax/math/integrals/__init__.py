"""QL-JAX Numerical integration methods."""

from ql_jax.math.integrals.gauss import (
    gauss_legendre_nodes_weights,
    gauss_laguerre_nodes_weights,
    gauss_hermite_nodes_weights,
    integrate_gauss_legendre,
    integrate_gauss_laguerre,
    integrate_gauss_hermite,
)
from ql_jax.math.integrals.simpson import integrate as simpson_integrate
from ql_jax.math.integrals.trapezoid import integrate as trapezoid_integrate
