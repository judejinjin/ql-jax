"""QL-JAX Random number generation.

Maps QuantLib RNG to JAX's functional PRNG system.
"""

from ql_jax.math.random.pseudo import make_key, split, uniform, normal, multivariate_normal
from ql_jax.math.random.quasi import sobol_sequence, halton_sequence
from ql_jax.math.random.sobol_brownian import generate_paths as sobol_brownian_paths
from ql_jax.math.random.faure import generate as faure_sequence
from ql_jax.math.random.inverse_cumulative import (
    inverse_cumulative_normal,
    transform_uniform_to_normal,
    moro_inverse_normal,
)
