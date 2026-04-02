"""QL-JAX 1D root solvers."""

from ql_jax.math.solvers.brent import solve as brent_solve
from ql_jax.math.solvers.newton import solve as newton_solve
from ql_jax.math.solvers.bisection import solve as bisection_solve
from ql_jax.math.solvers.secant import solve as secant_solve
from ql_jax.math.solvers.newton_safe import solve as newton_safe_solve
from ql_jax.math.solvers.halley import solve as halley_solve
from ql_jax.math.solvers.false_position import solve as false_position_solve
from ql_jax.math.solvers.ridder import solve as ridder_solve
