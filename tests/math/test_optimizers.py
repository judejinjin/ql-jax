"""Tests for optimization."""

import jax.numpy as jnp

from ql_jax.math.optimization import levenberg_marquardt, simplex
from ql_jax.math.optimization.end_criteria import EndCriteria


class TestLevenbergMarquardt:
    def test_rosenbrock_residuals(self):
        """Minimize Rosenbrock: f(x,y) = (1-x)^2 + 100*(y-x^2)^2
        as residuals: [1-x, 10*(y-x^2)]"""
        def residuals(params):
            x, y = params
            return jnp.array([1.0 - x, 10.0 * (y - x ** 2)])

        x0 = jnp.array([-1.0, 1.0])
        x_opt, cost, iters = levenberg_marquardt.minimize(residuals, x0)
        assert abs(float(x_opt[0]) - 1.0) < 1e-4
        assert abs(float(x_opt[1]) - 1.0) < 1e-4

    def test_linear_system(self):
        """r(x) = Ax - b"""
        A = jnp.array([[2.0, 1.0], [1.0, 3.0]])
        b = jnp.array([5.0, 7.0])
        x_exact = jnp.linalg.solve(A, b)

        def residuals(x):
            return A @ x - b

        x_opt, cost, _ = levenberg_marquardt.minimize(residuals, jnp.zeros(2))
        assert jnp.allclose(x_opt, x_exact, atol=1e-6)


class TestSimplex:
    def test_quadratic(self):
        """Minimize (x-3)^2 + (y-4)^2."""
        def f(x):
            return (x[0] - 3.0) ** 2 + (x[1] - 4.0) ** 2

        x_opt, val, _ = simplex.minimize(f, jnp.array([0.0, 0.0]))
        assert abs(float(x_opt[0]) - 3.0) < 0.01
        assert abs(float(x_opt[1]) - 4.0) < 0.01
        assert float(val) < 0.001
