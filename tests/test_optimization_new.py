"""Tests for optimization algorithms."""
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


class TestDifferentialEvolution:
    def test_rosenbrock(self):
        from ql_jax.math.optimization.differential_evolution import minimize
        def rosenbrock(x):
            return (1.0 - x[0])**2 + 100.0 * (x[1] - x[0]**2)**2

        bounds = jnp.array([[-5.0, 5.0], [-5.0, 5.0]])
        result = minimize(rosenbrock, bounds, population_size=30, max_iterations=500)
        x_opt = result['x']
        assert abs(float(x_opt[0]) - 1.0) < 0.5
        assert abs(float(x_opt[1]) - 1.0) < 0.5


class TestSimulatedAnnealing:
    def test_quadratic(self):
        from ql_jax.math.optimization.simulated_annealing import minimize
        def quadratic(x):
            return jnp.sum(x**2)

        x0 = jnp.array([5.0, 5.0])
        result = minimize(quadratic, x0, max_iterations=5000)
        x_opt = result['x']
        assert jnp.sum(x_opt**2) < 5.0


class TestBFGS:
    def test_quadratic(self):
        from ql_jax.math.optimization.bfgs import minimize
        def quadratic(x):
            return (x[0] - 3.0)**2 + (x[1] + 2.0)**2

        x0 = jnp.array([0.0, 0.0])
        result = minimize(quadratic, x0)
        x_opt = result['x']
        assert abs(float(x_opt[0]) - 3.0) < 0.1
        assert abs(float(x_opt[1]) + 2.0) < 0.1


class TestLevenbergMarquardt:
    def test_least_squares(self):
        from ql_jax.math.optimization.levenberg_marquardt import minimize
        x_data = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_data = jnp.array([2.1, 3.9, 6.1, 7.9, 10.1])

        def residuals(params):
            return params[0] * x_data + params[1] - y_data

        p0 = jnp.array([1.0, 0.0])
        p_opt, _, _ = minimize(residuals, p0)
        assert abs(float(p_opt[0]) - 2.0) < 0.2


class TestSimplex:
    def test_minimum(self):
        from ql_jax.math.optimization.simplex import minimize
        def sphere(x):
            return jnp.sum(x**2)

        x0 = jnp.array([3.0, 4.0])
        x_opt, _, _ = minimize(sphere, x0)
        assert jnp.sum(x_opt**2) < 1.0


class TestConstraints:
    def test_boundary_constraint(self):
        from ql_jax.math.optimization.constraint import BoundaryConstraint
        bc = BoundaryConstraint(lo=0.0, hi=1.0)
        assert bc.test(jnp.array([0.5])) is True or bc.test(jnp.array([0.5]))

    def test_positive_constraint(self):
        from ql_jax.math.optimization.constraint import PositiveConstraint
        pc = PositiveConstraint()
        assert pc.test(jnp.array([1.0, 2.0])) is True or pc.test(jnp.array([1.0, 2.0]))
