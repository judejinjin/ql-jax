"""Tests for interpolation methods."""

import jax
import jax.numpy as jnp
import pytest

from ql_jax.math.interpolations import linear, log, cubic, backward_flat, forward_flat


class TestLinearInterpolation:
    def test_exact_points(self):
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [1.0, 4.0, 9.0, 16.0]
        state = linear.build(xs, ys)
        for x, y in zip(xs, ys):
            assert abs(float(linear.evaluate(state, x)) - y) < 1e-12

    def test_midpoint(self):
        state = linear.build([0.0, 1.0], [0.0, 1.0])
        assert abs(float(linear.evaluate(state, 0.5)) - 0.5) < 1e-12

    def test_derivative(self):
        state = linear.build([0.0, 1.0, 2.0], [0.0, 2.0, 6.0])
        assert abs(float(linear.derivative(state, 0.5)) - 2.0) < 1e-12
        assert abs(float(linear.derivative(state, 1.5)) - 4.0) < 1e-12

    def test_ad_gradient(self):
        """Test that jax.grad works through linear interpolation."""
        xs = jnp.array([0.0, 1.0, 2.0])
        ys = jnp.array([0.0, 1.0, 4.0])
        state = linear.build(xs, ys)

        def f(x):
            return linear.evaluate(state, x)

        grad_f = jax.grad(f)
        # Between x=1 and x=2, slope = (4-1)/(2-1) = 3
        assert abs(float(grad_f(1.5)) - 3.0) < 1e-10

    def test_vectorized(self):
        state = linear.build([0.0, 1.0, 2.0], [0.0, 1.0, 4.0])
        xs = jnp.array([0.5, 1.0, 1.5])
        result = linear.evaluate_many(state, xs)
        expected = jnp.array([0.5, 1.0, 2.5])
        assert jnp.allclose(result, expected)


class TestLogLinearInterpolation:
    def test_exact_points(self):
        xs = [1.0, 2.0, 3.0]
        ys = [1.0, jnp.e, jnp.e ** 2]
        state = log.build(xs, ys)
        for x, y in zip(xs, ys):
            assert abs(float(log.evaluate(state, x)) - float(y)) < 1e-10


class TestCubicSpline:
    def test_exact_points(self):
        xs = [0.0, 1.0, 2.0, 3.0, 4.0]
        ys = [0.0, 1.0, 4.0, 9.0, 16.0]
        state = cubic.build(xs, ys)
        for x, y in zip(xs, ys):
            assert abs(float(cubic.evaluate(state, x)) - y) < 1e-10

    def test_smooth(self):
        """Cubic spline should be smoother than linear."""
        xs = [0.0, 1.0, 2.0, 3.0]
        ys = [0.0, 1.0, 0.0, 1.0]
        state = cubic.build(xs, ys)
        # Midpoint should be smoothly interpolated, not linearly
        mid = float(cubic.evaluate(state, 0.5))
        lin_mid = 0.5  # linear interp
        assert mid != pytest.approx(lin_mid, abs=0.1)  # Different from linear

    def test_derivatives(self):
        xs = [0.0, 1.0, 2.0, 3.0]
        ys = [0.0, 1.0, 4.0, 9.0]
        state = cubic.build(xs, ys)
        d1 = float(cubic.derivative(state, 1.5))
        d2 = float(cubic.second_derivative(state, 1.5))
        # Should have non-zero derivatives
        assert d1 != 0.0


class TestBackwardFlat:
    def test_basic(self):
        state = backward_flat.build([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
        assert float(backward_flat.evaluate(state, 1.5)) == 10.0
        assert float(backward_flat.evaluate(state, 2.0)) == 20.0


class TestForwardFlat:
    def test_basic(self):
        state = forward_flat.build([1.0, 2.0, 3.0], [10.0, 20.0, 30.0])
        # Forward flat: value at x is ys[next index]
        assert float(forward_flat.evaluate(state, 1.0)) == 10.0
        assert float(forward_flat.evaluate(state, 1.5)) == 20.0
