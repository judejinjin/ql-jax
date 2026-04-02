"""Tests for 1D root solvers."""

import math
from ql_jax.math.solvers import brent, newton, bisection, secant


class TestBrent:
    def test_simple_root(self):
        # x^2 - 4 = 0 => x = 2
        root = brent.solve(lambda x: x ** 2 - 4, 1e-12, guess=1.0, step=0.5)
        assert abs(root - 2.0) < 1e-10

    def test_bracketed(self):
        root = brent.solve(lambda x: x ** 2 - 4, 1e-12, guess=1.0, x_min=0.0, x_max=5.0)
        assert abs(root - 2.0) < 1e-10

    def test_cos(self):
        # cos(x) = 0 near pi/2
        root = brent.solve(math.cos, 1e-12, guess=1.0, x_min=0.0, x_max=3.0)
        assert abs(root - math.pi / 2) < 1e-10


class TestNewton:
    def test_simple_root(self):
        root = newton.solve(
            lambda x: x ** 2 - 4,
            lambda x: 2 * x,
            1e-12,
            guess=1.0,
        )
        assert abs(root - 2.0) < 1e-10

    def test_bracketed(self):
        root = newton.solve(
            lambda x: x ** 2 - 4,
            lambda x: 2 * x,
            1e-12,
            guess=1.0,
            x_min=0.0,
            x_max=5.0,
        )
        assert abs(root - 2.0) < 1e-10


class TestBisection:
    def test_simple(self):
        root = bisection.solve(lambda x: x ** 2 - 4, 1e-12, 0.0, 5.0)
        assert abs(root - 2.0) < 1e-10


class TestSecant:
    def test_simple(self):
        root = secant.solve(lambda x: x ** 2 - 4, 1e-10, guess=1.0, step=0.5)
        assert abs(root - 2.0) < 1e-8
