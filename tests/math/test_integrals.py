"""Tests for numerical integration."""

import jax.numpy as jnp
import math

from ql_jax.math.integrals import gauss, simpson, trapezoid


class TestGaussLegendre:
    def test_polynomial(self):
        # Integrate x^2 from 0 to 1 = 1/3
        result = gauss.integrate_gauss_legendre(lambda x: x ** 2, 0.0, 1.0, n=5)
        assert abs(float(result) - 1.0 / 3.0) < 1e-12

    def test_sin(self):
        # Integrate sin(x) from 0 to pi = 2
        result = gauss.integrate_gauss_legendre(jnp.sin, 0.0, jnp.pi, n=16)
        assert abs(float(result) - 2.0) < 1e-12

    def test_exponential(self):
        # Integrate e^x from 0 to 1 = e - 1
        result = gauss.integrate_gauss_legendre(jnp.exp, 0.0, 1.0, n=16)
        assert abs(float(result) - (math.e - 1.0)) < 1e-12


class TestSimpson:
    def test_polynomial(self):
        result = simpson.integrate(lambda x: x ** 2, 0.0, 1.0, n=100)
        assert abs(float(result) - 1.0 / 3.0) < 1e-8

    def test_sin(self):
        result = simpson.integrate(jnp.sin, 0.0, jnp.pi, n=100)
        assert abs(float(result) - 2.0) < 1e-7


class TestTrapezoid:
    def test_polynomial(self):
        result = trapezoid.integrate(lambda x: x ** 2, 0.0, 1.0, n=1000)
        assert abs(float(result) - 1.0 / 3.0) < 1e-5
