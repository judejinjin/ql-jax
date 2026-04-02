"""Tests for Phase 8 math module gaps."""

import jax
import jax.numpy as jnp
import pytest
import math

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Special functions
# ---------------------------------------------------------------------------

class TestSpecialFunctions:
    def test_factorial(self):
        from ql_jax.math.special import factorial
        assert float(factorial(0)) == pytest.approx(1.0)
        assert float(factorial(5)) == pytest.approx(120.0)
        assert float(factorial(10)) == pytest.approx(3628800.0)

    def test_double_factorial(self):
        from ql_jax.math.special import double_factorial
        # 5!! = 5*3*1 = 15
        assert float(double_factorial(5)) == pytest.approx(15.0, rel=0.01)
        # 6!! = 6*4*2 = 48
        assert float(double_factorial(6)) == pytest.approx(48.0, rel=0.01)

    def test_binomial_coefficient(self):
        from ql_jax.math.special import binomial_coefficient
        assert float(binomial_coefficient(5, 2)) == pytest.approx(10.0)
        assert float(binomial_coefficient(10, 0)) == pytest.approx(1.0)
        assert float(binomial_coefficient(10, 10)) == pytest.approx(1.0)

    def test_pascal_triangle_row(self):
        from ql_jax.math.special import pascal_triangle_row
        row = pascal_triangle_row(4)
        expected = jnp.array([1, 4, 6, 4, 1], dtype=jnp.float64)
        assert jnp.allclose(row, expected, atol=1e-10)

    def test_error_function(self):
        from ql_jax.math.special import error_function, error_function_complement
        assert float(error_function(0.0)) == pytest.approx(0.0, abs=1e-10)
        assert float(error_function(10.0)) == pytest.approx(1.0, abs=1e-10)
        assert float(error_function_complement(0.0)) == pytest.approx(1.0, abs=1e-10)
        # erf(x) + erfc(x) = 1
        x = 1.5
        assert float(error_function(x)) + float(error_function_complement(x)) == pytest.approx(1.0, abs=1e-10)

    def test_modified_bessel_first(self):
        from ql_jax.math.special import modified_bessel_first
        # I_0(0) = 1
        assert float(modified_bessel_first(0.0, 0.0)) == pytest.approx(1.0, abs=0.01)
        # I_0(1) ~ 1.2661
        assert float(modified_bessel_first(0.0, 1.0)) == pytest.approx(1.2661, abs=0.01)

    def test_incomplete_gamma(self):
        from ql_jax.math.special import incomplete_gamma
        # P(1, x) = 1 - exp(-x)
        val = incomplete_gamma(1.0, 1.0)
        assert float(val) == pytest.approx(1.0 - math.exp(-1.0), abs=0.01)


# ---------------------------------------------------------------------------
# B-spline / Bernstein
# ---------------------------------------------------------------------------

class TestBSpline:
    def test_basis_partition_of_unity(self):
        from ql_jax.math.bspline import bspline_values
        knots = jnp.array([0, 0, 0, 0, 1, 2, 3, 3, 3, 3], dtype=jnp.float64)
        x = 1.5
        vals = bspline_values(knots, degree=3, x=x)
        # Partition of unity: sum of all basis values = 1
        assert float(jnp.sum(vals)) == pytest.approx(1.0, abs=0.01)

    def test_bernstein_polynomial(self):
        from ql_jax.math.bspline import bernstein_polynomial
        # B_{0,1}(0) = 1, B_{0,1}(1) = 0
        assert float(bernstein_polynomial(1, 0, 0.0)) == pytest.approx(1.0, abs=1e-10)
        assert float(bernstein_polynomial(1, 0, 1.0)) == pytest.approx(0.0, abs=1e-10)
        # B_{1,1}(0) = 0, B_{1,1}(1) = 1
        assert float(bernstein_polynomial(1, 1, 0.0)) == pytest.approx(0.0, abs=1e-10)
        assert float(bernstein_polynomial(1, 1, 1.0)) == pytest.approx(1.0, abs=1e-10)

    def test_bernstein_expansion(self):
        from ql_jax.math.bspline import bernstein_expansion
        # Linear polynomial: f(x) = 2x => coeffs [0, 2] in Bernstein form for degree 1
        coeffs = jnp.array([0.0, 2.0])
        assert float(bernstein_expansion(coeffs, 0.5)) == pytest.approx(1.0, abs=0.01)


# ---------------------------------------------------------------------------
# FFT pricing
# ---------------------------------------------------------------------------

class TestFFT:
    def test_fft_ifft_roundtrip(self):
        from ql_jax.math.fft import fft, ifft
        x = jnp.array([1.0, 2.0, 3.0, 4.0]) + 0j
        y = ifft(fft(x))
        assert jnp.allclose(jnp.real(y), jnp.real(x), atol=1e-10)

    def test_fft_heston_call(self):
        from ql_jax.math.fft import fft_price_heston_call
        price = fft_price_heston_call(
            S=100, K=100, r=0.05, q=0.0, T=1.0,
            v0=0.04, kappa=1.5, theta=0.04, sigma=0.3, rho=-0.7,
            N=4096, eta=0.25, alpha=1.5,
        )
        # Should give a positive call price
        assert float(price) > 0
        assert float(price) < 50


# ---------------------------------------------------------------------------
# Richardson extrapolation
# ---------------------------------------------------------------------------

class TestRichardson:
    def test_extrapolation(self):
        from ql_jax.math.richardson import richardson_extrapolation
        # f(h) = sin(h)/h approaches 1 as h->0
        f = lambda h: jnp.sin(h) / h
        val = richardson_extrapolation(f, x=0.0, h=0.1, order=1)
        assert float(val) == pytest.approx(1.0, abs=0.01)

    def test_numerical_differentiation(self):
        from ql_jax.math.richardson import numerical_differentiation
        # d/dx sin(x) = cos(x) at x = pi/4
        f = lambda x: jnp.sin(x)
        deriv = numerical_differentiation(f, jnp.pi / 4, h=1e-5, order=1, method='central')
        assert float(deriv) == pytest.approx(float(jnp.cos(jnp.pi / 4)), abs=1e-6)

    def test_second_derivative(self):
        from ql_jax.math.richardson import numerical_differentiation
        f = lambda x: x ** 3
        # d2/dx2 x^3 = 6x, at x=2 => 12
        d2 = numerical_differentiation(f, 2.0, h=1e-4, order=2, method='central')
        assert float(d2) == pytest.approx(12.0, abs=0.01)


# ---------------------------------------------------------------------------
# Regression
# ---------------------------------------------------------------------------

class TestRegression:
    def test_linear_fit(self):
        from ql_jax.math.regression import general_linear_least_squares
        # y = 2x + 1
        x = jnp.arange(10, dtype=jnp.float64)
        y = 2.0 * x + 1.0
        X = jnp.column_stack([jnp.ones_like(x), x])
        beta = general_linear_least_squares(X, y)
        assert float(beta[0]) == pytest.approx(1.0, abs=1e-10)
        assert float(beta[1]) == pytest.approx(2.0, abs=1e-10)

    def test_polynomial_regression(self):
        from ql_jax.math.regression import polynomial_regression
        x = jnp.linspace(0, 1, 50)
        y = 3.0 * x ** 2 + 2.0 * x + 1.0
        coeffs = polynomial_regression(x, y, degree=2)
        assert float(coeffs[0]) == pytest.approx(1.0, abs=0.01)
        assert float(coeffs[1]) == pytest.approx(2.0, abs=0.01)
        assert float(coeffs[2]) == pytest.approx(3.0, abs=0.01)

    def test_weighted_regression(self):
        from ql_jax.math.regression import weighted_linear_least_squares
        x = jnp.arange(5, dtype=jnp.float64)
        y = 3.0 * x + 1.0
        X = jnp.column_stack([jnp.ones_like(x), x])
        w = jnp.ones(5)
        beta = weighted_linear_least_squares(X, y, w)
        assert float(beta[0]) == pytest.approx(1.0, abs=1e-8)
        assert float(beta[1]) == pytest.approx(3.0, abs=1e-8)


# ---------------------------------------------------------------------------
# Advanced integrals
# ---------------------------------------------------------------------------

class TestAdvancedIntegrals:
    def test_gauss_lobatto(self):
        from ql_jax.math.integrals_advanced import gauss_lobatto_integral
        # Integrate x^2 from 0 to 1 => 1/3
        result = gauss_lobatto_integral(lambda x: x ** 2, 0.0, 1.0, n=7)
        assert float(result) == pytest.approx(1.0 / 3, abs=1e-6)

    def test_gauss_kronrod(self):
        from ql_jax.math.integrals_advanced import gauss_kronrod_integral
        result, err = gauss_kronrod_integral(lambda x: jnp.exp(x), 0.0, 1.0)
        assert float(result) == pytest.approx(jnp.e - 1, abs=1e-6)
        assert float(err) < 1e-6

    def test_tanh_sinh(self):
        from ql_jax.math.integrals_advanced import tanh_sinh_integral
        # sin(x) from 0 to pi => 2
        result = tanh_sinh_integral(lambda x: jnp.sin(x), 0.0, jnp.pi, n=50)
        assert float(result) == pytest.approx(2.0, abs=0.01)

    def test_two_dimensional(self):
        from ql_jax.math.integrals_advanced import two_dimensional_integral
        # Integrate 1 over [0,1]x[0,1] => 1
        result = two_dimensional_integral(lambda x, y: 1.0, 0.0, 1.0, 0.0, 1.0)
        assert float(result) == pytest.approx(1.0, abs=1e-6)

    def test_discrete_trapezoid(self):
        from ql_jax.math.integrals_advanced import discrete_integral
        x = jnp.linspace(0, 1, 1001)
        y = x ** 2
        result = discrete_integral(x, y, method='trapezoid')
        assert float(result) == pytest.approx(1.0 / 3, abs=1e-4)


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

class TestIncrementalStatistics:
    def test_mean_variance(self):
        from ql_jax.math.statistics.advanced import IncrementalStatistics
        stats = IncrementalStatistics()
        data = [1.0, 2.0, 3.0, 4.0, 5.0]
        for x in data:
            stats.add(x)
        assert stats.count == 5
        assert stats.mean == pytest.approx(3.0)
        assert stats.variance == pytest.approx(2.5, rel=0.01)  # sample variance

    def test_min_max(self):
        from ql_jax.math.statistics.advanced import IncrementalStatistics
        stats = IncrementalStatistics()
        for x in [5.0, 1.0, 3.0, 9.0, 2.0]:
            stats.add(x)
        assert stats.min_value == pytest.approx(1.0)
        assert stats.max_value == pytest.approx(9.0)

    def test_batch(self):
        from ql_jax.math.statistics.advanced import IncrementalStatistics
        stats = IncrementalStatistics()
        stats.add_batch(jnp.array([10.0, 20.0, 30.0]))
        assert stats.count == 3
        assert stats.mean == pytest.approx(20.0)


class TestHistogram:
    def test_basic(self):
        from ql_jax.math.statistics.advanced import Histogram
        h = Histogram(bins=5, range_=(0, 10))
        h.add(jnp.array([1.0, 3.0, 5.0, 7.0, 9.0, 2.0, 4.0]))
        assert h.counts.shape[0] == 5
        assert jnp.sum(h.counts) == 7


class TestDiscrepancy:
    def test_low_discrepancy(self):
        from ql_jax.math.statistics.advanced import discrepancy
        # Regular grid in 2D should have low discrepancy
        n = 100
        x = jnp.linspace(0.01, 0.99, 10)
        pts = jnp.stack(jnp.meshgrid(x, x), axis=-1).reshape(-1, 2)
        d = discrepancy(pts)
        assert float(d) < 0.05  # Low discrepancy for regular grid


# ---------------------------------------------------------------------------
# Halton sequence
# ---------------------------------------------------------------------------

class TestHalton:
    def test_element(self):
        from ql_jax.math.random.halton import halton_element
        # Halton base 2: 1/2, 1/4, 3/4, 1/8, ...
        assert halton_element(1, 2) == pytest.approx(0.5)
        assert halton_element(2, 2) == pytest.approx(0.25)
        assert halton_element(3, 2) == pytest.approx(0.75)

    def test_sequence_shape(self):
        from ql_jax.math.random.halton import halton_sequence
        seq = halton_sequence(100, 3)
        assert seq.shape == (100, 3)
        # All values in (0, 1)
        assert jnp.all(seq > 0)
        assert jnp.all(seq < 1)

    def test_normal_shape(self):
        from ql_jax.math.random.halton import halton_normal
        seq = halton_normal(100, 2)
        assert seq.shape == (100, 2)
        # Normal samples: mean should be near 0
        assert abs(float(jnp.mean(seq))) < 0.5
