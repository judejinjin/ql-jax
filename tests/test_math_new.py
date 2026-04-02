"""Tests for math subpackages: distributions, interpolations, solvers, statistics."""
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


# ── Distributions ────────────────────────────────────────────────────────────

class TestNormalDistribution:
    def test_pdf_symmetry(self):
        from ql_jax.math.distributions.normal import pdf
        assert abs(float(pdf(1.0)) - float(pdf(-1.0))) < 1e-12

    def test_cdf_bounds(self):
        from ql_jax.math.distributions.normal import cdf
        assert float(cdf(-10.0)) < 1e-10
        assert float(cdf(10.0)) > 1.0 - 1e-10
        assert abs(float(cdf(0.0)) - 0.5) < 1e-10

    def test_inverse_cdf_roundtrip(self):
        from ql_jax.math.distributions.normal import cdf, inverse_cdf
        for p in [0.1, 0.25, 0.5, 0.75, 0.9]:
            x = float(inverse_cdf(p))
            p_back = float(cdf(x))
            assert abs(p_back - p) < 1e-10


class TestBivariateNormal:
    def test_independent(self):
        from ql_jax.math.distributions.bivariate import bivariate_normal_cdf
        from ql_jax.math.distributions.normal import cdf
        # Independent (rho=0): P(X<a, Y<b) = P(X<a) * P(Y<b)
        a, b = 1.0, 0.5
        biv = float(bivariate_normal_cdf(a, b, 0.0))
        indep = float(cdf(a)) * float(cdf(b))
        assert abs(biv - indep) < 1e-6


class TestGammaDistribution:
    def test_gamma_cdf_bounds(self):
        from ql_jax.math.distributions.gamma import cdf
        assert float(cdf(0.0, 1.0, 1.0)) < 1e-10
        assert float(cdf(100.0, 1.0, 1.0)) > 0.99


class TestStudentT:
    def test_student_t_symmetric(self):
        from ql_jax.math.distributions.student_t import pdf
        assert abs(float(pdf(1.0, 5)) - float(pdf(-1.0, 5))) < 1e-10


# ── Interpolations ───────────────────────────────────────────────────────────

class TestLinearInterpolation:
    def test_basic_interpolation(self):
        from ql_jax.math.interpolations.linear import build, evaluate
        xs = jnp.array([0.0, 1.0, 2.0, 3.0])
        ys = jnp.array([0.0, 1.0, 4.0, 9.0])
        state = build(xs, ys)
        # At data point
        assert abs(float(evaluate(state, 1.0)) - 1.0) < 1e-10
        # Between points
        val = float(evaluate(state, 1.5))
        assert abs(val - 2.5) < 1e-10  # linear between (1,1) and (2,4)


class TestCubicSpline:
    def test_cubic_at_data_points(self):
        from ql_jax.math.interpolations.cubic import build, evaluate
        xs = jnp.array([0.0, 1.0, 2.0, 3.0, 4.0])
        ys = jnp.array([0.0, 1.0, 0.0, -1.0, 0.0])
        state = build(xs, ys)
        # At data points (should be exact)
        for i in range(5):
            val = float(evaluate(state, xs[i]))
            assert abs(val - float(ys[i])) < 1e-8


class TestSABRInterpolation:
    def test_sabr_vol_positive(self):
        from ql_jax.math.interpolations.sabr import sabr_vol
        vol = sabr_vol(100.0, 100.0, 1.0, 0.2, 0.5, -0.3, 0.4)
        assert float(vol) > 0.0
        assert float(vol) < 1.0  # reasonable vol


# ── Solvers ──────────────────────────────────────────────────────────────────

class TestBrentSolver:
    def test_find_root(self):
        from ql_jax.math.solvers.brent import solve
        # Solve x^2 - 2 = 0 → x = sqrt(2)
        fn = lambda x: x**2 - 2.0
        root = float(solve(fn, 1e-12, 1.0, x_min=0.0, x_max=2.0))
        assert abs(root - jnp.sqrt(2.0)) < 1e-8

    def test_find_negative_root(self):
        from ql_jax.math.solvers.brent import solve
        fn = lambda x: jnp.exp(x) - 3.0
        root = float(solve(fn, 1e-12, 1.0, x_min=0.0, x_max=5.0))
        assert abs(root - jnp.log(3.0)) < 1e-8


class TestNewtonSolver:
    def test_find_root(self):
        from ql_jax.math.solvers.newton import solve
        fn = lambda x: x**2 - 4.0
        dfn = lambda x: 2.0 * x
        root = float(solve(fn, dfn, 1e-12, 3.0))
        assert abs(root - 2.0) < 1e-8


# ── Statistics ───────────────────────────────────────────────────────────────

class TestStatistics:
    def test_mean_variance(self):
        from ql_jax.math.statistics.general import mean, variance, standard_deviation
        x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(float(mean(x)) - 3.0) < 1e-10
        assert abs(float(variance(x)) - 2.5) < 1e-10  # sample variance
        assert float(standard_deviation(x)) > 0.0

    def test_risk_measures(self):
        from ql_jax.math.statistics.risk import (
            max_drawdown,
        )
        # Simple increasing path → zero drawdown
        prices = jnp.array([100.0, 101.0, 102.0, 103.0, 104.0])
        dd = float(max_drawdown(prices))
        assert dd == 0.0 or dd < 1e-10

        # Path with drawdown
        prices2 = jnp.array([100.0, 110.0, 95.0, 105.0, 90.0])
        dd2 = float(max_drawdown(prices2))
        assert abs(dd2) > 0.0  # non-zero drawdown (may be negative by convention)


# ── Integrals ────────────────────────────────────────────────────────────────

class TestIntegrals:
    def test_simpson(self):
        from ql_jax.math.integrals.simpson import integrate
        # Integrate x^2 from 0 to 1 → 1/3
        fn = lambda x: x**2
        result = float(integrate(fn, 0.0, 1.0, n=100))
        assert abs(result - 1.0 / 3.0) < 1e-6

    def test_gauss_legendre(self):
        from ql_jax.math.integrals.gauss import integrate_gauss_legendre
        fn = lambda x: jnp.exp(x)
        result = float(integrate_gauss_legendre(fn, 0.0, 1.0, n=10))
        assert abs(result - (jnp.e - 1.0)) < 1e-8
