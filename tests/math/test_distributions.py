"""Tests for probability distributions."""

import jax.numpy as jnp
import math

from ql_jax.math.distributions import normal


class TestNormalDistribution:
    def test_pdf_at_zero(self):
        expected = 1.0 / math.sqrt(2.0 * math.pi)
        assert abs(float(normal.pdf(0.0)) - expected) < 1e-12

    def test_cdf_at_zero(self):
        assert abs(float(normal.cdf(0.0)) - 0.5) < 1e-12

    def test_cdf_symmetry(self):
        assert abs(float(normal.cdf(1.0) + normal.cdf(-1.0)) - 1.0) < 1e-12

    def test_inverse_cdf_roundtrip(self):
        for p in [0.01, 0.1, 0.25, 0.5, 0.75, 0.9, 0.99]:
            x = float(normal.inverse_cdf(p))
            p_back = float(normal.cdf(x))
            assert abs(p_back - p) < 1e-10

    def test_custom_mean_sigma(self):
        assert abs(float(normal.cdf(5.0, mean=5.0, sigma=1.0)) - 0.5) < 1e-12

    def test_bivariate_uncorrelated(self):
        """Bivariate CDF with rho=0 should equal product of marginal CDFs."""
        x, y = 1.0, 1.0
        biv = float(normal.bivariate_cdf(x, y, 0.0))
        expected = float(normal.cdf(x)) * float(normal.cdf(y))
        assert abs(biv - expected) < 1e-6
