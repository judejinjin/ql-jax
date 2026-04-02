"""Tests for statistics."""

import jax.numpy as jnp
from ql_jax.math.statistics import general, risk


class TestGeneralStatistics:
    def test_mean(self):
        samples = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(float(general.mean(samples)) - 3.0) < 1e-12

    def test_variance(self):
        samples = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert abs(float(general.variance(samples)) - 2.5) < 1e-12

    def test_std(self):
        samples = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        std = float(general.standard_deviation(samples))
        assert abs(std ** 2 - 2.5) < 1e-10

    def test_min_max(self):
        samples = jnp.array([3.0, 1.0, 4.0, 1.0, 5.0])
        assert float(general.min_value(samples)) == 1.0
        assert float(general.max_value(samples)) == 5.0

    def test_median(self):
        samples = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0])
        assert float(general.median(samples)) == 3.0

    def test_weighted_mean(self):
        samples = jnp.array([1.0, 2.0, 3.0])
        weights = jnp.array([1.0, 2.0, 1.0])
        assert abs(float(general.mean(samples, weights)) - 2.0) < 1e-12


class TestRiskStatistics:
    def test_var(self):
        # Simple: 1000 samples uniform [0, 1]
        samples = jnp.linspace(0.0, 1.0, 1000)
        var_95 = float(risk.value_at_risk(samples, 0.95))
        # VaR at 95% for uniform[0,1] should be ~-0.05 (5th percentile negated)
        assert abs(var_95 - (-0.05)) < 0.01

    def test_max_drawdown(self):
        cum_returns = jnp.array([100.0, 110.0, 105.0, 95.0, 100.0])
        mdd = float(risk.max_drawdown(cum_returns))
        assert abs(mdd - (-15.0)) < 1e-6  # 110 -> 95
