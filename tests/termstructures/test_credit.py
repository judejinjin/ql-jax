"""Tests for credit term structures."""

import jax.numpy as jnp

from ql_jax.time.date import Date
from ql_jax.termstructures.credit.default_curves import (
    FlatHazardRate, InterpolatedSurvivalProbabilityCurve, PiecewiseDefaultCurve,
)


class TestFlatHazardRate:
    def test_survival_probability(self):
        ref = Date(1, 1, 2023)
        h = 0.02  # 2% hazard rate
        curve = FlatHazardRate(ref, h)
        sp = float(curve.survival_probability(1.0))
        assert abs(sp - jnp.exp(-0.02)) < 1e-10

    def test_default_probability(self):
        ref = Date(1, 1, 2023)
        curve = FlatHazardRate(ref, 0.02)
        pd = float(curve.default_probability(5.0))
        assert abs(pd - (1.0 - jnp.exp(-0.1))) < 1e-10

    def test_hazard_rate(self):
        ref = Date(1, 1, 2023)
        curve = FlatHazardRate(ref, 0.03)
        h = float(curve.hazard_rate(2.0))
        assert abs(h - 0.03) < 1e-10

    def test_default_density(self):
        ref = Date(1, 1, 2023)
        curve = FlatHazardRate(ref, 0.02)
        d = float(curve.default_density(1.0))
        expected = 0.02 * float(jnp.exp(-0.02))
        assert abs(d - expected) < 1e-10


class TestInterpolatedSurvivalCurve:
    def test_exact_pillar(self):
        ref = Date(1, 1, 2023)
        dates = [Date(1, 1, 2024), Date(1, 1, 2025), Date(1, 1, 2028)]
        surv = [0.99, 0.97, 0.90]
        curve = InterpolatedSurvivalProbabilityCurve(ref, dates, surv)
        from ql_jax.time.daycounter import year_fraction
        t = year_fraction(ref, Date(1, 1, 2024), 'Actual365Fixed')
        sp = float(curve.survival_probability(t))
        assert abs(sp - 0.99) < 1e-4

    def test_monotone_decreasing(self):
        ref = Date(1, 1, 2023)
        dates = [Date(1, 1, 2024), Date(1, 1, 2025)]
        surv = [0.98, 0.95]
        curve = InterpolatedSurvivalProbabilityCurve(ref, dates, surv)
        from ql_jax.time.daycounter import year_fraction
        t1 = year_fraction(ref, Date(1, 1, 2024), 'Actual365Fixed')
        t2 = year_fraction(ref, Date(1, 1, 2025), 'Actual365Fixed')
        assert float(curve.survival_probability(t1)) > float(curve.survival_probability(t2))
