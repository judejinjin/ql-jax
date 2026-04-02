"""Tests for inflation term structures."""

import jax.numpy as jnp

from ql_jax.time.date import Date
from ql_jax.termstructures.inflation.curves import (
    InterpolatedZeroInflationCurve, InterpolatedYoYInflationCurve, Seasonality,
)


class TestZeroInflation:
    def test_flat_rate(self):
        ref = Date(1, 1, 2023)
        dates = [Date(1, 1, 2024), Date(1, 1, 2025)]
        rates = [0.02, 0.02]
        curve = InterpolatedZeroInflationCurve(ref, dates, rates)
        from ql_jax.time.daycounter import year_fraction
        t = year_fraction(ref, Date(1, 1, 2024), 'Actual365Fixed')
        r = float(curve.zero_rate(t))
        assert abs(r - 0.02) < 1e-6

    def test_upward_sloping(self):
        ref = Date(1, 1, 2023)
        dates = [Date(1, 1, 2024), Date(1, 1, 2030)]
        rates = [0.02, 0.03]
        curve = InterpolatedZeroInflationCurve(ref, dates, rates)
        from ql_jax.time.daycounter import year_fraction
        t1 = year_fraction(ref, Date(1, 1, 2024), 'Actual365Fixed')
        t2 = year_fraction(ref, Date(1, 1, 2030), 'Actual365Fixed')
        assert float(curve.zero_rate(t2)) > float(curve.zero_rate(t1))


class TestYoYInflation:
    def test_flat_rate(self):
        ref = Date(1, 1, 2023)
        dates = [Date(1, 1, 2024), Date(1, 1, 2025)]
        rates = [0.025, 0.025]
        curve = InterpolatedYoYInflationCurve(ref, dates, rates)
        from ql_jax.time.daycounter import year_fraction
        t = year_fraction(ref, Date(1, 1, 2024), 'Actual365Fixed')
        r = float(curve.yoy_rate(t))
        assert abs(r - 0.025) < 1e-6


class TestSeasonality:
    def test_default_no_adjustment(self):
        s = Seasonality()
        assert abs(s.correction(1) - 1.0) < 1e-12
        assert abs(s.correction(12) - 1.0) < 1e-12

    def test_custom_factors(self):
        factors = [1.01, 0.99, 1.0, 1.02, 0.98, 1.0,
                   1.01, 0.99, 1.0, 1.01, 0.99, 1.0]
        s = Seasonality(factors)
        assert abs(s.correction(1) - 1.01) < 1e-12
        assert abs(s.correction(5) - 0.98) < 1e-12
