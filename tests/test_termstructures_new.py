"""Tests for term structure modules: yield, volatility, credit, inflation."""
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from ql_jax.time.date import Date


# ── Yield term structures ────────────────────────────────────────────────────

class TestFlatForward:
    def test_discount(self):
        from ql_jax.termstructures.yield_.flat_forward import FlatForward
        curve = FlatForward(reference_date=Date(1, 1, 2024), rate=0.05)
        assert abs(float(curve.discount(1.0)) - float(jnp.exp(-0.05))) < 1e-10

    def test_zero_rate(self):
        from ql_jax.termstructures.yield_.flat_forward import FlatForward
        curve = FlatForward(reference_date=Date(1, 1, 2024), rate=0.05)
        assert abs(float(curve.zero_rate(2.0)) - 0.05) < 1e-10


class TestZeroCurve:
    def test_interpolation(self):
        from ql_jax.termstructures.yield_.zero_curve import ZeroCurve
        ref = Date(1, 1, 2024)
        dates = [Date(1, 4, 2024), Date(1, 7, 2024), Date(1, 1, 2025),
                 Date(1, 1, 2026), Date(1, 1, 2029)]
        rates = [0.02, 0.025, 0.03, 0.035, 0.04]
        curve = ZeroCurve(reference_date=ref, dates=dates, zero_rates=rates)
        r = float(curve.zero_rate(1.5))
        assert 0.02 <= r <= 0.04

    def test_discount_from_zero_rate(self):
        from ql_jax.termstructures.yield_.zero_curve import ZeroCurve
        ref = Date(1, 1, 2024)
        dates = [Date(1, 1, 2025), Date(1, 1, 2026)]
        rates = [0.05, 0.05]
        curve = ZeroCurve(reference_date=ref, dates=dates, zero_rates=rates)
        d = float(curve.discount(1.0))
        assert abs(d - float(jnp.exp(-0.05))) < 0.01


class TestFittedBondCurve:
    def test_nelson_siegel(self):
        from ql_jax.termstructures.yield_.fitted_bond_curve import NelsonSiegel
        ns = NelsonSiegel(params=jnp.array([0.06, -0.02, 0.01, 1.5]))
        r = float(ns.zero_rate(5.0))
        assert 0.03 < r < 0.07


# ── Volatility term structures ───────────────────────────────────────────────

class TestBlackConstantVol:
    def test_vol(self):
        from ql_jax.termstructures.volatility.black_vol import BlackConstantVol
        vol_ts = BlackConstantVol(reference_date=Date(1, 1, 2024), vol=0.20)
        assert abs(float(vol_ts.black_vol(1.0, 100.0)) - 0.20) < 1e-10


class TestSmileSection:
    def test_flat_smile(self):
        from ql_jax.termstructures.volatility.smile_section import FlatSmileSection
        smile = FlatSmileSection(expiry_time=1.0, vol=0.20, forward=100.0)
        assert abs(float(smile.volatility(105.0)) - 0.20) < 1e-10


# ── Credit term structures ───────────────────────────────────────────────────

class TestDefaultCurves:
    def test_flat_hazard(self):
        from ql_jax.termstructures.credit.default_curves import FlatHazardRate
        curve = FlatHazardRate(reference_date=Date(1, 1, 2024), hazard_rate=0.02)
        sp = float(curve.survival_probability(5.0))
        assert abs(sp - float(jnp.exp(-0.02 * 5.0))) < 1e-10

    def test_default_probability(self):
        from ql_jax.termstructures.credit.default_curves import FlatHazardRate
        curve = FlatHazardRate(reference_date=Date(1, 1, 2024), hazard_rate=0.02)
        dp = float(curve.default_probability(5.0))
        assert abs(dp - (1.0 - float(jnp.exp(-0.02 * 5.0)))) < 1e-10


# ── Inflation term structures ────────────────────────────────────────────────

class TestInflationCurves:
    def test_zero_inflation_curve(self):
        from ql_jax.termstructures.inflation.curves import InterpolatedZeroInflationCurve
        ref = Date(1, 1, 2024)
        dates = [Date(1, 1, 2025), Date(1, 1, 2026), Date(1, 1, 2029), Date(1, 1, 2034)]
        rates = [0.02, 0.022, 0.025, 0.028]
        curve = InterpolatedZeroInflationCurve(reference_date=ref, dates=dates, rates=rates)
        r = float(curve.zero_rate(3.0))
        assert 0.02 < r < 0.03

    def test_yoy_inflation_curve(self):
        from ql_jax.termstructures.inflation.curves import InterpolatedYoYInflationCurve
        ref = Date(1, 1, 2024)
        dates = [Date(1, 1, 2025), Date(1, 1, 2026), Date(1, 1, 2029)]
        rates = [0.03, 0.025, 0.02]
        curve = InterpolatedYoYInflationCurve(reference_date=ref, dates=dates, rates=rates)
        r = float(curve.yoy_rate(3.0))
        assert 0.01 < r < 0.04
