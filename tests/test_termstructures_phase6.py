"""Tests for Phase 6 term structure gaps."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Spread curves
# ---------------------------------------------------------------------------

class TestInterpolatedSimpleZeroCurve:
    def test_discount(self):
        from ql_jax.termstructures.yield_.spread_curves import InterpolatedSimpleZeroCurve
        from ql_jax.time.date import Date

        curve = InterpolatedSimpleZeroCurve(
            reference_date_=Date(1, 1, 2024),
            times=jnp.array([0.25, 0.5, 1.0, 2.0, 5.0]),
            rates=jnp.array([0.04, 0.042, 0.045, 0.048, 0.05]),
        )
        # D(t) = 1/(1 + r*t), at t=1 with r~0.045: D ~ 1/1.045
        d = curve.discount_impl(1.0)
        assert float(d) == pytest.approx(1.0 / 1.045, rel=0.02)

    def test_discount_at_zero(self):
        from ql_jax.termstructures.yield_.spread_curves import InterpolatedSimpleZeroCurve
        from ql_jax.time.date import Date
        curve = InterpolatedSimpleZeroCurve(
            reference_date_=Date(1, 1, 2024),
            times=jnp.array([0.5, 1.0]),
            rates=jnp.array([0.05, 0.05]),
        )
        d = curve.discount_impl(0.0)
        assert float(d) == pytest.approx(1.0, abs=1e-10)


class TestSpreadDiscountCurve:
    def test_spread(self):
        from ql_jax.termstructures.yield_.spread_curves import SpreadDiscountCurve
        from ql_jax.termstructures.yield_.base import YieldTermStructure
        from ql_jax.time.date import Date
        # Simple mock base curve with flat rate
        class MockCurve(YieldTermStructure):
            def __init__(self):
                super().__init__(reference_date=Date(1, 1, 2024))
            def discount_impl(self, t):
                return jnp.exp(-0.03 * t)
        base = MockCurve()
        sc = SpreadDiscountCurve(base=base, spread=0.01)
        d = sc.discount_impl(1.0)
        # D_spread(t) = D_base(t) * exp(-spread * t) = exp(-0.03)*exp(-0.01) = exp(-0.04)
        assert float(d) == pytest.approx(float(jnp.exp(-0.04)), rel=0.02)


# ---------------------------------------------------------------------------
# Equity/FX vol surfaces
# ---------------------------------------------------------------------------

class TestBlackVolSurfaceDelta:
    def test_construction(self):
        from ql_jax.termstructures.volatility.equityfx_extended import BlackVolSurfaceDelta
        expiries = jnp.array([0.25, 0.5, 1.0])
        deltas = jnp.array([0.25, 0.50, 0.75])
        vols = jnp.array([
            [0.12, 0.10, 0.11],
            [0.13, 0.11, 0.12],
            [0.14, 0.12, 0.13],
        ])
        surf = BlackVolSurfaceDelta(expiries=expiries, deltas=deltas, vols=vols, spot=100.0)
        v = surf.black_vol(0.5)
        assert float(v) > 0
        assert float(v) < 1.0

class TestLocalVolCurve:
    def test_interpolation(self):
        from ql_jax.termstructures.volatility.equityfx_extended import LocalVolCurve
        times = jnp.array([0.0, 0.5, 1.0, 2.0])
        lvols = jnp.array([0.20, 0.22, 0.21, 0.19])
        lv = LocalVolCurve(times=times, local_vols=lvols)
        val = lv.local_vol(0.75)
        assert float(val) > 0.19
        assert float(val) < 0.23


class TestFixedLocalVolSurface:
    def test_bilinear(self):
        from ql_jax.termstructures.volatility.equityfx_extended import FixedLocalVolSurface
        times = jnp.array([0.0, 0.5, 1.0])
        spots = jnp.array([80.0, 100.0, 120.0])
        lvols = jnp.array([
            [0.25, 0.20, 0.22],
            [0.24, 0.19, 0.21],
            [0.23, 0.18, 0.20],
        ])
        flvs = FixedLocalVolSurface(times=times, spots=spots, local_vols=lvols)
        val = flvs.local_vol(0.25, 100.0)
        assert float(val) > 0.15
        assert float(val) < 0.25


class TestPiecewiseBlackVarianceSurface:
    def test_vol(self):
        from ql_jax.termstructures.volatility.equityfx_extended import PiecewiseBlackVarianceSurface
        times = jnp.array([0.25, 0.5, 1.0])
        strikes = jnp.array([90.0, 100.0, 110.0])
        # Total variance = sigma^2 * t
        total_var = jnp.array([
            [0.04 * 0.25, 0.04 * 0.25, 0.04 * 0.25],
            [0.04 * 0.5, 0.04 * 0.5, 0.04 * 0.5],
            [0.04 * 1.0, 0.04 * 1.0, 0.04 * 1.0],
        ])
        surf = PiecewiseBlackVarianceSurface(times=times, strikes=strikes, total_variances=total_var)
        v = surf.black_vol(0.5, 100.0)
        assert float(v) == pytest.approx(0.20, abs=0.02)


# ---------------------------------------------------------------------------
# Inflation vol
# ---------------------------------------------------------------------------

class TestConstantCPIVolatility:
    def test_constant(self):
        from ql_jax.termstructures.volatility.inflation_vol import ConstantCPIVolatility
        cv = ConstantCPIVolatility(vol_level=0.03)
        assert float(cv.vol(1.0)) == pytest.approx(0.03)
        assert float(cv.vol(5.0, 0.02)) == pytest.approx(0.03)


class TestYoYInflationOptionletVolatility:
    def test_interpolation(self):
        from ql_jax.termstructures.volatility.inflation_vol import YoYInflationOptionletVolatilityStructure
        expiries = jnp.array([1.0, 2.0, 3.0])
        strikes = jnp.array([0.01, 0.02, 0.03])
        vols = jnp.array([
            [0.005, 0.004, 0.006],
            [0.006, 0.005, 0.007],
            [0.007, 0.006, 0.008],
        ])
        yoy_vol = YoYInflationOptionletVolatilityStructure(expiries=expiries, strikes=strikes, vols=vols)
        v = yoy_vol.vol(1.5, 0.02)
        assert float(v) > 0.004
        assert float(v) < 0.007


# ---------------------------------------------------------------------------
# Smile sections
# ---------------------------------------------------------------------------

class TestAtmSmileSection:
    def test_constant(self):
        from ql_jax.termstructures.volatility.smile_section_ext import AtmSmileSection
        ss = AtmSmileSection(atm_vol=0.20, expiry=1.0)
        assert float(ss.vol(100.0)) == pytest.approx(0.20)
        assert float(ss.vol(80.0)) == pytest.approx(0.20)
        assert ss.expiry == 1.0


class TestKahaleSmileSection:
    def test_interpolation(self):
        from ql_jax.termstructures.volatility.smile_section_ext import KahaleSmileSection
        strikes = jnp.array([80.0, 90.0, 100.0, 110.0, 120.0])
        vols = jnp.array([0.25, 0.22, 0.20, 0.21, 0.24])
        ks = KahaleSmileSection(strikes=strikes, vols=vols, forward=100.0, expiry=1.0)
        v = ks.vol(100.0)
        assert float(v) == pytest.approx(0.20, abs=0.01)


class TestZABRSmileSection:
    def test_vol(self):
        from ql_jax.termstructures.volatility.smile_section_ext import ZABRSmileSection
        zabr = ZABRSmileSection(forward=0.05, alpha=0.03, beta=0.5,
                                 nu=0.3, rho=-0.3, gamma=1.0, expiry=1.0)
        v = zabr.vol(0.05)
        assert float(v) > 0


# ---------------------------------------------------------------------------
# Default density curve
# ---------------------------------------------------------------------------

class TestInterpolatedDefaultDensityCurve:
    def test_survival(self):
        from ql_jax.termstructures.credit.default_density import InterpolatedDefaultDensityCurve
        from ql_jax.time.date import Date
        # Constant density d=0.02
        curve = InterpolatedDefaultDensityCurve(
            reference_date_=Date(1, 1, 2024),
            times=jnp.array([0.0, 1.0, 2.0, 5.0]),
            densities=jnp.array([0.02, 0.02, 0.02, 0.02]),
        )
        # S(t) = 1 - integral(0,t) 0.02 ds = 1 - 0.02*t
        assert float(curve.survival_probability(1.0)) == pytest.approx(0.98, abs=0.01)
        assert float(curve.survival_probability(0.0)) == pytest.approx(1.0, abs=0.01)

    def test_hazard_rate(self):
        from ql_jax.termstructures.credit.default_density import InterpolatedDefaultDensityCurve
        from ql_jax.time.date import Date
        curve = InterpolatedDefaultDensityCurve(
            reference_date_=Date(1, 1, 2024),
            times=jnp.array([0.0, 1.0, 5.0]),
            densities=jnp.array([0.01, 0.01, 0.01]),
        )
        h = curve.hazard_rate(1.0)
        # h(t) = d(t) / S(t) = 0.01 / (1-0.01) ~ 0.0101
        assert float(h) == pytest.approx(0.01 / 0.99, abs=0.01)


# ---------------------------------------------------------------------------
# Optionlet stripping
# ---------------------------------------------------------------------------

class TestStrippedOptionlet:
    def test_construction(self):
        from ql_jax.termstructures.volatility.optionlet_extended import StrippedOptionlet
        so = StrippedOptionlet(
            expiries=jnp.array([0.5, 1.0, 2.0]),
            strikes=jnp.array([0.01, 0.02, 0.03]),
            optionlet_vols=jnp.ones((3, 3)) * 0.20,
        )
        assert so.optionlet_vols.shape == (3, 3)


class TestStrippedOptionletAdapter:
    def test_vol_interpolation(self):
        from ql_jax.termstructures.volatility.optionlet_extended import (
            StrippedOptionlet, StrippedOptionletAdapter,
        )
        so = StrippedOptionlet(
            expiries=jnp.array([0.5, 1.0, 2.0]),
            strikes=jnp.array([0.01, 0.02, 0.03]),
            optionlet_vols=jnp.ones((3, 3)) * 0.15,
        )
        adapter = StrippedOptionletAdapter(stripped=so)
        v = adapter.vol(0.75, 0.02)
        assert float(v) == pytest.approx(0.15, abs=0.02)
