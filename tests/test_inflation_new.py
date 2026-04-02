"""Tests for inflation instruments, cashflows, and term structures."""
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from ql_jax.time.date import Date


class TestInflationInstruments:
    def test_zero_coupon_inflation_swap(self):
        from ql_jax.instruments.inflation_swap import ZeroCouponInflationSwap
        swap = ZeroCouponInflationSwap(
            notional=1_000_000.0, fixed_rate=0.02, maturity=5.0,
            base_cpi=100.0, is_payer=True
        )
        assert swap.notional == 1_000_000.0
        assert swap.fixed_rate == 0.02

    def test_yoy_inflation_swap(self):
        from ql_jax.instruments.inflation_swap import YearOnYearInflationSwap
        swap = YearOnYearInflationSwap(
            notional=1_000_000.0, fixed_rate=0.02, maturity=5.0,
            is_payer=True
        )
        assert swap.maturity == 5.0


class TestInflationTermStructure:
    def test_zero_inflation_curve(self):
        from ql_jax.termstructures.inflation.curves import InterpolatedZeroInflationCurve
        ref_date = Date(1, 1, 2024)
        dates = [Date(1, 1, 2025), Date(1, 1, 2026), Date(1, 1, 2029), Date(1, 1, 2034)]
        rates = jnp.array([0.02, 0.022, 0.025, 0.03])
        curve = InterpolatedZeroInflationCurve(
            reference_date=ref_date, dates=dates, rates=rates
        )
        rate = float(curve.zero_rate(3.0))
        assert 0.01 < rate < 0.05

    def test_yoy_inflation_curve(self):
        from ql_jax.termstructures.inflation.curves import InterpolatedYoYInflationCurve
        ref_date = Date(1, 1, 2024)
        dates = [Date(1, 1, 2025), Date(1, 1, 2026), Date(1, 1, 2029), Date(1, 1, 2034)]
        rates = jnp.array([0.03, 0.028, 0.025, 0.022])
        curve = InterpolatedYoYInflationCurve(
            reference_date=ref_date, dates=dates, rates=rates
        )
        rate = float(curve.yoy_rate(3.0))
        assert 0.01 < rate < 0.05


class TestInflationEngine:
    def test_zc_inflation_swap_npv(self):
        from ql_jax.engines.inflation.capfloor import zero_coupon_inflation_swap_npv
        discount_fn = lambda t: jnp.exp(-0.05 * t)
        forward_cpi_fn = lambda t: 100.0 * (1.0 + 0.025) ** t
        npv = float(zero_coupon_inflation_swap_npv(
            notional=1_000_000.0, fixed_rate=0.02, maturity=5.0,
            discount_fn=discount_fn, forward_cpi_fn=forward_cpi_fn, base_cpi=100.0
        ))
        assert abs(npv) < 1_000_000.0
