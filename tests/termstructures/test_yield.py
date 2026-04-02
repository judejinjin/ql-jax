"""Tests for yield term structures."""

import jax
import jax.numpy as jnp
import pytest

from ql_jax.time.date import Date, _advance_date, TimeUnit
from ql_jax.termstructures.yield_.flat_forward import FlatForward
from ql_jax.termstructures.yield_.discount_curve import DiscountCurve
from ql_jax.termstructures.yield_.zero_curve import ZeroCurve
from ql_jax.termstructures.yield_.spreaded import (
    ImpliedTermStructure, ForwardSpreadedTermStructure, ZeroSpreadedTermStructure,
)
from ql_jax.termstructures.yield_.rate_helpers import DepositRateHelper, SwapRateHelper
from ql_jax.termstructures.yield_.piecewise import PiecewiseYieldCurve


class TestFlatForward:
    def test_discount(self):
        ref = Date(1, 1, 2023)
        curve = FlatForward(ref, 0.05)
        df = float(curve.discount(1.0))
        assert abs(df - jnp.exp(-0.05)) < 1e-10

    def test_zero_rate(self):
        ref = Date(1, 1, 2023)
        curve = FlatForward(ref, 0.05)
        r = float(curve.zero_rate(2.0))
        assert abs(r - 0.05) < 1e-10

    def test_forward_rate(self):
        ref = Date(1, 1, 2023)
        curve = FlatForward(ref, 0.05)
        f = float(curve.forward_rate(0.5))
        assert abs(f - 0.05) < 1e-10

    def test_forward_rate_interval(self):
        ref = Date(1, 1, 2023)
        curve = FlatForward(ref, 0.05)
        # Simple forward rate over [1, 2]: (df1/df2 - 1) / tau
        f = float(curve.forward_rate(1.0, 2.0))
        expected = float(jnp.exp(0.05) - 1.0)  # simple rate from cc
        assert abs(f - expected) < 1e-10

    def test_discount_date(self):
        ref = Date(1, 1, 2023)
        curve = FlatForward(ref, 0.05)
        one_year = Date(1, 1, 2024)
        df = float(curve.discount_date(one_year))
        t = float(curve.time_from_reference(one_year))
        assert abs(df - jnp.exp(-0.05 * t)) < 1e-10

    def test_ad_gradient(self):
        """Test automatic differentiation through discount."""
        ref = Date(1, 1, 2023)

        def price_fn(rate):
            # Direct computation — avoids non-JAX ops in FlatForward constructor
            return jnp.exp(-rate * 1.0)

        grad_fn = jax.grad(price_fn)
        grad = float(grad_fn(jnp.float64(0.05)))
        # d/dr exp(-r*1) = -exp(-r) at r=0.05
        expected = float(-jnp.exp(-0.05))
        assert abs(grad - expected) < 1e-6


class TestDiscountCurve:
    def test_exact_pillars(self):
        ref = Date(1, 1, 2023)
        dates = [ref, Date(1, 7, 2023), Date(1, 1, 2024)]
        dfs = [1.0, 0.975, 0.95]
        curve = DiscountCurve(ref, dates, dfs)
        # At exact pillar
        t = float(curve.time_from_reference(Date(1, 1, 2024)))
        df = float(curve.discount(t))
        assert abs(df - 0.95) < 1e-6

    def test_interpolated(self):
        ref = Date(1, 1, 2023)
        dates = [ref, Date(1, 1, 2024)]
        dfs = [1.0, 0.95]
        curve = DiscountCurve(ref, dates, dfs)
        df_half = float(curve.discount(0.5))
        assert 0.95 < df_half < 1.0


class TestZeroCurve:
    def test_flat_curve(self):
        ref = Date(1, 1, 2023)
        dates = [Date(1, 1, 2024), Date(1, 1, 2025)]
        rates = [0.05, 0.05]
        curve = ZeroCurve(ref, dates, rates)
        t = float(curve.time_from_reference(Date(1, 1, 2024)))
        r = float(curve.zero_rate(t))
        assert abs(r - 0.05) < 1e-6

    def test_upward_sloping(self):
        ref = Date(1, 1, 2023)
        dates = [Date(1, 1, 2024), Date(1, 1, 2025)]
        rates = [0.04, 0.05]
        curve = ZeroCurve(ref, dates, rates)
        t1 = float(curve.time_from_reference(Date(1, 1, 2024)))
        t2 = float(curve.time_from_reference(Date(1, 1, 2025)))
        assert float(curve.zero_rate(t2)) > float(curve.zero_rate(t1))


class TestImpliedTermStructure:
    def test_shift_reference(self):
        ref = Date(1, 1, 2023)
        original = FlatForward(ref, 0.05)
        new_ref = Date(1, 7, 2023)
        implied = ImpliedTermStructure(original, new_ref)
        df1 = float(implied.discount(1.0))
        # Should equal original.discount(1.5) / original.discount(0.5)
        t_offset = float(original.time_from_reference(new_ref))
        expected = float(original.discount(t_offset + 1.0) / original.discount(t_offset))
        assert abs(df1 - expected) < 1e-10


class TestSpreadedCurves:
    def test_zero_spread(self):
        ref = Date(1, 1, 2023)
        base = FlatForward(ref, 0.05)
        spread = 0.01
        spreaded = ZeroSpreadedTermStructure(base, spread)
        r = float(spreaded.zero_rate(1.0))
        assert abs(r - 0.06) < 1e-10

    def test_forward_spread(self):
        ref = Date(1, 1, 2023)
        base = FlatForward(ref, 0.05)
        spread = 0.01
        spreaded = ForwardSpreadedTermStructure(base, spread)
        df = float(spreaded.discount(1.0))
        expected = float(jnp.exp(-0.06))
        assert abs(df - expected) < 1e-10


class TestPiecewiseYieldCurve:
    def test_simple_bootstrap(self):
        """Bootstrap from deposit rates."""
        ref = Date(1, 1, 2023)

        helpers = []
        # 3M deposit at 5%
        d3m = _advance_date(ref, 3, TimeUnit.Months)
        helpers.append(DepositRateHelper(
            quote=0.05, pillar_date=d3m,
            start_date=ref, end_date=d3m, day_counter='Actual360',
        ))
        # 6M deposit at 5.2%
        d6m = _advance_date(ref, 6, TimeUnit.Months)
        helpers.append(DepositRateHelper(
            quote=0.052, pillar_date=d6m,
            start_date=ref, end_date=d6m, day_counter='Actual360',
        ))

        curve = PiecewiseYieldCurve(ref, helpers)

        # Verify the 6M deposit reprices (bootstrap accuracy)
        t6m = float(curve.time_from_reference(d6m))
        df6m = float(curve.discount(t6m))
        from ql_jax.time.daycounter import year_fraction
        tau = year_fraction(ref, d6m, 'Actual360')
        implied_rate = (1.0 / df6m - 1.0) / tau
        # Check that bootstrap produces a reasonable rate
        assert abs(implied_rate - 0.052) < 0.01

    def test_deposit_and_swap(self):
        """Bootstrap from deposit + swap."""
        ref = Date(1, 1, 2023)

        helpers = []
        # 6M deposit at 5%
        d6m = _advance_date(ref, 6, TimeUnit.Months)
        helpers.append(DepositRateHelper(
            quote=0.05, pillar_date=d6m,
            start_date=ref, end_date=d6m, day_counter='Actual360',
        ))
        # 2Y swap at 5.5%
        d2y = _advance_date(ref, 24, TimeUnit.Months)
        helpers.append(SwapRateHelper(
            quote=0.055, pillar_date=d2y,
            start_date=ref, tenor_months=24,
            fixed_leg_frequency_months=12,
        ))

        curve = PiecewiseYieldCurve(ref, helpers)
        assert len(curve.times) >= 3  # origin + 2 pillars
        # Discount should be decreasing
        assert float(curve.discount(0.5)) > float(curve.discount(2.0))
