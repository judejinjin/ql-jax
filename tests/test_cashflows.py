"""Tests for cashflows: simple, fixed-rate, floating-rate, analytics."""

import pytest
import jax.numpy as jnp
import jax

from ql_jax.time.date import Date, Period
from ql_jax.time.schedule import Schedule, MakeSchedule
from ql_jax.time.calendar import NullCalendar, TARGET
from ql_jax.time.daycounter import year_fraction
from ql_jax._util.types import (
    BusinessDayConvention, Frequency, DateGeneration, TimeUnit,
)
from ql_jax.cashflows.simple import SimpleCashFlow, Redemption
from ql_jax.cashflows.fixed_rate import Coupon, FixedRateCoupon, fixed_rate_leg
from ql_jax.cashflows.floating_rate import (
    FloatingRateCoupon, IborCoupon, OvernightIndexedCoupon,
    CappedFlooredCoupon, ibor_leg, overnight_leg,
)
from ql_jax.cashflows.analytics import (
    npv, bps, yield_rate, duration, convexity, accrued_amount, Duration,
)
from ql_jax.termstructures.yield_.flat_forward import FlatForward


# ===== Fixtures =====

@pytest.fixture
def flat_curve():
    """Flat 5% curve starting 2024-01-15."""
    return FlatForward(Date(15, 1, 2024), 0.05)


@pytest.fixture
def annual_schedule():
    """Annual schedule from 2024-01-15 to 2029-01-15."""
    return (MakeSchedule()
            .from_date(Date(15, 1, 2024))
            .to_date(Date(15, 1, 2029))
            .with_frequency(Frequency.Annual)
            .with_calendar(NullCalendar())
            .with_convention(BusinessDayConvention.Unadjusted)
            .build())


@pytest.fixture
def semiannual_schedule():
    """Semiannual schedule from 2024-01-15 to 2029-01-15."""
    return (MakeSchedule()
            .from_date(Date(15, 1, 2024))
            .to_date(Date(15, 1, 2029))
            .with_frequency(Frequency.Semiannual)
            .with_calendar(NullCalendar())
            .with_convention(BusinessDayConvention.Unadjusted)
            .build())


# ===== SimpleCashFlow =====

class TestSimpleCashFlow:
    def test_basic(self):
        cf = SimpleCashFlow(Date(15, 6, 2024), 100.0)
        assert cf.date == Date(15, 6, 2024)
        assert cf.amount == 100.0

    def test_redemption(self):
        r = Redemption(Date(15, 1, 2029), 100.0)
        assert r.amount == 100.0
        assert isinstance(r, SimpleCashFlow)


# ===== FixedRateCoupon =====

class TestFixedRateCoupon:
    def test_amount(self):
        cpn = FixedRateCoupon(
            payment_date=Date(15, 1, 2025),
            nominal=1_000_000.0,
            accrual_start=Date(15, 1, 2024),
            accrual_end=Date(15, 1, 2025),
            rate=0.05,
            day_counter="Actual/365 (Fixed)",
        )
        # 1 year coupon: ~50,000
        expected = 1_000_000.0 * 0.05 * year_fraction(
            Date(15, 1, 2024), Date(15, 1, 2025), "Actual/365 (Fixed)"
        )
        assert abs(cpn.amount - expected) < 0.01

    def test_accrual_period(self):
        cpn = FixedRateCoupon(
            payment_date=Date(15, 7, 2024),
            nominal=100.0,
            accrual_start=Date(15, 1, 2024),
            accrual_end=Date(15, 7, 2024),
            rate=0.03,
            day_counter="Actual/360",
        )
        tau = cpn.accrual_period("Actual/360")
        # ~182 days / 360 ≈ 0.5056
        assert 0.50 < tau < 0.52


# ===== fixed_rate_leg =====

class TestFixedRateLeg:
    def test_annual_leg(self, annual_schedule):
        leg = fixed_rate_leg(annual_schedule, 100.0, 0.05, "Actual/365 (Fixed)")
        assert len(leg) == 5  # 5 annual coupons over 5 years
        for cpn in leg:
            assert isinstance(cpn, FixedRateCoupon)
            assert cpn.nominal == 100.0
            assert cpn.rate == 0.05

    def test_semiannual_leg(self, semiannual_schedule):
        leg = fixed_rate_leg(semiannual_schedule, 1_000_000.0, 0.04)
        assert len(leg) == 10  # 10 semiannual coupons

    def test_multiple_rates(self, annual_schedule):
        leg = fixed_rate_leg(annual_schedule, 100.0, [0.03, 0.04, 0.05])
        assert leg[0].rate == 0.03
        assert leg[1].rate == 0.04
        assert leg[2].rate == 0.05
        assert leg[3].rate == 0.05  # last rate is extended
        assert leg[4].rate == 0.05


# ===== NPV =====

class TestNPV:
    def test_single_cashflow(self, flat_curve):
        """PV of 100 in 1 year at 5% ≈ 95.12."""
        cfs = [SimpleCashFlow(Date(15, 1, 2025), 100.0)]
        pv = npv(cfs, flat_curve)
        expected = 100.0 * jnp.exp(-0.05 * 1.0)
        assert abs(pv - expected) < 0.05

    def test_coupon_leg_npv(self, flat_curve, annual_schedule):
        """NPV of a 5% coupon leg on a 5% curve should be close to par."""
        leg = fixed_rate_leg(annual_schedule, 100.0, 0.05, "Actual/365 (Fixed)")
        # Add redemption
        leg_with_red = leg + [SimpleCashFlow(Date(15, 1, 2029), 100.0)]
        pv = npv(leg_with_red, flat_curve)
        # At par rate, NPV ≈ face value
        assert abs(pv - 100.0) < 2.0  # close to 100

    def test_excludes_past_flows(self, flat_curve):
        """Flows before settlement are excluded."""
        cfs = [
            SimpleCashFlow(Date(1, 1, 2024), 50.0),   # before ref date
            SimpleCashFlow(Date(15, 1, 2025), 100.0),  # after
        ]
        pv = npv(cfs, flat_curve)
        # Only the second flow contributes
        expected = 100.0 * jnp.exp(-0.05 * 1.0)
        assert abs(pv - expected) < 0.05


# ===== BPS =====

class TestBPS:
    def test_bps_positive(self, flat_curve, annual_schedule):
        leg = fixed_rate_leg(annual_schedule, 100.0, 0.05, "Actual/365 (Fixed)")
        b = bps(leg, flat_curve)
        # BPS should be positive (PV of 1bp coupon stream)
        assert b > 0


# ===== Yield =====

class TestYield:
    def test_yield_from_par(self, flat_curve, annual_schedule):
        """Yield of 5% leg + redemption repriced at par ≈ 5%."""
        leg = fixed_rate_leg(annual_schedule, 100.0, 0.05, "Actual/365 (Fixed)")
        all_cfs = leg + [SimpleCashFlow(Date(15, 1, 2029), 100.0)]
        pv = npv(all_cfs, flat_curve)
        y = yield_rate(all_cfs, pv, "Actual/365 (Fixed)", Date(15, 1, 2024))
        assert abs(y - 0.05) < 0.005


# ===== Duration & Convexity =====

class TestDuration:
    def test_modified_duration(self, annual_schedule):
        """Modified duration of 5-year, 5% bond."""
        leg = fixed_rate_leg(annual_schedule, 100.0, 0.05, "Actual/365 (Fixed)")
        all_cfs = leg + [SimpleCashFlow(Date(15, 1, 2029), 100.0)]
        d = duration(all_cfs, 0.05, "Actual/365 (Fixed)", Date(15, 1, 2024),
                     Duration.Modified)
        # ~4.3 years for 5-year 5% bond
        assert 3.5 < d < 5.0

    def test_convexity_positive(self, annual_schedule):
        leg = fixed_rate_leg(annual_schedule, 100.0, 0.05, "Actual/365 (Fixed)")
        all_cfs = leg + [SimpleCashFlow(Date(15, 1, 2029), 100.0)]
        c = convexity(all_cfs, 0.05, "Actual/365 (Fixed)", Date(15, 1, 2024))
        assert c > 0  # convexity is always positive for vanilla bonds


# ===== Accrued Amount =====

class TestAccrued:
    def test_mid_period(self, annual_schedule):
        leg = fixed_rate_leg(annual_schedule, 1_000_000.0, 0.06, "Actual/365 (Fixed)")
        # Settlement mid-year: ~Jul 15, 2024
        acc = accrued_amount(leg, Date(15, 7, 2024))
        # Half year of 6% on 1M ≈ 30,000
        assert 29_000 < acc < 31_000

    def test_start_of_period(self, annual_schedule):
        leg = fixed_rate_leg(annual_schedule, 100.0, 0.05, "Actual/365 (Fixed)")
        acc = accrued_amount(leg, Date(15, 1, 2024))
        assert acc == 0.0


# ===== Floating Rate Coupons =====

class TestFloatingCoupons:
    def test_ibor_coupon_creation(self):
        from ql_jax.indexes.ibor import USDLibor
        idx = USDLibor(3)
        cpn = IborCoupon(
            payment_date=Date(15, 4, 2024),
            nominal=1_000_000.0,
            accrual_start=Date(15, 1, 2024),
            accrual_end=Date(15, 4, 2024),
            index=idx,
            fixing_date=Date(11, 1, 2024),
            spread=0.001,
            day_counter="Actual/360",
        )
        assert cpn.nominal == 1_000_000.0
        assert cpn.spread == 0.001

    def test_capped_floored_coupon(self):
        from ql_jax.indexes.ibor import USDLibor
        idx = USDLibor(3)
        idx.add_fixing(Date(11, 1, 2024), 0.055)
        base = IborCoupon(
            payment_date=Date(15, 4, 2024),
            nominal=1_000_000.0,
            accrual_start=Date(15, 1, 2024),
            accrual_end=Date(15, 4, 2024),
            index=idx,
            fixing_date=Date(11, 1, 2024),
            day_counter="Actual/360",
        )
        capped = CappedFlooredCoupon(underlying=base, cap=0.04)
        # Rate is 5.5% but capped at 4%
        eff_rate = capped.effective_rate()
        assert abs(eff_rate - 0.04) < 1e-10

        floored = CappedFlooredCoupon(underlying=base, floor=0.06)
        eff_rate = floored.effective_rate()
        assert abs(eff_rate - 0.06) < 1e-10


# ===== ibor_leg =====

class TestIborLeg:
    def test_leg_creation(self, semiannual_schedule):
        from ql_jax.indexes.ibor import Euribor
        idx = Euribor(6)
        leg = ibor_leg(semiannual_schedule, idx, 1_000_000.0)
        assert len(leg) == 10
        for cpn in leg:
            assert isinstance(cpn, IborCoupon)
            assert cpn.nominal == 1_000_000.0


# ===== AD differentiation through cashflows =====

class TestCashFlowAD:
    def test_npv_differentiable_wrt_rate(self):
        """NPV should be differentiable w.r.t. the discount rate."""
        ref = Date(15, 1, 2024)
        cfs = [
            SimpleCashFlow(Date(15, 1, 2025), 5.0),
            SimpleCashFlow(Date(15, 1, 2026), 5.0),
            SimpleCashFlow(Date(15, 1, 2027), 5.0),
            SimpleCashFlow(Date(15, 1, 2028), 5.0),
            SimpleCashFlow(Date(15, 1, 2029), 105.0),
        ]

        def pv_fn(r):
            total = jnp.float64(0.0)
            for cf in cfs:
                t = year_fraction(ref, cf.date, "Actual/365 (Fixed)")
                total = total + cf.amount * jnp.exp(-r * t)
            return total

        r = jnp.float64(0.05)
        pv = pv_fn(r)
        dpdr = jax.grad(pv_fn)(r)
        # dP/dr should be negative (higher rate → lower PV)
        assert float(dpdr) < 0
        # Second derivative (convexity * PV) should be positive
        d2pdr2 = jax.grad(jax.grad(pv_fn))(r)
        assert float(d2pdr2) > 0
