"""Tests for bond instruments and pricing engines."""

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
from ql_jax.instruments.bond import (
    Bond, make_fixed_rate_bond, make_zero_coupon_bond,
    make_floating_rate_bond, make_amortizing_fixed_rate_bond,
)
from ql_jax.engines.bond.discounting import (
    discounting_bond_npv, discounting_bond_clean_price,
    discounting_bond_dirty_price, BondFunctions,
)
from ql_jax.termstructures.yield_.flat_forward import FlatForward


@pytest.fixture
def flat_curve():
    return FlatForward(Date(15, 1, 2024), 0.05)


@pytest.fixture
def annual_schedule():
    return (MakeSchedule()
            .from_date(Date(15, 1, 2024))
            .to_date(Date(15, 1, 2029))
            .with_frequency(Frequency.Annual)
            .with_calendar(NullCalendar())
            .with_convention(BusinessDayConvention.Unadjusted)
            .build())


class TestFixedRateBond:
    def test_creation(self, annual_schedule):
        bond = make_fixed_rate_bond(
            settlement_days=0,
            face_amount=100.0,
            schedule=annual_schedule,
            coupons=0.05,
            day_counter="Actual/365 (Fixed)",
        )
        assert bond.notional == 100.0
        assert len(bond.coupons) == 5
        assert len(bond.redemptions) == 1
        assert bond.redemptions[0].amount == 100.0

    def test_cashflows_sorted(self, annual_schedule):
        bond = make_fixed_rate_bond(
            settlement_days=0,
            face_amount=100.0,
            schedule=annual_schedule,
            coupons=0.05,
        )
        all_cfs = bond.cashflows
        dates = [cf.payment_date if hasattr(cf, 'payment_date') else cf.date
                 for cf in all_cfs]
        for i in range(len(dates) - 1):
            assert dates[i] <= dates[i + 1]

    def test_maturity(self, annual_schedule):
        bond = make_fixed_rate_bond(
            settlement_days=0,
            face_amount=100.0,
            schedule=annual_schedule,
            coupons=0.05,
        )
        assert bond.maturity_date == Date(15, 1, 2029)


class TestZeroCouponBond:
    def test_creation(self):
        bond = make_zero_coupon_bond(
            settlement_days=0,
            calendar=NullCalendar(),
            face_amount=100.0,
            maturity_date=Date(15, 1, 2029),
        )
        assert len(bond.coupons) == 0
        assert len(bond.redemptions) == 1
        assert bond.maturity_date == Date(15, 1, 2029)


class TestBondPricing:
    def test_par_bond_at_par_rate(self, flat_curve, annual_schedule):
        """A 5% bond on a 5% flat curve should price near par."""
        bond = make_fixed_rate_bond(
            settlement_days=0,
            face_amount=100.0,
            schedule=annual_schedule,
            coupons=0.05,
            day_counter="Actual/365 (Fixed)",
        )
        price = discounting_bond_dirty_price(bond, flat_curve)
        # Should be close to 100
        assert abs(price - 100.0) < 2.0

    def test_zero_coupon_pricing(self, flat_curve):
        bond = make_zero_coupon_bond(
            settlement_days=0,
            calendar=NullCalendar(),
            face_amount=100.0,
            maturity_date=Date(15, 1, 2029),
        )
        raw_npv = discounting_bond_npv(bond, flat_curve)
        # PV = 100 * exp(-0.05 * 5) ≈ 77.88
        expected = 100.0 * float(jnp.exp(-0.05 * 5.0))
        assert abs(raw_npv - expected) < 1.0

    def test_higher_rate_lower_price(self, annual_schedule):
        """Bond priced on higher rate curve ⇒ lower price."""
        bond = make_fixed_rate_bond(
            settlement_days=0,
            face_amount=100.0,
            schedule=annual_schedule,
            coupons=0.05,
        )
        curve_low = FlatForward(Date(15, 1, 2024), 0.03)
        curve_high = FlatForward(Date(15, 1, 2024), 0.07)
        pv_low = discounting_bond_npv(bond, curve_low)
        pv_high = discounting_bond_npv(bond, curve_high)
        assert pv_low > pv_high


class TestBondFunctions:
    def test_yield_round_trip(self, flat_curve, annual_schedule):
        """Yield from a clean price should round-trip."""
        bond = make_fixed_rate_bond(
            settlement_days=0,
            face_amount=100.0,
            schedule=annual_schedule,
            coupons=0.05,
            day_counter="Actual/365 (Fixed)",
        )
        clean = BondFunctions.clean_price(bond, flat_curve)
        y = BondFunctions.yield_rate(
            bond, clean, "Actual/365 (Fixed)",
            settlement_date=Date(15, 1, 2024),
        )
        assert abs(y - 0.05) < 0.01

    def test_duration(self, annual_schedule):
        bond = make_fixed_rate_bond(
            settlement_days=0,
            face_amount=100.0,
            schedule=annual_schedule,
            coupons=0.05,
            day_counter="Actual/365 (Fixed)",
        )
        d = BondFunctions.duration(bond, 0.05, "Actual/365 (Fixed)",
                                   settlement_date=Date(15, 1, 2024))
        # Modified duration of 5-year 5% bond ≈ 4.3
        assert 3.5 < d < 5.0

    def test_convexity(self, annual_schedule):
        bond = make_fixed_rate_bond(
            settlement_days=0,
            face_amount=100.0,
            schedule=annual_schedule,
            coupons=0.05,
            day_counter="Actual/365 (Fixed)",
        )
        c = BondFunctions.convexity(bond, 0.05, "Actual/365 (Fixed)",
                                    settlement_date=Date(15, 1, 2024))
        assert c > 0

    def test_z_spread(self, flat_curve, annual_schedule):
        """Z-spread of a par bond on its own curve should be ~0."""
        bond = make_fixed_rate_bond(
            settlement_days=0,
            face_amount=100.0,
            schedule=annual_schedule,
            coupons=0.05,
            day_counter="Actual/365 (Fixed)",
        )
        settle = Date(15, 1, 2024)
        clean = BondFunctions.clean_price(bond, flat_curve, settlement_date=settle)
        z = BondFunctions.z_spread(
            bond, flat_curve, clean, "Actual/365 (Fixed)",
            settlement_date=settle,
        )
        assert abs(z) < 0.01


class TestBondAD:
    def test_dv01_via_ad(self, annual_schedule):
        """DV01 via AD through bond pricing."""
        bond = make_fixed_rate_bond(
            settlement_days=0,
            face_amount=100.0,
            schedule=annual_schedule,
            coupons=0.05,
            day_counter="Actual/365 (Fixed)",
        )
        ref = Date(15, 1, 2024)

        def price_fn(r):
            total = jnp.float64(0.0)
            for cf in bond.cashflows:
                d = cf.payment_date if hasattr(cf, 'payment_date') else cf.date
                a = cf.amount if not callable(cf.amount) else cf.amount()
                t = year_fraction(ref, d, "Actual/365 (Fixed)")
                total = total + a * jnp.exp(-r * t)
            return total

        r = jnp.float64(0.05)
        dv01 = -jax.grad(price_fn)(r) * 0.0001
        # DV01 should be positive and roughly = duration * price * 1bp
        assert float(dv01) > 0
        assert float(dv01) < 0.1  # for 100 face
