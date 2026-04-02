"""Tests for Phase 1 instruments: futures, inflation cap/floor, energy, swap extensions, bond extensions."""

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# IR Futures
# ---------------------------------------------------------------------------

class TestIRFuture:
    def test_implied_rate(self):
        from ql_jax.instruments.futures import IRFuture
        fut = IRFuture(price=97.5, maturity=0.25, tenor=0.25)
        assert fut.implied_rate == pytest.approx(0.025)

    def test_implied_rate_zero(self):
        from ql_jax.instruments.futures import IRFuture
        fut = IRFuture(price=100.0, maturity=0.25)
        assert fut.implied_rate == pytest.approx(0.0)

    def test_defaults(self):
        from ql_jax.instruments.futures import IRFuture
        fut = IRFuture(price=95.0, maturity=1.0)
        assert fut.notional == 1_000_000.0
        assert fut.tenor == 0.25
        assert fut.convention == "Actual/360"


class TestOvernightIndexFuture:
    def test_implied_rate(self):
        from ql_jax.instruments.futures import OvernightIndexFuture
        fut = OvernightIndexFuture(price=95.5)
        assert fut.implied_rate == pytest.approx(0.045)

    def test_default_maturity(self):
        from ql_jax.instruments.futures import OvernightIndexFuture
        fut = OvernightIndexFuture()
        assert fut.maturity_date == 0.25


class TestPerpetualFuture:
    def test_funding_payment(self):
        from ql_jax.instruments.futures import PerpetualFuture
        pf = PerpetualFuture(underlying_price=50000, funding_rate=0.001, notional=2.0, mark_price=49500)
        expected = 2.0 * 0.001 * 49500
        assert pf.funding_payment() == pytest.approx(expected)

    def test_zero_funding(self):
        from ql_jax.instruments.futures import PerpetualFuture
        pf = PerpetualFuture(funding_rate=0.0, mark_price=100.0)
        assert pf.funding_payment() == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# CPI / YoY Inflation cap/floor
# ---------------------------------------------------------------------------

class TestCPICapFloor:
    def test_cap_itm(self):
        from ql_jax.instruments.inflation_capfloor import CPICapFloor
        cf = CPICapFloor(option_type=1, notional=1e6, base_cpi=100.0, strike=0.03, maturity=1.0)
        # CPI goes to 105 => realized rate = 0.05 > 0.03
        assert float(cf.payoff(105.0)) == pytest.approx(1e6 * 0.02)

    def test_cap_otm(self):
        from ql_jax.instruments.inflation_capfloor import CPICapFloor
        cf = CPICapFloor(option_type=1, notional=1e6, base_cpi=100.0, strike=0.10, maturity=1.0)
        assert float(cf.payoff(105.0)) == pytest.approx(0.0)

    def test_floor_itm(self):
        from ql_jax.instruments.inflation_capfloor import CPICapFloor
        cf = CPICapFloor(option_type=-1, notional=1e6, base_cpi=100.0, strike=0.05, maturity=1.0)
        # realized rate = 0.02 < 0.05
        assert float(cf.payoff(102.0)) == pytest.approx(1e6 * 0.03)

    def test_floor_otm(self):
        from ql_jax.instruments.inflation_capfloor import CPICapFloor
        cf = CPICapFloor(option_type=-1, notional=1e6, base_cpi=100.0, strike=0.01, maturity=1.0)
        assert float(cf.payoff(105.0)) == pytest.approx(0.0)


class TestYoYInflationCapFloor:
    def test_make_cap(self):
        from ql_jax.instruments.inflation_capfloor import make_yoy_inflation_cap
        cap = make_yoy_inflation_cap(1e6, 0.03, 5, frequency=1)
        assert cap.option_type == 1
        assert cap.n_periods == 5
        assert float(cap.payment_times[-1]) == pytest.approx(5.0)

    def test_make_floor(self):
        from ql_jax.instruments.inflation_capfloor import make_yoy_inflation_floor
        floor = make_yoy_inflation_floor(1e6, 0.02, 3, frequency=2)
        assert floor.option_type == -1
        assert floor.n_periods == 6  # 3 years * 2 freq
        assert float(floor.strikes[0]) == pytest.approx(0.02)

    def test_accrual_fractions(self):
        from ql_jax.instruments.inflation_capfloor import make_yoy_inflation_cap
        cap = make_yoy_inflation_cap(1e6, 0.03, 2, frequency=4)
        assert cap.n_periods == 8
        assert float(cap.accrual_fractions[0]) == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Energy instruments
# ---------------------------------------------------------------------------

class TestVanillaStorageOption:
    def test_construction(self):
        from ql_jax.instruments.energy import VanillaStorageOption
        dates = jnp.array([0.25, 0.5, 0.75, 1.0])
        vso = VanillaStorageOption(strike=50.0, maturity=1.0, exercise_dates=dates,
                                   min_volume=0.0, max_volume=100.0)
        assert vso.n_exercises == 4
        assert vso.max_volume == 100.0

    def test_default_rates(self):
        from ql_jax.instruments.energy import VanillaStorageOption
        dates = jnp.array([0.5])
        vso = VanillaStorageOption(strike=10.0, maturity=1.0, exercise_dates=dates)
        assert vso.injection_rate == 1.0
        assert vso.withdrawal_rate == 1.0
        assert vso.initial_volume == 0.0


class TestVanillaSwingOption:
    def test_construction(self):
        from ql_jax.instruments.energy import VanillaSwingOption
        dates = jnp.array([0.25, 0.5, 0.75, 1.0])
        swing = VanillaSwingOption(strike=100.0, exercise_dates=dates,
                                   min_exercises=1, max_exercises=3)
        assert swing.n_dates == 4
        assert float(swing.maturity) == pytest.approx(1.0)

    def test_single_exercise(self):
        from ql_jax.instruments.energy import VanillaSwingOption
        dates = jnp.array([0.5, 1.0])
        swing = VanillaSwingOption(strike=50.0, exercise_dates=dates, max_exercises=1)
        assert swing.max_exercises == 1


# ---------------------------------------------------------------------------
# Swap extensions
# ---------------------------------------------------------------------------

class TestZeroCouponSwap:
    def test_construction(self):
        from ql_jax.instruments.swap import ZeroCouponSwap, SwapType
        zcs = ZeroCouponSwap(type_=SwapType.Payer, nominal=1e6, fixed_rate=0.05)
        assert zcs.type_ == 1
        assert zcs.nominal == 1e6


class TestMultipleResetsSwap:
    def test_construction(self):
        from ql_jax.instruments.swap import MultipleResetsSwap, SwapType
        mrs = MultipleResetsSwap(type_=SwapType.Receiver, nominal=5e6,
                                  fixed_rate=0.03, resets_per_period=3)
        assert mrs.type_ == -1
        assert mrs.resets_per_period == 3
        assert mrs.averaging is False


# ---------------------------------------------------------------------------
# Bond extensions
# ---------------------------------------------------------------------------

class TestCPIBond:
    def test_make_cpi_bond(self):
        from ql_jax.instruments.bond import make_cpi_bond
        from ql_jax.time.date import Date
        from ql_jax.time.schedule import MakeSchedule
        from ql_jax.time.calendar import NullCalendar
        from ql_jax._util.types import Frequency

        sch = (MakeSchedule()
               .from_date(Date(15, 1, 2024))
               .to_date(Date(15, 1, 2029))
               .with_frequency(Frequency.Semiannual)
               .with_calendar(NullCalendar())
               .build())
        bond = make_cpi_bond(settlement_days=3, face_amount=100.0,
                             schedule=sch, coupon_rate=0.01, base_cpi=260.0)
        assert bond.is_inflation_linked is True
        assert bond.base_cpi == 260.0
        assert bond.notional == pytest.approx(100.0)


class TestAmortizingFloatingRateBond:
    def test_construction(self):
        from ql_jax.instruments.bond import make_amortizing_floating_rate_bond
        from ql_jax.time.date import Date, Period
        from ql_jax.time.calendar import NullCalendar
        from ql_jax._util.types import Frequency, TimeUnit
        from ql_jax.indexes.ibor import SOFR

        bond = make_amortizing_floating_rate_bond(
            settlement_days=2,
            calendar=NullCalendar(),
            face_amount=1e6,
            start_date=Date(15, 1, 2024),
            tenor=Period(5, TimeUnit.Years),
            frequency=Frequency.Annual,
            index=SOFR(),
        )
        assert bond.notional == pytest.approx(1e6)
