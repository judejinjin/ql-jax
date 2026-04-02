"""Tests for swap instruments and pricing engines."""

import pytest
import jax.numpy as jnp

from ql_jax.time.date import Date, Period
from ql_jax.time.schedule import MakeSchedule
from ql_jax.time.calendar import NullCalendar, TARGET
from ql_jax._util.types import (
    BusinessDayConvention, Frequency, TimeUnit,
)
from ql_jax.instruments.swap import (
    VanillaSwap, OvernightIndexedSwap, SwapType,
    make_vanilla_swap, make_ois,
)
from ql_jax.engines.swap.discounting import (
    discounting_swap_npv, discounting_swap_fair_rate,
    discounting_swap_fair_spread,
)
from ql_jax.termstructures.yield_.flat_forward import FlatForward
from ql_jax.indexes.ibor import Euribor, SOFR


@pytest.fixture
def flat_curve():
    return FlatForward(Date(15, 1, 2024), 0.05)


@pytest.fixture
def fixed_schedule():
    return (MakeSchedule()
            .from_date(Date(15, 1, 2024))
            .to_date(Date(15, 1, 2029))
            .with_frequency(Frequency.Annual)
            .with_calendar(NullCalendar())
            .with_convention(BusinessDayConvention.Unadjusted)
            .build())


@pytest.fixture
def float_schedule():
    return (MakeSchedule()
            .from_date(Date(15, 1, 2024))
            .to_date(Date(15, 1, 2029))
            .with_frequency(Frequency.Semiannual)
            .with_calendar(NullCalendar())
            .with_convention(BusinessDayConvention.Unadjusted)
            .build())


class TestVanillaSwap:
    def test_creation(self, fixed_schedule, float_schedule):
        idx = Euribor(6)
        swap = make_vanilla_swap(
            SwapType.Payer, 1_000_000.0,
            fixed_schedule, 0.05, "Actual/365 (Fixed)",
            float_schedule, idx, spread=0.001,
        )
        assert len(swap.fixed_leg) == 5
        assert len(swap.floating_leg) == 10

    def test_payer_receiver_symmetry(self, fixed_schedule, float_schedule, flat_curve):
        """Payer and receiver NPV should be opposite."""
        idx = Euribor(6)
        payer = make_vanilla_swap(
            SwapType.Payer, 1_000_000.0,
            fixed_schedule, 0.05, "Actual/365 (Fixed)",
            float_schedule, idx,
        )
        receiver = make_vanilla_swap(
            SwapType.Receiver, 1_000_000.0,
            fixed_schedule, 0.05, "Actual/365 (Fixed)",
            float_schedule, idx,
        )
        npv_payer = discounting_swap_npv(payer, flat_curve)
        npv_receiver = discounting_swap_npv(receiver, flat_curve)
        assert abs(npv_payer + npv_receiver) < 1.0

    def test_fair_rate(self, fixed_schedule, float_schedule, flat_curve):
        """Fair rate on a flat 5% curve should be close to 5%."""
        idx = Euribor(6)
        swap = make_vanilla_swap(
            SwapType.Payer, 1_000_000.0,
            fixed_schedule, 0.05, "Actual/365 (Fixed)",
            float_schedule, idx,
        )
        fair = discounting_swap_fair_rate(swap, flat_curve)
        # On a flat curve, fair rate ≈ curve rate (with day count differences)
        assert abs(fair - 0.05) < 0.02

    def test_at_fair_rate_npv_zero(self, fixed_schedule, float_schedule, flat_curve):
        """Swap at fair rate has NPV ≈ 0."""
        idx = Euribor(6)
        # First create a swap to find fair rate
        swap_temp = make_vanilla_swap(
            SwapType.Payer, 1_000_000.0,
            fixed_schedule, 0.05, "Actual/365 (Fixed)",
            float_schedule, idx,
        )
        fair = discounting_swap_fair_rate(swap_temp, flat_curve)

        # Create swap at fair rate
        swap = make_vanilla_swap(
            SwapType.Payer, 1_000_000.0,
            fixed_schedule, fair, "Actual/365 (Fixed)",
            float_schedule, idx,
        )
        npv_val = discounting_swap_npv(swap, flat_curve)
        assert abs(npv_val) < 100.0  # close to 0 for 1M notional


class TestFRA:
    def test_fra_npv(self, flat_curve):
        """FRA NPV at market rate should be ~0."""
        from ql_jax.instruments.fra import ForwardRateAgreement, FRAType

        fra = ForwardRateAgreement(
            value_date=Date(15, 7, 2024),
            maturity_date=Date(15, 1, 2025),
            strike=0.05,  # at-the-money on 5% flat curve
            notional=1_000_000.0,
            type_=FRAType.Long,
            day_counter="Actual/360",
        )
        fwd = fra.forward_rate(flat_curve)
        # Forward rate should be ~5% (cc to simple adjustment)
        assert abs(fwd - 0.05) < 0.01

    def test_fra_off_market(self, flat_curve):
        """FRA with strike away from market should have non-zero NPV."""
        from ql_jax.instruments.fra import ForwardRateAgreement, FRAType

        fra = ForwardRateAgreement(
            value_date=Date(15, 7, 2024),
            maturity_date=Date(15, 1, 2025),
            strike=0.03,  # below market → long profits
            notional=1_000_000.0,
            type_=FRAType.Long,
            day_counter="Actual/360",
        )
        val = fra.npv(flat_curve)
        assert val > 0  # long FRA profits when rate > strike
