"""Bond base class and common bond types."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ql_jax.time.date import Date, Period
from ql_jax.time.schedule import Schedule, MakeSchedule
from ql_jax.time.calendar import Calendar, NullCalendar
from ql_jax.cashflows.simple import SimpleCashFlow, Redemption
from ql_jax.cashflows.fixed_rate import FixedRateCoupon, fixed_rate_leg
from ql_jax.cashflows.floating_rate import IborCoupon, ibor_leg
from ql_jax._util.types import (
    BusinessDayConvention, Frequency, DateGeneration, TimeUnit,
)


@dataclass
class Bond:
    """Base bond instrument.

    Holds a collection of cash flows (coupons + redemptions).
    Pricing is done externally by engines.
    """
    settlement_days: int = 0
    calendar: Calendar = field(default_factory=NullCalendar)
    issue_date: Date | None = None
    cashflows_: list = field(default_factory=list)
    redemptions_: list = field(default_factory=list)
    notional_: float = 100.0

    @property
    def cashflows(self) -> list:
        """All cash flows (coupons + redemptions), sorted by date."""
        return sorted(
            self.cashflows_ + self.redemptions_,
            key=lambda cf: _cf_date(cf).serial,
        )

    @property
    def coupons(self) -> list:
        return list(self.cashflows_)

    @property
    def redemptions(self) -> list:
        return list(self.redemptions_)

    @property
    def notional(self) -> float:
        return self.notional_

    @property
    def maturity_date(self) -> Date | None:
        if self.redemptions_:
            return max(_cf_date(r) for r in self.redemptions_)
        if self.cashflows_:
            return max(_cf_date(cf) for cf in self.cashflows_)
        return None

    def settlement_date(self, from_date: Date | None = None) -> Date:
        if from_date is None:
            from ql_jax.patterns.observable import Settings
            from_date = Settings.evaluation_date()
        if self.settlement_days == 0:
            return from_date
        return self.calendar.advance(
            from_date, self.settlement_days, TimeUnit.Days,
            BusinessDayConvention.Following,
        )


def _cf_date(cf) -> Date:
    if hasattr(cf, "payment_date"):
        return cf.payment_date
    return cf.date


# ---------------------------------------------------------------------------
# Fixed-rate bond
# ---------------------------------------------------------------------------

def make_fixed_rate_bond(
    settlement_days: int,
    face_amount: float,
    schedule: Schedule,
    coupons: list[float] | float,
    day_counter: str = "Actual/365 (Fixed)",
    payment_convention: int = BusinessDayConvention.Following,
    redemption: float = 100.0,
    issue_date: Date | None = None,
    calendar: Calendar | None = None,
    payment_lag: int = 0,
) -> Bond:
    """Create a fixed-rate bond from a schedule and coupon rates."""
    cal = calendar or schedule.calendar
    legs = fixed_rate_leg(
        schedule, face_amount, coupons, day_counter,
        payment_lag=payment_lag,
        payment_calendar=cal,
        payment_convention=payment_convention,
    )
    redemption_date = schedule[-1]
    if cal is not None:
        redemption_date = cal.adjust(redemption_date, payment_convention)
    reds = [Redemption(date=redemption_date, amount=redemption)]

    return Bond(
        settlement_days=settlement_days,
        calendar=cal,
        issue_date=issue_date,
        cashflows_=legs,
        redemptions_=reds,
        notional_=face_amount,
    )


# ---------------------------------------------------------------------------
# Zero-coupon bond
# ---------------------------------------------------------------------------

def make_zero_coupon_bond(
    settlement_days: int,
    calendar: Calendar,
    face_amount: float,
    maturity_date: Date,
    payment_convention: int = BusinessDayConvention.Following,
    redemption: float = 100.0,
    issue_date: Date | None = None,
) -> Bond:
    """Create a zero-coupon bond."""
    pay_date = calendar.adjust(maturity_date, payment_convention)
    reds = [Redemption(date=pay_date, amount=redemption)]
    return Bond(
        settlement_days=settlement_days,
        calendar=calendar,
        issue_date=issue_date,
        cashflows_=[],
        redemptions_=reds,
        notional_=face_amount,
    )


# ---------------------------------------------------------------------------
# Floating-rate bond
# ---------------------------------------------------------------------------

def make_floating_rate_bond(
    settlement_days: int,
    face_amount: float,
    schedule: Schedule,
    index,
    day_counter: str = "Actual/360",
    payment_convention: int = BusinessDayConvention.Following,
    fixing_days: int = 2,
    gearings: float = 1.0,
    spreads: float = 0.0,
    redemption: float = 100.0,
    issue_date: Date | None = None,
    calendar: Calendar | None = None,
    in_arrears: bool = False,
) -> Bond:
    """Create a floating-rate bond."""
    cal = calendar or schedule.calendar
    legs = ibor_leg(
        schedule, index, face_amount, day_counter,
        gearing=gearings,
        spread=spreads,
        fixing_days=fixing_days,
        in_arrears=in_arrears,
        payment_calendar=cal,
        payment_convention=payment_convention,
    )
    redemption_date = schedule[-1]
    if cal is not None:
        redemption_date = cal.adjust(redemption_date, payment_convention)
    reds = [Redemption(date=redemption_date, amount=redemption)]

    return Bond(
        settlement_days=settlement_days,
        calendar=cal,
        issue_date=issue_date,
        cashflows_=legs,
        redemptions_=reds,
        notional_=face_amount,
    )


# ---------------------------------------------------------------------------
# Amortizing fixed-rate bond
# ---------------------------------------------------------------------------

def make_amortizing_fixed_rate_bond(
    settlement_days: int,
    calendar: Calendar,
    face_amount: float,
    start_date: Date,
    tenor: Period,
    frequency: int,
    coupon_rate: float,
    day_counter: str = "Actual/365 (Fixed)",
    payment_convention: int = BusinessDayConvention.Following,
    issue_date: Date | None = None,
) -> Bond:
    """Create an amortizing fixed-rate bond with level payments."""
    from ql_jax.time.daycounter import year_fraction

    # Build schedule
    sched = (MakeSchedule()
             .from_date(start_date)
             .to_date(start_date + tenor)
             .with_frequency(frequency)
             .with_calendar(calendar)
             .with_convention(payment_convention)
             .build())

    n = len(sched) - 1
    if n <= 0:
        return Bond(settlement_days=settlement_days, calendar=calendar)

    # Level payment amortization
    # PMT = P * r / (1 - (1+r)^-n)
    period_rate = coupon_rate / frequency
    if period_rate > 0:
        pmt = face_amount * period_rate / (1.0 - (1.0 + period_rate) ** (-n))
    else:
        pmt = face_amount / n

    balance = face_amount
    coupons_list = []
    redemptions = []
    for i in range(n):
        start = sched[i]
        end = sched[i + 1]
        interest = balance * coupon_rate * year_fraction(start, end, day_counter)
        principal = pmt - interest
        if principal > balance:
            principal = balance
        pay_date = calendar.adjust(end, payment_convention)
        coupons_list.append(FixedRateCoupon(
            payment_date=pay_date,
            nominal=balance,
            accrual_start=start,
            accrual_end=end,
            rate=coupon_rate,
            day_counter=day_counter,
        ))
        redemptions.append(Redemption(date=pay_date, amount=principal))
        balance -= principal

    return Bond(
        settlement_days=settlement_days,
        calendar=calendar,
        issue_date=issue_date,
        cashflows_=coupons_list,
        redemptions_=redemptions,
        notional_=face_amount,
    )
