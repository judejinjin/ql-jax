"""Interest-rate swap instruments."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from ql_jax.time.date import Date, Period
from ql_jax.time.schedule import Schedule, MakeSchedule
from ql_jax.time.calendar import Calendar, NullCalendar
from ql_jax.cashflows.fixed_rate import FixedRateCoupon, fixed_rate_leg
from ql_jax.cashflows.floating_rate import (
    IborCoupon, OvernightIndexedCoupon, ibor_leg, overnight_leg,
)
from ql_jax._util.types import (
    BusinessDayConvention, Frequency, DateGeneration, TimeUnit,
)


class SwapType:
    Payer = 1     # pay fixed, receive floating
    Receiver = -1  # receive fixed, pay floating


@dataclass
class VanillaSwap:
    """Plain-vanilla fixed-vs-floating interest rate swap.

    Attributes
    ----------
    type_ : int
        SwapType.Payer or SwapType.Receiver.
    nominal : float
        Notional amount.
    fixed_schedule : Schedule
    fixed_rate : float
    fixed_day_counter : str
    float_schedule : Schedule
    ibor_index : InterestRateIndex
    spread : float
    float_day_counter : str
    fixed_leg : list
        Generated fixed-rate coupons.
    floating_leg : list
        Generated floating-rate coupons.
    """
    type_: int = SwapType.Payer
    nominal: float = 1_000_000.0
    fixed_schedule: Schedule | None = None
    fixed_rate: float = 0.0
    fixed_day_counter: str = "Actual/365 (Fixed)"
    float_schedule: Schedule | None = None
    ibor_index: Any = None
    spread: float = 0.0
    float_day_counter: str = "Actual/360"
    payment_convention: int = BusinessDayConvention.Following
    fixed_leg: list = field(default_factory=list)
    floating_leg: list = field(default_factory=list)

    def __post_init__(self):
        if self.fixed_schedule is not None and not self.fixed_leg:
            self.fixed_leg = fixed_rate_leg(
                self.fixed_schedule,
                self.nominal,
                self.fixed_rate,
                self.fixed_day_counter,
                payment_convention=self.payment_convention,
            )
        if self.float_schedule is not None and self.ibor_index is not None and not self.floating_leg:
            self.floating_leg = ibor_leg(
                self.float_schedule,
                self.ibor_index,
                self.nominal,
                self.float_day_counter,
                spread=self.spread,
                payment_convention=self.payment_convention,
            )

    @property
    def maturity_date(self) -> Date | None:
        dates = []
        if self.fixed_schedule and len(self.fixed_schedule) > 0:
            dates.append(self.fixed_schedule[-1])
        if self.float_schedule and len(self.float_schedule) > 0:
            dates.append(self.float_schedule[-1])
        return max(dates) if dates else None


@dataclass
class OvernightIndexedSwap:
    """Overnight-indexed swap (OIS): fixed vs compounded overnight.

    Attributes
    ----------
    type_ : int
        SwapType.Payer or SwapType.Receiver.
    nominal : float
    fixed_schedule : Schedule
    fixed_rate : float
    fixed_day_counter : str
    overnight_schedule : Schedule
    overnight_index : OvernightIndex
    spread : float
    overnight_day_counter : str
    """
    type_: int = SwapType.Payer
    nominal: float = 1_000_000.0
    fixed_schedule: Schedule | None = None
    fixed_rate: float = 0.0
    fixed_day_counter: str = "Actual/365 (Fixed)"
    overnight_schedule: Schedule | None = None
    overnight_index: Any = None
    spread: float = 0.0
    overnight_day_counter: str = "Actual/360"
    payment_convention: int = BusinessDayConvention.Following
    averaging: bool = False
    fixed_leg: list = field(default_factory=list)
    overnight_leg_: list = field(default_factory=list)

    def __post_init__(self):
        if self.fixed_schedule is not None and not self.fixed_leg:
            self.fixed_leg = fixed_rate_leg(
                self.fixed_schedule,
                self.nominal,
                self.fixed_rate,
                self.fixed_day_counter,
                payment_convention=self.payment_convention,
            )
        if (self.overnight_schedule is not None
                and self.overnight_index is not None
                and not self.overnight_leg_):
            self.overnight_leg_ = overnight_leg(
                self.overnight_schedule,
                self.overnight_index,
                self.nominal,
                self.overnight_day_counter,
                spread=self.spread,
                averaging=self.averaging,
                payment_convention=self.payment_convention,
            )

    @property
    def floating_leg(self) -> list:
        return self.overnight_leg_


# ---------------------------------------------------------------------------
# Builder helpers
# ---------------------------------------------------------------------------

def make_vanilla_swap(
    type_: int,
    nominal: float,
    fixed_schedule: Schedule,
    fixed_rate: float,
    fixed_day_counter: str,
    float_schedule: Schedule,
    ibor_index,
    spread: float = 0.0,
    float_day_counter: str | None = None,
    payment_convention: int = BusinessDayConvention.Following,
) -> VanillaSwap:
    """Convenience constructor for a vanilla swap."""
    fdc = float_day_counter or ibor_index.day_counter_convention
    return VanillaSwap(
        type_=type_,
        nominal=nominal,
        fixed_schedule=fixed_schedule,
        fixed_rate=fixed_rate,
        fixed_day_counter=fixed_day_counter,
        float_schedule=float_schedule,
        ibor_index=ibor_index,
        spread=spread,
        float_day_counter=fdc,
        payment_convention=payment_convention,
    )


def make_ois(
    type_: int,
    nominal: float,
    fixed_schedule: Schedule,
    fixed_rate: float,
    fixed_day_counter: str,
    overnight_schedule: Schedule,
    overnight_index,
    spread: float = 0.0,
    overnight_day_counter: str | None = None,
    payment_convention: int = BusinessDayConvention.Following,
    averaging: bool = False,
) -> OvernightIndexedSwap:
    """Convenience constructor for an OIS."""
    odc = overnight_day_counter or overnight_index.day_counter_convention
    return OvernightIndexedSwap(
        type_=type_,
        nominal=nominal,
        fixed_schedule=fixed_schedule,
        fixed_rate=fixed_rate,
        fixed_day_counter=fixed_day_counter,
        overnight_schedule=overnight_schedule,
        overnight_index=overnight_index,
        spread=spread,
        overnight_day_counter=odc,
        payment_convention=payment_convention,
        averaging=averaging,
    )


# ---------------------------------------------------------------------------
# Zero-coupon swap
# ---------------------------------------------------------------------------

@dataclass
class ZeroCouponSwap:
    """Zero-coupon swap — exchanges a single fixed payment for a floating leg.

    Parameters
    ----------
    type_ : SwapType.Payer or SwapType.Receiver
    nominal : notional amount
    start_date : start date
    maturity_date : maturity date
    fixed_rate : annualized fixed rate
    float_schedule : Schedule for floating leg
    ibor_index : floating rate index
    day_counter : day count convention
    spread : floating spread
    """
    type_: int = SwapType.Payer
    nominal: float = 1_000_000.0
    start_date: Date | None = None
    maturity_date_: Date | None = None
    fixed_rate: float = 0.0
    float_schedule: Schedule | None = None
    ibor_index: Any = None
    day_counter: str = "Actual/365 (Fixed)"
    spread: float = 0.0
    payment_convention: int = BusinessDayConvention.Following
    floating_leg: list = field(default_factory=list)

    def __post_init__(self):
        if (self.float_schedule is not None
                and self.ibor_index is not None
                and not self.floating_leg):
            self.floating_leg = ibor_leg(
                self.float_schedule,
                self.ibor_index,
                self.nominal,
                self.day_counter,
                spread=self.spread,
                payment_convention=self.payment_convention,
            )

    @property
    def maturity_date(self) -> Date | None:
        return self.maturity_date_


# ---------------------------------------------------------------------------
# Multiple-resets swap  (sub-periods per coupon)
# ---------------------------------------------------------------------------

@dataclass
class MultipleResetsSwap:
    """Swap with multiple index resets per coupon period.

    Each floating coupon accumulates sub-period fixings
    (compounded or averaged).

    Parameters
    ----------
    type_ : SwapType.Payer or SwapType.Receiver
    nominal : notional
    fixed_schedule : Schedule for fixed leg
    fixed_rate : fixed rate
    fixed_day_counter : day count for fixed leg
    float_schedule : Schedule for floating leg / outer periods
    ibor_index : IBOR index for sub-period fixings
    float_day_counter : day count for float leg
    spread : floating spread
    resets_per_period : number of index fixings per coupon period
    averaging : True = simple avg, False = compound
    """
    type_: int = SwapType.Payer
    nominal: float = 1_000_000.0
    fixed_schedule: Schedule | None = None
    fixed_rate: float = 0.0
    fixed_day_counter: str = "Actual/365 (Fixed)"
    float_schedule: Schedule | None = None
    ibor_index: Any = None
    float_day_counter: str = "Actual/360"
    spread: float = 0.0
    resets_per_period: int = 3
    averaging: bool = False
    payment_convention: int = BusinessDayConvention.Following
    fixed_leg: list = field(default_factory=list)
    floating_leg: list = field(default_factory=list)

    def __post_init__(self):
        if self.fixed_schedule is not None and not self.fixed_leg:
            self.fixed_leg = fixed_rate_leg(
                self.fixed_schedule,
                self.nominal,
                self.fixed_rate,
                self.fixed_day_counter,
                payment_convention=self.payment_convention,
            )

    @property
    def maturity_date(self) -> Date | None:
        dates = []
        if self.fixed_schedule and len(self.fixed_schedule) > 0:
            dates.append(self.fixed_schedule[-1])
        if self.float_schedule and len(self.float_schedule) > 0:
            dates.append(self.float_schedule[-1])
        return max(dates) if dates else None
