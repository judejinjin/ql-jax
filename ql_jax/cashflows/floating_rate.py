"""Floating-rate coupons: IBOR, overnight-indexed, and CMS."""

from __future__ import annotations

from dataclasses import dataclass

from ql_jax.time.date import Date
from ql_jax.time.daycounter import year_fraction
from ql_jax.time.schedule import Schedule
from ql_jax.cashflows.fixed_rate import Coupon


@dataclass(frozen=True)
class FloatingRateCoupon(Coupon):
    """Base floating-rate coupon with fixing, gearing, and spread."""
    index: object = None          # InterestRateIndex
    fixing_date: Date | None = None
    gearing: float = 1.0
    spread: float = 0.0
    day_counter: str = "Actual/360"
    is_in_arrears: bool = False

    def index_fixing(self, forecast_curve=None) -> float:
        """Return the index fixing for this coupon."""
        fd = self.fixing_date or self.accrual_start
        return self.index.fixing(fd, forecast_curve)

    def adjusted_fixing(self, forecast_curve=None) -> float:
        """Gearing * fixing + spread."""
        return self.gearing * self.index_fixing(forecast_curve) + self.spread

    def amount_with_curve(self, forecast_curve=None) -> float:
        """Coupon amount using a forecast curve."""
        rate = self.adjusted_fixing(forecast_curve)
        tau = self.accrual_period(self.day_counter)
        return self.nominal * rate * tau


@dataclass(frozen=True)
class IborCoupon(FloatingRateCoupon):
    """IBOR-based floating rate coupon."""
    pass


@dataclass(frozen=True)
class OvernightIndexedCoupon(Coupon):
    """Coupon accruing compounded or averaged overnight rates.

    In practice, the rate is the geometric (compounded) or arithmetic
    (averaged) mean of daily overnight rates over the accrual period.
    """
    index: object = None          # OvernightIndex
    gearing: float = 1.0
    spread: float = 0.0
    day_counter: str = "Actual/360"
    rateCutoff: int = 0           # number of days before end to freeze rate
    lookback_days: int = 0        # observation shift
    averaging: bool = False       # True = simple average; False = compound

    def compounded_rate(self, forecast_curve) -> float:
        """Compute the compounded overnight rate over the accrual period."""
        from ql_jax.time.date import Date
        from ql_jax._util.types import TimeUnit

        cal = self.index.calendar
        dc = self.day_counter
        d = self.accrual_start
        end = self.accrual_end
        compound = 1.0
        total_weight = 0.0
        total_rate = 0.0

        while d < end:
            next_d = cal.advance(d, 1, TimeUnit.Days)
            if next_d > end:
                next_d = end
            dt = year_fraction(d, next_d, dc)
            if dt <= 0:
                d = next_d
                continue
            rate = self.index.fixing(d, forecast_curve)
            if self.averaging:
                total_rate += rate * dt
                total_weight += dt
            else:
                compound *= (1.0 + rate * dt)
            d = next_d

        if self.averaging and total_weight > 0:
            return total_rate / total_weight
        elif not self.averaging:
            tau = year_fraction(self.accrual_start, self.accrual_end, dc)
            if tau > 0:
                return (compound - 1.0) / tau
        return 0.0

    def adjusted_rate(self, forecast_curve) -> float:
        return self.gearing * self.compounded_rate(forecast_curve) + self.spread

    def amount_with_curve(self, forecast_curve) -> float:
        rate = self.adjusted_rate(forecast_curve)
        tau = self.accrual_period(self.day_counter)
        return self.nominal * rate * tau


@dataclass(frozen=True)
class CappedFlooredCoupon:
    """A coupon with an optional cap and/or floor on the rate."""
    underlying: FloatingRateCoupon | OvernightIndexedCoupon
    cap: float | None = None
    floor: float | None = None

    @property
    def payment_date(self) -> Date:
        return self.underlying.payment_date

    @property
    def nominal(self) -> float:
        return self.underlying.nominal

    def effective_rate(self, forecast_curve=None) -> float:
        if isinstance(self.underlying, OvernightIndexedCoupon):
            rate = self.underlying.adjusted_rate(forecast_curve)
        else:
            rate = self.underlying.adjusted_fixing(forecast_curve)
        if self.cap is not None:
            rate = min(rate, self.cap)
        if self.floor is not None:
            rate = max(rate, self.floor)
        return rate

    def amount_with_curve(self, forecast_curve=None) -> float:
        rate = self.effective_rate(forecast_curve)
        dc = self.underlying.day_counter
        tau = self.underlying.accrual_period(dc)
        return self.underlying.nominal * rate * tau


# ---------------------------------------------------------------------------
# Leg builders
# ---------------------------------------------------------------------------

def ibor_leg(
    schedule: Schedule,
    index,
    notionals: list[float] | float,
    day_counter: str | None = None,
    gearing: float = 1.0,
    spread: float = 0.0,
    payment_lag: int = 0,
    payment_calendar=None,
    payment_convention: int | None = None,
    fixing_days: int | None = None,
    in_arrears: bool = False,
) -> list[IborCoupon]:
    """Generate a leg of IBOR coupons from a schedule."""
    if isinstance(notionals, (int, float)):
        notionals = [float(notionals)]

    dc = day_counter or index.day_counter_convention
    cal = payment_calendar or schedule.calendar
    conv = payment_convention if payment_convention is not None else schedule.convention
    fix_days = fixing_days if fixing_days is not None else index.fixing_days

    n = len(schedule) - 1
    coupons: list[IborCoupon] = []
    for i in range(n):
        start = schedule[i]
        end = schedule[i + 1]
        notional = notionals[min(i, len(notionals) - 1)]

        # Fixing date
        if in_arrears:
            fixing_ref = end
        else:
            fixing_ref = start
        fix_date = _advance_bdays(fixing_ref, -fix_days, cal) if fix_days > 0 else fixing_ref

        # Payment date
        pay_date = end
        if payment_lag > 0 and cal is not None:
            from ql_jax._util.types import TimeUnit
            pay_date = cal.advance(end, payment_lag, TimeUnit.Days, conv)
        elif cal is not None:
            pay_date = cal.adjust(end, conv)

        coupons.append(IborCoupon(
            payment_date=pay_date,
            nominal=notional,
            accrual_start=start,
            accrual_end=end,
            index=index,
            fixing_date=fix_date,
            gearing=gearing,
            spread=spread,
            day_counter=dc,
            is_in_arrears=in_arrears,
        ))
    return coupons


def overnight_leg(
    schedule: Schedule,
    index,
    notionals: list[float] | float,
    day_counter: str | None = None,
    gearing: float = 1.0,
    spread: float = 0.0,
    averaging: bool = False,
    payment_lag: int = 0,
    payment_calendar=None,
    payment_convention: int | None = None,
) -> list[OvernightIndexedCoupon]:
    """Generate a leg of overnight-indexed coupons."""
    if isinstance(notionals, (int, float)):
        notionals = [float(notionals)]

    dc = day_counter or index.day_counter_convention
    cal = payment_calendar or schedule.calendar
    conv = payment_convention if payment_convention is not None else schedule.convention

    n = len(schedule) - 1
    coupons: list[OvernightIndexedCoupon] = []
    for i in range(n):
        start = schedule[i]
        end = schedule[i + 1]
        notional = notionals[min(i, len(notionals) - 1)]

        pay_date = end
        if payment_lag > 0 and cal is not None:
            from ql_jax._util.types import TimeUnit
            pay_date = cal.advance(end, payment_lag, TimeUnit.Days, conv)
        elif cal is not None:
            pay_date = cal.adjust(end, conv)

        coupons.append(OvernightIndexedCoupon(
            payment_date=pay_date,
            nominal=notional,
            accrual_start=start,
            accrual_end=end,
            index=index,
            gearing=gearing,
            spread=spread,
            day_counter=dc,
            averaging=averaging,
        ))
    return coupons


def _advance_bdays(d: Date, n: int, calendar) -> Date:
    """Advance d by n business days (n can be negative)."""
    from ql_jax._util.types import TimeUnit, BusinessDayConvention
    if n >= 0:
        return calendar.advance(d, n, TimeUnit.Days, BusinessDayConvention.Following)
    else:
        return calendar.advance(d, n, TimeUnit.Days, BusinessDayConvention.Preceding)
