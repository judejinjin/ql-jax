"""Fixed-rate coupons and leg generation."""

from __future__ import annotations

from dataclasses import dataclass, field

from ql_jax.time.date import Date, Period
from ql_jax.time.daycounter import year_fraction
from ql_jax.time.schedule import Schedule


@dataclass(frozen=True)
class Coupon:
    """Base coupon: known payment date, nominal, and accrual period."""
    payment_date: Date
    nominal: float
    accrual_start: Date
    accrual_end: Date
    ref_start: Date | None = None
    ref_end: Date | None = None

    def accrual_period(self, day_counter: str) -> float:
        return year_fraction(self.accrual_start, self.accrual_end, day_counter,
                             self.ref_start, self.ref_end)


@dataclass(frozen=True)
class FixedRateCoupon(Coupon):
    """Coupon paying a fixed rate over an accrual period."""
    rate: float = 0.0
    day_counter: str = "Actual/365 (Fixed)"

    @property
    def amount(self) -> float:
        return self.nominal * self.rate * self.accrual_period(self.day_counter)


def fixed_rate_leg(
    schedule: Schedule,
    notionals: list[float] | float,
    coupon_rates: list[float] | float,
    day_counter: str = "Actual/365 (Fixed)",
    payment_lag: int = 0,
    payment_calendar=None,
    payment_convention: int | None = None,
) -> list[FixedRateCoupon]:
    """Generate a leg of fixed-rate coupons from a schedule.

    Parameters
    ----------
    schedule : Schedule
        Coupon schedule (N+1 dates for N coupons).
    notionals : float or list[float]
        Notional amount(s). Last value is extended if list is shorter than coupons.
    coupon_rates : float or list[float]
        Coupon rate(s). Last value is extended if list is shorter than coupons.
    day_counter : str
        Day count convention for accrual calculation.
    payment_lag : int
        Business days after accrual end for payment.
    payment_calendar : Calendar or None
        Calendar for adjusting payment dates; defaults to schedule calendar.
    payment_convention : int or None
        Convention for adjusting payment dates; defaults to schedule convention.

    Returns
    -------
    list[FixedRateCoupon]
    """
    if isinstance(notionals, (int, float)):
        notionals = [float(notionals)]
    if isinstance(coupon_rates, (int, float)):
        coupon_rates = [float(coupon_rates)]
    elif not isinstance(coupon_rates, list):
        # Handle JAX scalars and numpy arrays
        coupon_rates = [float(coupon_rates)]

    cal = payment_calendar or schedule.calendar
    conv = payment_convention if payment_convention is not None else schedule.convention

    n = len(schedule) - 1
    coupons: list[FixedRateCoupon] = []
    for i in range(n):
        start = schedule[i]
        end = schedule[i + 1]
        notional = notionals[min(i, len(notionals) - 1)]
        rate = coupon_rates[min(i, len(coupon_rates) - 1)]

        # Payment date: accrual end + lag, adjusted
        pay_date = end
        if payment_lag > 0 and cal is not None:
            from ql_jax._util.types import TimeUnit
            pay_date = cal.advance(end, payment_lag, TimeUnit.Days, conv)
        elif cal is not None and conv is not None:
            pay_date = cal.adjust(end, conv)

        coupons.append(FixedRateCoupon(
            payment_date=pay_date,
            nominal=notional,
            accrual_start=start,
            accrual_end=end,
            rate=rate,
            day_counter=day_counter,
        ))
    return coupons
