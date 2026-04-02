"""Inflation-linked cash flows: CPI and YoY coupons."""

from __future__ import annotations

from dataclasses import dataclass

from ql_jax.time.date import Date
from ql_jax.time.daycounter import year_fraction
from ql_jax.time.schedule import Schedule
from ql_jax.cashflows.fixed_rate import Coupon


@dataclass(frozen=True)
class ZeroInflationCashFlow:
    """Cash flow indexed to cumulative zero-coupon inflation."""
    payment_date: Date
    nominal: float
    observation_date: Date
    base_index: float          # CPI at base date
    observation_lag_months: int = 3

    def amount_with_curve(self, inflation_curve) -> float:
        """Compute amount given an inflation term structure."""
        t = inflation_curve.time_from_reference(self.observation_date)
        index_ratio = inflation_curve.zero_rate(t)
        # For zero inflation: index(t) / base = (1 + r)^t
        tau = t
        if tau > 0:
            growth = (1.0 + index_ratio) ** tau
        else:
            growth = 1.0
        return self.nominal * growth


@dataclass(frozen=True)
class CPICoupon(Coupon):
    """CPI-indexed coupon: nominal * fixedRate * CPI(t)/CPI(base)."""
    rate: float = 0.0
    day_counter: str = "Actual/365 (Fixed)"
    base_cpi: float = 100.0
    observation_lag_months: int = 3
    cpi_at_payment: float | None = None   # known CPI fixing

    @property
    def amount(self) -> float:
        """Amount if CPI at payment is known."""
        if self.cpi_at_payment is not None and self.base_cpi > 0:
            index_ratio = self.cpi_at_payment / self.base_cpi
        else:
            index_ratio = 1.0
        tau = self.accrual_period(self.day_counter)
        return self.nominal * self.rate * tau * index_ratio


@dataclass(frozen=True)
class YoYInflationCoupon(Coupon):
    """Year-on-year inflation coupon: nominal * (yoy_rate + spread) * tau."""
    gearing: float = 1.0
    spread: float = 0.0
    day_counter: str = "Actual/365 (Fixed)"
    yoy_rate: float | None = None   # known YoY rate

    @property
    def amount(self) -> float:
        if self.yoy_rate is not None:
            rate = self.gearing * self.yoy_rate + self.spread
        else:
            rate = self.spread
        tau = self.accrual_period(self.day_counter)
        return self.nominal * rate * tau


# ---------------------------------------------------------------------------
# Leg builders
# ---------------------------------------------------------------------------

def cpi_leg(
    schedule: Schedule,
    notionals: list[float] | float,
    fixed_rate: float,
    base_cpi: float,
    day_counter: str = "Actual/365 (Fixed)",
    observation_lag_months: int = 3,
) -> list[CPICoupon]:
    """Generate a CPI-indexed coupon leg."""
    if isinstance(notionals, (int, float)):
        notionals = [float(notionals)]

    n = len(schedule) - 1
    coupons = []
    for i in range(n):
        start = schedule[i]
        end = schedule[i + 1]
        notional = notionals[min(i, len(notionals) - 1)]
        coupons.append(CPICoupon(
            payment_date=end,
            nominal=notional,
            accrual_start=start,
            accrual_end=end,
            rate=fixed_rate,
            day_counter=day_counter,
            base_cpi=base_cpi,
            observation_lag_months=observation_lag_months,
        ))
    return coupons


def yoy_inflation_leg(
    schedule: Schedule,
    notionals: list[float] | float,
    gearing: float = 1.0,
    spread: float = 0.0,
    day_counter: str = "Actual/365 (Fixed)",
) -> list[YoYInflationCoupon]:
    """Generate a year-on-year inflation coupon leg."""
    if isinstance(notionals, (int, float)):
        notionals = [float(notionals)]

    n = len(schedule) - 1
    coupons = []
    for i in range(n):
        start = schedule[i]
        end = schedule[i + 1]
        notional = notionals[min(i, len(notionals) - 1)]
        coupons.append(YoYInflationCoupon(
            payment_date=end,
            nominal=notional,
            accrual_start=start,
            accrual_end=end,
            gearing=gearing,
            spread=spread,
            day_counter=day_counter,
        ))
    return coupons
