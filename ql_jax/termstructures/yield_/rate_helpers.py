"""Rate helpers for piecewise yield curve bootstrapping.

Each helper represents a market instrument (deposit, FRA, swap, etc.)
and knows how to compute its implied rate from a discount curve.
"""

from __future__ import annotations
from dataclasses import dataclass

import jax.numpy as jnp

from ql_jax.time.date import Date, _advance_date, TimeUnit
from ql_jax.time.daycounter import year_fraction


@dataclass
class RateHelper:
    """Base class for rate helpers."""
    quote: float  # Market quote (rate or price)
    pillar_date: Date  # The date this helper is associated with

    def implied_quote(self, curve) -> float:
        """Compute the implied quote from the curve. Override in subclass."""
        raise NotImplementedError


@dataclass
class DepositRateHelper(RateHelper):
    """Deposit rate helper.

    A deposit: lend at t_start, receive at t_end.
    The quoted rate r is a simple rate:
        df(t_start) / df(t_end) = 1 + r * tau
    """
    start_date: Date = None
    end_date: Date = None
    day_counter: str = 'Actual360'

    def __post_init__(self):
        if self.end_date is not None:
            self.pillar_date = self.end_date

    def implied_quote(self, curve) -> float:
        t_start = curve.time_from_reference(self.start_date)
        t_end = curve.time_from_reference(self.end_date)
        df_start = curve.discount(t_start)
        df_end = curve.discount(t_end)
        tau = year_fraction(self.start_date, self.end_date, self.day_counter)
        return (df_start / df_end - 1.0) / tau


@dataclass
class FraRateHelper(RateHelper):
    """Forward rate agreement helper."""
    start_date: Date = None
    end_date: Date = None
    day_counter: str = 'Actual360'

    def __post_init__(self):
        if self.end_date is not None:
            self.pillar_date = self.end_date

    def implied_quote(self, curve) -> float:
        t_start = curve.time_from_reference(self.start_date)
        t_end = curve.time_from_reference(self.end_date)
        df_start = curve.discount(t_start)
        df_end = curve.discount(t_end)
        tau = year_fraction(self.start_date, self.end_date, self.day_counter)
        return (df_start / df_end - 1.0) / tau


@dataclass
class SwapRateHelper(RateHelper):
    """Vanilla swap rate helper.

    A swap with fixed rate equal to the quoted swap rate has zero NPV.
    We compute the fair swap rate from the curve.
    """
    start_date: Date = None
    tenor_months: int = 60  # 5Y
    fixed_leg_frequency_months: int = 12  # Annual
    fixed_day_counter: str = 'Thirty360'
    float_day_counter: str = 'Actual360'

    def __post_init__(self):
        if self.start_date is not None and self.pillar_date is None:
            self.pillar_date = _advance_date(
                self.start_date, self.tenor_months, TimeUnit.Months
            )

    def implied_quote(self, curve) -> float:
        """Fair swap rate from the curve."""
        start = self.start_date
        n_periods = self.tenor_months // self.fixed_leg_frequency_months
        annuity = 0.0
        prev = start
        for i in range(1, n_periods + 1):
            payment = _advance_date(
                start, i * self.fixed_leg_frequency_months, TimeUnit.Months
            )
            tau = year_fraction(prev, payment, self.fixed_day_counter)
            df = float(curve.discount(curve.time_from_reference(payment)))
            annuity += tau * df
            prev = payment

        end = _advance_date(start, self.tenor_months, TimeUnit.Months)
        df_start = float(curve.discount(curve.time_from_reference(start)))
        df_end = float(curve.discount(curve.time_from_reference(end)))

        if annuity <= 0:
            return 0.0
        return (df_start - df_end) / annuity


@dataclass
class OISRateHelper(RateHelper):
    """Overnight Index Swap rate helper.

    Similar to SwapRateHelper but the floating leg uses compounded overnight rates.
    """
    start_date: Date = None
    tenor_months: int = 12
    day_counter: str = 'Actual360'

    def __post_init__(self):
        if self.start_date is not None and self.pillar_date is None:
            self.pillar_date = _advance_date(
                self.start_date, self.tenor_months, TimeUnit.Months
            )

    def implied_quote(self, curve) -> float:
        end = _advance_date(self.start_date, self.tenor_months, TimeUnit.Months)
        t_start = curve.time_from_reference(self.start_date)
        t_end = curve.time_from_reference(end)
        df_start = curve.discount(t_start)
        df_end = curve.discount(t_end)
        tau = year_fraction(self.start_date, end, self.day_counter)
        if tau <= 0:
            return 0.0
        return (df_start / df_end - 1.0) / tau


@dataclass
class FuturesRateHelper(RateHelper):
    """Futures rate helper (e.g. Eurodollar futures).

    Futures price = 100 - rate * 100
    """
    start_date: Date = None
    end_date: Date = None
    day_counter: str = 'Actual360'
    convexity_adjustment: float = 0.0

    def __post_init__(self):
        if self.end_date is not None:
            self.pillar_date = self.end_date

    def implied_quote(self, curve) -> float:
        t_start = curve.time_from_reference(self.start_date)
        t_end = curve.time_from_reference(self.end_date)
        df_start = curve.discount(t_start)
        df_end = curve.discount(t_end)
        tau = year_fraction(self.start_date, self.end_date, self.day_counter)
        rate = (df_start / df_end - 1.0) / tau
        # Futures price = 100 * (1 - rate - convexity)
        return 100.0 * (1.0 - rate - self.convexity_adjustment)
