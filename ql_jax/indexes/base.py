"""Interest rate index base classes.

QuantLib's InterestRateIndex is the base class for IBOR indexes, overnight
indexes, and swap indexes. In QL-JAX, indexes are frozen dataclasses holding
metadata, with fixing lookup via dict and rate forecasting via term structure.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from ql_jax.time.date import Date
from ql_jax.time.calendar import Calendar, WeekendsOnly
from ql_jax._util.types import BusinessDayConvention, Frequency


@dataclass
class InterestRateIndex:
    """Base interest rate index."""
    family_name: str
    tenor_months: int  # 0 for overnight indexes
    fixing_days: int
    currency_code: str
    day_counter_convention: str  # e.g. 'Actual360'
    calendar: Calendar = field(default_factory=WeekendsOnly)

    # Fixing history — mutable dict for adding fixings
    _fixings: dict[int, float] = field(default_factory=dict, repr=False)

    @property
    def name(self) -> str:
        if self.tenor_months == 0:
            return self.family_name
        return f"{self.family_name}{self.tenor_months}M"

    def add_fixing(self, date: Date, value: float):
        """Add a historical fixing."""
        self._fixings[date.serial] = value

    def add_fixings(self, dates: list[Date], values: list[float]):
        """Add multiple historical fixings."""
        for d, v in zip(dates, values):
            self._fixings[d.serial] = v

    def fixing(self, date: Date, forecast_curve=None) -> float:
        """Get fixing for date: historical if available, else forecast."""
        if date.serial in self._fixings:
            return self._fixings[date.serial]
        if forecast_curve is not None:
            return self._forecast_fixing(date, forecast_curve)
        raise KeyError(f"No fixing for {self.name} on {date}")

    def _forecast_fixing(self, date: Date, curve) -> float:
        """Forecast rate from term structure."""
        raise NotImplementedError("Subclass must implement _forecast_fixing")

    def fixing_date(self, value_date: Date) -> Date:
        """Get the fixing date for a given value date."""
        from ql_jax._util.types import TimeUnit, BusinessDayConvention
        d = value_date
        for _ in range(self.fixing_days):
            d = self.calendar.advance(d, -1, TimeUnit.Days, BusinessDayConvention.Preceding)
        return d

    def value_date(self, fixing_date: Date) -> Date:
        """Get the value date for a given fixing date."""
        from ql_jax._util.types import TimeUnit, BusinessDayConvention
        d = fixing_date
        for _ in range(self.fixing_days):
            d = self.calendar.advance(d, 1, TimeUnit.Days, BusinessDayConvention.Following)
        return d

    def maturity_date(self, value_date: Date) -> Date:
        """Get the maturity date for a given value date."""
        from ql_jax.time.date import _advance_date, TimeUnit
        return _advance_date(value_date, self.tenor_months, TimeUnit.Months)

    def is_valid_fixing_date(self, date: Date) -> bool:
        return self.calendar.is_business_day(date)

    def clear_fixings(self):
        self._fixings.clear()
