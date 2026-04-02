"""Schedule generation matching QuantLib's Schedule class.

Generates sequences of payment/reset dates given start, end, tenor,
calendar, business day convention, and date generation rule.
"""

from __future__ import annotations

from ql_jax._util.types import (
    BusinessDayConvention,
    DateGeneration,
    Frequency,
    TimeUnit,
)
from ql_jax.time.date import Date, Period
from ql_jax.time.calendar import Calendar, NullCalendar


class Schedule:
    """A schedule of dates."""

    def __init__(
        self,
        dates: list[Date] | None = None,
        effective_date: Date | None = None,
        termination_date: Date | None = None,
        tenor: Period | None = None,
        calendar: Calendar | None = None,
        convention: int = BusinessDayConvention.Following,
        termination_convention: int | None = None,
        rule: int = DateGeneration.Backward,
        end_of_month: bool = False,
        first_date: Date | None = None,
        next_to_last_date: Date | None = None,
    ):
        self._calendar = calendar or NullCalendar()
        self._tenor = tenor
        self._convention = convention
        self._termination_convention = termination_convention if termination_convention is not None else convention
        self._rule = rule
        self._end_of_month = end_of_month
        self._is_regular: list[bool] = []

        if dates is not None:
            self._dates = list(dates)
            self._is_regular = [True] * max(0, len(dates) - 1)
            return

        if effective_date is None or termination_date is None or tenor is None:
            self._dates = []
            return

        self._dates = _generate_dates(
            effective_date,
            termination_date,
            tenor,
            self._calendar,
            convention,
            self._termination_convention,
            rule,
            end_of_month,
            first_date,
            next_to_last_date,
        )
        self._is_regular = [True] * max(0, len(self._dates) - 1)

    @property
    def dates(self) -> list[Date]:
        return self._dates

    def __len__(self) -> int:
        return len(self._dates)

    def __getitem__(self, i: int) -> Date:
        return self._dates[i]

    @property
    def start_date(self) -> Date:
        return self._dates[0] if self._dates else Date()

    @property
    def end_date(self) -> Date:
        return self._dates[-1] if self._dates else Date()

    @property
    def tenor(self) -> Period | None:
        return self._tenor

    @property
    def calendar(self) -> Calendar:
        return self._calendar

    @property
    def convention(self) -> int:
        return self._convention

    @property
    def rule(self) -> int:
        return self._rule

    @property
    def end_of_month_flag(self) -> bool:
        return self._end_of_month

    def is_regular(self, i: int) -> bool:
        if 0 <= i < len(self._is_regular):
            return self._is_regular[i]
        return True


def _generate_dates(
    effective: Date,
    termination: Date,
    tenor: Period,
    calendar: Calendar,
    convention: int,
    termination_convention: int,
    rule: int,
    end_of_month: bool,
    first_date: Date | None,
    next_to_last_date: Date | None,
) -> list[Date]:
    """Generate schedule dates using Forward or Backward rule."""

    if rule == DateGeneration.Zero:
        return [effective, termination]

    dates: list[Date] = []

    if rule == DateGeneration.Backward:
        dates.append(termination)

        seed = termination
        if next_to_last_date and next_to_last_date != termination:
            dates.insert(0, next_to_last_date)
            seed = next_to_last_date

        periods = 1
        while True:
            d = termination - Period(tenor.length * periods, tenor.units)
            if end_of_month and tenor.units == TimeUnit.Months:
                d = Date.end_of_month(d)
            if d <= effective:
                break
            dates.insert(0, d)
            periods += 1

        if dates[0] != effective:
            dates.insert(0, effective)

        if first_date and len(dates) > 1 and first_date > dates[0] and first_date < dates[-1]:
            # Insert first_date stub
            if first_date not in dates:
                # Find position and insert
                for i in range(1, len(dates)):
                    if dates[i] > first_date:
                        dates.insert(i, first_date)
                        break

    elif rule in (DateGeneration.Forward, DateGeneration.ThirdWednesday):
        dates.append(effective)

        seed = effective
        if first_date and first_date != effective:
            dates.append(first_date)
            seed = first_date

        periods = 1
        while True:
            d = effective + Period(tenor.length * periods, tenor.units)
            if end_of_month and tenor.units == TimeUnit.Months:
                d = Date.end_of_month(d)
            if d >= termination:
                break
            dates.append(d)
            periods += 1

        if dates[-1] != termination:
            dates.append(termination)

        if next_to_last_date and len(dates) > 1 and next_to_last_date > dates[0] and next_to_last_date < dates[-1]:
            if next_to_last_date not in dates:
                for i in range(len(dates) - 1, 0, -1):
                    if dates[i - 1] < next_to_last_date:
                        dates.insert(i, next_to_last_date)
                        break

    else:
        # Fallback to backward
        return _generate_dates(
            effective, termination, tenor, calendar, convention,
            termination_convention, DateGeneration.Backward,
            end_of_month, first_date, next_to_last_date,
        )

    # Adjust for business days (skip first and last for their own conventions)
    adjusted = []
    for i, d in enumerate(dates):
        if i == 0:
            adjusted.append(calendar.adjust(d, convention))
        elif i == len(dates) - 1:
            adjusted.append(calendar.adjust(d, termination_convention))
        else:
            adjusted.append(calendar.adjust(d, convention))

    # Remove duplicates while preserving order
    result = []
    for d in adjusted:
        if not result or d != result[-1]:
            result.append(d)

    return result


# ---------------------------------------------------------------------------
# MakeSchedule builder
# ---------------------------------------------------------------------------

class MakeSchedule:
    """Builder pattern for Schedule construction."""

    def __init__(self):
        self._effective = None
        self._termination = None
        self._tenor = None
        self._calendar = NullCalendar()
        self._convention = BusinessDayConvention.Following
        self._termination_convention = None
        self._rule = DateGeneration.Backward
        self._end_of_month = False
        self._first_date = None
        self._next_to_last_date = None

    def from_date(self, d: Date) -> MakeSchedule:
        self._effective = d
        return self

    def to_date(self, d: Date) -> MakeSchedule:
        self._termination = d
        return self

    def with_tenor(self, p: Period) -> MakeSchedule:
        self._tenor = p
        return self

    def with_frequency(self, f: int) -> MakeSchedule:
        self._tenor = Period.from_frequency(f)
        return self

    def with_calendar(self, c: Calendar) -> MakeSchedule:
        self._calendar = c
        return self

    def with_convention(self, c: int) -> MakeSchedule:
        self._convention = c
        return self

    def with_termination_convention(self, c: int) -> MakeSchedule:
        self._termination_convention = c
        return self

    def with_rule(self, r: int) -> MakeSchedule:
        self._rule = r
        return self

    def forwards(self) -> MakeSchedule:
        self._rule = DateGeneration.Forward
        return self

    def backwards(self) -> MakeSchedule:
        self._rule = DateGeneration.Backward
        return self

    def with_end_of_month(self, eom: bool = True) -> MakeSchedule:
        self._end_of_month = eom
        return self

    def with_first_date(self, d: Date) -> MakeSchedule:
        self._first_date = d
        return self

    def with_next_to_last_date(self, d: Date) -> MakeSchedule:
        self._next_to_last_date = d
        return self

    def build(self) -> Schedule:
        return Schedule(
            effective_date=self._effective,
            termination_date=self._termination,
            tenor=self._tenor,
            calendar=self._calendar,
            convention=self._convention,
            termination_convention=self._termination_convention,
            rule=self._rule,
            end_of_month=self._end_of_month,
            first_date=self._first_date,
            next_to_last_date=self._next_to_last_date,
        )
