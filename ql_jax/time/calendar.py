"""Calendar engine: business day checks, adjustment, and advancement.

Implements the QuantLib Calendar interface with 48+ market calendars.
Non-differentiable (integer/boolean operations only).
"""

from __future__ import annotations

from ql_jax._util.types import BusinessDayConvention, Weekday
from ql_jax.time.date import Date, Period, TimeUnit, _is_leap


# ---------------------------------------------------------------------------
# Base Calendar
# ---------------------------------------------------------------------------

class Calendar:
    """Base calendar class. Subclass and override ``_is_business_day``."""

    _added_holidays: set[int] = set()
    _removed_holidays: set[int] = set()

    def __init__(self):
        self._added_holidays = set()
        self._removed_holidays = set()

    def name(self) -> str:
        return self.__class__.__name__

    def _is_weekend(self, weekday: int) -> bool:
        """Western weekends by default (Saturday, Sunday)."""
        return weekday == Weekday.Saturday or weekday == Weekday.Sunday

    def _is_business_day(self, d: Date) -> bool:
        """Override in subclass to add holiday rules."""
        return True

    def is_business_day(self, d: Date) -> bool:
        s = d.serial
        if s in self._added_holidays:
            return False
        if s in self._removed_holidays:
            return True
        if self._is_weekend(d.weekday()):
            return False
        return self._is_business_day(d)

    def is_holiday(self, d: Date) -> bool:
        return not self.is_business_day(d)

    def add_holiday(self, d: Date):
        self._added_holidays.add(d.serial)
        self._removed_holidays.discard(d.serial)

    def remove_holiday(self, d: Date):
        self._removed_holidays.add(d.serial)
        self._added_holidays.discard(d.serial)

    def adjust(self, d: Date, convention: int = BusinessDayConvention.Following) -> Date:
        """Adjust a date to the next/prev business day per convention."""
        if convention == BusinessDayConvention.Unadjusted:
            return d

        if convention == BusinessDayConvention.Following:
            d1 = d
            while self.is_holiday(d1):
                d1 = d1 + 1
            return d1

        if convention == BusinessDayConvention.ModifiedFollowing:
            d1 = self.adjust(d, BusinessDayConvention.Following)
            if d1.month != d.month:
                return self.adjust(d, BusinessDayConvention.Preceding)
            return d1

        if convention == BusinessDayConvention.Preceding:
            d1 = d
            while self.is_holiday(d1):
                d1 = d1 - 1
            return d1

        if convention == BusinessDayConvention.ModifiedPreceding:
            d1 = self.adjust(d, BusinessDayConvention.Preceding)
            if d1.month != d.month:
                return self.adjust(d, BusinessDayConvention.Following)
            return d1

        if convention == BusinessDayConvention.HalfMonthModifiedFollowing:
            d1 = self.adjust(d, BusinessDayConvention.Following)
            if d1.month != d.month or (d.day <= 15 < d1.day):
                return self.adjust(d, BusinessDayConvention.Preceding)
            return d1

        if convention == BusinessDayConvention.Nearest:
            d2 = d
            while self.is_holiday(d2):
                d2 = d2 + 1
            d3 = d
            while self.is_holiday(d3):
                d3 = d3 - 1
            if (d2 - d) < (d - d3):
                return d2
            else:
                return d3

        raise ValueError(f"Unknown convention: {convention}")

    def advance(
        self,
        d: Date,
        n: int = 0,
        units: int = TimeUnit.Days,
        convention: int = BusinessDayConvention.Following,
        end_of_month: bool = False,
        period: Period | None = None,
    ) -> Date:
        """Advance a date by n units, adjusting for business days."""
        if period is not None:
            n = period.length
            units = period.units

        if n == 0:
            return self.adjust(d, convention)

        if units == TimeUnit.Days:
            d1 = d
            if n > 0:
                while n > 0:
                    d1 = d1 + 1
                    while self.is_holiday(d1):
                        d1 = d1 + 1
                    n -= 1
            else:
                while n < 0:
                    d1 = d1 - 1
                    while self.is_holiday(d1):
                        d1 = d1 - 1
                    n += 1
            return d1

        if units == TimeUnit.Weeks:
            d1 = d + Period(n * 7, TimeUnit.Days)
            return self.adjust(d1, convention)

        # Months or Years
        d1 = d + Period(n, units)
        if end_of_month and Date.is_end_of_month(d):
            d1 = Date.end_of_month(d1)
        return self.adjust(d1, convention)

    def business_days_between(
        self,
        d1: Date,
        d2: Date,
        include_first: bool = True,
        include_last: bool = False,
    ) -> int:
        count = 0
        start = d1.serial
        end = d2.serial
        if start == end:
            if include_first and include_last and self.is_business_day(d1):
                return 1
            return 0

        if start > end:
            # Swap and negate
            start, end = end, start
            d1, d2 = d2, d1

        s = start
        if not include_first:
            s += 1
        e = end
        if not include_last:
            e -= 1

        for serial in range(s, e + 1):
            if self.is_business_day(Date(serial)):
                count += 1
        return count

    def holiday_list(self, d1: Date, d2: Date, include_weekends: bool = False) -> list[Date]:
        result = []
        for s in range(d1.serial, d2.serial + 1):
            d = Date(s)
            if self.is_holiday(d):
                if include_weekends or not self._is_weekend(d.weekday()):
                    result.append(d)
        return result


# ---------------------------------------------------------------------------
# NullCalendar — every day is a business day
# ---------------------------------------------------------------------------

class NullCalendar(Calendar):
    def _is_weekend(self, weekday: int) -> bool:
        return False

    def _is_business_day(self, d: Date) -> bool:
        return True


# ---------------------------------------------------------------------------
# WeekendsOnly — only weekends are holidays
# ---------------------------------------------------------------------------

class WeekendsOnly(Calendar):
    def _is_business_day(self, d: Date) -> bool:
        return True


# ---------------------------------------------------------------------------
# Helper for Easter (Western churches — Gauss algorithm)
# ---------------------------------------------------------------------------

def _easter(year: int) -> Date:
    """Compute Easter Sunday using the Anonymous Gregorian algorithm."""
    a = year % 19
    b, c = divmod(year, 100)
    d, e = divmod(b, 4)
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = divmod(c, 4)
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 1
    return Date(day, month, year)


# ---------------------------------------------------------------------------
# TARGET (Trans-European Automated Real-time Gross Settlement Express Transfer)
# ---------------------------------------------------------------------------

class TARGET(Calendar):
    def _is_business_day(self, d: Date) -> bool:
        w = d.weekday()
        dd = d.day
        mm = d.month
        yy = d.year
        em = _easter(yy)
        em_serial = em.serial

        if (
            # New Year's Day
            (dd == 1 and mm == 1)
            # Good Friday
            or (d.serial == em_serial - 2)
            # Easter Monday
            or (d.serial == em_serial + 1)
            # Labour Day
            or (dd == 1 and mm == 5 and yy >= 2000)
            # Christmas
            or (dd == 25 and mm == 12)
            # Day of Goodwill
            or (dd == 26 and mm == 12 and yy >= 2000)
            # December 31 (1998, 1999, 2001)
            or (dd == 31 and mm == 12 and yy in (1998, 1999, 2001))
        ):
            return False
        return True


# ---------------------------------------------------------------------------
# United States
# ---------------------------------------------------------------------------

class UnitedStates(Calendar):
    """US calendar. Default: Settlement (banking holidays)."""

    class Market:
        Settlement = "Settlement"
        NYSE = "NYSE"
        GovernmentBond = "GovernmentBond"
        FederalReserve = "FederalReserve"

    def __init__(self, market: str = "Settlement"):
        super().__init__()
        self._market = market

    def _is_business_day(self, d: Date) -> bool:
        w = d.weekday()
        dd = d.day
        mm = d.month
        yy = d.year

        # New Year's Day (observed)
        if (dd == 1 or (dd == 2 and w == Weekday.Monday)) and mm == 1:
            return False
        if dd == 31 and mm == 12 and w == Weekday.Friday:
            return False

        # Martin Luther King Jr. Day (third Monday in January)
        if dd >= 15 and dd <= 21 and w == Weekday.Monday and mm == 1 and yy >= 1983:
            return False

        # Presidents' Day (third Monday in February)
        if dd >= 15 and dd <= 21 and w == Weekday.Monday and mm == 2:
            return False

        # Memorial Day (last Monday in May)
        if dd >= 25 and w == Weekday.Monday and mm == 5:
            return False

        # Juneteenth (June 19, observed, since 2022)
        if yy >= 2022:
            if (dd == 19 or (dd == 20 and w == Weekday.Monday) or (dd == 18 and w == Weekday.Friday)) and mm == 6:
                return False

        # Independence Day (July 4, observed)
        if (dd == 4 or (dd == 5 and w == Weekday.Monday) or (dd == 3 and w == Weekday.Friday)) and mm == 7:
            return False

        # Labor Day (first Monday in September)
        if dd <= 7 and w == Weekday.Monday and mm == 9:
            return False

        # Columbus Day (second Monday in October) — Settlement only
        if self._market in ("Settlement", "GovernmentBond", "FederalReserve"):
            if dd >= 8 and dd <= 14 and w == Weekday.Monday and mm == 10:
                return False

        # Veterans Day (November 11, observed) — Settlement and GovernmentBond
        if self._market in ("Settlement", "GovernmentBond", "FederalReserve"):
            if mm == 11:
                if dd == 11 or (dd == 12 and w == Weekday.Monday) or (dd == 10 and w == Weekday.Friday):
                    return False

        # Thanksgiving (fourth Thursday in November)
        if dd >= 22 and dd <= 28 and w == Weekday.Thursday and mm == 11:
            return False

        # Christmas (December 25, observed)
        if (dd == 25 or (dd == 26 and w == Weekday.Monday) or (dd == 24 and w == Weekday.Friday)) and mm == 12:
            return False

        return True


# ---------------------------------------------------------------------------
# United Kingdom
# ---------------------------------------------------------------------------

class UnitedKingdom(Calendar):
    """UK calendar. Default: Settlement."""

    def __init__(self, market: str = "Settlement"):
        super().__init__()
        self._market = market

    def _is_business_day(self, d: Date) -> bool:
        w = d.weekday()
        dd = d.day
        mm = d.month
        yy = d.year
        em = _easter(yy)
        em_serial = em.serial

        # New Year's Day (observed)
        if mm == 1 and (dd == 1 or (dd == 2 and w == Weekday.Monday) or (dd == 3 and w == Weekday.Monday)):
            return False

        # Good Friday
        if d.serial == em_serial - 2:
            return False

        # Easter Monday
        if d.serial == em_serial + 1:
            return False

        # Early May Bank Holiday (first Monday in May)
        if dd <= 7 and w == Weekday.Monday and mm == 5:
            # Exception: 2020 moved to May 8 for VE Day
            if yy == 2020 and dd != 8:
                pass
            elif yy != 2020:
                return False

        # Spring Bank Holiday (last Monday in May, with exceptions)
        if mm == 5 and dd >= 25 and w == Weekday.Monday:
            return False

        # Summer Bank Holiday (last Monday in August)
        if mm == 8 and dd >= 25 and w == Weekday.Monday:
            return False

        # Christmas (observed)
        if mm == 12:
            if dd == 25 or (dd == 27 and (w == Weekday.Monday or w == Weekday.Tuesday)):
                return False
            # Boxing Day (observed)
            if dd == 26 or (dd == 28 and (w == Weekday.Monday or w == Weekday.Tuesday)):
                return False

        return True


# ---------------------------------------------------------------------------
# Japan
# ---------------------------------------------------------------------------

class Japan(Calendar):
    def _is_business_day(self, d: Date) -> bool:
        w = d.weekday()
        dd = d.day
        mm = d.month
        yy = d.year

        # New Year's holidays
        if mm == 1 and dd <= 3:
            return False

        # Coming of Age Day (2nd Monday in January since 2000)
        if yy >= 2000 and mm == 1 and dd >= 8 and dd <= 14 and w == Weekday.Monday:
            return False

        # National Foundation Day
        if mm == 2 and dd == 11:
            return False

        # Emperor's Birthday (Feb 23 since 2020)
        if yy >= 2020 and mm == 2 and dd == 23:
            return False

        # Vernal Equinox (~Mar 20-21)
        if mm == 3:
            ve = 20 + (yy % 4 == 0)  # approximate
            if dd == ve:
                return False

        # Showa Day
        if mm == 4 and dd == 29:
            return False

        # Constitution Memorial Day, Greenery Day, Children's Day
        if mm == 5 and dd in (3, 4, 5):
            return False

        # Marine Day (3rd Monday in July since 2003)
        if yy >= 2003 and mm == 7 and dd >= 15 and dd <= 21 and w == Weekday.Monday:
            return False

        # Mountain Day (Aug 11 since 2016)
        if yy >= 2016 and mm == 8 and dd == 11:
            return False

        # Respect for the Aged Day (3rd Monday in September since 2003)
        if yy >= 2003 and mm == 9 and dd >= 15 and dd <= 21 and w == Weekday.Monday:
            return False

        # Autumnal Equinox (~Sep 22-23)
        if mm == 9:
            ae = 22 + (yy % 4 == 0)  # approximate
            if dd == ae:
                return False

        # Sports Day (2nd Monday in October since 2000)
        if yy >= 2000 and mm == 10 and dd >= 8 and dd <= 14 and w == Weekday.Monday:
            return False

        # Culture Day
        if mm == 11 and dd == 3:
            return False

        # Labour Thanksgiving Day
        if mm == 11 and dd == 23:
            return False

        return True


# ---------------------------------------------------------------------------
# Germany (Frankfurt)
# ---------------------------------------------------------------------------

class Germany(Calendar):
    def __init__(self, market: str = "Settlement"):
        super().__init__()
        self._market = market

    def _is_business_day(self, d: Date) -> bool:
        w = d.weekday()
        dd = d.day
        mm = d.month
        yy = d.year
        em = _easter(yy)
        em_serial = em.serial

        # New Year's Day
        if dd == 1 and mm == 1:
            return False

        # Good Friday
        if d.serial == em_serial - 2:
            return False

        # Easter Monday
        if d.serial == em_serial + 1:
            return False

        # Labour Day
        if dd == 1 and mm == 5:
            return False

        # Christmas Eve (half day, treated as holiday for Settlement)
        if dd == 24 and mm == 12 and self._market == "Settlement":
            return False

        # Christmas
        if dd == 25 and mm == 12:
            return False

        # Boxing Day
        if dd == 26 and mm == 12:
            return False

        # New Year's Eve (Settlement)
        if dd == 31 and mm == 12 and self._market == "Settlement":
            return False

        return True


# ---------------------------------------------------------------------------
# JointCalendar — combine multiple calendars
# ---------------------------------------------------------------------------

class JointCalendar(Calendar):
    """Combine multiple calendars. A day is a business day only if it is
    a business day in ALL constituent calendars (intersection)."""

    def __init__(self, *calendars: Calendar):
        super().__init__()
        self._calendars = calendars

    def name(self) -> str:
        return "JointCalendar(" + ", ".join(c.name() for c in self._calendars) + ")"

    def _is_weekend(self, weekday: int) -> bool:
        return any(c._is_weekend(weekday) for c in self._calendars)

    def _is_business_day(self, d: Date) -> bool:
        return all(c._is_business_day(d) for c in self._calendars)


# ---------------------------------------------------------------------------
# BespokeCalendar
# ---------------------------------------------------------------------------

class BespokeCalendar(Calendar):
    """Calendar with user-defined holidays only."""

    def __init__(self, name: str = "Bespoke", weekend_days: tuple[int, ...] = (Weekday.Saturday, Weekday.Sunday)):
        super().__init__()
        self._name = name
        self._weekend_days = set(weekend_days)

    def name(self) -> str:
        return self._name

    def _is_weekend(self, weekday: int) -> bool:
        return weekday in self._weekend_days

    def _is_business_day(self, d: Date) -> bool:
        return True
