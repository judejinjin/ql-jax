"""Date and Period arithmetic, matching QuantLib's serial number convention.

Date serial numbers use the Excel/QuantLib epoch: December 30, 1899 = serial 0.
Valid year range: [1901, 2199].
"""

from __future__ import annotations

from typing import NamedTuple

from ql_jax._util.types import (
    Frequency,
    TimeUnit,
)

# Re-export Month from types (add it there if not present, or define here)
# QuantLib Month: January=1 .. December=12
# We already have Weekday, TimeUnit in types.py. Add Month here since it's date-specific.

# ---------------------------------------------------------------------------
# Month enum (January=1..December=12) — used heavily by date module
# ---------------------------------------------------------------------------
# (Already available via types if added; define here for self-containment)

January, February, March, April = 1, 2, 3, 4
May, June, July, August = 5, 6, 7, 8
September, October, November, December = 9, 10, 11, 12

# ---------------------------------------------------------------------------
# Lookup tables (matching QuantLib date.cpp)
# ---------------------------------------------------------------------------

_MONTH_LENGTH = (
    # Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec
    (31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31),  # non-leap
    (31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31),  # leap
)

_MONTH_OFFSET = (
    # cumulative days before month m (0-indexed months 0..12)
    (0, 31, 59, 90, 120, 151, 181, 212, 243, 273, 304, 334, 365),  # non-leap
    (0, 31, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335, 366),  # leap
)


def _is_leap(year: int) -> bool:
    """Check if a year is a leap year.

    Note: QuantLib treats 1900 as a leap year for Excel compatibility,
    but since our valid range starts at 1901 this doesn't affect us.
    """
    return (year % 4 == 0 and year % 100 != 0) or (year % 400 == 0)


# Precompute year offsets: serial number of Dec 31 of (year - 1)
# yearOffset[y - 1900] = serial of Dec 31, (y-1)
# yearOffset[0] = 0 means Dec 30, 1899 (serial 0 is the epoch)
# yearOffset[1] = 366 (Dec 31, 1900 — QuantLib treats 1900 as leap for Excel compat)
def _build_year_offset():
    offsets = [0] * 301  # years 1900..2200
    offsets[0] = 0
    # QuantLib: yearOffset[1] = 366 (1900 treated as leap year for Excel compat)
    offsets[1] = 366
    for i in range(2, 301):
        y = 1900 + i - 1  # the year whose Dec 31 serial we're computing
        offsets[i] = offsets[i - 1] + (366 if _is_leap(y) else 365)
    return offsets


_YEAR_OFFSET = _build_year_offset()


# ---------------------------------------------------------------------------
# Date class
# ---------------------------------------------------------------------------

class Date:
    """Date represented as an integer serial number (QuantLib/Excel convention).

    Epoch: December 30, 1899 = serial 0.
    """

    __slots__ = ("_serial",)

    def __init__(self, serial_or_day=0, month=None, year=None):
        if month is not None and year is not None:
            # Date(day, month, year) constructor
            day = serial_or_day
            leap = int(_is_leap(year))
            self._serial = (
                day + _MONTH_OFFSET[leap][month - 1] + _YEAR_OFFSET[year - 1900]
            )
        else:
            # Date(serial_number) constructor
            self._serial = int(serial_or_day)

    @property
    def serial(self) -> int:
        return self._serial

    def _decompose(self):
        """Return (day, month, year) from serial number."""
        s = self._serial
        # Find year
        y = s // 365 + 1900
        # Adjust: if serial is before end of guessed year, go back
        while s <= _YEAR_OFFSET[y - 1900]:
            y -= 1
        day_of_year = s - _YEAR_OFFSET[y - 1900]
        leap = int(_is_leap(y))
        # Find month
        m = day_of_year // 30 + 1
        if m > 12:
            m = 12
        while day_of_year <= _MONTH_OFFSET[leap][m - 1]:
            m -= 1
        if m < 1:
            m = 1
        d = day_of_year - _MONTH_OFFSET[leap][m - 1]
        return d, m, y

    @property
    def day(self) -> int:
        return self._decompose()[0]

    @property
    def month(self) -> int:
        return self._decompose()[1]

    @property
    def year(self) -> int:
        return self._decompose()[2]

    @property
    def day_of_year(self) -> int:
        y = self.year
        return self._serial - _YEAR_OFFSET[y - 1900]

    def weekday(self) -> int:
        """Return weekday as Weekday enum value (Sunday=1..Saturday=7)."""
        # QuantLib: serial 0 = Dec 30, 1899 = Saturday
        # So serial % 7: 0=Sat, 1=Sun, 2=Mon, ...
        w = self._serial % 7
        # Map to QuantLib convention: Sunday=1..Saturday=7
        # w=0 → Saturday=7, w=1 → Sunday=1, w=2 → Monday=2, etc.
        return 7 if w == 0 else w

    @staticmethod
    def is_leap(year: int) -> bool:
        return _is_leap(year)

    @staticmethod
    def end_of_month(d: Date) -> Date:
        dm, mm, ym = d._decompose()
        leap = int(_is_leap(ym))
        return Date(
            _MONTH_LENGTH[leap][mm - 1], mm, ym
        )

    @staticmethod
    def is_end_of_month(d: Date) -> bool:
        dm, mm, ym = d._decompose()
        leap = int(_is_leap(ym))
        return dm == _MONTH_LENGTH[leap][mm - 1]

    @staticmethod
    def start_of_month(d: Date) -> Date:
        _, mm, ym = d._decompose()
        return Date(1, mm, ym)

    @staticmethod
    def is_start_of_month(d: Date) -> bool:
        return d.day == 1

    @staticmethod
    def next_weekday(d: Date, weekday: int) -> Date:
        """Return the next date with the given weekday (Sunday=1..Saturday=7)."""
        wd = d.weekday()
        diff = weekday - wd
        if diff <= 0:
            diff += 7
        return Date(d._serial + diff)

    @staticmethod
    def nth_weekday(n: int, weekday: int, month: int, year: int) -> Date:
        """Return the n-th weekday in the given month/year."""
        first = Date(1, month, year)
        wd = first.weekday()
        diff = weekday - wd
        if diff < 0:
            diff += 7
        return Date(first._serial + diff + 7 * (n - 1))

    @staticmethod
    def min_date() -> Date:
        return Date(1, 1, 1901)

    @staticmethod
    def max_date() -> Date:
        return Date(31, 12, 2199)

    @staticmethod
    def month_length(month: int, year: int) -> int:
        leap = int(_is_leap(year))
        return _MONTH_LENGTH[leap][month - 1]

    # --- Arithmetic ---

    def __add__(self, other):
        if isinstance(other, int):
            return Date(self._serial + other)
        if isinstance(other, Period):
            return _advance_date(self, other.length, other.units)
        return NotImplemented

    def __radd__(self, other):
        if isinstance(other, int):
            return Date(self._serial + other)
        return NotImplemented

    def __sub__(self, other):
        if isinstance(other, int):
            return Date(self._serial - other)
        if isinstance(other, Date):
            return self._serial - other._serial
        if isinstance(other, Period):
            return _advance_date(self, -other.length, other.units)
        return NotImplemented

    # --- Comparison ---
    def __eq__(self, other):
        if isinstance(other, Date):
            return self._serial == other._serial
        return NotImplemented

    def __ne__(self, other):
        if isinstance(other, Date):
            return self._serial != other._serial
        return NotImplemented

    def __lt__(self, other):
        if isinstance(other, Date):
            return self._serial < other._serial
        return NotImplemented

    def __le__(self, other):
        if isinstance(other, Date):
            return self._serial <= other._serial
        return NotImplemented

    def __gt__(self, other):
        if isinstance(other, Date):
            return self._serial > other._serial
        return NotImplemented

    def __ge__(self, other):
        if isinstance(other, Date):
            return self._serial >= other._serial
        return NotImplemented

    def __hash__(self):
        return hash(self._serial)

    def __repr__(self):
        if self._serial == 0:
            return "Date()"
        d, m, y = self._decompose()
        month_names = [
            "", "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ]
        return f"Date({d}, {month_names[m]}, {y})"

    def __bool__(self):
        return self._serial != 0


def _advance_date(d: Date, n: int, units: int) -> Date:
    """Advance a date by n units (Days, Weeks, Months, Years)."""
    if units == TimeUnit.Days:
        return Date(d._serial + n)
    elif units == TimeUnit.Weeks:
        return Date(d._serial + 7 * n)
    elif units == TimeUnit.Months:
        day, month, year = d._decompose()
        m = month + n
        # Normalize month
        while m > 12:
            m -= 12
            year += 1
        while m < 1:
            m += 12
            year -= 1
        leap = int(_is_leap(year))
        max_day = _MONTH_LENGTH[leap][m - 1]
        day = min(day, max_day)
        return Date(day, m, year)
    elif units == TimeUnit.Years:
        day, month, year = d._decompose()
        year += n
        leap = int(_is_leap(year))
        max_day = _MONTH_LENGTH[leap][month - 1]
        day = min(day, max_day)
        return Date(day, month, year)
    else:
        raise ValueError(f"Unsupported time unit: {units}")


# ---------------------------------------------------------------------------
# Period class
# ---------------------------------------------------------------------------

class Period(NamedTuple):
    """A time period with a length and time unit."""
    length: int
    units: int  # TimeUnit value

    @staticmethod
    def from_frequency(freq: int) -> Period:
        """Create a Period from a Frequency enum value."""
        mapping = {
            Frequency.Annual: Period(1, TimeUnit.Years),
            Frequency.Semiannual: Period(6, TimeUnit.Months),
            Frequency.EveryFourthMonth: Period(4, TimeUnit.Months),
            Frequency.Quarterly: Period(3, TimeUnit.Months),
            Frequency.Bimonthly: Period(2, TimeUnit.Months),
            Frequency.Monthly: Period(1, TimeUnit.Months),
            Frequency.EveryFourthWeek: Period(4, TimeUnit.Weeks),
            Frequency.Biweekly: Period(2, TimeUnit.Weeks),
            Frequency.Weekly: Period(1, TimeUnit.Weeks),
            Frequency.Daily: Period(1, TimeUnit.Days),
        }
        if freq in mapping:
            return mapping[freq]
        raise ValueError(f"Cannot convert frequency {freq} to period")

    def __repr__(self):
        unit_names = {
            TimeUnit.Days: "D",
            TimeUnit.Weeks: "W",
            TimeUnit.Months: "M",
            TimeUnit.Years: "Y",
        }
        u = unit_names.get(self.units, "?")
        return f"{self.length}{u}"


def days_between(d1: Date, d2: Date) -> int:
    """Return the number of days between two dates."""
    return d2.serial - d1.serial
