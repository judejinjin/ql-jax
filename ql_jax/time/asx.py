"""ASX (Australian Securities Exchange) date utilities.

ASX futures settle on the 2nd Friday of the contract month.
Main cycle months are March, June, September, December (same as IMM).
"""

from __future__ import annotations

from ql_jax.time.date import Date


def is_asx_date(d: Date, main_cycle: bool = True) -> bool:
    """Check whether *d* is an ASX settlement date (2nd Friday of month)."""
    if d.weekday() != 6:  # Friday (Sunday=1..Saturday=7)
        return False
    if d.day < 8 or d.day > 14:
        return False
    if main_cycle and d.month not in (3, 6, 9, 12):
        return False
    return True


def next_asx_date(d: Date, main_cycle: bool = True) -> Date:
    """Return the next ASX settlement date on or after *d*."""
    months = [3, 6, 9, 12] if main_cycle else list(range(1, 13))

    year = d.year
    for _ in range(24):
        for m in months:
            if year == d.year and m < d.month:
                continue
            asx = _second_friday(year, m)
            if asx.serial >= d.serial:
                return asx
        year += 1
    return d  # fallback


def asx_code(d: Date) -> str:
    """Return the ASX code for a date (same letter convention as IMM)."""
    codes = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
             7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}
    return codes[d.month] + str(d.year % 10)


def asx_date_from_code(code: str, ref_date: Date | None = None) -> Date:
    """Parse an ASX code and return the corresponding ASX date."""
    letter_to_month = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
                       'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
    if len(code) < 2:
        raise ValueError(f"Invalid ASX code: {code}")
    month = letter_to_month.get(code[0].upper())
    if month is None:
        raise ValueError(f"Invalid ASX month letter: {code[0]}")
    year_digit = int(code[1])
    if ref_date is None:
        ref_date = Date(1, 1, 2020)
    base_year = ref_date.year - (ref_date.year % 10) + year_digit
    if base_year < ref_date.year:
        base_year += 10
    return _second_friday(base_year, month)


def _second_friday(year: int, month: int) -> Date:
    """Return the 2nd Friday of the given month/year."""
    first = Date(1, month, year)
    # weekday: 1=Sun, 2=Mon, ..., 6=Fri, 7=Sat
    wd = first.weekday()
    # days until first Friday
    days_to_friday = (6 - wd) % 7
    first_friday_day = 1 + days_to_friday
    second_friday_day = first_friday_day + 7
    return Date(second_friday_day, month, year)
