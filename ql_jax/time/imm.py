"""IMM (International Monetary Market) date utilities."""

from __future__ import annotations

from ql_jax.time.date import Date


def is_imm_date(d: Date, main_cycle: bool = True) -> bool:
    """Check if a date is an IMM date.

    IMM dates are the 3rd Wednesday of Mar, Jun, Sep, Dec (main cycle)
    or any month (all months).

    Parameters
    ----------
    d : Date
    main_cycle : if True, only quarterly months; if False, all months

    Returns
    -------
    bool
    """
    if d.weekday() != 4:  # Wednesday (Sunday=1..Saturday=7)
        return False
    if d.day < 15 or d.day > 21:
        return False
    if main_cycle and d.month not in (3, 6, 9, 12):
        return False
    return True


def next_imm_date(d: Date, main_cycle: bool = True) -> Date:
    """Return the next IMM date on or after d.

    Parameters
    ----------
    d : Date
    main_cycle : bool

    Returns
    -------
    Date
    """
    if main_cycle:
        months = [3, 6, 9, 12]
    else:
        months = list(range(1, 13))

    year = d.year
    for _ in range(24):  # search up to 2 years
        for m in months:
            if year == d.year and m < d.month:
                continue
            imm = _third_wednesday(year, m)
            if imm.serial >= d.serial:
                return imm
        year += 1
    return d  # fallback


def imm_code(d: Date) -> str:
    """Return the IMM code for a date (e.g., 'H5' for Mar 2025).

    Format: single letter (month) + last digit(s) of year.
    Month codes: F=Jan, G=Feb, H=Mar, J=Apr, K=May, M=Jun,
                 N=Jul, Q=Aug, U=Sep, V=Oct, X=Nov, Z=Dec
    """
    codes = {1: 'F', 2: 'G', 3: 'H', 4: 'J', 5: 'K', 6: 'M',
             7: 'N', 8: 'Q', 9: 'U', 10: 'V', 11: 'X', 12: 'Z'}
    return codes[d.month] + str(d.year % 10)


def imm_date_from_code(code: str, ref_date: Date = None) -> Date:
    """Parse an IMM code and return the corresponding IMM date.

    Parameters
    ----------
    code : str like 'H5', 'Z4'
    ref_date : Date (used to determine century/decade)

    Returns
    -------
    Date
    """
    month_map = {'F': 1, 'G': 2, 'H': 3, 'J': 4, 'K': 5, 'M': 6,
                 'N': 7, 'Q': 8, 'U': 9, 'V': 10, 'X': 11, 'Z': 12}
    m = month_map[code[0].upper()]
    y_digit = int(code[1:])

    if ref_date is not None:
        year_base = ref_date.year - ref_date.year % 10
    else:
        year_base = 2020
    year = year_base + y_digit
    return _third_wednesday(year, m)


def _third_wednesday(year, month):
    """Find the 3rd Wednesday of the given month."""
    # First day of month
    d = Date(1, month, year)
    wd = d.weekday()
    # Wednesday = 4 (Sunday=1..Saturday=7)
    first_wed = 1 + (4 - wd) % 7
    third_wed = first_wed + 14
    return Date(third_wed, month, year)
