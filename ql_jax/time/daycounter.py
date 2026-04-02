"""Day count conventions, matching QuantLib's day counter implementations.

The key function ``year_fraction(d1, d2, convention)`` returns a float
suitable for use in pricing (the bridge from integer dates to JAX floats).
"""

from __future__ import annotations

from ql_jax.time.date import Date, _is_leap, _MONTH_LENGTH


# ---------------------------------------------------------------------------
# Day count convention identifiers
# ---------------------------------------------------------------------------

class DayCountConvention:
    Actual360 = "Actual/360"
    Actual365Fixed = "Actual/365 (Fixed)"
    Actual365NoLeap = "Actual/365 (No Leap)"
    Actual364 = "Actual/364"
    ActualActualISDA = "Actual/Actual (ISDA)"
    ActualActualAFB = "Actual/Actual (AFB)"
    Thirty360BondBasis = "30/360 (Bond Basis)"
    Thirty360EurobondBasis = "30E/360 (Eurobond Basis)"
    Thirty360ISDA = "30/360 (ISDA)"
    Thirty360NASD = "30/360 (NASD)"
    Thirty365 = "30/365"
    Business252 = "Business/252"
    One = "1/1"
    Simple = "Simple"


# ---------------------------------------------------------------------------
# Day count computation
# ---------------------------------------------------------------------------

def day_count(d1: Date, d2: Date, convention: str = DayCountConvention.Actual360) -> int:
    """Return the day count between two dates under a given convention."""
    if convention in (
        DayCountConvention.Actual360,
        DayCountConvention.Actual365Fixed,
        DayCountConvention.Actual365NoLeap,
        DayCountConvention.Actual364,
        DayCountConvention.ActualActualISDA,
        DayCountConvention.ActualActualAFB,
        DayCountConvention.Simple,
    ):
        return d2.serial - d1.serial

    if convention in (
        DayCountConvention.Thirty360BondBasis,
        DayCountConvention.Thirty360EurobondBasis,
        DayCountConvention.Thirty360ISDA,
        DayCountConvention.Thirty360NASD,
        DayCountConvention.Thirty365,
    ):
        return _thirty360_day_count(d1, d2, convention)

    if convention == DayCountConvention.One:
        return 0 if d1 == d2 else 1

    return d2.serial - d1.serial


def year_fraction(
    d1: Date,
    d2: Date,
    convention: str = DayCountConvention.Actual360,
    ref_start: Date | None = None,
    ref_end: Date | None = None,
) -> float:
    """Return the year fraction between two dates under a given convention.

    This is the primary bridge from integer date domain to float pricing domain.
    """
    if d1 == d2:
        return 0.0

    if convention == DayCountConvention.Actual360:
        return (d2.serial - d1.serial) / 360.0

    if convention == DayCountConvention.Actual365Fixed:
        return (d2.serial - d1.serial) / 365.0

    if convention == DayCountConvention.Actual365NoLeap:
        return _actual365_noleap_yf(d1, d2)

    if convention == DayCountConvention.Actual364:
        return (d2.serial - d1.serial) / 364.0

    if convention == DayCountConvention.ActualActualISDA:
        return _actual_actual_isda_yf(d1, d2)

    if convention == DayCountConvention.ActualActualAFB:
        return _actual_actual_afb_yf(d1, d2)

    if convention in (
        DayCountConvention.Thirty360BondBasis,
        DayCountConvention.Thirty360EurobondBasis,
        DayCountConvention.Thirty360ISDA,
        DayCountConvention.Thirty360NASD,
    ):
        return _thirty360_day_count(d1, d2, convention) / 360.0

    if convention == DayCountConvention.Thirty365:
        return _thirty360_day_count(d1, d2, DayCountConvention.Thirty360BondBasis) / 365.0

    if convention == DayCountConvention.One:
        return 0.0 if d1 == d2 else 1.0

    if convention == DayCountConvention.Simple:
        return (d2.serial - d1.serial) / 365.25

    # Fallback
    return (d2.serial - d1.serial) / 365.0


# ---------------------------------------------------------------------------
# Implementation helpers
# ---------------------------------------------------------------------------

def _thirty360_day_count(d1: Date, d2: Date, convention: str) -> int:
    """Compute 30/360 day count using convention-specific day adjustments."""
    dd1, mm1, yy1 = d1.day, d1.month, d1.year
    dd2, mm2, yy2 = d2.day, d2.month, d2.year

    if convention == DayCountConvention.Thirty360BondBasis:
        if dd1 == 31:
            dd1 = 30
        if dd2 == 31 and dd1 >= 30:
            dd2 = 30

    elif convention == DayCountConvention.Thirty360EurobondBasis:
        if dd1 == 31:
            dd1 = 30
        if dd2 == 31:
            dd2 = 30

    elif convention == DayCountConvention.Thirty360ISDA:
        is_last_feb1 = (mm1 == 2 and dd1 == _MONTH_LENGTH[int(_is_leap(yy1))][1])
        is_last_feb2 = (mm2 == 2 and dd2 == _MONTH_LENGTH[int(_is_leap(yy2))][1])
        if dd1 == 31 or is_last_feb1:
            dd1 = 30
        if dd2 == 31 or is_last_feb2:
            if dd1 >= 30:
                dd2 = 30

    elif convention == DayCountConvention.Thirty360NASD:
        if dd1 == 31:
            dd1 = 30
        if dd2 == 31:
            if dd1 >= 30:
                dd2 = 30
            else:
                dd2 = 31  # keep as is per NASD rules

    return 360 * (yy2 - yy1) + 30 * (mm2 - mm1) + (dd2 - dd1)


def _actual_actual_isda_yf(d1: Date, d2: Date) -> float:
    """Actual/Actual (ISDA) year fraction."""
    if d1 == d2:
        return 0.0

    y1 = d1.year
    y2 = d2.year

    if y1 == y2:
        days_in_year = 366.0 if _is_leap(y1) else 365.0
        return (d2.serial - d1.serial) / days_in_year

    # Split across years
    eoy1 = Date(31, 12, y1)
    boy2 = Date(1, 1, y2)

    yf = 0.0
    # Fraction of first year
    days_in_y1 = 366.0 if _is_leap(y1) else 365.0
    yf += (eoy1.serial - d1.serial + 1) / days_in_y1

    # Full years in between
    for y in range(y1 + 1, y2):
        yf += 1.0

    # Fraction of last year
    days_in_y2 = 366.0 if _is_leap(y2) else 365.0
    yf += (d2.serial - boy2.serial) / days_in_y2

    return yf


def _actual_actual_afb_yf(d1: Date, d2: Date) -> float:
    """Actual/Actual (AFB/Euro) year fraction."""
    if d1 == d2:
        return 0.0

    # Count full years backward from d2
    d_temp = d2
    full_years = 0
    while True:
        try:
            d_prev = Date(d_temp.day, d_temp.month, d_temp.year - 1)
        except Exception:
            d_prev = Date(d_temp.day - 1, d_temp.month, d_temp.year - 1)
        if d_prev < d1:
            break
        d_temp = d_prev
        full_years += 1

    # Remaining fraction
    remaining_days = d_temp.serial - d1.serial
    # Denominator: 366 if leap year contains Feb 29 in the period, else 365
    den = 365.0
    if _is_leap(d1.year):
        feb29 = Date(29, 2, d1.year)
        if d1 <= feb29 and feb29 < d_temp:
            den = 366.0
    elif _is_leap(d_temp.year):
        feb29 = Date(29, 2, d_temp.year)
        if d1 <= feb29 and feb29 < d_temp:
            den = 366.0

    return full_years + remaining_days / den


def _actual365_noleap_yf(d1: Date, d2: Date) -> float:
    """Actual/365 (No Leap) — excludes Feb 29 days from the count."""
    if d1 == d2:
        return 0.0

    actual_days = d2.serial - d1.serial

    # Subtract Feb 29 occurrences in [d1, d2)
    y1 = d1.year
    y2 = d2.year
    feb29_count = 0
    for y in range(y1, y2 + 1):
        if _is_leap(y):
            feb29 = Date(29, 2, y)
            if d1 < feb29 and feb29 <= d2:
                feb29_count += 1

    return (actual_days - feb29_count) / 365.0
