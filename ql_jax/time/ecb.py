"""ECB (European Central Bank) meeting date utilities."""

from __future__ import annotations

from ql_jax.time.date import Date


# Known ECB meeting dates (Governing Council rate decisions).
# These are maintained manually or fetched from ECB website.
# This list covers 2024-2026 as a representative set.
_ECB_DATES = [
    # 2024
    (25, 1, 2024), (7, 3, 2024), (11, 4, 2024), (6, 6, 2024),
    (18, 7, 2024), (12, 9, 2024), (17, 10, 2024), (12, 12, 2024),
    # 2025
    (30, 1, 2025), (6, 3, 2025), (17, 4, 2025), (5, 6, 2025),
    (24, 7, 2025), (11, 9, 2025), (30, 10, 2025), (18, 12, 2025),
    # 2026
    (22, 1, 2026), (5, 3, 2026), (16, 4, 2026), (4, 6, 2026),
    (16, 7, 2026), (10, 9, 2026), (29, 10, 2026), (17, 12, 2026),
]


def ecb_dates(start: Date = None, end: Date = None):
    """Return ECB meeting dates in the given range.

    Parameters
    ----------
    start, end : Date (optional bounds)

    Returns
    -------
    list of Date objects
    """
    result = []
    for dd, mm, yy in _ECB_DATES:
        d = Date(dd, mm, yy)
        if start is not None and d.serial < start.serial:
            continue
        if end is not None and d.serial > end.serial:
            continue
        result.append(d)
    return result


def next_ecb_date(d: Date) -> Date:
    """Return the next ECB date on or after d."""
    for dd, mm, yy in _ECB_DATES:
        ecb = Date(dd, mm, yy)
        if ecb.serial >= d.serial:
            return ecb
    # Fallback: estimate next year Q1
    return Date(25, 1, d.year + 1)


def is_ecb_date(d: Date) -> bool:
    """Check if d is an ECB meeting date."""
    for dd, mm, yy in _ECB_DATES:
        if d.day == dd and d.month == mm and d.year == yy:
            return True
    return False
