"""Additional market calendars for QL-JAX.

Supplements the core calendars in time/calendar.py with additional markets.
"""

from ql_jax.time.calendar import Calendar
from ql_jax.time.date import Date


class Canada(Calendar):
    """Canadian calendar (TSX / Settlement)."""

    def __init__(self, market="Settlement"):
        super().__init__()
        self._market = market

    def name(self):
        return f"Canada({self._market})"

    def _is_business_day(self, d: Date) -> bool:
        dd, mm, yy = d.day, d.month, d.year
        w = d.weekday()
        em = _easter_monday(yy)

        # New Year's
        if (dd == 1 or (dd == 2 and w == 1)) and mm == 1:
            return False
        # Family Day (3rd Monday of Feb, since 2008)
        if yy >= 2008 and mm == 2 and w == 1 and 15 <= dd <= 21:
            return False
        # Good Friday
        if d.serial == em - 3:
            return False
        # Victoria Day (Monday before May 25)
        if mm == 5 and w == 1 and 18 <= dd <= 24:
            return False
        # Canada Day
        if (dd == 1 or (dd == 2 and w == 1) or (dd == 3 and w == 1)) and mm == 7:
            return False
        # Civic Holiday (1st Monday of August)
        if mm == 8 and w == 1 and dd <= 7:
            return False
        # Labour Day (1st Monday of September)
        if mm == 9 and w == 1 and dd <= 7:
            return False
        # National Day for Truth and Reconciliation (Sept 30, since 2021)
        if yy >= 2021 and mm == 9 and (dd == 30 or (dd == 29 and w == 5)):
            return False
        # Thanksgiving (2nd Monday of October)
        if mm == 10 and w == 1 and 8 <= dd <= 14:
            return False
        # Remembrance Day
        if mm == 11 and (dd == 11 or (dd == 12 and w == 1)):
            return False
        # Christmas
        if mm == 12 and (dd == 25 or (dd == 27 and (w == 1 or w == 2))):
            return False
        # Boxing Day
        if mm == 12 and (dd == 26 or (dd == 28 and (w == 1 or w == 2))):
            return False

        return True


class Australia(Calendar):
    """Australian calendar."""

    def __init__(self, market="Settlement"):
        super().__init__()
        self._market = market

    def name(self):
        return f"Australia({self._market})"

    def _is_business_day(self, d: Date) -> bool:
        dd, mm, yy = d.day, d.month, d.year
        w = d.weekday()
        em = _easter_monday(yy)

        if (dd == 1 or (dd == 2 and w == 1)) and mm == 1:
            return False
        if mm == 1 and w == 1 and 22 <= dd <= 28:
            return False  # Australia Day (approx)
        if d.serial == em - 3:
            return False  # Good Friday
        if d.serial == em - 2:
            return False  # Easter Saturday
        if d.serial == em:
            return False  # Easter Monday
        if mm == 4 and dd == 25:
            return False  # ANZAC Day
        if mm == 6 and w == 1 and 8 <= dd <= 14:
            return False  # Queen's Birthday (approx)
        if mm == 12 and dd == 25:
            return False
        if mm == 12 and dd == 26:
            return False

        return True


class HongKong(Calendar):
    """Hong Kong SAR calendar."""

    def name(self):
        return "HongKong"

    def _is_business_day(self, d: Date) -> bool:
        dd, mm, yy = d.day, d.month, d.year
        w = d.weekday()
        em = _easter_monday(yy)

        if dd == 1 and mm == 1:
            return False
        if d.serial == em - 3:
            return False  # Good Friday
        if d.serial == em - 2:
            return False  # Easter Saturday
        if d.serial == em:
            return False  # Easter Monday
        if mm == 5 and dd == 1:
            return False  # Labour Day
        if mm == 7 and dd == 1:
            return False  # SAR Establishment Day
        if mm == 10 and dd == 1:
            return False  # National Day
        if mm == 12 and dd == 25:
            return False
        if mm == 12 and dd == 26:
            return False

        return True


class Singapore(Calendar):
    """Singapore calendar."""

    def name(self):
        return "Singapore"

    def _is_business_day(self, d: Date) -> bool:
        dd, mm, yy = d.day, d.month, d.year
        w = d.weekday()
        em = _easter_monday(yy)

        if dd == 1 and mm == 1:
            return False
        if d.serial == em - 3:
            return False  # Good Friday
        if mm == 5 and dd == 1:
            return False  # Labour Day
        if mm == 8 and dd == 9:
            return False  # National Day
        if mm == 12 and dd == 25:
            return False

        return True


class SouthKorea(Calendar):
    """South Korea (KRX) calendar."""

    def name(self):
        return "SouthKorea"

    def _is_business_day(self, d: Date) -> bool:
        dd, mm, yy = d.day, d.month, d.year
        w = d.weekday()

        if dd == 1 and mm == 1:
            return False
        if mm == 3 and dd == 1:
            return False  # Independence Movement Day
        if mm == 5 and dd == 5:
            return False  # Children's Day
        if mm == 6 and dd == 6:
            return False  # Memorial Day
        if mm == 8 and dd == 15:
            return False  # Liberation Day
        if mm == 10 and dd == 3:
            return False  # National Foundation Day
        if mm == 10 and dd == 9:
            return False  # Hangul Day
        if mm == 12 and dd == 25:
            return False

        return True


class China(Calendar):
    """China (SSE/IB) calendar (simplified – major holidays only)."""

    def name(self):
        return "China"

    def _is_business_day(self, d: Date) -> bool:
        dd, mm = d.day, d.month

        # New Year
        if mm == 1 and dd <= 3:
            return False
        # Labour Day
        if mm == 5 and dd <= 3:
            return False
        # National Day (Golden Week)
        if mm == 10 and dd <= 7:
            return False

        return True


class Switzerland(Calendar):
    """Swiss calendar."""

    def name(self):
        return "Switzerland"

    def _is_business_day(self, d: Date) -> bool:
        dd, mm, yy = d.day, d.month, d.year
        em = _easter_monday(yy)

        if dd == 1 and mm == 1:
            return False
        if dd == 2 and mm == 1:
            return False  # Berchtoldstag
        if d.serial == em - 3:
            return False  # Good Friday
        if d.serial == em:
            return False  # Easter Monday
        if d.serial == em + 38:
            return False  # Ascension
        if d.serial == em + 49:
            return False  # Whit Monday
        if mm == 8 and dd == 1:
            return False  # National Day
        if mm == 12 and dd == 25:
            return False
        if mm == 12 and dd == 26:
            return False

        return True


class France(Calendar):
    """French calendar."""

    def name(self):
        return "France"

    def _is_business_day(self, d: Date) -> bool:
        dd, mm, yy = d.day, d.month, d.year
        em = _easter_monday(yy)

        if dd == 1 and mm == 1:
            return False
        if d.serial == em:
            return False  # Easter Monday
        if mm == 5 and dd == 1:
            return False  # Labour Day
        if mm == 5 and dd == 8:
            return False  # Victory Day
        if d.serial == em + 38:
            return False  # Ascension
        if d.serial == em + 49:
            return False  # Whit Monday
        if mm == 7 and dd == 14:
            return False  # Bastille Day
        if mm == 8 and dd == 15:
            return False  # Assumption
        if mm == 11 and dd == 1:
            return False  # All Saints
        if mm == 11 and dd == 11:
            return False  # Armistice
        if mm == 12 and dd == 25:
            return False

        return True


class Italy(Calendar):
    """Italian calendar (Borsa Italiana / Settlement)."""

    def name(self):
        return "Italy"

    def _is_business_day(self, d: Date) -> bool:
        dd, mm, yy = d.day, d.month, d.year
        em = _easter_monday(yy)

        if dd == 1 and mm == 1:
            return False
        if dd == 6 and mm == 1:
            return False  # Epiphany
        if d.serial == em:
            return False  # Easter Monday
        if mm == 4 and dd == 25:
            return False  # Liberation Day
        if mm == 5 and dd == 1:
            return False  # Labour Day
        if mm == 6 and dd == 2:
            return False  # Republic Day
        if mm == 8 and dd == 15:
            return False  # Assumption
        if mm == 11 and dd == 1:
            return False  # All Saints
        if mm == 12 and dd == 8:
            return False  # Immaculate Conception
        if mm == 12 and dd == 25:
            return False
        if mm == 12 and dd == 26:
            return False

        return True


class Brazil(Calendar):
    """Brazilian calendar (BM&F Bovespa)."""

    def name(self):
        return "Brazil"

    def _is_business_day(self, d: Date) -> bool:
        dd, mm, yy = d.day, d.month, d.year
        em = _easter_monday(yy)

        if dd == 1 and mm == 1:
            return False
        if d.serial == em - 49 or d.serial == em - 48:
            return False  # Carnival
        if d.serial == em - 3:
            return False  # Good Friday
        if d.serial == em - 2:
            return False  # Passion of Christ
        if mm == 4 and dd == 21:
            return False  # Tiradentes
        if mm == 5 and dd == 1:
            return False  # Labour Day
        if d.serial == em + 59:
            return False  # Corpus Christi
        if mm == 9 and dd == 7:
            return False  # Independence
        if mm == 10 and dd == 12:
            return False  # Our Lady
        if mm == 11 and dd == 2:
            return False  # All Souls
        if mm == 11 and dd == 15:
            return False  # Republic Day
        if mm == 12 and dd == 25:
            return False

        return True


class India(Calendar):
    """Indian calendar (NSE)."""

    def name(self):
        return "India"

    def _is_business_day(self, d: Date) -> bool:
        dd, mm = d.day, d.month

        if dd == 26 and mm == 1:
            return False  # Republic Day
        if dd == 15 and mm == 8:
            return False  # Independence Day
        if dd == 2 and mm == 10:
            return False  # Gandhi Jayanti
        if dd == 25 and mm == 12:
            return False

        return True


class SouthAfrica(Calendar):
    """South African calendar."""

    def name(self):
        return "SouthAfrica"

    def _is_business_day(self, d: Date) -> bool:
        dd, mm, yy = d.day, d.month, d.year
        em = _easter_monday(yy)

        if dd == 1 and mm == 1:
            return False
        if dd == 21 and mm == 3:
            return False  # Human Rights Day
        if d.serial == em - 3:
            return False  # Good Friday
        if d.serial == em:
            return False  # Family Day (Easter Monday)
        if dd == 27 and mm == 4:
            return False  # Freedom Day
        if dd == 1 and mm == 5:
            return False  # Workers Day
        if dd == 16 and mm == 6:
            return False  # Youth Day
        if dd == 9 and mm == 8:
            return False  # National Women's Day
        if dd == 24 and mm == 9:
            return False  # Heritage Day
        if dd == 16 and mm == 12:
            return False  # Day of Reconciliation
        if dd == 25 and mm == 12:
            return False
        if dd == 26 and mm == 12:
            return False  # Day of Goodwill

        return True


class Mexico(Calendar):
    """Mexican calendar (BMV)."""

    def name(self):
        return "Mexico"

    def _is_business_day(self, d: Date) -> bool:
        dd, mm, yy = d.day, d.month, d.year
        w = d.weekday()
        em = _easter_monday(yy)

        if dd == 1 and mm == 1:
            return False
        if mm == 2 and w == 1 and dd >= 1 and dd <= 7:
            return False  # Constitution Day (1st Monday Feb)
        if mm == 3 and w == 1 and 15 <= dd <= 21:
            return False  # Benito Juárez Birthday
        if d.serial == em - 4:
            return False  # Holy Thursday
        if d.serial == em - 3:
            return False  # Good Friday
        if mm == 5 and dd == 1:
            return False  # Labour Day
        if mm == 9 and dd == 16:
            return False  # Independence Day
        if mm == 11 and w == 1 and 15 <= dd <= 21:
            return False  # Revolution Day
        if mm == 12 and dd == 25:
            return False

        return True


class Sweden(Calendar):
    """Swedish calendar."""

    def name(self):
        return "Sweden"

    def _is_business_day(self, d: Date) -> bool:
        dd, mm, yy = d.day, d.month, d.year
        em = _easter_monday(yy)

        if dd == 1 and mm == 1:
            return False
        if dd == 6 and mm == 1:
            return False  # Epiphany
        if d.serial == em - 3:
            return False  # Good Friday
        if d.serial == em:
            return False  # Easter Monday
        if mm == 5 and dd == 1:
            return False  # Labour Day
        if d.serial == em + 38:
            return False  # Ascension
        if mm == 6 and dd == 6:
            return False  # National Day
        if mm == 12 and dd == 24:
            return False  # Christmas Eve
        if mm == 12 and dd == 25:
            return False
        if mm == 12 and dd == 26:
            return False
        if mm == 12 and dd == 31:
            return False

        return True


class Norway(Calendar):
    """Norwegian calendar."""

    def name(self):
        return "Norway"

    def _is_business_day(self, d: Date) -> bool:
        dd, mm, yy = d.day, d.month, d.year
        em = _easter_monday(yy)

        if dd == 1 and mm == 1:
            return False
        if d.serial == em - 4:
            return False  # Holy Thursday
        if d.serial == em - 3:
            return False  # Good Friday
        if d.serial == em:
            return False  # Easter Monday
        if mm == 5 and dd == 1:
            return False  # Labour Day
        if mm == 5 and dd == 17:
            return False  # Constitution Day
        if d.serial == em + 38:
            return False  # Ascension
        if d.serial == em + 49:
            return False  # Whit Monday
        if mm == 12 and dd == 25:
            return False
        if mm == 12 and dd == 26:
            return False

        return True


class Denmark(Calendar):
    """Danish calendar."""

    def name(self):
        return "Denmark"

    def _is_business_day(self, d: Date) -> bool:
        dd, mm, yy = d.day, d.month, d.year
        em = _easter_monday(yy)

        if dd == 1 and mm == 1:
            return False
        if d.serial == em - 4:
            return False  # Holy Thursday
        if d.serial == em - 3:
            return False  # Good Friday
        if d.serial == em:
            return False  # Easter Monday
        if d.serial == em + 25:
            return False  # General Prayer Day
        if d.serial == em + 38:
            return False  # Ascension
        if d.serial == em + 49:
            return False  # Whit Monday
        if mm == 6 and dd == 5:
            return False  # Constitution Day
        if mm == 12 and dd == 24:
            return False
        if mm == 12 and dd == 25:
            return False
        if mm == 12 and dd == 26:
            return False
        if mm == 12 and dd == 31:
            return False

        return True


def _easter_monday(year):
    """Return serial number of Easter Monday for given year (Meeus algorithm)."""
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = (h + l - 7 * m + 114) % 31 + 2  # +1 for Sunday, +1 for Monday
    return Date(day if day <= 30 else 1, month if day <= 30 else month + 1, year).serial
