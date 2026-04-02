"""Additional market calendars — Phase 11 extension.

Missing calendars: Argentina, Austria, Chile, CzechRepublic, Finland,
Hungary, Iceland, Indonesia, Israel, NewZealand, Poland, Romania, Russia,
SaudiArabia, Slovakia, Taiwan, Thailand, Turkey, Ukraine, Botswana.
"""

from ql_jax.time.calendar import Calendar
from ql_jax.time.date import Date
from ql_jax.time.calendars import _easter_monday


class Argentina(Calendar):
    """Argentine calendar (Buenos Aires Exchange)."""
    def name(self): return "Argentina"
    def _is_business_day(self, d: Date) -> bool:
        dd, mm, yy, w = d.day, d.month, d.year, d.weekday()
        em = _easter_monday(yy)
        if w >= 6: return False
        if dd == 1 and mm == 1: return False  # New Year
        if d.serial == em - 4 or d.serial == em - 3: return False  # Carnival
        if d.serial == em - 3: return False  # Good Friday
        if dd == 24 and mm == 3: return False  # Truth & Memory
        if dd == 2 and mm == 4: return False  # Malvinas Day
        if dd == 1 and mm == 5: return False  # Labour
        if dd == 25 and mm == 5: return False  # May Revolution
        if dd == 20 and mm == 6: return False  # Flag Day (approx)
        if dd == 9 and mm == 7: return False  # Independence
        if dd == 8 and mm == 12: return False  # Immaculate Conception
        if dd == 25 and mm == 12: return False  # Christmas
        return True


class Austria(Calendar):
    """Austrian calendar."""
    def name(self): return "Austria"
    def _is_business_day(self, d: Date) -> bool:
        dd, mm, yy, w = d.day, d.month, d.year, d.weekday()
        em = _easter_monday(yy)
        if w >= 6: return False
        if dd == 1 and mm == 1: return False
        if dd == 6 and mm == 1: return False  # Epiphany
        if d.serial == em - 1: return False  # Easter Monday
        if d.serial == em - 3: return False  # Good Friday
        if dd == 1 and mm == 5: return False  # Labour
        if d.serial == em + 38: return False  # Ascension
        if d.serial == em + 49: return False  # Whit Monday
        if d.serial == em + 59: return False  # Corpus Christi
        if dd == 15 and mm == 8: return False  # Assumption
        if dd == 26 and mm == 10: return False  # National Day
        if dd == 1 and mm == 11: return False  # All Saints
        if dd == 8 and mm == 12: return False  # Immaculate Conception
        if dd == 25 and mm == 12: return False
        if dd == 26 and mm == 12: return False
        return True


class Chile(Calendar):
    """Chilean calendar."""
    def name(self): return "Chile"
    def _is_business_day(self, d: Date) -> bool:
        dd, mm, yy, w = d.day, d.month, d.year, d.weekday()
        em = _easter_monday(yy)
        if w >= 6: return False
        if dd == 1 and mm == 1: return False
        if d.serial == em - 3: return False  # Good Friday
        if d.serial == em - 2: return False  # Holy Saturday
        if dd == 1 and mm == 5: return False
        if dd == 21 and mm == 5: return False  # Navy Day
        if dd == 18 and mm == 9: return False  # Independence
        if dd == 19 and mm == 9: return False  # Army Day
        if dd == 12 and mm == 10: return False  # Columbus Day (approx)
        if dd == 1 and mm == 11: return False
        if dd == 8 and mm == 12: return False
        if dd == 25 and mm == 12: return False
        return True


class CzechRepublic(Calendar):
    """Czech Republic calendar."""
    def name(self): return "CzechRepublic"
    def _is_business_day(self, d: Date) -> bool:
        dd, mm, yy, w = d.day, d.month, d.year, d.weekday()
        em = _easter_monday(yy)
        if w >= 6: return False
        if dd == 1 and mm == 1: return False
        if d.serial == em - 3: return False  # Good Friday
        if d.serial == em - 1: return False  # Easter Monday
        if dd == 1 and mm == 5: return False
        if dd == 8 and mm == 5: return False  # Liberation Day
        if dd == 5 and mm == 7: return False  # Cyril & Methodius
        if dd == 6 and mm == 7: return False  # Jan Hus Day
        if dd == 28 and mm == 9: return False  # Czech Statehood
        if dd == 28 and mm == 10: return False  # Independence
        if dd == 17 and mm == 11: return False  # Freedom Day
        if dd == 24 and mm == 12: return False
        if dd == 25 and mm == 12: return False
        if dd == 26 and mm == 12: return False
        return True


class Finland(Calendar):
    """Finnish calendar."""
    def name(self): return "Finland"
    def _is_business_day(self, d: Date) -> bool:
        dd, mm, yy, w = d.day, d.month, d.year, d.weekday()
        em = _easter_monday(yy)
        if w >= 6: return False
        if dd == 1 and mm == 1: return False
        if dd == 6 and mm == 1: return False  # Epiphany
        if d.serial == em - 3: return False  # Good Friday
        if d.serial == em - 1: return False  # Easter Monday
        if dd == 1 and mm == 5: return False
        if d.serial == em + 38: return False  # Ascension
        if mm == 6 and w == 5 and 20 <= dd <= 26: return False  # Midsummer Eve
        if dd == 6 and mm == 12: return False  # Independence
        if dd == 24 and mm == 12: return False
        if dd == 25 and mm == 12: return False
        if dd == 26 and mm == 12: return False
        return True


class Hungary(Calendar):
    """Hungarian calendar."""
    def name(self): return "Hungary"
    def _is_business_day(self, d: Date) -> bool:
        dd, mm, yy, w = d.day, d.month, d.year, d.weekday()
        em = _easter_monday(yy)
        if w >= 6: return False
        if dd == 1 and mm == 1: return False
        if dd == 15 and mm == 3: return False  # 1848 Revolution
        if d.serial == em - 3: return False
        if d.serial == em - 1: return False
        if dd == 1 and mm == 5: return False
        if d.serial == em + 49: return False  # Whit Monday
        if dd == 20 and mm == 8: return False  # St. Stephen
        if dd == 23 and mm == 10: return False  # 1956 Revolution
        if dd == 1 and mm == 11: return False
        if dd == 25 and mm == 12: return False
        if dd == 26 and mm == 12: return False
        return True


class Iceland(Calendar):
    """Icelandic calendar."""
    def name(self): return "Iceland"
    def _is_business_day(self, d: Date) -> bool:
        dd, mm, yy, w = d.day, d.month, d.year, d.weekday()
        em = _easter_monday(yy)
        if w >= 6: return False
        if dd == 1 and mm == 1: return False
        if d.serial == em - 4: return False  # Holy Thursday
        if d.serial == em - 3: return False
        if d.serial == em - 1: return False
        if mm == 4 and w == 4 and 19 <= dd <= 25: return False  # 1st day of summer
        if dd == 1 and mm == 5: return False
        if d.serial == em + 38: return False  # Ascension
        if d.serial == em + 49: return False
        if dd == 17 and mm == 6: return False  # Independence
        if dd == 24 and mm == 12: return False
        if dd == 25 and mm == 12: return False
        if dd == 26 and mm == 12: return False
        if dd == 31 and mm == 12: return False
        return True


class Indonesia(Calendar):
    """Indonesian calendar (IDX)."""
    def name(self): return "Indonesia"
    def _is_business_day(self, d: Date) -> bool:
        dd, mm, yy, w = d.day, d.month, d.year, d.weekday()
        if w >= 6: return False
        if dd == 1 and mm == 1: return False
        if dd == 17 and mm == 8: return False  # Independence Day
        if dd == 25 and mm == 12: return False
        # Various Islamic holidays move, simplified
        return True


class Israel(Calendar):
    """Israeli calendar (TASE)."""
    def name(self): return "Israel"
    def _is_business_day(self, d: Date) -> bool:
        dd, mm, yy, w = d.day, d.month, d.year, d.weekday()
        # Israel uses Friday-Saturday weekend
        if w >= 5: return False
        # Major holidays (approximate civil dates)
        if dd == 1 and mm == 1: return False
        if dd == 11 and mm == 4: return False  # Passover start (varies)
        if dd == 17 and mm == 4: return False  # Passover end
        if dd == 29 and mm == 4: return False  # Independence Day (varies)
        if dd == 31 and mm == 5: return False  # Shavuot
        if dd == 17 and mm == 9: return False  # Rosh Hashanah
        if dd == 26 and mm == 9: return False  # Yom Kippur
        if dd == 1 and mm == 10: return False  # Sukkot
        return True


class NewZealand(Calendar):
    """New Zealand calendar."""
    def name(self): return "New Zealand"
    def _is_business_day(self, d: Date) -> bool:
        dd, mm, yy, w = d.day, d.month, d.year, d.weekday()
        em = _easter_monday(yy)
        if w >= 6: return False
        if (dd == 1 or (dd == 3 and w == 1)) and mm == 1: return False
        if (dd == 2 or (dd == 4 and w == 1)) and mm == 1: return False  # Day after NY
        if mm == 1 and w == 1 and 22 <= dd <= 28: return False  # Auckland Anniversary (approx)
        if dd == 6 and mm == 2: return False  # Waitangi Day
        if d.serial == em - 3: return False
        if d.serial == em - 1: return False
        if dd == 25 and mm == 4: return False  # ANZAC
        if mm == 6 and w == 1 and dd <= 7: return False  # Queen's Birthday
        if mm == 10 and w == 1 and 22 <= dd <= 28: return False  # Labour Day
        if dd == 25 and mm == 12: return False
        if dd == 26 and mm == 12: return False
        return True


class Poland(Calendar):
    """Polish calendar (WSE)."""
    def name(self): return "Poland"
    def _is_business_day(self, d: Date) -> bool:
        dd, mm, yy, w = d.day, d.month, d.year, d.weekday()
        em = _easter_monday(yy)
        if w >= 6: return False
        if dd == 1 and mm == 1: return False
        if dd == 6 and mm == 1: return False  # Epiphany
        if d.serial == em - 1: return False
        if dd == 1 and mm == 5: return False
        if dd == 3 and mm == 5: return False  # Constitution
        if d.serial == em + 59: return False  # Corpus Christi
        if dd == 15 and mm == 8: return False
        if dd == 1 and mm == 11: return False
        if dd == 11 and mm == 11: return False  # Independence
        if dd == 25 and mm == 12: return False
        if dd == 26 and mm == 12: return False
        return True


class Romania(Calendar):
    """Romanian calendar (BVB)."""
    def name(self): return "Romania"
    def _is_business_day(self, d: Date) -> bool:
        dd, mm, yy, w = d.day, d.month, d.year, d.weekday()
        em = _easter_monday(yy)
        if w >= 6: return False
        if dd == 1 and mm == 1: return False
        if dd == 2 and mm == 1: return False
        if dd == 24 and mm == 1: return False  # Union Day
        if d.serial == em - 3: return False
        if d.serial == em - 1: return False
        if dd == 1 and mm == 5: return False
        if dd == 1 and mm == 6: return False  # Children's Day
        if dd == 15 and mm == 8: return False
        if dd == 30 and mm == 11: return False  # St. Andrew
        if dd == 1 and mm == 12: return False  # National Day
        if dd == 25 and mm == 12: return False
        if dd == 26 and mm == 12: return False
        return True


class Russia(Calendar):
    """Russian calendar (MOEX)."""
    def name(self): return "Russia"
    def _is_business_day(self, d: Date) -> bool:
        dd, mm, yy, w = d.day, d.month, d.year, d.weekday()
        if w >= 6: return False
        # Extended New Year holidays (Jan 1-8)
        if mm == 1 and dd <= 8: return False
        if dd == 23 and mm == 2: return False  # Defender of Fatherland
        if dd == 8 and mm == 3: return False  # Women's Day
        if dd == 1 and mm == 5: return False  # Spring & Labour
        if dd == 9 and mm == 5: return False  # Victory Day
        if dd == 12 and mm == 6: return False  # Russia Day
        if dd == 4 and mm == 11: return False  # Unity Day
        return True


class SaudiArabia(Calendar):
    """Saudi Arabian calendar (Tadawul)."""
    def name(self): return "SaudiArabia"
    def _is_business_day(self, d: Date) -> bool:
        dd, mm, yy, w = d.day, d.month, d.year, d.weekday()
        # Friday-Saturday weekend
        if w == 4 or w == 5: return False  # Friday/Saturday
        if dd == 23 and mm == 9: return False  # National Day
        # Eid al-Fitr and Eid al-Adha move with Islamic calendar
        return True


class Slovakia(Calendar):
    """Slovak calendar (BSSE)."""
    def name(self): return "Slovakia"
    def _is_business_day(self, d: Date) -> bool:
        dd, mm, yy, w = d.day, d.month, d.year, d.weekday()
        em = _easter_monday(yy)
        if w >= 6: return False
        if dd == 1 and mm == 1: return False
        if dd == 6 and mm == 1: return False
        if d.serial == em - 3: return False
        if d.serial == em - 1: return False
        if dd == 1 and mm == 5: return False
        if dd == 8 and mm == 5: return False
        if dd == 5 and mm == 7: return False  # Cyril & Methodius
        if dd == 29 and mm == 8: return False  # SNP Day
        if dd == 1 and mm == 9: return False  # Constitution
        if dd == 15 and mm == 9: return False  # Our Lady of Seven Sorrows
        if dd == 1 and mm == 11: return False
        if dd == 17 and mm == 11: return False  # Velvet Revolution
        if dd == 24 and mm == 12: return False
        if dd == 25 and mm == 12: return False
        if dd == 26 and mm == 12: return False
        return True


class Taiwan(Calendar):
    """Taiwanese calendar (TWSE)."""
    def name(self): return "Taiwan"
    def _is_business_day(self, d: Date) -> bool:
        dd, mm, yy, w = d.day, d.month, d.year, d.weekday()
        if w >= 6: return False
        if dd == 1 and mm == 1: return False
        # Lunar New Year (approx late Jan/early Feb, varies)
        if mm == 2 and dd <= 3: return False  # approximate
        if dd == 28 and mm == 2: return False  # Peace Memorial
        if dd == 4 and mm == 4: return False  # Children's Day
        if dd == 5 and mm == 4: return False  # Tomb Sweeping
        if dd == 1 and mm == 5: return False
        if dd == 10 and mm == 10: return False  # National Day
        return True


class Thailand(Calendar):
    """Thai calendar (SET)."""
    def name(self): return "Thailand"
    def _is_business_day(self, d: Date) -> bool:
        dd, mm, yy, w = d.day, d.month, d.year, d.weekday()
        if w >= 6: return False
        if dd == 1 and mm == 1: return False
        if dd == 6 and mm == 4: return False  # Chakri Memorial
        if dd == 13 and mm == 4 or dd == 14 and mm == 4 or dd == 15 and mm == 4: return False  # Songkran
        if dd == 1 and mm == 5: return False
        if dd == 4 and mm == 5: return False  # Coronation Day (post-2019)
        if dd == 3 and mm == 6: return False  # Queen's Birthday
        if dd == 28 and mm == 7: return False  # King's Birthday
        if dd == 12 and mm == 8: return False  # Queen Mother Birthday
        if dd == 13 and mm == 10: return False  # King Bhumibol Memorial
        if dd == 23 and mm == 10: return False  # Chulalongkorn Day
        if dd == 5 and mm == 12: return False  # King Bhumibol Birthday
        if dd == 10 and mm == 12: return False  # Constitution Day
        if dd == 31 and mm == 12: return False
        return True


class Turkey(Calendar):
    """Turkish calendar (Borsa Istanbul)."""
    def name(self): return "Turkey"
    def _is_business_day(self, d: Date) -> bool:
        dd, mm, yy, w = d.day, d.month, d.year, d.weekday()
        if w >= 6: return False
        if dd == 1 and mm == 1: return False
        if dd == 23 and mm == 4: return False  # National Sovereignty
        if dd == 1 and mm == 5: return False
        if dd == 19 and mm == 5: return False  # Youth & Sports
        if dd == 15 and mm == 7: return False  # Democracy Day
        if dd == 30 and mm == 8: return False  # Victory Day
        if dd == 29 and mm == 10: return False  # Republic Day
        # Eid holidays move with Islamic calendar, simplified
        return True


class Ukraine(Calendar):
    """Ukrainian calendar."""
    def name(self): return "Ukraine"
    def _is_business_day(self, d: Date) -> bool:
        dd, mm, yy, w = d.day, d.month, d.year, d.weekday()
        if w >= 6: return False
        if dd == 1 and mm == 1: return False
        if dd == 7 and mm == 1: return False  # Orthodox Christmas
        if dd == 8 and mm == 3: return False  # Women's Day
        if dd == 1 and mm == 5: return False
        if dd == 9 and mm == 5: return False  # Victory Day (pre-2023)
        if dd == 28 and mm == 6: return False  # Constitution Day
        if dd == 24 and mm == 8: return False  # Independence Day
        if dd == 25 and mm == 12: return False
        return True


class Botswana(Calendar):
    """Botswana calendar (BSE)."""
    def name(self): return "Botswana"
    def _is_business_day(self, d: Date) -> bool:
        dd, mm, yy, w = d.day, d.month, d.year, d.weekday()
        em = _easter_monday(yy)
        if w >= 6: return False
        if dd == 1 and mm == 1: return False
        if d.serial == em - 3: return False
        if d.serial == em - 2: return False
        if d.serial == em - 1: return False
        if dd == 1 and mm == 5: return False
        if d.serial == em + 38: return False  # Ascension
        if dd == 1 and mm == 7: return False  # Sir Seretse Khama
        if mm == 7 and w == 1 and 15 <= dd <= 21: return False  # President's Day
        if mm == 7 and w == 2 and 16 <= dd <= 22: return False  # President's Day holiday
        if dd == 30 and mm == 9: return False  # Botswana Day
        if dd == 1 and mm == 10: return False  # Botswana Day holiday
        if dd == 25 and mm == 12: return False
        if dd == 26 and mm == 12: return False
        return True
