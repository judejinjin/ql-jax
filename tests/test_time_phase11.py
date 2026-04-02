"""Tests for Phase 11: extended calendars, IMM dates, ECB dates."""

import jax
import pytest

jax.config.update("jax_enable_x64", True)


# ---------------------------------------------------------------------------
# Extended calendars
# ---------------------------------------------------------------------------

class TestExtendedCalendarsBasic:
    """Smoke tests for all 20 extended calendars."""

    def _test_calendar(self, CalendarClass, name_substr):
        from ql_jax.time.date import Date
        cal = CalendarClass()
        assert name_substr.lower() in cal.name().lower()
        # Weekends should be holidays
        sat = Date(1, 6, 2024)  # Saturday
        sun = Date(2, 6, 2024)  # Sunday
        assert not cal.is_business_day(sat)
        assert not cal.is_business_day(sun)
        # Monday should generally be business day (unless holiday)
        mon = Date(3, 6, 2024)  # Monday
        # Just check it returns a bool
        assert isinstance(cal.is_business_day(mon), bool)

    def test_argentina(self):
        from ql_jax.time.calendars_extended import Argentina
        self._test_calendar(Argentina, "Argentina")

    def test_austria(self):
        from ql_jax.time.calendars_extended import Austria
        self._test_calendar(Austria, "Austria")

    def test_chile(self):
        from ql_jax.time.calendars_extended import Chile
        self._test_calendar(Chile, "Chile")

    def test_czech_republic(self):
        from ql_jax.time.calendars_extended import CzechRepublic
        self._test_calendar(CzechRepublic, "Czech")

    def test_finland(self):
        from ql_jax.time.calendars_extended import Finland
        self._test_calendar(Finland, "Finland")

    def test_hungary(self):
        from ql_jax.time.calendars_extended import Hungary
        self._test_calendar(Hungary, "Hungary")

    def test_iceland(self):
        from ql_jax.time.calendars_extended import Iceland
        self._test_calendar(Iceland, "Iceland")

    def test_indonesia(self):
        from ql_jax.time.calendars_extended import Indonesia
        self._test_calendar(Indonesia, "Indonesia")

    def test_israel(self):
        from ql_jax.time.calendars_extended import Israel
        self._test_calendar(Israel, "Israel")

    def test_new_zealand(self):
        from ql_jax.time.calendars_extended import NewZealand
        self._test_calendar(NewZealand, "New Zealand")

    def test_poland(self):
        from ql_jax.time.calendars_extended import Poland
        self._test_calendar(Poland, "Poland")

    def test_romania(self):
        from ql_jax.time.calendars_extended import Romania
        self._test_calendar(Romania, "Romania")

    def test_russia(self):
        from ql_jax.time.calendars_extended import Russia
        self._test_calendar(Russia, "Russia")

    def test_saudi_arabia(self):
        from ql_jax.time.calendars_extended import SaudiArabia
        self._test_calendar(SaudiArabia, "Saudi")

    def test_slovakia(self):
        from ql_jax.time.calendars_extended import Slovakia
        self._test_calendar(Slovakia, "Slovakia")

    def test_taiwan(self):
        from ql_jax.time.calendars_extended import Taiwan
        self._test_calendar(Taiwan, "Taiwan")

    def test_thailand(self):
        from ql_jax.time.calendars_extended import Thailand
        self._test_calendar(Thailand, "Thailand")

    def test_turkey(self):
        from ql_jax.time.calendars_extended import Turkey
        self._test_calendar(Turkey, "Turkey")

    def test_ukraine(self):
        from ql_jax.time.calendars_extended import Ukraine
        self._test_calendar(Ukraine, "Ukraine")

    def test_botswana(self):
        from ql_jax.time.calendars_extended import Botswana
        self._test_calendar(Botswana, "Botswana")


class TestCalendarHolidays:
    """Specific holiday tests."""

    def test_argentina_new_year(self):
        from ql_jax.time.calendars_extended import Argentina
        from ql_jax.time.date import Date
        cal = Argentina()
        # Jan 1 2024 is Monday
        assert not cal.is_business_day(Date(1, 1, 2024))

    def test_russia_new_year_holidays(self):
        from ql_jax.time.calendars_extended import Russia
        from ql_jax.time.date import Date
        cal = Russia()
        # Russia has Jan 1-8 holidays
        assert not cal.is_business_day(Date(2, 1, 2024))

    def test_israel_friday_holiday(self):
        from ql_jax.time.calendars_extended import Israel
        from ql_jax.time.date import Date
        cal = Israel()
        # Israel: Friday/Saturday weekend
        fri = Date(7, 6, 2024)
        assert not cal.is_business_day(fri)


# ---------------------------------------------------------------------------
# IMM dates
# ---------------------------------------------------------------------------

class TestIMM:
    def test_is_imm_date(self):
        from ql_jax.time.imm import is_imm_date
        from ql_jax.time.date import Date
        # March 20, 2024 is the 3rd Wednesday of March
        assert is_imm_date(Date(20, 3, 2024))
        # March 21, 2024 is Thursday
        assert not is_imm_date(Date(21, 3, 2024))

    def test_next_imm_date(self):
        from ql_jax.time.imm import next_imm_date
        from ql_jax.time.date import Date
        nxt = next_imm_date(Date(1, 1, 2024))
        assert nxt.month in [3, 6, 9, 12]
        assert nxt > Date(1, 1, 2024)

    def test_imm_code(self):
        from ql_jax.time.imm import imm_code
        from ql_jax.time.date import Date
        # March 2024 => 'H4'
        code = imm_code(Date(20, 3, 2024))
        assert code[0] == 'H'  # March
        assert code[1] == '4'  # 2024

    def test_imm_code_roundtrip(self):
        from ql_jax.time.imm import imm_code, imm_date_from_code
        from ql_jax.time.date import Date
        d = Date(19, 6, 2024)  # 3rd Wednesday of June
        code = imm_code(d)
        d2 = imm_date_from_code(code, ref_date=Date(1, 1, 2024))
        assert d2 == d


# ---------------------------------------------------------------------------
# ECB dates
# ---------------------------------------------------------------------------

class TestECB:
    def test_ecb_dates(self):
        from ql_jax.time.ecb import ecb_dates
        from ql_jax.time.date import Date
        dates = ecb_dates(start=Date(1, 1, 2024), end=Date(31, 12, 2024))
        assert len(dates) > 0
        # ECB typically has 8 meetings per year
        assert len(dates) >= 6

    def test_next_ecb_date(self):
        from ql_jax.time.ecb import next_ecb_date
        from ql_jax.time.date import Date
        nxt = next_ecb_date(Date(1, 1, 2024))
        assert nxt > Date(1, 1, 2024)
        assert nxt.year == 2024

    def test_is_ecb_date(self):
        from ql_jax.time.ecb import is_ecb_date, ecb_dates
        from ql_jax.time.date import Date
        dates = ecb_dates(start=Date(1, 1, 2024), end=Date(31, 12, 2024))
        if dates:
            assert is_ecb_date(dates[0])
        assert not is_ecb_date(Date(1, 1, 2024))  # Likely not an ECB date
