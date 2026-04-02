"""Tests for Date and Period arithmetic — porting QuantLib test-suite/dates.cpp."""

from ql_jax.time.date import Date, Period, days_between
from ql_jax._util.types import TimeUnit, Weekday


class TestDateConstruction:
    def test_null_date(self):
        d = Date()
        assert d.serial == 0
        assert not d

    def test_serial_constructor(self):
        d = Date(45000)
        assert d.serial == 45000

    def test_dmy_constructor(self):
        d = Date(15, 3, 2023)
        assert d.day == 15
        assert d.month == 3
        assert d.year == 2023

    def test_roundtrip(self):
        """d/m/y -> serial -> d/m/y roundtrip."""
        for y in (1901, 1950, 2000, 2023, 2100, 2199):
            for m in (1, 2, 6, 12):
                for day in (1, 15, 28):
                    d = Date(day, m, y)
                    d2 = Date(d.serial)
                    assert d2.day == day
                    assert d2.month == m
                    assert d2.year == y

    def test_leap_year(self):
        assert Date.is_leap(2000)
        assert Date.is_leap(2024)
        assert not Date.is_leap(1900)
        assert not Date.is_leap(2023)
        assert not Date.is_leap(2100)

    def test_feb29_leap(self):
        d = Date(29, 2, 2024)
        assert d.day == 29
        assert d.month == 2
        assert d.year == 2024

    def test_end_of_month(self):
        d = Date(15, 3, 2023)
        eom = Date.end_of_month(d)
        assert eom.day == 31
        assert eom.month == 3

        d = Date(15, 2, 2024)
        eom = Date.end_of_month(d)
        assert eom.day == 29

    def test_start_of_month(self):
        d = Date(15, 6, 2023)
        som = Date.start_of_month(d)
        assert som.day == 1
        assert som.month == 6

    def test_is_end_of_month(self):
        assert Date.is_end_of_month(Date(31, 1, 2023))
        assert Date.is_end_of_month(Date(28, 2, 2023))
        assert not Date.is_end_of_month(Date(28, 2, 2024))
        assert Date.is_end_of_month(Date(29, 2, 2024))


class TestDateArithmetic:
    def test_add_days(self):
        d = Date(15, 3, 2023)
        d2 = d + 10
        assert d2.day == 25
        assert d2.month == 3

    def test_sub_days(self):
        d = Date(15, 3, 2023)
        d2 = d - 15
        assert d2.month == 2

    def test_date_diff(self):
        d1 = Date(1, 1, 2023)
        d2 = Date(1, 1, 2024)
        assert d2 - d1 == 365

    def test_add_months(self):
        d = Date(31, 1, 2023)
        d2 = d + Period(1, TimeUnit.Months)
        assert d2.day == 28  # Feb 2023 has 28 days
        assert d2.month == 2

    def test_add_years(self):
        d = Date(29, 2, 2024)
        d2 = d + Period(1, TimeUnit.Years)
        assert d2.day == 28  # 2025 not leap
        assert d2.month == 2
        assert d2.year == 2025

    def test_add_weeks(self):
        d = Date(1, 1, 2023)  # Sunday
        d2 = d + Period(2, TimeUnit.Weeks)
        assert d2 - d == 14

    def test_days_between(self):
        d1 = Date(1, 3, 2023)
        d2 = Date(1, 6, 2023)
        assert days_between(d1, d2) == 92


class TestDateComparison:
    def test_equal(self):
        assert Date(1, 1, 2023) == Date(1, 1, 2023)

    def test_not_equal(self):
        assert Date(1, 1, 2023) != Date(2, 1, 2023)

    def test_less_than(self):
        assert Date(1, 1, 2023) < Date(2, 1, 2023)

    def test_greater_than(self):
        assert Date(2, 1, 2023) > Date(1, 1, 2023)

    def test_hash(self):
        d1 = Date(1, 1, 2023)
        d2 = Date(1, 1, 2023)
        assert hash(d1) == hash(d2)
        s = {d1, d2}
        assert len(s) == 1


class TestDateWeekday:
    def test_known_weekdays(self):
        # Jan 1, 2023 was a Sunday
        d = Date(1, 1, 2023)
        assert d.weekday() == Weekday.Sunday

        # Jan 2, 2023 was a Monday
        d = Date(2, 1, 2023)
        assert d.weekday() == Weekday.Monday

    def test_next_weekday(self):
        d = Date(1, 1, 2023)  # Sunday
        nxt = Date.next_weekday(d, Weekday.Wednesday)
        assert nxt.weekday() == Weekday.Wednesday
        assert nxt.day == 4

    def test_nth_weekday(self):
        # 3rd Monday of January 2023
        d = Date.nth_weekday(3, Weekday.Monday, 1, 2023)
        assert d.weekday() == Weekday.Monday
        assert d.day == 16  # Jan 16, 2023


class TestPeriod:
    def test_construction(self):
        p = Period(3, TimeUnit.Months)
        assert p.length == 3
        assert p.units == TimeUnit.Months

    def test_from_frequency(self):
        from ql_jax._util.types import Frequency
        p = Period.from_frequency(Frequency.Quarterly)
        assert p.length == 3
        assert p.units == TimeUnit.Months

    def test_repr(self):
        p = Period(6, TimeUnit.Months)
        assert repr(p) == "6M"
