"""Tests for Calendar — porting QuantLib test-suite/calendars.cpp."""

from ql_jax.time.date import Date, Period
from ql_jax.time.calendar import (
    Calendar,
    NullCalendar,
    WeekendsOnly,
    TARGET,
    UnitedStates,
    UnitedKingdom,
    Japan,
    Germany,
    JointCalendar,
    BespokeCalendar,
)
from ql_jax._util.types import BusinessDayConvention, Weekday, TimeUnit


class TestNullCalendar:
    def test_every_day_is_business(self):
        cal = NullCalendar()
        # Saturday
        d = Date(7, 1, 2023)
        assert d.weekday() == Weekday.Saturday
        assert cal.is_business_day(d)

    def test_no_holidays(self):
        cal = NullCalendar()
        for day in range(1, 32):
            d = Date(day, 1, 2023)
            assert cal.is_business_day(d)


class TestWeekendsOnly:
    def test_weekdays_are_business(self):
        cal = WeekendsOnly()
        d = Date(2, 1, 2023)  # Monday
        assert cal.is_business_day(d)

    def test_weekends_are_holidays(self):
        cal = WeekendsOnly()
        d = Date(7, 1, 2023)  # Saturday
        assert not cal.is_business_day(d)
        d = Date(8, 1, 2023)  # Sunday
        assert not cal.is_business_day(d)


class TestTARGET:
    def test_new_year_2023(self):
        cal = TARGET()
        assert not cal.is_business_day(Date(1, 1, 2023))

    def test_good_friday_2023(self):
        cal = TARGET()
        assert not cal.is_business_day(Date(7, 4, 2023))

    def test_easter_monday_2023(self):
        cal = TARGET()
        assert not cal.is_business_day(Date(10, 4, 2023))

    def test_labour_day_2023(self):
        cal = TARGET()
        assert not cal.is_business_day(Date(1, 5, 2023))

    def test_christmas_2023(self):
        cal = TARGET()
        assert not cal.is_business_day(Date(25, 12, 2023))
        assert not cal.is_business_day(Date(26, 12, 2023))

    def test_normal_business_day(self):
        cal = TARGET()
        assert cal.is_business_day(Date(3, 1, 2023))  # Tuesday


class TestUnitedStates:
    def test_new_year_2023(self):
        cal = UnitedStates()
        d = Date(1, 1, 2023)  # Sunday -> observed Monday Jan 2
        assert not cal.is_business_day(Date(2, 1, 2023))

    def test_mlk_day_2023(self):
        cal = UnitedStates()
        assert not cal.is_business_day(Date(16, 1, 2023))  # 3rd Monday

    def test_thanksgiving_2023(self):
        cal = UnitedStates()
        assert not cal.is_business_day(Date(23, 11, 2023))  # 4th Thursday

    def test_christmas_2023(self):
        cal = UnitedStates()
        assert not cal.is_business_day(Date(25, 12, 2023))

    def test_normal_business_day(self):
        cal = UnitedStates()
        assert cal.is_business_day(Date(3, 1, 2023))  # Tuesday


class TestCalendarAdjust:
    def test_following(self):
        cal = WeekendsOnly()
        d = Date(7, 1, 2023)  # Saturday
        adj = cal.adjust(d, BusinessDayConvention.Following)
        assert adj.weekday() == Weekday.Monday
        assert adj == Date(9, 1, 2023)

    def test_preceding(self):
        cal = WeekendsOnly()
        d = Date(7, 1, 2023)  # Saturday
        adj = cal.adjust(d, BusinessDayConvention.Preceding)
        assert adj.weekday() == Weekday.Friday
        assert adj == Date(6, 1, 2023)

    def test_modified_following(self):
        cal = WeekendsOnly()
        # Sat Dec 31, 2022 -> Modified Following should go backward to Dec 30 (Fri)
        d = Date(31, 12, 2022)  # Saturday
        adj = cal.adjust(d, BusinessDayConvention.ModifiedFollowing)
        # Next Monday is Jan 2, different month -> go back to Friday
        assert adj == Date(30, 12, 2022)

    def test_unadjusted(self):
        cal = WeekendsOnly()
        d = Date(7, 1, 2023)
        adj = cal.adjust(d, BusinessDayConvention.Unadjusted)
        assert adj == d


class TestCalendarAdvance:
    def test_advance_days(self):
        cal = WeekendsOnly()
        d = Date(6, 1, 2023)  # Friday
        adv = cal.advance(d, 1, TimeUnit.Days)
        assert adv == Date(9, 1, 2023)  # skip weekend to Monday

    def test_advance_months(self):
        cal = WeekendsOnly()
        d = Date(31, 1, 2023)  # Tuesday
        adv = cal.advance(d, 1, TimeUnit.Months)
        # Feb 28, 2023 is a Tuesday
        assert adv.month == 2
        assert adv.day == 28

    def test_business_days_between(self):
        cal = WeekendsOnly()
        d1 = Date(2, 1, 2023)  # Monday
        d2 = Date(6, 1, 2023)  # Friday
        assert cal.business_days_between(d1, d2) == 4  # Mon-Thu (include_last=False)


class TestAddRemoveHoliday:
    def test_add_holiday(self):
        cal = WeekendsOnly()
        d = Date(3, 1, 2023)  # Tuesday
        assert cal.is_business_day(d)
        cal.add_holiday(d)
        assert not cal.is_business_day(d)

    def test_remove_holiday(self):
        cal = WeekendsOnly()
        d = Date(7, 1, 2023)  # Saturday (weekend)
        assert not cal.is_business_day(d)
        cal.remove_holiday(d)
        assert cal.is_business_day(d)


class TestJointCalendar:
    def test_intersection(self):
        us = UnitedStates()
        uk = UnitedKingdom()
        joint = JointCalendar(us, uk)
        # A day that's a holiday in US but not UK should be a holiday in joint
        # MLK day 2023 (US holiday, not UK)
        assert not joint.is_business_day(Date(16, 1, 2023))
