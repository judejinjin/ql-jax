"""Tests for Schedule generation — porting QuantLib test-suite/schedule.cpp."""

from ql_jax.time.date import Date, Period
from ql_jax.time.schedule import Schedule, MakeSchedule
from ql_jax.time.calendar import TARGET, WeekendsOnly, NullCalendar
from ql_jax._util.types import (
    BusinessDayConvention,
    DateGeneration,
    Frequency,
    TimeUnit,
)


class TestScheduleBasic:
    def test_from_dates(self):
        dates = [Date(1, 1, 2023), Date(1, 4, 2023), Date(1, 7, 2023)]
        s = Schedule(dates=dates)
        assert len(s) == 3
        assert s[0] == dates[0]
        assert s[-1] == dates[-1]

    def test_backward_quarterly(self):
        s = Schedule(
            effective_date=Date(1, 1, 2023),
            termination_date=Date(1, 1, 2024),
            tenor=Period(3, TimeUnit.Months),
            calendar=NullCalendar(),
            convention=BusinessDayConvention.Unadjusted,
            rule=DateGeneration.Backward,
        )
        assert len(s) >= 5  # 4 quarters + 1
        assert s.start_date == Date(1, 1, 2023)
        assert s.end_date == Date(1, 1, 2024)

    def test_forward_monthly(self):
        s = Schedule(
            effective_date=Date(1, 1, 2023),
            termination_date=Date(1, 7, 2023),
            tenor=Period(1, TimeUnit.Months),
            calendar=NullCalendar(),
            convention=BusinessDayConvention.Unadjusted,
            rule=DateGeneration.Forward,
        )
        assert len(s) == 7  # Jan, Feb, Mar, Apr, May, Jun, Jul 1
        assert s[0] == Date(1, 1, 2023)
        assert s[-1] == Date(1, 7, 2023)

    def test_zero_rule(self):
        s = Schedule(
            effective_date=Date(1, 1, 2023),
            termination_date=Date(1, 1, 2024),
            tenor=Period(3, TimeUnit.Months),
            rule=DateGeneration.Zero,
        )
        assert len(s) == 2
        assert s[0] == Date(1, 1, 2023)
        assert s[1] == Date(1, 1, 2024)


class TestMakeSchedule:
    def test_builder(self):
        s = (
            MakeSchedule()
            .from_date(Date(1, 1, 2023))
            .to_date(Date(1, 1, 2024))
            .with_frequency(Frequency.Quarterly)
            .with_calendar(NullCalendar())
            .with_convention(BusinessDayConvention.Unadjusted)
            .backwards()
            .build()
        )
        assert len(s) >= 5
        assert s.start_date == Date(1, 1, 2023)
        assert s.end_date == Date(1, 1, 2024)

    def test_semiannual(self):
        s = (
            MakeSchedule()
            .from_date(Date(15, 3, 2023))
            .to_date(Date(15, 3, 2025))
            .with_frequency(Frequency.Semiannual)
            .with_calendar(TARGET())
            .with_convention(BusinessDayConvention.ModifiedFollowing)
            .build()
        )
        assert len(s) >= 5


class TestScheduleWithCalendar:
    def test_adjusted_dates(self):
        s = Schedule(
            effective_date=Date(1, 1, 2023),  # Sunday
            termination_date=Date(1, 7, 2023),
            tenor=Period(1, TimeUnit.Months),
            calendar=WeekendsOnly(),
            convention=BusinessDayConvention.Following,
            rule=DateGeneration.Forward,
        )
        # All dates should be business days
        cal = WeekendsOnly()
        for d in s.dates:
            assert cal.is_business_day(d), f"{d} is not a business day"
