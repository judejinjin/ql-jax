"""Tests for day counters and calendar coverage."""
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from ql_jax.time.date import Date
from ql_jax.time.daycounter import day_count, year_fraction, DayCountConvention


# ── Day Count Conventions ────────────────────────────────────────────────────

class TestDayCountConventions:
    def test_actual360(self):
        d1, d2 = Date(1, 1, 2024), Date(1, 4, 2024)  # 91 days
        dc = day_count(d1, d2, DayCountConvention.Actual360)
        yf = year_fraction(d1, d2, DayCountConvention.Actual360)
        assert dc == 91
        assert abs(yf - 91.0 / 360.0) < 1e-10

    def test_actual365fixed(self):
        d1, d2 = Date(1, 1, 2024), Date(1, 1, 2025)  # leap year = 366 days
        dc = day_count(d1, d2, DayCountConvention.Actual365Fixed)
        yf = year_fraction(d1, d2, DayCountConvention.Actual365Fixed)
        assert dc == 366
        assert abs(yf - 366.0 / 365.0) < 1e-10

    def test_thirty360_bond_basis(self):
        d1, d2 = Date(1, 1, 2024), Date(1, 7, 2024)
        yf = year_fraction(d1, d2, DayCountConvention.Thirty360BondBasis)
        assert abs(yf - 0.5) < 1e-10  # 180/360

    def test_thirty360_european(self):
        d1, d2 = Date(1, 1, 2024), Date(1, 7, 2024)
        yf = year_fraction(d1, d2, DayCountConvention.Thirty360EurobondBasis)
        assert abs(yf - 0.5) < 1e-10

    def test_actual_actual_isda(self):
        d1, d2 = Date(1, 1, 2024), Date(1, 1, 2025)
        yf = year_fraction(d1, d2, DayCountConvention.ActualActualISDA)
        assert abs(yf - 1.0) < 0.01  # approximately 1 year

    def test_actual365_noleap(self):
        d1, d2 = Date(1, 1, 2024), Date(1, 1, 2025)
        yf = year_fraction(d1, d2, DayCountConvention.Actual365NoLeap)
        # 365 out of 366 actual (skips Feb 29)
        assert 0.99 < yf < 1.01


# ── Additional Calendars ─────────────────────────────────────────────────────

class TestCalendars:
    def test_canada(self):
        from ql_jax.time.calendars import Canada
        cal = Canada()
        # Canada Day: July 1
        d = Date(1, 7, 2024)
        assert cal.is_holiday(d)

    def test_china(self):
        from ql_jax.time.calendars import China
        cal = China()
        # Chinese New Year is a holiday (varies by year)
        # Weekend test
        sat = Date(6, 1, 2024)  # Saturday
        assert not cal.is_business_day(sat)

    def test_hongkong(self):
        from ql_jax.time.calendars import HongKong
        cal = HongKong()
        # New Year
        ny = Date(1, 1, 2024)
        assert cal.is_holiday(ny)

    def test_singapore(self):
        from ql_jax.time.calendars import Singapore
        cal = Singapore()
        ny = Date(1, 1, 2024)
        assert cal.is_holiday(ny)

    def test_india(self):
        from ql_jax.time.calendars import India
        cal = India()
        # Republic Day: Jan 26
        d = Date(26, 1, 2024)
        assert cal.is_holiday(d)

    def test_brazil(self):
        from ql_jax.time.calendars import Brazil
        cal = Brazil()
        # Tiradentes Day: April 21
        d = Date(21, 4, 2024)
        assert cal.is_holiday(d)

    def test_south_africa(self):
        from ql_jax.time.calendars import SouthAfrica
        cal = SouthAfrica()
        # Freedom Day: April 27
        d = Date(27, 4, 2024)  # Saturday in 2024
        # Check Human Rights Day: March 21
        d2 = Date(21, 3, 2024)
        assert cal.is_holiday(d2)

    def test_mexico(self):
        from ql_jax.time.calendars import Mexico
        cal = Mexico()
        # Independence Day: Sep 16 (Monday in 2024)
        d = Date(16, 9, 2024)
        assert cal.is_holiday(d)

    def test_sweden(self):
        from ql_jax.time.calendars import Sweden
        cal = Sweden()
        # National Day: June 6
        d = Date(6, 6, 2024)
        assert cal.is_holiday(d)

    def test_switzerland(self):
        from ql_jax.time.calendars import Switzerland
        cal = Switzerland()
        # Swiss National Day: Aug 1
        d = Date(1, 8, 2024)
        assert cal.is_holiday(d)

    def test_target(self):
        from ql_jax.time.calendar import TARGET
        cal = TARGET()
        # May 1 is TARGET holiday
        d = Date(1, 5, 2024)
        assert cal.is_holiday(d)

    def test_business_day_count(self):
        from ql_jax.time.calendar import TARGET
        cal = TARGET()
        d1 = Date(1, 1, 2024)
        d2 = Date(31, 1, 2024)
        bdays = cal.business_days_between(d1, d2)
        assert 18 <= bdays <= 23  # roughly 22 business days

    def test_advance(self):
        from ql_jax.time.calendar import TARGET
        cal = TARGET()
        d = Date(1, 1, 2024)  # holiday
        next_bd = cal.advance(d, 1)
        assert cal.is_business_day(next_bd)


# ── Joint Calendar ───────────────────────────────────────────────────────────

class TestJointCalendar:
    def test_joint_calendar(self):
        from ql_jax.time.calendar import TARGET, UnitedStates, JointCalendar
        cal = JointCalendar(TARGET(), UnitedStates())
        # Both should agree weekends are non-business days
        sat = Date(6, 1, 2024)
        assert not cal.is_business_day(sat)
