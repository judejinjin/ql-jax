"""Tests for day counters — porting QuantLib test-suite/daycounters.cpp."""

import math
from ql_jax.time.date import Date
from ql_jax.time.daycounter import year_fraction, day_count, DayCountConvention


class TestActual360:
    def test_basic(self):
        d1 = Date(1, 1, 2023)
        d2 = Date(1, 4, 2023)  # 90 days
        yf = year_fraction(d1, d2, DayCountConvention.Actual360)
        assert abs(yf - 90 / 360.0) < 1e-12

    def test_full_year(self):
        d1 = Date(1, 1, 2023)
        d2 = Date(1, 1, 2024)
        yf = year_fraction(d1, d2, DayCountConvention.Actual360)
        assert abs(yf - 365 / 360.0) < 1e-12


class TestActual365Fixed:
    def test_basic(self):
        d1 = Date(1, 1, 2023)
        d2 = Date(1, 7, 2023)  # 181 days
        yf = year_fraction(d1, d2, DayCountConvention.Actual365Fixed)
        assert abs(yf - 181 / 365.0) < 1e-12


class TestActualActualISDA:
    def test_same_year(self):
        d1 = Date(1, 1, 2023)
        d2 = Date(1, 7, 2023)
        yf = year_fraction(d1, d2, DayCountConvention.ActualActualISDA)
        assert abs(yf - 181 / 365.0) < 1e-12

    def test_leap_year(self):
        d1 = Date(1, 1, 2024)
        d2 = Date(1, 1, 2025)
        yf = year_fraction(d1, d2, DayCountConvention.ActualActualISDA)
        assert abs(yf - 1.0) < 1e-12

    def test_cross_year(self):
        d1 = Date(1, 11, 2023)
        d2 = Date(1, 3, 2024)
        yf = year_fraction(d1, d2, DayCountConvention.ActualActualISDA)
        # 61 days in 2023 (365 day year) + 60 days in 2024 (366 day year)
        expected = 61 / 365.0 + 60 / 366.0
        assert abs(yf - expected) < 1e-10


class TestThirty360:
    def test_bond_basis(self):
        d1 = Date(1, 1, 2023)
        d2 = Date(1, 7, 2023)
        yf = year_fraction(d1, d2, DayCountConvention.Thirty360BondBasis)
        # 30/360: 6 months = 180/360 = 0.5
        assert abs(yf - 0.5) < 1e-12

    def test_eurobond_basis(self):
        d1 = Date(31, 1, 2023)
        d2 = Date(28, 2, 2023)
        yf = year_fraction(d1, d2, DayCountConvention.Thirty360EurobondBasis)
        dc = day_count(d1, d2, DayCountConvention.Thirty360EurobondBasis)
        # 30E/360: d1=30 -> 30, d2=28 -> 28, so 30*(2-1)+(28-30) = 28 days
        assert dc == 28

    def test_full_year_bond_basis(self):
        d1 = Date(1, 1, 2023)
        d2 = Date(1, 1, 2024)
        yf = year_fraction(d1, d2, DayCountConvention.Thirty360BondBasis)
        assert abs(yf - 1.0) < 1e-12


class TestOne:
    def test_same_date(self):
        d = Date(1, 1, 2023)
        assert year_fraction(d, d, DayCountConvention.One) == 0.0

    def test_different_dates(self):
        d1 = Date(1, 1, 2023)
        d2 = Date(1, 6, 2023)
        assert year_fraction(d1, d2, DayCountConvention.One) == 1.0


class TestSameDate:
    def test_all_conventions_zero(self):
        d = Date(15, 6, 2023)
        for conv in (
            DayCountConvention.Actual360,
            DayCountConvention.Actual365Fixed,
            DayCountConvention.ActualActualISDA,
            DayCountConvention.Thirty360BondBasis,
            DayCountConvention.One,
        ):
            assert year_fraction(d, d, conv) == 0.0
