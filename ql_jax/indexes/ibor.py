"""IBOR index implementations.

Covers Libor, Euribor, SOFR, SONIA, and other major IBOR indexes.
"""

from __future__ import annotations
from dataclasses import dataclass, field

from ql_jax.indexes.base import InterestRateIndex
from ql_jax.time.calendar import Calendar, WeekendsOnly, TARGET, UnitedStates, UnitedKingdom, Japan
from ql_jax.time.date import Date, _advance_date, TimeUnit
from ql_jax.time.daycounter import year_fraction


@dataclass
class IborIndex(InterestRateIndex):
    """IBOR-type index with term structure forecasting."""

    end_of_month: bool = True

    def _forecast_fixing(self, date: Date, curve) -> float:
        """Forecast from yield curve: (1/df(t1,t2) - 1) / tau."""
        vd = self.value_date(date)
        md = self.maturity_date(vd)
        t1 = curve.time_from_reference(vd)
        t2 = curve.time_from_reference(md)
        df1 = curve.discount(t1)
        df2 = curve.discount(t2)
        tau = float(t2 - t1)
        if tau <= 0:
            return 0.0
        return float((df1 / df2 - 1.0) / tau)


@dataclass
class OvernightIndex(InterestRateIndex):
    """Overnight rate index (SOFR, SONIA, ESTR, etc.)."""

    def __post_init__(self):
        self.tenor_months = 0

    def _forecast_fixing(self, date: Date, curve) -> float:
        """Forecast overnight rate from curve."""
        vd = self.value_date(date)
        md = self.calendar.advance_by_days(vd, 1)
        t1 = curve.time_from_reference(vd)
        t2 = curve.time_from_reference(md)
        df1 = curve.discount(t1)
        df2 = curve.discount(t2)
        tau = float(t2 - t1)
        if tau <= 0:
            return 0.0
        return float((df1 / df2 - 1.0) / tau)


# ---------------------------------------------------------------------------
# USD Indexes
# ---------------------------------------------------------------------------

def USDLibor(tenor_months: int = 3) -> IborIndex:
    return IborIndex(
        family_name="USDLibor",
        tenor_months=tenor_months,
        fixing_days=2,
        currency_code="USD",
        day_counter_convention="Actual360",
        calendar=UnitedStates(),
    )


def SOFR() -> OvernightIndex:
    return OvernightIndex(
        family_name="SOFR",
        tenor_months=0,
        fixing_days=0,
        currency_code="USD",
        day_counter_convention="Actual360",
        calendar=UnitedStates(),
    )


def FedFunds() -> OvernightIndex:
    return OvernightIndex(
        family_name="FedFunds",
        tenor_months=0,
        fixing_days=0,
        currency_code="USD",
        day_counter_convention="Actual360",
        calendar=UnitedStates(),
    )


# ---------------------------------------------------------------------------
# EUR Indexes
# ---------------------------------------------------------------------------

def Euribor(tenor_months: int = 6) -> IborIndex:
    return IborIndex(
        family_name="Euribor",
        tenor_months=tenor_months,
        fixing_days=2,
        currency_code="EUR",
        day_counter_convention="Actual360",
        calendar=TARGET(),
    )


def EURLibor(tenor_months: int = 6) -> IborIndex:
    return IborIndex(
        family_name="EURLibor",
        tenor_months=tenor_months,
        fixing_days=2,
        currency_code="EUR",
        day_counter_convention="Actual360",
        calendar=TARGET(),
    )


def ESTR() -> OvernightIndex:
    return OvernightIndex(
        family_name="ESTR",
        tenor_months=0,
        fixing_days=0,
        currency_code="EUR",
        day_counter_convention="Actual360",
        calendar=TARGET(),
    )


def Eonia() -> OvernightIndex:
    return OvernightIndex(
        family_name="Eonia",
        tenor_months=0,
        fixing_days=0,
        currency_code="EUR",
        day_counter_convention="Actual360",
        calendar=TARGET(),
    )


# ---------------------------------------------------------------------------
# GBP Indexes
# ---------------------------------------------------------------------------

def GBPLibor(tenor_months: int = 3) -> IborIndex:
    return IborIndex(
        family_name="GBPLibor",
        tenor_months=tenor_months,
        fixing_days=0,
        currency_code="GBP",
        day_counter_convention="Actual365Fixed",
        calendar=UnitedKingdom(),
    )


def SONIA() -> OvernightIndex:
    return OvernightIndex(
        family_name="SONIA",
        tenor_months=0,
        fixing_days=0,
        currency_code="GBP",
        day_counter_convention="Actual365Fixed",
        calendar=UnitedKingdom(),
    )


# ---------------------------------------------------------------------------
# JPY Indexes
# ---------------------------------------------------------------------------

def JPYLibor(tenor_months: int = 6) -> IborIndex:
    return IborIndex(
        family_name="JPYLibor",
        tenor_months=tenor_months,
        fixing_days=2,
        currency_code="JPY",
        day_counter_convention="Actual360",
        calendar=Japan(),
    )


def Tibor(tenor_months: int = 3) -> IborIndex:
    return IborIndex(
        family_name="Tibor",
        tenor_months=tenor_months,
        fixing_days=2,
        currency_code="JPY",
        day_counter_convention="Actual365Fixed",
        calendar=Japan(),
    )


def TONA() -> OvernightIndex:
    return OvernightIndex(
        family_name="TONA",
        tenor_months=0,
        fixing_days=0,
        currency_code="JPY",
        day_counter_convention="Actual365Fixed",
        calendar=Japan(),
    )


# ---------------------------------------------------------------------------
# CHF Indexes
# ---------------------------------------------------------------------------

def CHFLibor(tenor_months: int = 3) -> IborIndex:
    return IborIndex(
        family_name="CHFLibor",
        tenor_months=tenor_months,
        fixing_days=2,
        currency_code="CHF",
        day_counter_convention="Actual360",
        calendar=WeekendsOnly(),
    )


def SARON() -> OvernightIndex:
    return OvernightIndex(
        family_name="SARON",
        tenor_months=0,
        fixing_days=0,
        currency_code="CHF",
        day_counter_convention="Actual360",
        calendar=WeekendsOnly(),
    )


# ---------------------------------------------------------------------------
# Other Indexes
# ---------------------------------------------------------------------------

def AONIA() -> OvernightIndex:
    return OvernightIndex(
        family_name="AONIA",
        tenor_months=0,
        fixing_days=0,
        currency_code="AUD",
        day_counter_convention="Actual365Fixed",
        calendar=WeekendsOnly(),
    )


def CORRA() -> OvernightIndex:
    return OvernightIndex(
        family_name="CORRA",
        tenor_months=0,
        fixing_days=0,
        currency_code="CAD",
        day_counter_convention="Actual365Fixed",
        calendar=WeekendsOnly(),
    )


# ---------------------------------------------------------------------------
# Additional IBOR / Overnight indexes (Phase 12)
# ---------------------------------------------------------------------------

def AUDLibor(tenor_months: int = 3) -> IborIndex:
    return IborIndex(family_name="AUDLibor", tenor_months=tenor_months, fixing_days=2, currency_code="AUD", day_counter_convention="Actual360", calendar=WeekendsOnly())


def BBSW(tenor_months: int = 3) -> IborIndex:
    """Bank Bill Swap Rate (Australia)."""
    return IborIndex(family_name="BBSW", tenor_months=tenor_months, fixing_days=0, currency_code="AUD", day_counter_convention="Actual365Fixed", calendar=WeekendsOnly())


def CADLibor(tenor_months: int = 3) -> IborIndex:
    return IborIndex(family_name="CADLibor", tenor_months=tenor_months, fixing_days=2, currency_code="CAD", day_counter_convention="Actual360", calendar=WeekendsOnly())


def CDOR(tenor_months: int = 3) -> IborIndex:
    """Canadian Dollar Offered Rate."""
    return IborIndex(family_name="CDOR", tenor_months=tenor_months, fixing_days=0, currency_code="CAD", day_counter_convention="Actual365Fixed", calendar=WeekendsOnly())


def CDI() -> OvernightIndex:
    """Brazilian CDI rate."""
    return OvernightIndex(family_name="CDI", tenor_months=0, fixing_days=0, currency_code="BRL", day_counter_convention="Business252", calendar=WeekendsOnly())


def DESTR() -> OvernightIndex:
    """Danish short-term rate."""
    return OvernightIndex(family_name="DESTR", tenor_months=0, fixing_days=0, currency_code="DKK", day_counter_convention="Actual360", calendar=WeekendsOnly())


def DKKLibor(tenor_months: int = 3) -> IborIndex:
    return IborIndex(family_name="DKKLibor", tenor_months=tenor_months, fixing_days=2, currency_code="DKK", day_counter_convention="Actual360", calendar=WeekendsOnly())


def Jibar(tenor_months: int = 3) -> IborIndex:
    """Johannesburg Interbank Agreed Rate."""
    return IborIndex(family_name="Jibar", tenor_months=tenor_months, fixing_days=0, currency_code="ZAR", day_counter_convention="Actual365Fixed", calendar=WeekendsOnly())


def KOFR() -> OvernightIndex:
    """Korean overnight financing rate."""
    return OvernightIndex(family_name="KOFR", tenor_months=0, fixing_days=0, currency_code="KRW", day_counter_convention="Actual365Fixed", calendar=WeekendsOnly())


def NZDLibor(tenor_months: int = 3) -> IborIndex:
    return IborIndex(family_name="NZDLibor", tenor_months=tenor_months, fixing_days=2, currency_code="NZD", day_counter_convention="Actual360", calendar=WeekendsOnly())


def NZOCR() -> OvernightIndex:
    """New Zealand Official Cash Rate."""
    return OvernightIndex(family_name="NZOCR", tenor_months=0, fixing_days=0, currency_code="NZD", day_counter_convention="Actual365Fixed", calendar=WeekendsOnly())


def Pribor(tenor_months: int = 3) -> IborIndex:
    """Prague Interbank Offered Rate."""
    return IborIndex(family_name="Pribor", tenor_months=tenor_months, fixing_days=2, currency_code="CZK", day_counter_convention="Actual360", calendar=WeekendsOnly())


def SEKLibor(tenor_months: int = 3) -> IborIndex:
    return IborIndex(family_name="SEKLibor", tenor_months=tenor_months, fixing_days=2, currency_code="SEK", day_counter_convention="Actual360", calendar=WeekendsOnly())


def Shibor(tenor_months: int = 3) -> IborIndex:
    """Shanghai Interbank Offered Rate."""
    return IborIndex(family_name="Shibor", tenor_months=tenor_months, fixing_days=0, currency_code="CNY", day_counter_convention="Actual360", calendar=WeekendsOnly())


def SWESTR() -> OvernightIndex:
    """Swedish short-term rate."""
    return OvernightIndex(family_name="SWESTR", tenor_months=0, fixing_days=0, currency_code="SEK", day_counter_convention="Actual360", calendar=WeekendsOnly())


def Tonar() -> OvernightIndex:
    """Tokyo Overnight Average Rate (TONA alias)."""
    return TONA()


def Wibor(tenor_months: int = 3) -> IborIndex:
    """Warsaw Interbank Offered Rate."""
    return IborIndex(family_name="Wibor", tenor_months=tenor_months, fixing_days=2, currency_code="PLN", day_counter_convention="Actual365Fixed", calendar=WeekendsOnly())
