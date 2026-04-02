"""Swap indexes.

A swap index represents a fixed-for-floating interest rate swap.
Used as the underlying for swaptions.
"""

from __future__ import annotations
from dataclasses import dataclass, field

from ql_jax.indexes.base import InterestRateIndex
from ql_jax.indexes.ibor import IborIndex, Euribor, USDLibor, GBPLibor, JPYLibor, CHFLibor
from ql_jax.time.calendar import TARGET, UnitedStates, UnitedKingdom, Japan, WeekendsOnly


@dataclass
class SwapIndex(InterestRateIndex):
    """Swap rate index."""
    fixed_leg_tenor_months: int = 12  # Annual fixed leg
    fixed_leg_convention: str = "Actual365Fixed"
    ibor_index_name: str = ""

    def _forecast_fixing(self, date, curve) -> float:
        """Forecast fair swap rate from discount curve."""
        from ql_jax.time.date import _advance_date, TimeUnit
        vd = self.value_date(date)
        # Build fixed leg schedule
        n_periods = self.tenor_months // self.fixed_leg_tenor_months
        if n_periods <= 0:
            n_periods = 1
        annuity = 0.0
        prev_date = vd
        for i in range(1, n_periods + 1):
            payment_date = _advance_date(vd, i * self.fixed_leg_tenor_months, TimeUnit.Months)
            t_prev = curve.time_from_reference(prev_date)
            t_pay = curve.time_from_reference(payment_date)
            tau = float(t_pay - t_prev)
            df = curve.discount(t_pay)
            annuity += tau * float(df)
            prev_date = payment_date

        # Swap rate = (df_start - df_end) / annuity
        t_start = curve.time_from_reference(vd)
        t_end = curve.time_from_reference(
            _advance_date(vd, self.tenor_months, TimeUnit.Months)
        )
        df_start = curve.discount(t_start)
        df_end = curve.discount(t_end)
        if annuity <= 0:
            return 0.0
        return float((df_start - df_end) / annuity)


# ---------------------------------------------------------------------------
# Standard swap index constructors
# ---------------------------------------------------------------------------

def EuriborSwapIsdaFixA(tenor_years: int = 10) -> SwapIndex:
    return SwapIndex(
        family_name="EuriborSwapIsdaFixA",
        tenor_months=tenor_years * 12,
        fixing_days=2,
        currency_code="EUR",
        day_counter_convention="Thirty360",
        calendar=TARGET(),
        fixed_leg_tenor_months=12,
        fixed_leg_convention="Thirty360",
        ibor_index_name="Euribor6M",
    )


def UsdLiborSwapIsdaFixAm(tenor_years: int = 10) -> SwapIndex:
    return SwapIndex(
        family_name="UsdLiborSwapIsdaFixAm",
        tenor_months=tenor_years * 12,
        fixing_days=2,
        currency_code="USD",
        day_counter_convention="Thirty360",
        calendar=UnitedStates(),
        fixed_leg_tenor_months=6,
        fixed_leg_convention="Thirty360",
        ibor_index_name="USDLibor3M",
    )


def GbpLiborSwapIsdaFix(tenor_years: int = 10) -> SwapIndex:
    return SwapIndex(
        family_name="GbpLiborSwapIsdaFix",
        tenor_months=tenor_years * 12,
        fixing_days=0,
        currency_code="GBP",
        day_counter_convention="Actual365Fixed",
        calendar=UnitedKingdom(),
        fixed_leg_tenor_months=6,
        fixed_leg_convention="Actual365Fixed",
        ibor_index_name="GBPLibor6M",
    )


def JpyLiborSwapIsdaFixAm(tenor_years: int = 10) -> SwapIndex:
    return SwapIndex(
        family_name="JpyLiborSwapIsdaFixAm",
        tenor_months=tenor_years * 12,
        fixing_days=2,
        currency_code="JPY",
        day_counter_convention="Actual365Fixed",
        calendar=Japan(),
        fixed_leg_tenor_months=6,
        fixed_leg_convention="Actual365Fixed",
        ibor_index_name="JPYLibor6M",
    )


def ChfLiborSwapIsdaFix(tenor_years: int = 10) -> SwapIndex:
    return SwapIndex(
        family_name="ChfLiborSwapIsdaFix",
        tenor_months=tenor_years * 12,
        fixing_days=2,
        currency_code="CHF",
        day_counter_convention="Thirty360",
        calendar=WeekendsOnly(),
        fixed_leg_tenor_months=12,
        fixed_leg_convention="Thirty360",
        ibor_index_name="CHFLibor6M",
    )
