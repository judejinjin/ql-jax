"""Equity cash flow – dividend or equity amount leg."""

import jax.numpy as jnp
from dataclasses import dataclass


@dataclass(frozen=True)
class EquityCashFlow:
    """Equity-linked cash flow.

    Pays notional * (S(T) / S(0) - 1) at payment date,
    optionally with dividends reinvested.

    Parameters
    ----------
    notional : float
    base_price : float – initial equity price S(0)
    fixing_date : float – observation date for S(T)
    payment_date : float
    dividend_factor : float – 1.0 for price return, or accumulated div factor
    """
    notional: float
    base_price: float
    fixing_date: float
    payment_date: float
    dividend_factor: float = 1.0

    def amount(self, spot_at_fixing):
        """Cash flow = notional * (div_factor * S(T)/S(0) - 1)."""
        return self.notional * (self.dividend_factor * spot_at_fixing / self.base_price - 1.0)


@dataclass(frozen=True)
class EquityLeg:
    """A leg of equity cash flows (e.g., equity swap floating side).

    Parameters
    ----------
    cashflows : list of EquityCashFlow
    """
    cashflows: tuple

    def npv(self, spot_prices, discount_fn):
        """Price the equity leg.

        Parameters
        ----------
        spot_prices : array – forward equity prices at each fixing
        discount_fn : callable(t) -> DF

        Returns
        -------
        float
        """
        pv = 0.0
        for i, cf in enumerate(self.cashflows):
            pv += cf.amount(spot_prices[i]) * discount_fn(cf.payment_date)
        return pv


def equity_swap_npv(equity_leg, fixed_rate, notional, schedule_dates,
                    day_fractions, spot_prices, discount_fn):
    """NPV of equity swap = equity leg - fixed leg.

    Parameters
    ----------
    equity_leg : EquityLeg
    fixed_rate : float
    notional : float
    schedule_dates : array – payment dates for fixed leg
    day_fractions : array – accrual fractions for fixed leg
    spot_prices : array – forward equity prices at fixing dates
    discount_fn : callable(t) -> DF

    Returns
    -------
    float – NPV (positive = receive equity)
    """
    eq_pv = equity_leg.npv(spot_prices, discount_fn)

    fixed_pv = 0.0
    for i in range(len(day_fractions)):
        fixed_pv += notional * fixed_rate * day_fractions[i] * discount_fn(schedule_dates[i])

    return eq_pv - fixed_pv
