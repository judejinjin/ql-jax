"""Indexed cash flow – principal adjusted by an index ratio."""

import jax.numpy as jnp
from dataclasses import dataclass


@dataclass(frozen=True)
class IndexedCashFlow:
    """Cash flow whose amount is adjusted by an index ratio.

    amount = notional * index(fixing_date) / base_index

    Parameters
    ----------
    notional : float – base notional amount
    base_index : float – index level at base date
    fixing_date : float – date to observe the index
    payment_date : float – cash flow payment date
    growth_only : bool – if True, pays only the growth (ratio - 1)
    """
    notional: float
    base_index: float
    fixing_date: float
    payment_date: float
    growth_only: bool = False

    def index_ratio(self, index_at_fixing):
        """Compute index ratio."""
        return index_at_fixing / self.base_index

    def amount(self, index_at_fixing):
        """Compute the cash flow amount."""
        ratio = self.index_ratio(index_at_fixing)
        if self.growth_only:
            return self.notional * (ratio - 1.0)
        return self.notional * ratio


def indexed_cashflow_npv(cashflow, index_at_fixing, discount_fn):
    """Present value of an indexed cash flow.

    Parameters
    ----------
    cashflow : IndexedCashFlow
    index_at_fixing : float – projected or realized index level
    discount_fn : callable(t) -> DF

    Returns
    -------
    float
    """
    return cashflow.amount(index_at_fixing) * discount_fn(cashflow.payment_date)


def indexed_leg_npv(cashflows, index_values, discount_fn):
    """NPV of a leg of indexed cash flows.

    Parameters
    ----------
    cashflows : list of IndexedCashFlow
    index_values : array – index levels at each fixing
    discount_fn : callable(t) -> DF

    Returns
    -------
    float
    """
    pv = 0.0
    for i, cf in enumerate(cashflows):
        pv += indexed_cashflow_npv(cf, index_values[i], discount_fn)
    return pv
