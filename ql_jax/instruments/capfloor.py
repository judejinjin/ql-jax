"""Cap/Floor instruments.

A cap is a portfolio of caplets, a floor is a portfolio of floorlets.
Each caplet pays max(L - K, 0) * tau * notional at the end of the accrual period.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import jax.numpy as jnp


@dataclass(frozen=True)
class CapFloor:
    """Interest rate cap or floor.

    Parameters
    ----------
    cap_or_floor : 'cap' or 'floor'
    strike : fixed strike rate
    payment_dates : array of payment dates (year fractions from today)
    reset_dates : array of reset dates
    accrual_fractions : day count fractions for each period
    notional : notional amount
    """
    cap_or_floor: str  # 'cap' or 'floor'
    strike: float
    payment_dates: jnp.ndarray
    reset_dates: jnp.ndarray
    accrual_fractions: jnp.ndarray
    notional: float = 1.0

    @property
    def is_cap(self):
        return self.cap_or_floor == 'cap'

    @property
    def n_periods(self):
        return self.payment_dates.shape[0]


def make_cap(strike, start, maturity, frequency=0.25, notional=1.0):
    """Create a standard cap with regular reset dates.

    Parameters
    ----------
    strike : cap strike rate
    start : start time (year fraction)
    maturity : maturity time (year fraction)
    frequency : payment frequency in years (0.25 = quarterly)
    notional : notional amount

    Returns
    -------
    CapFloor
    """
    n = int(round((maturity - start) / frequency))
    reset_dates = jnp.array([start + i * frequency for i in range(n)], dtype=jnp.float64)
    payment_dates = jnp.array([start + (i + 1) * frequency for i in range(n)], dtype=jnp.float64)
    accrual_fractions = jnp.full(n, frequency, dtype=jnp.float64)

    return CapFloor(
        cap_or_floor='cap', strike=strike,
        payment_dates=payment_dates, reset_dates=reset_dates,
        accrual_fractions=accrual_fractions, notional=notional,
    )


def make_floor(strike, start, maturity, frequency=0.25, notional=1.0):
    """Create a standard floor."""
    cap = make_cap(strike, start, maturity, frequency, notional)
    return CapFloor(
        cap_or_floor='floor', strike=cap.strike,
        payment_dates=cap.payment_dates, reset_dates=cap.reset_dates,
        accrual_fractions=cap.accrual_fractions, notional=cap.notional,
    )
