"""Forward Rate Agreement (FRA) instrument."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from ql_jax.time.date import Date
from ql_jax.time.daycounter import year_fraction


class FRAType:
    Long = 1    # agree to borrow (benefit from rising rates)
    Short = -1  # agree to lend (benefit from falling rates)


@dataclass
class ForwardRateAgreement:
    """Forward Rate Agreement.

    A FRA is a forward contract where one party pays a fixed rate
    and receives a floating rate (IBOR) over a specified future period.

    Attributes
    ----------
    value_date : Date
        Start of the FRA period.
    maturity_date : Date
        End of the FRA period.
    strike : float
        Agreed fixed rate.
    notional : float
    type_ : int
        FRAType.Long or FRAType.Short.
    index : InterestRateIndex
        The reference IBOR index.
    day_counter : str
    """
    value_date: Date
    maturity_date: Date
    strike: float
    notional: float = 1_000_000.0
    type_: int = FRAType.Long
    index: Any = None
    day_counter: str = "Actual/360"

    def forward_rate(self, discount_curve) -> float:
        """Implied forward rate from discount curve."""
        t1 = discount_curve.time_from_reference(self.value_date)
        t2 = discount_curve.time_from_reference(self.maturity_date)
        df1 = discount_curve.discount(t1)
        df2 = discount_curve.discount(t2)
        tau = year_fraction(self.value_date, self.maturity_date, self.day_counter)
        if tau > 0:
            return (df1 / df2 - 1.0) / tau
        return 0.0

    def npv(self, discount_curve) -> float:
        """Compute NPV of the FRA.

        NPV = type * notional * tau * (forward - strike) * df(maturity)
        """
        fwd = self.forward_rate(discount_curve)
        tau = year_fraction(self.value_date, self.maturity_date, self.day_counter)
        t2 = discount_curve.time_from_reference(self.maturity_date)
        df = discount_curve.discount(t2)
        return float(self.type_ * self.notional * tau * (fwd - self.strike) * df)

    def forward_value(self, discount_curve) -> float:
        """Forward value (undiscounted payoff)."""
        fwd = self.forward_rate(discount_curve)
        tau = year_fraction(self.value_date, self.maturity_date, self.day_counter)
        return float(self.type_ * self.notional * tau * (fwd - self.strike))

    def implied_yield(self, discount_curve) -> float:
        """Return the implied forward rate."""
        return self.forward_rate(discount_curve)
