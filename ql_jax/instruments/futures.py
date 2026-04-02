"""Interest-rate futures instruments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp


@dataclass(frozen=True)
class IRFuture:
    """Generic interest-rate future.

    Parameters
    ----------
    price : futures price (e.g. 97.5 => implied rate 2.5%)
    maturity : maturity date as year fraction
    tenor : underlying tenor in years (e.g. 0.25 for 3M)
    notional : contract notional
    convention : day count convention string
    """
    price: float
    maturity: float
    tenor: float = 0.25
    notional: float = 1_000_000.0
    convention: str = "Actual/360"

    @property
    def implied_rate(self) -> float:
        return (100.0 - self.price) / 100.0


@dataclass(frozen=True)
class OvernightIndexFuture:
    """Overnight-index future (e.g. SOFR future).

    Parameters
    ----------
    overnight_index : reference overnight rate index
    value_date : start of accrual period (year frac)
    maturity_date : end of accrual period (year frac)
    notional : contract notional
    price : quoted price
    """
    overnight_index: Any = None
    value_date: float = 0.0
    maturity_date: float = 0.25
    notional: float = 1_000_000.0
    price: float = 100.0

    @property
    def implied_rate(self) -> float:
        return (100.0 - self.price) / 100.0


@dataclass(frozen=True)
class PerpetualFuture:
    """Perpetual futures contract.

    Parameters
    ----------
    underlying_price : current spot price
    funding_rate : periodic funding rate
    notional : contract notional
    mark_price : current mark price
    """
    underlying_price: float = 100.0
    funding_rate: float = 0.0
    notional: float = 1.0
    mark_price: float = 100.0

    def funding_payment(self) -> float:
        """Compute funding payment for one period."""
        return self.notional * self.funding_rate * self.mark_price
