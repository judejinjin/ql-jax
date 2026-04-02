"""Dividend and equity cash flows."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp

from ql_jax.time.date import Date
from ql_jax.cashflows.simple import SimpleCashFlow


@dataclass(frozen=True)
class FixedDividend(SimpleCashFlow):
    """Dividend of a fixed dollar amount."""
    pass


@dataclass(frozen=True)
class FractionalDividend:
    """Dividend as a fraction of the stock price."""
    date: Date
    rate: float           # dividend yield fraction
    reference_price: float | None = None

    @property
    def amount(self) -> float:
        if self.reference_price is not None:
            return self.rate * self.reference_price
        return 0.0
