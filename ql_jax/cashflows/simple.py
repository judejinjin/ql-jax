"""Simple cash flows — fixed amounts at specified dates."""

from __future__ import annotations

from dataclasses import dataclass

from ql_jax.time.date import Date


@dataclass(frozen=True)
class SimpleCashFlow:
    """A single cash flow of a known amount on a known date."""
    date: Date
    amount: float


@dataclass(frozen=True)
class Redemption(SimpleCashFlow):
    """Bond redemption (principal repayment)."""
    pass
