"""Exercise types for option instruments."""

from __future__ import annotations

from dataclasses import dataclass

from ql_jax.time.date import Date


@dataclass(frozen=True)
class EuropeanExercise:
    """Exercise only at expiry."""
    date: Date


@dataclass(frozen=True)
class AmericanExercise:
    """Exercise any time between earliest and latest dates."""
    earliest_date: Date
    latest_date: Date


@dataclass(frozen=True)
class BermudanExercise:
    """Exercise at discrete dates."""
    dates: tuple[Date, ...]
