"""Callability schedule for callable bonds."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
import jax.numpy as jnp


class CallabilityType(IntEnum):
    CALL = 1
    PUT = -1


@dataclass(frozen=True)
class Callability:
    """Single call or put provision.

    Parameters
    ----------
    type_ : CallabilityType.CALL or PUT
    price : exercise price (clean)
    date : exercise date (float year fraction)
    """
    type_: CallabilityType
    price: float
    date: float


@dataclass(frozen=True)
class CallabilitySchedule:
    """Schedule of call/put provisions.

    Parameters
    ----------
    callabilities : tuple of Callability objects, sorted by date
    """
    callabilities: tuple

    @classmethod
    def from_dates_prices(cls, dates, prices, type_=CallabilityType.CALL):
        """Convenience constructor from parallel arrays."""
        cals = tuple(
            Callability(type_, float(prices[i]), float(dates[i]))
            for i in range(len(dates))
        )
        return cls(cals)

    def dates(self):
        return jnp.array([c.date for c in self.callabilities])

    def prices(self):
        return jnp.array([c.price for c in self.callabilities])

    def next_callability(self, t):
        """Find next call/put provision after time t."""
        for c in self.callabilities:
            if c.date > t:
                return c
        return None
