"""Basket options — options on multiple underlying assets."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import jax.numpy as jnp


class BasketType(Enum):
    MIN = "min"
    MAX = "max"
    AVERAGE = "average"


@dataclass(frozen=True)
class BasketPayoff:
    """Payoff for basket options.

    Aggregates multiple asset prices into a single basket value,
    then applies a standard payoff.
    """
    basket_type: BasketType
    strike: float
    is_call: bool

    def aggregate(self, prices):
        """Aggregate asset prices into basket value.

        Parameters
        ----------
        prices : array of individual asset prices
        """
        if self.basket_type == BasketType.MIN:
            return jnp.min(prices)
        elif self.basket_type == BasketType.MAX:
            return jnp.max(prices)
        else:  # AVERAGE
            return jnp.mean(prices)

    def __call__(self, prices):
        basket_val = self.aggregate(prices)
        if self.is_call:
            return jnp.maximum(basket_val - self.strike, 0.0)
        return jnp.maximum(self.strike - basket_val, 0.0)


@dataclass(frozen=True)
class BasketOption:
    """Option on a basket of assets.

    Parameters
    ----------
    payoff : BasketPayoff
    maturity : time to maturity
    exercise_type : 'european' or 'american'
    """
    payoff: BasketPayoff
    maturity: float
    exercise_type: str = "european"


@dataclass(frozen=True)
class SpreadOption:
    """Option on the spread between two assets.

    payoff = max(S1 - S2 - K, 0) for call
    """
    strike: float
    is_call: bool
    maturity: float

    def payoff_value(self, s1, s2):
        spread = s1 - s2
        if self.is_call:
            return jnp.maximum(spread - self.strike, 0.0)
        return jnp.maximum(self.strike - spread, 0.0)
