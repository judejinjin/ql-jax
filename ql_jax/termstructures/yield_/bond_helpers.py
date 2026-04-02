"""Bond helpers for yield curve bootstrapping."""

import jax.numpy as jnp
from dataclasses import dataclass


@dataclass(frozen=True)
class BondHelper:
    """Generic bond helper for bootstrapping.

    Parameters
    ----------
    clean_price : float – market clean price
    coupon_rate : float – annual coupon rate
    settlement_days : int
    face_amount : float
    schedule_times : array – payment times (year fractions)
    day_count_fraction : float – year fraction per period
    """
    clean_price: float
    coupon_rate: float
    settlement_days: int
    face_amount: float
    schedule_times: jnp.ndarray
    day_count_fraction: float

    def implied_quote(self, discount_fn):
        """Compute model clean price given a discount function."""
        n = len(self.schedule_times)
        pv = 0.0
        for i in range(n):
            t = self.schedule_times[i]
            coupon = self.coupon_rate * self.day_count_fraction * self.face_amount
            if i == n - 1:
                coupon += self.face_amount  # principal
            pv += coupon * discount_fn(t)
        return pv

    def quote_error(self, discount_fn):
        """Difference between model and market price."""
        return self.implied_quote(discount_fn) - self.clean_price


@dataclass(frozen=True)
class FixedRateBondHelper(BondHelper):
    """Specialized for fixed-rate bonds (identical to BondHelper)."""
    pass


@dataclass(frozen=True)
class ZeroCouponBondHelper:
    """Helper for zero-coupon bonds.

    Parameters
    ----------
    market_price : float – market price per unit face
    maturity : float – maturity in year fractions
    face_amount : float
    """
    market_price: float
    maturity: float
    face_amount: float = 100.0

    def implied_quote(self, discount_fn):
        return self.face_amount * discount_fn(self.maturity)

    def quote_error(self, discount_fn):
        return self.implied_quote(discount_fn) - self.market_price

    def implied_zero_rate(self):
        """Zero rate implied by market price."""
        return -jnp.log(self.market_price / self.face_amount) / self.maturity
