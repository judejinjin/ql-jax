"""Convertible bond instrument."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class ConvertibleBond:
    """Convertible bond — bond that can be converted into equity shares.

    Parameters
    ----------
    face_value : par value of the bond
    coupon_rate : annual coupon rate
    conversion_ratio : number of shares per bond upon conversion
    maturity : time to maturity
    coupon_frequency : coupons per year
    call_price : price at which issuer can call the bond (None = not callable)
    put_price : price at which holder can put the bond (None = not putable)
    call_date : earliest call date
    put_date : earliest put date
    """
    face_value: float
    coupon_rate: float
    conversion_ratio: float
    maturity: float
    coupon_frequency: int = 2
    call_price: float = None
    put_price: float = None
    call_date: float = None
    put_date: float = None

    @property
    def conversion_price(self):
        """Implied conversion price per share."""
        return self.face_value / self.conversion_ratio

    def conversion_value(self, stock_price):
        """Value if converted to equity."""
        return self.conversion_ratio * stock_price

    def parity(self, stock_price):
        """Conversion parity (conversion value / face)."""
        return self.conversion_value(stock_price) / self.face_value
