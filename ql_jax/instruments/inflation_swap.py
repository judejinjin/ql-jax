"""Inflation swap instruments — CPI, Zero-Coupon Inflation, YoY Inflation."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class ZeroCouponInflationSwap:
    """Zero-coupon inflation swap.

    One party pays a fixed amount, the other pays
    notional * (I(T)/I(0) - 1) where I is the inflation index.

    Parameters
    ----------
    notional : notional amount
    fixed_rate : fixed rate paid
    maturity : time to maturity in years
    base_cpi : base CPI at inception
    is_payer : True if paying fixed (receiving inflation)
    """
    notional: float
    fixed_rate: float
    maturity: float
    base_cpi: float
    is_payer: bool = True

    def npv(self, current_cpi, discount_factor):
        """NPV of the swap.

        Parameters
        ----------
        current_cpi : estimated CPI at maturity
        discount_factor : discount factor to maturity
        """
        inflation_leg = self.notional * (current_cpi / self.base_cpi - 1.0)
        fixed_leg = self.notional * ((1.0 + self.fixed_rate) ** self.maturity - 1.0)

        if self.is_payer:
            return (inflation_leg - fixed_leg) * discount_factor
        return (fixed_leg - inflation_leg) * discount_factor


@dataclass(frozen=True)
class YearOnYearInflationSwap:
    """Year-on-year inflation swap.

    Periodic payments based on year-on-year inflation rates
    vs fixed rate payments.

    Parameters
    ----------
    notional : notional amount
    fixed_rate : fixed rate per period
    maturity : total maturity in years
    payment_frequency : number of payments per year
    is_payer : True if paying fixed
    """
    notional: float
    fixed_rate: float
    maturity: float
    payment_frequency: int = 1
    is_payer: bool = True


@dataclass(frozen=True)
class CPISwap:
    """CPI swap.

    Similar to zero-coupon inflation swap but with periodic
    CPI-linked coupon payments.

    Parameters
    ----------
    notional : notional amount
    fixed_rate : fixed rate per period
    maturity : total maturity
    base_cpi : base CPI at inception
    payment_frequency : coupons per year
    is_payer : True if paying fixed
    """
    notional: float
    fixed_rate: float
    maturity: float
    base_cpi: float
    payment_frequency: int = 1
    is_payer: bool = True
