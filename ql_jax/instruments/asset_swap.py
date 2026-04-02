"""Additional swap types — asset swap, basis swaps, equity TRS, non-standard."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class AssetSwap:
    """Asset swap — exchanging bond cash flows for floating rate payments.

    Parameters
    ----------
    notional : notional amount
    bond_price : dirty price of the bond
    bond_coupon : bond coupon rate
    spread : spread over the floating rate
    maturity : swap maturity
    is_par : True for par asset swap, False for market value
    """
    notional: float
    bond_price: float
    bond_coupon: float
    spread: float
    maturity: float
    is_par: bool = True


@dataclass(frozen=True)
class BMASwap:
    """BMA/Municipal swap — fixed vs BMA floating rate.

    Parameters
    ----------
    notional : notional amount
    fixed_rate : fixed rate
    maturity : swap maturity
    """
    notional: float
    fixed_rate: float
    maturity: float


@dataclass(frozen=True)
class FloatFloatSwap:
    """Float-float swap (basis swap) — two floating legs referencing different indexes.

    Parameters
    ----------
    notional : notional amount
    spread1 : spread on first floating leg
    spread2 : spread on second floating leg
    maturity : swap maturity
    """
    notional: float
    spread1: float = 0.0
    spread2: float = 0.0
    maturity: float = 5.0


@dataclass(frozen=True)
class NonStandardSwap:
    """Non-standard swap with amortizing notional or variable spreads.

    Parameters
    ----------
    notionals : array of notionals per period
    fixed_rates : array of fixed rates per period
    spreads : array of spreads on floating leg per period
    payment_dates : payment times
    """
    notionals: tuple
    fixed_rates: tuple
    spreads: tuple
    payment_dates: tuple


@dataclass(frozen=True)
class EquityTotalReturnSwap:
    """Equity total return swap.

    One party pays equity total return (price change + dividends),
    the other pays floating rate + spread.

    Parameters
    ----------
    notional : notional amount
    equity_spot : initial equity price
    spread : spread over floating rate
    maturity : swap maturity
    """
    notional: float
    equity_spot: float
    spread: float
    maturity: float

    def payoff(self, equity_final, floating_payments):
        equity_return = (equity_final / self.equity_spot - 1.0) * self.notional
        float_total = jnp.sum(floating_payments)
        return equity_return - float_total
