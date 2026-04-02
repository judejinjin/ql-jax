"""Bond forward, FX forward, and other miscellaneous instruments."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class BondForward:
    """Forward contract on a bond.

    Parameters
    ----------
    forward_date : delivery date (year fraction)
    bond_maturity : underlying bond maturity
    coupon_rate : bond coupon rate
    face_value : bond face value
    forward_price : agreed forward price (None = compute fair price)
    """
    forward_date: float
    bond_maturity: float
    coupon_rate: float
    face_value: float = 100.0
    forward_price: float | None = None


@dataclass(frozen=True)
class FXForward:
    """Foreign exchange forward contract.

    Parameters
    ----------
    domestic_notional : notional in domestic currency
    foreign_notional : notional in foreign currency
    maturity : delivery date
    spot_fx : current spot exchange rate (domestic per foreign)
    """
    domestic_notional: float
    foreign_notional: float
    maturity: float
    spot_fx: float


def fx_forward_price(spot_fx, r_domestic, r_foreign, T):
    """Compute FX forward price via interest rate parity.

    F = S * exp((r_d - r_f) * T)

    Parameters
    ----------
    spot_fx : spot exchange rate
    r_domestic : domestic risk-free rate
    r_foreign : foreign risk-free rate
    T : time to maturity

    Returns
    -------
    forward FX rate
    """
    return spot_fx * jnp.exp((r_domestic - r_foreign) * T)


def fx_forward_npv(fwd, discount_domestic_fn, discount_foreign_fn):
    """NPV of an FX forward.

    Parameters
    ----------
    fwd : FXForward
    discount_domestic_fn : P_d(0,T)
    discount_foreign_fn : P_f(0,T)

    Returns
    -------
    npv in domestic currency
    """
    T = fwd.maturity
    Pd = discount_domestic_fn(T)
    Pf = discount_foreign_fn(T)

    # Pay domestic_notional, receive foreign_notional * spot
    return fwd.foreign_notional * fwd.spot_fx * Pf - fwd.domestic_notional * Pd


@dataclass(frozen=True)
class Stock:
    """Simple stock instrument for pricing."""
    spot: float
    dividend_yield: float = 0.0
    name: str = ""


@dataclass(frozen=True)
class CompositeInstrument:
    """A portfolio of instruments with weights.

    Used for decomposing complex products.
    """
    instruments: tuple
    weights: tuple

    def npv(self, pricer_fn):
        """Compute weighted sum of NPVs."""
        total = 0.0
        for inst, w in zip(self.instruments, self.weights):
            total = total + w * pricer_fn(inst)
        return total
