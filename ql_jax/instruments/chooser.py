"""Chooser options — simple and complex."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class SimpleChooserOption:
    """Simple chooser option.

    At choose_date the holder can choose between a call and a put
    with the same strike and maturity.

    Parameters
    ----------
    strike : common strike for both call and put
    choose_date : time at which the choice is made
    maturity : common maturity for call and put
    """
    strike: float
    choose_date: float
    maturity: float


@dataclass(frozen=True)
class ComplexChooserOption:
    """Complex chooser option.

    At choose_date the holder chooses between a call (with strike_call,
    maturity_call) and a put (with strike_put, maturity_put).
    """
    strike_call: float
    strike_put: float
    choose_date: float
    maturity_call: float
    maturity_put: float
