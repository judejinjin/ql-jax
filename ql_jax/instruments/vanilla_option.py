"""Vanilla option instrument."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class VanillaOption:
    """European or American vanilla option.

    Attributes
    ----------
    strike : float
    option_type : int
        OptionType.Call or OptionType.Put.
    exercise : EuropeanExercise, AmericanExercise, or BermudanExercise.
    """
    strike: float
    option_type: int
    exercise: object
