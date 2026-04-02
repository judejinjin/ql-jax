"""Energy/commodity derivatives — storage and swing options."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class VanillaStorageOption:
    """Gas/energy storage option.

    Models a storage facility with injection/withdrawal constraints.

    Parameters
    ----------
    strike : strike price (cost of injection/withdrawal)
    maturity : final maturity (year frac)
    exercise_dates : array of possible exercise dates (year fracs)
    min_volume : minimum stored volume
    max_volume : maximum stored volume
    injection_rate : max injection rate per period
    withdrawal_rate : max withdrawal rate per period
    initial_volume : starting stored volume
    """
    strike: float
    maturity: float
    exercise_dates: jnp.ndarray
    min_volume: float = 0.0
    max_volume: float = 1.0
    injection_rate: float = 1.0
    withdrawal_rate: float = 1.0
    initial_volume: float = 0.0

    @property
    def n_exercises(self) -> int:
        return self.exercise_dates.shape[0]


@dataclass(frozen=True)
class VanillaSwingOption:
    """Energy swing option (take-or-pay).

    Gives the holder the right to exercise a specified number of times
    at a fixed strike price.

    Parameters
    ----------
    strike : strike price per unit
    exercise_dates : array of possible exercise dates (year fracs)
    min_exercises : minimum number of exercises
    max_exercises : maximum number of exercises
    notional_per_exercise : volume per exercise
    """
    strike: float
    exercise_dates: jnp.ndarray
    min_exercises: int = 0
    max_exercises: int = 1
    notional_per_exercise: float = 1.0

    @property
    def n_dates(self) -> int:
        return self.exercise_dates.shape[0]

    @property
    def maturity(self) -> float:
        return float(self.exercise_dates[-1])
