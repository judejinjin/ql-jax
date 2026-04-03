"""Swing and storage options for energy markets."""

from __future__ import annotations

from dataclasses import dataclass
import jax.numpy as jnp


@dataclass(frozen=True)
class VanillaSwingOption:
    """Swing option (multiple exercise rights).

    Parameters
    ----------
    exercise_dates : available exercise dates
    strike : exercise price
    min_exercises : minimum required exercises
    max_exercises : maximum allowed exercises
    notional_per_exercise : notional per single exercise
    """
    exercise_dates: jnp.ndarray
    strike: float
    min_exercises: int = 0
    max_exercises: int = 1
    notional_per_exercise: float = 1.0

    @property
    def n_dates(self):
        return len(self.exercise_dates)


@dataclass(frozen=True)
class VanillaStorageOption:
    """Storage option (e.g., natural gas storage).

    Parameters
    ----------
    injection_dates : dates when injection is possible
    withdrawal_dates : dates when withdrawal is possible
    max_capacity : maximum storage capacity (volume)
    max_injection_rate : max injection per period
    max_withdrawal_rate : max withdrawal per period
    injection_cost : cost per unit injected
    withdrawal_cost : cost per unit withdrawn
    """
    injection_dates: jnp.ndarray
    withdrawal_dates: jnp.ndarray
    max_capacity: float = 1.0
    max_injection_rate: float = 0.1
    max_withdrawal_rate: float = 0.1
    injection_cost: float = 0.0
    withdrawal_cost: float = 0.0
