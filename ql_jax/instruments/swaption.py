"""Swaption instruments."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class Swaption:
    """European swaption — option to enter an interest rate swap.

    Parameters
    ----------
    exercise_time : exercise date (year fraction)
    swap_payment_dates : array of fixed leg payment dates
    swap_accrual_fractions : accrual fractions for each period
    fixed_rate : fixed leg rate
    notional : notional amount
    payer : True for payer swaption (right to pay fixed)
    """
    exercise_time: float
    swap_payment_dates: jnp.ndarray
    swap_accrual_fractions: jnp.ndarray
    fixed_rate: float
    notional: float = 1.0
    payer: bool = True

    @property
    def swap_maturity(self):
        return float(self.swap_payment_dates[-1])

    @property
    def n_periods(self):
        return self.swap_payment_dates.shape[0]


def make_swaption(exercise_time, swap_tenor, fixed_rate, frequency=0.5,
                  notional=1.0, payer=True):
    """Create a standard European swaption.

    Parameters
    ----------
    exercise_time : exercise date
    swap_tenor : length of underlying swap in years
    fixed_rate : fixed rate
    frequency : payment frequency in years
    notional : notional
    payer : True for payer swaption

    Returns
    -------
    Swaption
    """
    n = int(round(swap_tenor / frequency))
    payment_dates = jnp.array(
        [exercise_time + (i + 1) * frequency for i in range(n)],
        dtype=jnp.float64,
    )
    accrual_fracs = jnp.full(n, frequency, dtype=jnp.float64)

    return Swaption(
        exercise_time=exercise_time,
        swap_payment_dates=payment_dates,
        swap_accrual_fractions=accrual_fracs,
        fixed_rate=fixed_rate,
        notional=notional,
        payer=payer,
    )
