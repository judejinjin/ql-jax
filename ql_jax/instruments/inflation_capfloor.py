"""Inflation cap/floor instruments — CPI and YoY inflation caps/floors."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class CPICapFloor:
    """CPI (zero-coupon) inflation cap or floor.

    Parameters
    ----------
    option_type : 1 for cap, -1 for floor
    notional : notional amount
    base_cpi : base CPI at inception
    strike : strike inflation rate
    maturity : time to maturity (years)
    """
    option_type: int  # 1 = cap, -1 = floor
    notional: float
    base_cpi: float
    strike: float
    maturity: float

    def payoff(self, cpi_at_maturity: float) -> float:
        """Terminal payoff of CPI cap/floor."""
        realized_rate = cpi_at_maturity / self.base_cpi - 1.0
        if self.option_type == 1:
            return self.notional * jnp.maximum(realized_rate - self.strike, 0.0)
        return self.notional * jnp.maximum(self.strike - realized_rate, 0.0)


@dataclass(frozen=True)
class YoYInflationCapFloor:
    """Year-on-year inflation cap or floor.

    Parameters
    ----------
    option_type : 1 for cap, -1 for floor
    notional : notional amount
    strikes : array of strikes per period
    payment_times : array of payment times (year fracs)
    accrual_fractions : day count fractions for each period
    """
    option_type: int  # 1 = cap, -1 = floor
    notional: float
    strikes: jnp.ndarray
    payment_times: jnp.ndarray
    accrual_fractions: jnp.ndarray

    @property
    def n_periods(self) -> int:
        return self.payment_times.shape[0]


def make_yoy_inflation_cap(notional, strike, maturity, frequency=1):
    """Create a year-on-year inflation cap."""
    n = int(maturity * frequency)
    times = jnp.array([(i + 1) / frequency for i in range(n)], dtype=jnp.float64)
    fracs = jnp.full(n, 1.0 / frequency, dtype=jnp.float64)
    strikes = jnp.full(n, strike, dtype=jnp.float64)
    return YoYInflationCapFloor(
        option_type=1, notional=notional, strikes=strikes,
        payment_times=times, accrual_fractions=fracs,
    )


def make_yoy_inflation_floor(notional, strike, maturity, frequency=1):
    """Create a year-on-year inflation floor."""
    n = int(maturity * frequency)
    times = jnp.array([(i + 1) / frequency for i in range(n)], dtype=jnp.float64)
    fracs = jnp.full(n, 1.0 / frequency, dtype=jnp.float64)
    strikes = jnp.full(n, strike, dtype=jnp.float64)
    return YoYInflationCapFloor(
        option_type=-1, notional=notional, strikes=strikes,
        payment_times=times, accrual_fractions=fracs,
    )
