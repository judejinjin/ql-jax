"""Partial-time and soft barrier options."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass(frozen=True)
class PartialTimeBarrierOption:
    """Partial-time barrier option.

    The barrier is only monitored during part of the option's life.

    Parameters
    ----------
    strike : strike price
    barrier : barrier level
    is_call : True for call
    is_knock_in : True for knock-in, False for knock-out
    barrier_above : True for up barrier
    barrier_start : start of barrier monitoring (0 = from inception)
    barrier_end : end of barrier monitoring (maturity = full monitoring)
    maturity : time to maturity
    rebate : rebate paid if knocked out
    """
    strike: float
    barrier: float
    is_call: bool
    is_knock_in: bool
    barrier_above: bool
    barrier_start: float
    barrier_end: float
    maturity: float
    rebate: float = 0.0


@dataclass(frozen=True)
class SoftBarrierOption:
    """Soft barrier option.

    Instead of instantaneous knock-in/out, the barrier effect
    is smoothed over a range [barrier - width, barrier + width].

    Parameters
    ----------
    strike : strike price
    barrier : barrier level
    is_call : True for call
    is_knock_in : True for knock-in
    barrier_above : True for up barrier
    maturity : time to maturity
    rebate : rebate
    """
    strike: float
    barrier: float
    is_call: bool
    is_knock_in: bool
    barrier_above: bool
    maturity: float
    rebate: float = 0.0
